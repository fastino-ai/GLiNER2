"""
GLiNER2 Extractor Model with Optimized Batch Processing

This module contains the core Extractor model that accepts PreprocessedBatch
directly for efficient GPU-only forward passes.
"""

import importlib
import logging
import os
import tempfile
from typing import Dict, List, Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)
from gliner.modeling.span_rep import SpanRepLayer
from gliner2.layers import CountLSTMoE, CountLSTM, create_mlp, CountLSTMv2
from gliner2.processor import SchemaTransformer, PreprocessedBatch, SamplingConfig
from safetensors.torch import save_file, load_file
from transformers import (
    PretrainedConfig,
    PreTrainedModel,
    AutoModel,
    AutoConfig,
    AutoTokenizer,
)

IS_FLASHDEBERTA = importlib.util.find_spec("flashdeberta") is not None
if IS_FLASHDEBERTA:
    from flashdeberta import FlashDebertaV2Model


class ExtractorConfig(PretrainedConfig):
    """Configuration for the Extractor model."""
    model_type = "extractor"

    def __init__(
            self,
            model_name: str = "bert-base-uncased",
            max_width: int = 8,
            counting_layer: str = "count_lstm",
            token_pooling: str = "first",
            max_len: int = None,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.model_name = model_name
        self.max_width = max_width
        self.counting_layer = counting_layer
        self.token_pooling = token_pooling
        self.max_len = max_len


class Extractor(PreTrainedModel):
    """
    GLiNER2 Extractor Model.

    This model accepts PreprocessedBatch for efficient training.
    Use processor.collate_fn_train() to create batches.

    Example:
        >>> processor = SchemaTransformer(model_name)
        >>> model = Extractor.from_pretrained(repo_id)
        >>> 
        >>> # Training
        >>> loader = DataLoader(dataset, collate_fn=processor.collate_fn_train)
        >>> for batch in loader:
        ...     batch = batch.to(device)
        ...     loss = model(batch)["total_loss"]
    """
    config_class = ExtractorConfig

    def __init__(self, config: ExtractorConfig, encoder_config=None, tokenizer=None):
        super().__init__(config)
        self.config = config
        self.max_width = config.max_width

        # Initialize processor
        if tokenizer is not None:
            self.processor = SchemaTransformer(
                tokenizer=tokenizer,
                token_pooling=config.token_pooling
            )
        else:
            self.processor = SchemaTransformer(
                config.model_name,
                token_pooling=config.token_pooling
            )

        # Load encoder
        self.encoder = self._load_encoder(config.model_name, encoder_config)

        self.encoder.resize_token_embeddings(len(self.processor.tokenizer))
        self.hidden_size = self.encoder.config.hidden_size

        # Span representation layer
        self.span_rep = SpanRepLayer(
            span_mode="markerV0",
            hidden_size=self.hidden_size,
            max_width=self.max_width,
            dropout=0.1,
        )

        # Classifier for classification tasks
        self.classifier = create_mlp(
            input_dim=self.hidden_size,
            intermediate_dims=[self.hidden_size * 2],
            output_dim=1,
            dropout=0.,
            activation="relu",
            add_layer_norm=False
        )

        # Count prediction layer
        self.count_pred = create_mlp(
            input_dim=self.hidden_size,
            intermediate_dims=[self.hidden_size * 2],
            output_dim=20,
            dropout=0.,
            activation="relu",
            add_layer_norm=False
        )

        # Count embedding module
        if config.counting_layer == "count_lstm":
            self.count_embed = CountLSTM(self.hidden_size)
        elif config.counting_layer == "count_lstm_moe":
            self.count_embed = CountLSTMoE(
                hidden_size=self.hidden_size,
                n_experts=4,
                ffn_mult=2,
                dropout=0.1
            )
        elif config.counting_layer == "count_lstm_v2":
            self.count_embed = CountLSTMv2(hidden_size=self.hidden_size)

        # LoRA adapter state
        self._lora_layers = {}
        self._adapter_config = None

        self._print_config(config)

    def _print_config(self, config):
        _attn_impl, _ = self._resolve_attn_impl()
        _actual = getattr(self.encoder.config, "_attn_implementation", None)
        print("=" * 60)
        print("🧠 Model Configuration")
        print("=" * 60)
        print(f"Encoder model      : {config.model_name}")
        print(f"Counting layer     : {config.counting_layer}")
        print(f"Token pooling      : {config.token_pooling}")
        print(f"Attention backend  : requested={_attn_impl!r}, actual={_actual!r}")
        print("=" * 60)

    @staticmethod
    def _resolve_attn_impl() -> tuple:
        """Resolve attention implementation and dtype from environment.

        FLASH_ATTN=1 (or "true"/"yes") enables flash_attention_2.
        USE_FLASHDEBERTA=1 enables the FlashDeberta backend for DeBERTa-v2 models.

        Returns:
            (attn_impl, dtype) — both may be None when flash attention is off.
        """
        _flash = os.environ.get("FLASH_ATTN", "").lower() in ("1", "true", "yes")
        if _flash:
            _attn_impl = "flash_attention_2"
            _dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        else:
            _attn_impl = None
            _dtype = None
        return _attn_impl, _dtype

    @staticmethod
    def _load_encoder(model_name: str, encoder_config=None) -> nn.Module:
        """Load the transformer encoder, using optimized backends when available.

        Checks for FlashDeberta support when the encoder is DebertaV2-based
        (USE_FLASHDEBERTA env var), and for generic flash_attention_2 support
        via the FLASH_ATTN env var.

        Args:
            model_name: Name or path of the pretrained model.
            encoder_config: Optional pre-loaded encoder config. If provided,
                the model is initialized from config; otherwise from pretrained.

        Returns:
            The initialized encoder module.
        """
        _attn_impl, _dtype = Extractor._resolve_attn_impl()

        use_flashdeberta = (
            IS_FLASHDEBERTA
            and os.environ.get("USE_FLASHDEBERTA", "")
        )

        if encoder_config is not None:
            config_name = encoder_config.__class__.__name__
            if config_name == "DebertaV2Config" and use_flashdeberta:
                print("Using FlashDeberta backend.")
                return FlashDebertaV2Model(encoder_config)
            return AutoModel.from_config(encoder_config, trust_remote_code=True)

        pretrained_config = AutoConfig.from_pretrained(model_name)
        config_name = pretrained_config.__class__.__name__
        if config_name == "DebertaV2Config" and use_flashdeberta:
            print("Using FlashDeberta backend.")
            return FlashDebertaV2Model.from_pretrained(model_name)

        return AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            attn_implementation=_attn_impl,
            **({"torch_dtype": _dtype} if _dtype is not None else {}),
        )

    # =========================================================================
    # Main Forward Pass
    # =========================================================================

    def forward(
            self,
            batch: PreprocessedBatch,
            return_individual_losses: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass on preprocessed batch.

        Args:
            batch: PreprocessedBatch from processor.collate_fn_train()
            return_individual_losses: If True, return per-sample losses

        Returns:
            Dict with:
                - total_loss: Sum of all losses
                - classification_loss: Classification task loss
                - structure_loss: Span extraction loss
                - count_loss: Count prediction loss
                - batch_size: Number of valid samples
        """
        if len(batch) == 0:
            return self._empty_loss_dict()

        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype
        batch = batch.to(device, dtype if dtype != torch.float32 else None)

        # Encode batch through transformer
        all_token_embs, all_schema_embs = self._encode_batch(batch)

        # Batch span rep for samples that need it
        span_samples = []
        for i in range(len(batch)):
            has_span = any(t != "classifications" for t in batch.task_types[i])
            if has_span and all_token_embs[i].numel() > 0:
                span_samples.append(i)

        all_span_info = [None] * len(batch)
        if span_samples:
            span_embs = [all_token_embs[i] for i in span_samples]
            span_results = self.compute_span_rep_batched(span_embs)
            for idx, si in zip(span_samples, span_results):
                all_span_info[idx] = si

        # Compute losses for each sample
        cls_losses = []
        struct_losses = []
        count_losses = []
        individual = []
        valid_samples = 0

        for i in range(len(batch)):
            try:
                sample_losses = self._compute_sample_loss(
                    token_embeddings=all_token_embs[i],
                    embs_per_schema=all_schema_embs[i],
                    task_types=batch.task_types[i],
                    structure_labels=batch.structure_labels[i],
                    device=device,
                    span_info=all_span_info[i]
                )

                cls_losses.append(sample_losses["classification"])
                struct_losses.append(sample_losses["structure"])
                count_losses.append(sample_losses["count"])

                if return_individual_losses:
                    individual.append({
                        "total_loss": (
                                sample_losses["classification"] +
                                sample_losses["structure"] +
                                sample_losses["count"]
                        ).item(),
                        "classification_loss": sample_losses["classification"].item(),
                        "structure_loss": sample_losses["structure"].item(),
                        "count_loss": sample_losses["count"].item(),
                    })

                valid_samples += 1

            except Exception as e:
                print(f"Error processing sample {i}: {e}")
                zero = torch.tensor(0.0, device=device)
                cls_losses.append(zero)
                struct_losses.append(zero)
                count_losses.append(zero)

                if return_individual_losses:
                    individual.append({
                        "total_loss": 0.0,
                        "classification_loss": 0.0,
                        "structure_loss": 0.0,
                        "count_loss": 0.0,
                        "error": str(e)
                    })

        if valid_samples == 0:
            result = self._empty_loss_dict()
            if return_individual_losses:
                result["individual_losses"] = individual
            return result

        # Aggregate losses
        total_cls = torch.stack(cls_losses).sum()
        total_struct = torch.stack(struct_losses).sum()
        total_count = torch.stack(count_losses).sum()
        total_loss = total_cls + total_struct + total_count

        result = {
            "total_loss": total_loss,
            "classification_loss": total_cls,
            "structure_loss": total_struct,
            "count_loss": total_count,
            "batch_size": valid_samples
        }

        if return_individual_losses:
            result["individual_losses"] = individual

        return result

    def _empty_loss_dict(self) -> Dict[str, torch.Tensor]:
        """Return empty loss dictionary."""
        device = next(self.parameters()).device
        return {
            "total_loss": torch.tensor(0.0, device=device, requires_grad=True),
            "classification_loss": torch.tensor(0.0, device=device),
            "structure_loss": torch.tensor(0.0, device=device),
            "count_loss": torch.tensor(0.0, device=device),
            "batch_size": 0
        }

    # =========================================================================
    # Encoding
    # =========================================================================

    def _encode_batch(
            self,
            batch: PreprocessedBatch
    ) -> Tuple[List[torch.Tensor], List[List[torch.Tensor]]]:
        """
        Encode batch through transformer and extract embeddings.

        Args:
            batch: PreprocessedBatch with input_ids and attention_mask

        Returns:
            - all_token_embs: List of (text_len, hidden) per sample
            - all_schema_embs: List of schema embeddings per sample
        """
        # Forward through encoder
        outputs = self.encoder(
            input_ids=batch.input_ids,
            attention_mask=batch.attention_mask
        )
        token_embeddings = outputs.last_hidden_state

        # Extract embeddings using processor
        return self.processor.extract_embeddings_from_batch(
            token_embeddings,
            batch.input_ids,
            batch
        )

    # =========================================================================
    # Loss Computation
    # =========================================================================

    def _compute_sample_loss(
            self,
            token_embeddings: torch.Tensor,
            embs_per_schema: List[List[torch.Tensor]],
            task_types: List[str],
            structure_labels: List[Any],
            device: torch.device,
            span_info: Optional[Dict[str, Any]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute all losses for a single sample.

        Args:
            token_embeddings: (text_len, hidden) text token embeddings
            embs_per_schema: List of schema embeddings
            task_types: Task type for each schema
            structure_labels: Labels for each schema
            device: Computation device
            span_info: Pre-computed span representations (from batched computation).
                       If None, computed on-the-fly for this sample.

        Returns:
            Dict with classification, structure, and count losses
        """
        cls_loss = torch.tensor(0.0, device=device)
        struct_loss = torch.tensor(0.0, device=device)
        count_loss = torch.tensor(0.0, device=device)

        # Compute span representations if needed and not pre-computed
        if span_info is None:
            has_span_task = any(t != "classifications" for t in task_types)
            if has_span_task and token_embeddings.numel() > 0:
                span_info = self.compute_span_rep(token_embeddings)

        all_counts = []
        all_p_embs = []

        for i, task_type in enumerate(task_types):
            if not embs_per_schema[i]:
                continue

            schema_emb = torch.stack(embs_per_schema[i])

            if task_type == "classifications":
                # Classification loss
                cls_embeds = schema_emb[1:]  # Skip [P] token
                logits = self.classifier(cls_embeds).squeeze(-1)
                labels = torch.tensor(structure_labels[i], dtype=torch.float, device=device)
                cls_loss = cls_loss + F.binary_cross_entropy_with_logits(
                    logits, labels, reduction="sum"
                )
            else:
                # Structure loss
                structure = structure_labels[i]

                if structure[0] == 0:
                    # No instances to extract
                    continue

                if span_info is not None:
                    struct_loss = struct_loss + self.compute_struct_loss(
                        span_info["span_rep"],
                        schema_emb,
                        structure,
                        span_info["span_mask"]
                    )

                # Collect for count loss (skip entities)
                if task_type != "entities":
                    all_counts.append(min(structure[0], 19))
                    all_p_embs.append(schema_emb[0])

        # Count loss
        if all_counts and all_p_embs:
            counts = torch.tensor(all_counts, dtype=torch.long, device=device)
            p_embs = torch.stack(all_p_embs)
            count_loss = F.cross_entropy(self.count_pred(p_embs), counts, reduction="sum")

        return {
            "classification": cls_loss,
            "structure": struct_loss,
            "count": count_loss
        }

    # =========================================================================
    # Span Representation
    # =========================================================================

    def compute_span_rep(self, token_embeddings: torch.Tensor) -> Dict[str, Any]:
        """
        Compute span representations for token embeddings.

        Args:
            token_embeddings: (text_len, hidden) token embeddings

        Returns:
            Dict with span_rep, spans_idx, and span_mask
        """
        text_length = len(token_embeddings)
        device = token_embeddings.device

        # Vectorized span index generation
        starts = torch.arange(text_length, device=device).unsqueeze(1).expand(-1, self.max_width)
        offsets = torch.arange(self.max_width, device=device).unsqueeze(0)
        ends = starts + offsets
        valid = ends < text_length

        starts_flat = starts.reshape(-1)
        ends_flat = ends.reshape(-1)
        invalid = ~valid.reshape(-1)
        starts_flat = torch.where(invalid, torch.tensor(-1, device=device), starts_flat)
        ends_flat = torch.where(invalid, torch.tensor(-1, device=device), ends_flat)
        spans_idx = torch.stack([starts_flat, ends_flat], dim=-1).unsqueeze(0)

        # Mask invalid spans
        span_mask = invalid.unsqueeze(0)

        # Replace invalid with (0, 0) for safe indexing
        safe_spans = torch.where(
            span_mask.unsqueeze(-1),
            torch.zeros_like(spans_idx),
            spans_idx
        )

        # Compute span representations
        span_rep = self.span_rep(
            token_embeddings.unsqueeze(0),
            safe_spans
        ).squeeze(0)

        return {
            "span_rep": span_rep,
            "spans_idx": spans_idx,
            "span_mask": span_mask
        }

    def compute_span_rep_batched(
            self,
            token_embs_list: List[torch.Tensor],
    ) -> List[Dict[str, Any]]:
        """
        Batch span rep computation across multiple samples.

        Pads token embeddings to the max text length, builds span indices once
        for the padded length, then runs a single forward pass through
        SpanMarkerV0. The results are unpacked per-sample with correct shapes.

        Bit-identical to calling compute_span_rep per sample because
        SpanMarkerV0 uses pointwise MLPs + gather (no cross-position mixing),
        and we only use valid-position outputs.

        Args:
            token_embs_list: List of (text_len_i, hidden) tensors

        Returns:
            List of dicts with span_rep, spans_idx, span_mask per sample
        """
        if not token_embs_list:
            return []

        device = token_embs_list[0].device
        text_lengths = [len(t) for t in token_embs_list]
        max_text_len = max(text_lengths)
        batch_size = len(token_embs_list)
        hidden = token_embs_list[0].shape[-1]

        # Pad variable-length list into a single dense tensor (stays eager
        # so torch.compile doesn't guard on per-element shapes).
        padded = torch.zeros(batch_size, max_text_len, hidden,
                             device=device, dtype=token_embs_list[0].dtype)
        for i, emb in enumerate(token_embs_list):
            padded[i, :text_lengths[i]] = emb

        text_len_t = torch.tensor(text_lengths, device=device)

        # Dense tensor path — safe for torch.compile
        span_rep, safe_spans, span_mask = self._compute_span_rep_core(
            padded, text_len_t,
        )

        # Unpack per-sample results (Python dicts, stays eager)
        results = []
        for i in range(batch_size):
            tl = text_lengths[i]
            n_spans = tl * self.max_width  # actual number of spans for this sample
            results.append({
                "span_rep": span_rep[i, :tl, :, :],
                "spans_idx": safe_spans[i:i+1, :n_spans, :],
                "span_mask": span_mask[i:i+1, :n_spans],
            })
        return results

    def _compute_span_rep_core(
            self,
            padded: torch.Tensor,
            text_len_t: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Dense-tensor span computation (compile-friendly).

        Args:
            padded: (batch, max_text_len, hidden) — padded token embeddings
            text_len_t: (batch,) — actual text lengths per sample

        Returns:
            span_rep: (batch, max_text_len, max_width, hidden)
            safe_spans: (batch, N, 2)
            span_mask: (batch, N) — True for invalid
        """
        batch_size, max_text_len, _ = padded.shape
        device = padded.device

        # Vectorized span indices for max_text_len
        starts = torch.arange(max_text_len, device=device).unsqueeze(1).expand(-1, self.max_width)
        offsets = torch.arange(self.max_width, device=device).unsqueeze(0)
        ends = starts + offsets  # (max_text_len, max_width)

        # Per-sample validity: span (i, i+j) valid iff i+j < text_lengths[sample]
        ends_expanded = ends.unsqueeze(0).expand(batch_size, -1, -1)
        valid = ends_expanded < text_len_t.view(-1, 1, 1)

        starts_flat = starts.reshape(-1).unsqueeze(0).expand(batch_size, -1)
        ends_flat = ends.reshape(-1).unsqueeze(0).expand(batch_size, -1)
        valid_flat = valid.reshape(batch_size, -1)

        safe_starts = torch.where(valid_flat, starts_flat, torch.zeros_like(starts_flat))
        safe_ends = torch.where(valid_flat, ends_flat, torch.zeros_like(ends_flat))
        safe_spans = torch.stack([safe_starts, safe_ends], dim=-1)  # (batch, N, 2)
        span_mask = ~valid_flat  # (batch, N) — True for invalid

        # Single batched forward pass through SpanMarkerV0
        span_rep = self.span_rep(padded, safe_spans)  # (batch, max_text_len, max_width, hidden)

        return span_rep, safe_spans, span_mask

    def compute_struct_loss(
            self,
            span_rep: torch.Tensor,
            schema_emb: torch.Tensor,
            structure: List[Any],
            span_mask: torch.Tensor,
            masking_rate: float = 0.5
    ) -> torch.Tensor:
        """
        Compute structure extraction loss with negative span masking.

        Args:
            span_rep: (num_spans, hidden) span representations
            schema_emb: (num_fields + 1, hidden) schema embeddings
            structure: [count, spans] structure labels
            span_mask: (1, num_spans) mask for invalid spans
            masking_rate: Probability of masking negative spans

        Returns:
            Structure loss tensor
        """
        gold_count = min(structure[0], 19)
        struct_proj = self.count_embed(schema_emb[1:], gold_count)
        scores = torch.einsum('lkd,bpd->bplk', span_rep, struct_proj)

        # Create label tensor
        labs = torch.zeros_like(scores)

        for i in range(gold_count):
            gold_spans = structure[1][i]
            for k, span in enumerate(gold_spans):
                if span is None or span == (-1, -1):
                    continue
                if isinstance(span, tuple):
                    start, end = span
                    width = end - start
                    if 0 <= start < scores.shape[2] and 0 <= width < scores.shape[3]:
                        labs[i, k, start, width] = 1
                elif isinstance(span, list):
                    for sub in span:
                        if sub is None or sub == (-1, -1):
                            continue
                        start, end = sub
                        width = end - start
                        if 0 <= start < scores.shape[2] and 0 <= width < scores.shape[3]:
                            labs[i, k, start, width] = 1

        # Apply negative masking
        if masking_rate > 0.0 and self.training:
            negative = (labs == 0)
            random_mask = torch.rand_like(scores) < masking_rate
            to_mask = negative & random_mask
            loss_mask = (~to_mask).float()
        else:
            loss_mask = torch.ones_like(scores)

        # Compute masked loss
        loss = F.binary_cross_entropy_with_logits(scores, labs, reduction="none")
        loss = loss * loss_mask
        loss = loss.view(loss.shape[0], loss.shape[1], -1) * (~span_mask[0]).float()

        return loss.sum()

    # =========================================================================
    # Hugging Face Methods
    # =========================================================================

    def push_to_hub(self, repo_id: str, private: bool = True):
        """Push model to Hugging Face Hub."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            self.save_pretrained(tmp_dir)
            super().push_to_hub(repo_id=repo_id, save_dir=tmp_dir, private=private)
            self.processor.tokenizer.push_to_hub(repo_id)

    @classmethod
    def from_pretrained(cls, repo_or_dir: str, **kwargs):
        """
        Load model from Hugging Face Hub or local directory.

        Args:
            repo_or_dir: HuggingFace repo ID or local directory path.
            quantize: If True, convert model to fp16 after loading.
            compile: If True, torch.compile the encoder and span-rep
                with ``dynamic=True`` for fused GPU kernels.
            map_location: Device to load the model onto (e.g. "cpu", "cuda").
            **kwargs: Additional keyword arguments.

        To use a LoRA adapter:
            1. Load the base model first
            2. Then load the adapter using model.load_adapter()

        Example:
            model = Extractor.from_pretrained("base-model-name")
            model.load_adapter("path/to/adapter")
        """
        from huggingface_hub import hf_hub_download

        quantize = kwargs.pop("quantize", False)
        compile_model = kwargs.pop("compile", False)
        map_location = kwargs.pop("map_location", None)

        def download_or_local(repo, filename):
            if os.path.isdir(repo):
                return os.path.join(repo, filename)
            return hf_hub_download(repo, filename)

        config_path = download_or_local(repo_or_dir, "config.json")
        config = cls.config_class.from_pretrained(config_path)

        encoder_config_path = download_or_local(repo_or_dir, "encoder_config/config.json")
        encoder_config = AutoConfig.from_pretrained(encoder_config_path)

        tokenizer = AutoTokenizer.from_pretrained(repo_or_dir)
        model = cls(config, encoder_config=encoder_config, tokenizer=tokenizer)

        # Load weights
        try:
            model_path = download_or_local(repo_or_dir, "model.safetensors")
            state_dict = load_file(model_path)
        except Exception:
            model_path = download_or_local(repo_or_dir, "pytorch_model.bin")
            state_dict = torch.load(model_path, map_location="cpu")

        # Handle embedding size mismatch
        try:
            saved_emb = state_dict["encoder.embeddings.word_embeddings.weight"]
            model_emb = model.encoder.embeddings.word_embeddings.weight
            if saved_emb.shape[0] != model_emb.shape[0]:
                extra = model_emb.shape[0] - saved_emb.shape[0]
                state_dict["encoder.embeddings.word_embeddings.weight"] = torch.cat([
                    saved_emb,
                    torch.randn(extra, saved_emb.shape[1]) * 0.02
                ], dim=0)
        except KeyError:
            pass

        model.load_state_dict(state_dict)

        if map_location is not None:
            model = model.to(map_location)

        if quantize:
            model.quantize()

        if compile_model:
            model.compile()

        return model

    # =========================================================================
    # Quantization
    # =========================================================================

    def quantize(self) -> 'Extractor':
        """Convert all model parameters to float16 for faster inference.

        Returns:
            self (for method chaining).

        Example::

            model = GLiNER2.from_pretrained("fastino/gliner2-base-v1")
            model.quantize()

            model = GLiNER2.from_pretrained("fastino/gliner2-base-v1",
                                            map_location="cuda")
            model.quantize()
        """
        self.half()
        logger.info("Converted model to fp16")
        return self

    # =========================================================================
    # torch.compile
    # =========================================================================

    def compile(self) -> 'Extractor':
        """Compile tensor subgraphs with ``torch.compile(dynamic=True)``.

        Three components are compiled (all verified 0 graph breaks):

        - **encoder** (DeBERTa backbone)
        - **_compute_span_rep_core** (span index + MLP)
        - **count_embed** (CompileSafeGRU + DownscaledTransformer)

        The list-of-tensors padding in ``compute_span_rep_batched`` and the
        per-sample Python decode path are left in eager mode.

        The first call triggers tracing and is slow; subsequent calls
        with similar shapes use the cached compiled graph.

        Returns:
            self (for method chaining).

        Example::

            model = GLiNER2.from_pretrained("fastino/gliner2-base-v1",
                                            map_location="cuda")
            model.compile()
        """
        self.encoder = torch.compile(self.encoder, dynamic=True)
        self._compute_span_rep_core = torch.compile(
            self._compute_span_rep_core, dynamic=True,
        )
        self.count_embed = torch.compile(self.count_embed, dynamic=True)
        logger.info("Compiled encoder, span-rep, and count-embed with torch.compile(dynamic=True)")
        return self

    def load_adapter(self, adapter_path: str) -> 'Extractor':
        """
        Load a LoRA adapter onto this model.

        If an adapter is already loaded, it will be unloaded first.

        Args:
            adapter_path: Path to adapter directory

        Returns:
            self for method chaining
            
        Example:
            model.load_adapter("./legal_adapter")
            results = model.extract_entities(text, entities)
        """
        from gliner2.training.lora import load_lora_adapter, LoRAAdapterConfig
        
        # Load adapter config
        config = LoRAAdapterConfig.load(adapter_path)
        
        self._lora_layers = load_lora_adapter(self, adapter_path, auto_unload=True)
        self._adapter_config = config
        return self
    
    def unload_adapter(self) -> 'Extractor':
        """
        Unload current LoRA adapter, restoring base model.
        
        Returns:
            self for method chaining
        """
        from gliner2.training.lora import unload_lora_adapter
        
        if self._lora_layers:
            unload_lora_adapter(self)
            self._lora_layers = {}
            self._adapter_config = None
        return self
    
    def merge_lora(self) -> 'Extractor':
        """
        Merge LoRA weights into base model and remove adapter structure.
        
        After calling this, the model will have standard Linear layers with
        merged weights. LoRA adapters are permanently removed.
        
        Returns:
            self for method chaining
            
        Raises:
            ValueError: If no adapter is loaded
            
        Example:
            model.load_adapter("./my_adapter")
            model.merge_lora()  # Now model has merged weights, no LoRA
            model.save_pretrained("./merged_model")
        """
        if not self._lora_layers:
            raise ValueError("No adapter loaded. Nothing to merge.")
        
        from gliner2.training.lora import merge_lora_weights
        merge_lora_weights(self)
        self._lora_layers = {}
        self._adapter_config = None
        return self
    
    def save_adapter(self, save_path: str) -> None:
        """
        Save only the LoRA adapter (not full model).
        
        Args:
            save_path: Directory to save adapter
            
        Raises:
            ValueError: If no adapter is loaded
        """
        if not self._lora_layers:
            raise ValueError("No adapter loaded. Use save_pretrained for full model.")
        
        from gliner2.training.lora import save_lora_adapter
        save_lora_adapter(self, save_path)
    
    @property
    def has_adapter(self) -> bool:
        """Check if an adapter is currently loaded."""
        return bool(self._lora_layers)
    
    @property
    def adapter_config(self):
        """Get config of loaded adapter, or None."""
        return self._adapter_config
    
    def save_pretrained(
        self, 
        save_directory: str,
        save_adapter_only: bool = False,
        merge_lora: bool = True,
        **kwargs
    ):
        """
        Save model to directory.
        
        Args:
            save_directory: Where to save
            save_adapter_only: If True and adapter loaded, save only adapter
            merge_lora: If True and LoRA active, merge LoRA weights into base
                       model and remove adapter structure before saving.
                       WARNING: This permanently removes LoRA from the model instance.
        """
        if save_adapter_only:
            if not self._lora_layers:
                raise ValueError("save_adapter_only=True but no adapter loaded")
            self.save_adapter(save_directory)
            return
        
        # Handle LoRA merging if requested
        if merge_lora and self._lora_layers:
            self.merge_lora()
        
        # Original save logic
        os.makedirs(save_directory, exist_ok=True)
        self.config.save_pretrained(save_directory)

        encoder_config_path = os.path.join(save_directory, "encoder_config")
        os.makedirs(encoder_config_path, exist_ok=True)
        self.encoder.config.save_pretrained(encoder_config_path)

        model_path = os.path.join(save_directory, "model.safetensors")
        save_file(self.state_dict(), model_path)

        self.processor.tokenizer.save_pretrained(save_directory)