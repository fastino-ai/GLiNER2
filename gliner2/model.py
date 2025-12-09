"""
GLiNER2 Extractor Model with Optimized Batch Processing

This module contains the core Extractor model that accepts PreprocessedBatch
directly for efficient GPU-only forward passes.
"""

import os
import tempfile
from typing import Dict, List, Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
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


class ExtractorConfig(PretrainedConfig):
    """Configuration for the Extractor model."""
    model_type = "extractor"

    def __init__(
            self,
            model_name: str = "bert-base-uncased",
            max_width: int = 8,
            counting_layer: str = "count_lstm",
            token_pooling: str = "first",
            **kwargs
    ):
        super().__init__(**kwargs)
        self.model_name = model_name
        self.max_width = max_width
        self.counting_layer = counting_layer
        self.token_pooling = token_pooling


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
        if encoder_config is not None:
            self.encoder = AutoModel.from_config(encoder_config, trust_remote_code=True)
        else:
            self.encoder = AutoModel.from_pretrained(config.model_name, trust_remote_code=True)

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

        self._print_config(config)

    def _print_config(self, config):
        print("=" * 60)
        print("🧠 Model Configuration")
        print("=" * 60)
        print(f"Encoder model      : {config.model_name}")
        print(f"Counting layer     : {config.counting_layer}")
        print(f"Token pooling      : {config.token_pooling}")
        print("=" * 60)

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
        batch = batch.to(device)

        # Encode batch through transformer
        all_token_embs, all_schema_embs = self._encode_batch(batch)

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
                    device=device
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
            device: torch.device
    ) -> Dict[str, torch.Tensor]:
        """
        Compute all losses for a single sample.

        Args:
            token_embeddings: (text_len, hidden) text token embeddings
            embs_per_schema: List of schema embeddings
            task_types: Task type for each schema
            structure_labels: Labels for each schema
            device: Computation device

        Returns:
            Dict with classification, structure, and count losses
        """
        cls_loss = torch.tensor(0.0, device=device)
        struct_loss = torch.tensor(0.0, device=device)
        count_loss = torch.tensor(0.0, device=device)

        # Compute span representations if needed
        has_span_task = any(t != "classifications" for t in task_types)
        span_info = None
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

        spans_idx = []
        for i in range(text_length):
            for j in range(self.max_width):
                if i + j < text_length:
                    spans_idx.append((i, i + j))
                else:
                    spans_idx.append((-1, -1))

        spans_idx = torch.tensor([spans_idx], dtype=torch.long, device=device)

        # Mask invalid spans
        span_mask = (spans_idx[:, :, 0] == -1) | (spans_idx[:, :, 1] == -1)

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
        """Load model from Hugging Face Hub or local directory."""
        from huggingface_hub import hf_hub_download

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
        return model

    def save_pretrained(self, save_directory: str, **kwargs):
        """Save model to directory."""
        os.makedirs(save_directory, exist_ok=True)
        self.config.save_pretrained(save_directory)

        encoder_config_path = os.path.join(save_directory, "encoder_config")
        os.makedirs(encoder_config_path, exist_ok=True)
        self.encoder.config.save_pretrained(encoder_config_path)

        model_path = os.path.join(save_directory, "model.safetensors")
        save_file(self.state_dict(), model_path)

        self.processor.tokenizer.save_pretrained(save_directory)