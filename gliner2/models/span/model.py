"""
Span (legacy) architecture core model.

This module hosts the concrete ``SpanExtractorModel`` implementation. It was
moved here from ``gliner2.model`` (which now re-exports it as a compatibility
shim) so the span architecture lives under its architecture-scoped path.

Module attribute names (``encoder``, ``span_rep``, ``classifier``,
``count_pred``, ``count_embed``) are preserved verbatim so serialized
state-dict keys and numerical parity with legacy checkpoints are unchanged.

This model accepts PreprocessedBatch directly for efficient GPU-only forward
passes.
"""

import logging
import os
import tempfile
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Any, Optional, Tuple

if TYPE_CHECKING:
    from peft import PeftModel

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)
from gliner2.layers import CountLSTMoE, CountLSTM, create_mlp, CountLSTMv2, SpanRepLayer
from gliner2.processor import SchemaTransformer, PreprocessedBatch
from safetensors.torch import save_file
from transformers import AutoConfig

# ``ExtractorConfig`` lives in ``gliner2.configuration`` so both span and
# boundary architectures share one validated config.
from gliner2.configuration import ExtractorConfig
from gliner2.models.base import BaseExtractorModel, load_extractor_tokenizer
from gliner2.models.loading import (
    apply_post_load_options,
    checkpoint_file,
    load_checkpoint_state_dict,
    reconcile_encoder_embeddings,
    split_load_kwargs,
)


class SpanExtractorModel(BaseExtractorModel):
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
    architecture = "span"

    def task_module_names(self) -> tuple:
        """Names of task-specific (non-encoder) submodules for LoRA targeting."""
        return ("span_rep", "classifier", "count_embed", "count_pred")

    def __init__(
        self,
        config: ExtractorConfig,
        encoder_config=None,
        tokenizer=None,
        use_flashdeberta: Optional[bool] = None,
        word_splitter=None,
    ):
        super().__init__(config)
        self.config = config
        self.max_width = config.max_width

        # Initialize processor
        if tokenizer is not None:
            self.processor = SchemaTransformer(
                tokenizer=tokenizer,
                token_pooling=config.token_pooling,
                word_splitter=word_splitter,
            )
        else:
            self.processor = SchemaTransformer(
                config.model_name,
                token_pooling=config.token_pooling,
                word_splitter=word_splitter,
            )

        # Load encoder
        self.encoder = self._load_encoder(
            config.model_name,
            encoder_config,
            getattr(config, "attn_implementation", "sdpa"),
            use_flashdeberta=use_flashdeberta,
        )

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
        dtype = next(self.parameters()).dtype
        batch = batch.to(device, dtype if dtype != torch.float32 else None)

        # Encode batch through transformer
        all_token_embs, all_schema_embs = self._encode_batch(batch)

        # Build count-aware structural queries for every span task, concatenate
        # them per sample, and score them directly without materializing
        # [B, L, W, H] span representations.  ``mask_invalid=False`` keeps BCE
        # inputs finite; the loss applies the legacy validity mask afterwards.
        all_span_scores = self._compute_training_span_scores_batched(
            all_token_embs,
            all_schema_embs,
            batch.task_types,
            batch.structure_labels,
        )

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
                    span_scores_per_schema=all_span_scores[i]
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
            span_scores_per_schema: Optional[List[Optional[torch.Tensor]]] = None
    ) -> Dict[str, torch.Tensor]:
        """Compute all losses for a single sample from direct span logits."""
        cls_loss = torch.tensor(0.0, device=device)
        struct_loss = torch.tensor(0.0, device=device)
        count_loss = torch.tensor(0.0, device=device)
        span_scores_per_schema = span_scores_per_schema or [None] * len(task_types)

        all_counts = []
        all_p_embs = []

        for i, task_type in enumerate(task_types):
            if not embs_per_schema[i]:
                continue

            schema_emb = torch.stack(embs_per_schema[i])

            if task_type == "classifications":
                cls_embeds = schema_emb[1:]  # Skip [P] token
                logits = self.classifier(cls_embeds).squeeze(-1)
                labels = torch.tensor(structure_labels[i], dtype=torch.float, device=device)
                cls_loss = cls_loss + F.binary_cross_entropy_with_logits(
                    logits, labels, reduction="sum"
                )
            else:
                structure = structure_labels[i]
                if structure[0] == 0:
                    continue

                scores = span_scores_per_schema[i]
                if scores is not None:
                    struct_loss = struct_loss + self.compute_struct_loss_from_scores(
                        scores, structure
                    )

                # Preserve legacy count-loss behavior (entities are skipped).
                if task_type != "entities":
                    all_counts.append(min(structure[0], 19))
                    all_p_embs.append(schema_emb[0])

        if all_counts and all_p_embs:
            counts = torch.tensor(all_counts, dtype=torch.long, device=device)
            p_embs = torch.stack(all_p_embs)
            count_loss = F.cross_entropy(self.count_pred(p_embs), counts, reduction="sum")

        return {
            "classification": cls_loss,
            "structure": struct_loss,
            "count": count_loss
        }

    def _compute_training_span_scores_batched(
            self,
            token_embs_list: List[torch.Tensor],
            all_schema_embs: List[List[List[torch.Tensor]]],
            all_task_types: List[List[str]],
            all_structure_labels: List[List[Any]],
    ) -> List[List[Optional[torch.Tensor]]]:
        """Directly score all training span tasks with one token-side pass/sample.

        Each schema's ``count_embed`` output is flattened to query vectors.
        Queries from all span schemas in a sample are concatenated, passed once
        through the factorized scorer, then restored to ``[count, fields, L, W]``.
        """
        batch_size = len(token_embs_list)
        outputs: List[List[Optional[torch.Tensor]]] = [
            [None] * len(all_task_types[i]) for i in range(batch_size)
        ]
        query_chunks: List[List[torch.Tensor]] = [[] for _ in range(batch_size)]
        query_slices: Dict[Tuple[int, int], Tuple[int, int, int, int]] = {}
        offsets = [0] * batch_size

        for sample_idx in range(batch_size):
            if token_embs_list[sample_idx].numel() == 0:
                continue
            for task_idx, task_type in enumerate(all_task_types[sample_idx]):
                if task_type == "classifications":
                    continue
                if task_idx >= len(all_schema_embs[sample_idx]):
                    continue
                if not all_schema_embs[sample_idx][task_idx]:
                    continue

                structure = all_structure_labels[sample_idx][task_idx]
                if structure[0] == 0:
                    continue

                schema_emb = torch.stack(all_schema_embs[sample_idx][task_idx])
                gold_count = min(structure[0], 19)
                struct_proj = self.count_embed(schema_emb[1:], gold_count)
                if struct_proj.numel() == 0:
                    continue

                count_dim, field_dim, hidden = struct_proj.shape
                flat_queries = struct_proj.reshape(count_dim * field_dim, hidden)
                start = offsets[sample_idx]
                stop = start + flat_queries.shape[0]
                offsets[sample_idx] = stop
                query_chunks[sample_idx].append(flat_queries)
                query_slices[(sample_idx, task_idx)] = (
                    start, stop, count_dim, field_dim
                )

        active_indices = [i for i, chunks in enumerate(query_chunks) if chunks]
        if not active_indices:
            return outputs

        direct_scores = self.compute_span_scores_batched(
            [token_embs_list[i] for i in active_indices],
            [torch.cat(query_chunks[i], dim=0) for i in active_indices],
            mask_invalid=False,
        )
        scores_by_sample = dict(zip(active_indices, direct_scores))

        for (sample_idx, task_idx), (start, stop, count_dim, field_dim) in query_slices.items():
            task_scores = scores_by_sample[sample_idx][start:stop]
            outputs[sample_idx][task_idx] = task_scores.reshape(
                count_dim, field_dim, task_scores.shape[-2], task_scores.shape[-1]
            )

        return outputs

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

    def compute_span_scores_batched(
            self,
            token_embs_list: List[torch.Tensor],
            query_embs_list: List[torch.Tensor],
            *,
            mask_invalid: bool = True,
    ) -> List[torch.Tensor]:
        """Score variable-length token batches directly against queries."""
        if len(token_embs_list) != len(query_embs_list):
            raise ValueError("token/query batch sizes must match")
        if not token_embs_list:
            return []

        device = token_embs_list[0].device
        dtype = token_embs_list[0].dtype
        text_lengths = [len(t) for t in token_embs_list]
        query_lengths = [len(q) for q in query_embs_list]
        max_text_len = max(text_lengths)
        max_queries = max(query_lengths, default=0)
        batch_size = len(token_embs_list)
        hidden = token_embs_list[0].shape[-1]

        padded_tokens = torch.zeros(
            batch_size, max_text_len, hidden, device=device, dtype=dtype
        )
        padded_queries = torch.zeros(
            batch_size, max_queries, hidden, device=device, dtype=dtype
        )
        for i, (tokens, queries) in enumerate(zip(token_embs_list, query_embs_list)):
            padded_tokens[i, :text_lengths[i]] = tokens
            padded_queries[i, :query_lengths[i]] = queries

        text_len_t = torch.tensor(text_lengths, device=device)
        logits = self._compute_span_scores_core(
            padded_tokens, padded_queries, text_len_t, mask_invalid
        )
        return [
            logits[i, :query_lengths[i], :text_lengths[i], :]
            for i in range(batch_size)
        ]

    def _compute_span_scores_core(
            self,
            padded: torch.Tensor,
            queries: torch.Tensor,
            text_len_t: torch.Tensor,
            mask_invalid: bool = True,
    ) -> torch.Tensor:
        """Dense direct-score core kept separate for ``torch.compile``."""
        return self.span_rep.score_queries(
            padded, queries, text_len_t, mask_invalid=mask_invalid
        )

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

    def compute_struct_loss_from_scores(
            self,
            scores: torch.Tensor,
            structure: List[Any],
            masking_rate: float = 0.5
    ) -> torch.Tensor:
        """Compute the legacy structure loss from direct raw span logits."""
        gold_count = min(structure[0], 19)
        if scores.shape[0] != gold_count:
            raise ValueError(
                f"score count dimension {scores.shape[0]} != gold count {gold_count}"
            )

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

        if masking_rate > 0.0 and self.training:
            negative = (labs == 0)
            random_mask = torch.rand_like(scores) < masking_rate
            loss_mask = (~(negative & random_mask)).float()
        else:
            loss_mask = torch.ones_like(scores)

        loss = F.binary_cross_entropy_with_logits(scores, labs, reduction="none")
        loss = loss * loss_mask

        length, width = scores.shape[-2:]
        starts = torch.arange(length, device=scores.device).unsqueeze(1)
        offsets = torch.arange(width, device=scores.device).unsqueeze(0)
        valid = (starts + offsets) < length
        loss = loss * valid.unsqueeze(0).unsqueeze(0).to(loss.dtype)
        return loss.sum()

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
        from huggingface_hub import HfApi

        with tempfile.TemporaryDirectory() as tmp_dir:
            self.save_pretrained(tmp_dir)
            api = HfApi()
            api.create_repo(repo_id=repo_id, private=private, exist_ok=True)
            api.upload_folder(repo_id=repo_id, folder_path=tmp_dir)

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
            use_flashdeberta: If True, use the optional FlashDeBERTa backend
                for a compatible DeBERTaV2 encoder. If omitted, defer to the
                ``USE_FLASHDEBERTA`` environment variable.
            word_splitter: Built-in splitter name (``"whitespace"`` default,
                or ``"char"``) or a custom callable. Runtime-only; saved
                checkpoints reload with the default unless supplied again.
            **kwargs: Additional keyword arguments.

        To use a LoRA adapter:
            1. Load the base model first
            2. Then load the adapter using model.load_adapter()

        Example:
            model = Extractor.from_pretrained("base-model-name")
            model.load_adapter("path/to/adapter")
        """
        config = kwargs.pop("config", None)
        model_options, hub_kwargs = split_load_kwargs(
            kwargs, context=f"{cls.__name__}.from_pretrained"
        )
        quantize = model_options.pop("quantize", False)
        compile_model = model_options.pop("compile", False)
        map_location = model_options.pop("map_location", None)
        use_flashdeberta = model_options.pop("use_flashdeberta", None)
        word_splitter = model_options.pop("word_splitter", None)

        if config is None:
            config_path = checkpoint_file(repo_or_dir, "config.json", hub_kwargs)
            config = cls.config_class.from_pretrained(config_path)

        encoder_config_path = checkpoint_file(
            repo_or_dir, "encoder_config/config.json", hub_kwargs
        )
        encoder_config = AutoConfig.from_pretrained(encoder_config_path)

        tokenizer_source = repo_or_dir
        if os.path.isdir(str(repo_or_dir)) and hub_kwargs.get("subfolder"):
            tokenizer_source = os.path.join(
                str(repo_or_dir), str(hub_kwargs["subfolder"])
            )
        tokenizer = load_extractor_tokenizer(tokenizer_source)
        model = cls(
            config,
            encoder_config=encoder_config,
            tokenizer=tokenizer,
            use_flashdeberta=use_flashdeberta,
            word_splitter=word_splitter,
        )

        state_dict = load_checkpoint_state_dict(repo_or_dir, hub_kwargs)
        reconcile_encoder_embeddings(model, state_dict)
        try:
            model.load_state_dict(state_dict)
        except RuntimeError as exc:
            raise RuntimeError(
                f"Checkpoint at {repo_or_dir!r} does not match this model's "
                "span-head configuration. Ensure max_width, counting_layer, "
                f"and encoder settings match. Original error:\n{exc}"
            ) from exc

        # Mirror HF PreTrainedModel.from_pretrained semantics so downstream
        # PEFT saves derive ``base_model_name_or_path`` correctly. PEFT reads
        # ``self.base_model.model.__dict__.get("name_or_path")`` when serialising
        # an adapter; without an explicit instance attribute that lookup misses
        # the ``PreTrainedModel.name_or_path`` property and writes "" into
        # ``adapter_config.json``, which breaks downstream resolvers (e.g.
        # vLLM's ``lora_filesystem_resolver``) that match on this field.
        model.config._name_or_path = repo_or_dir
        model.name_or_path = repo_or_dir

        return apply_post_load_options(
            model,
            map_location=map_location,
            quantize=quantize,
            compile_model=compile_model,
        )

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

        Four components are compiled:

        - **encoder** (DeBERTa backbone)
        - **_compute_span_rep_core** (legacy span index + MLP)
        - **_compute_span_scores_core** (factorized direct span scoring)
        - **count_embed** (CompileSafeGRU + DownscaledTransformer)

        The list-of-tensors padding in ``compute_span_rep_batched`` /
        ``compute_span_scores_batched`` and the per-sample Python decode path
        are left in eager mode.

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
        self._compute_span_scores_core = torch.compile(
            self._compute_span_scores_core, dynamic=True,
        )
        self.count_embed = torch.compile(self.count_embed, dynamic=True)
        logger.info(
            "Compiled encoder, span-rep/direct-score, and count-embed with torch.compile(dynamic=True)"
        )
        return self

    # =========================================================================
    # LoRA / PEFT — new blessed API
    # =========================================================================

    def apply_lora(
        self,
        r: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
        targets: Optional[List[str]] = None,
        use_dora: bool = False,
    ) -> "PeftModel":
        """Apply LoRA adapters and return a PeftModel ready for training.

        Args:
            r: LoRA rank.
            alpha: LoRA alpha scaling factor.
            dropout: LoRA dropout probability.
            targets: High-level target group names. Defaults to ``["encoder"]``.
            use_dora: Whether to use DoRA weight decomposition.

        Returns:
            PeftModel wrapping this Extractor.
        """
        from peft import LoraConfig as PeftLoraConfig, get_peft_model
        from gliner2.training.lora import _resolve_targets, _cast_lora_dtype

        # Pin ``base_model_name_or_path`` up front so adapters saved via
        # ``PeftModel.save_pretrained`` always carry a concrete identifier
        # (matches what the base model was loaded from in ``from_pretrained``).
        base_id = (
            getattr(self, "name_or_path", "")
            or getattr(self.config, "_name_or_path", "")
            or None
        )
        cfg = PeftLoraConfig(
            r=r, lora_alpha=alpha, lora_dropout=dropout,
            target_modules=_resolve_targets(self, targets or ["encoder"]),
            bias="none", use_dora=use_dora,
            base_model_name_or_path=base_id,
        )
        peft_model = get_peft_model(self, cfg)
        _cast_lora_dtype(peft_model)
        return peft_model

    # =========================================================================
    # Legacy adapter API (deprecated — PendingDeprecationWarning)
    # =========================================================================

    def load_adapter(self, adapter_path: str) -> 'Extractor':
        """Load a LoRA adapter onto this model."""
        warnings.warn(
            "Extractor.load_adapter is deprecated; use PeftModel.from_pretrained() "
            "or Extractor.apply_lora().",
            PendingDeprecationWarning, stacklevel=2)
        from gliner2.training.lora import load_lora_adapter, LoRAAdapterConfig
        self._lora_layers = load_lora_adapter(self, adapter_path, auto_unload=True)
        p = Path(adapter_path)
        self._adapter_config = LoRAAdapterConfig.load(p) if (p / "adapter_config.json").exists() else None
        return self

    def unload_adapter(self) -> 'Extractor':
        """Unload current LoRA adapter, restoring base model."""
        warnings.warn(
            "Extractor.unload_adapter is deprecated; use PeftModel.merge_and_unload().",
            PendingDeprecationWarning, stacklevel=2)
        from gliner2.training.lora import unload_lora_adapter
        if self._lora_layers:
            unload_lora_adapter(self)
            self._lora_layers = {}
            self._adapter_config = None
        return self

    def merge_lora(self) -> 'Extractor':
        """Merge LoRA weights into base model and remove adapter structure."""
        warnings.warn(
            "Extractor.merge_lora is deprecated; use PeftModel.merge_and_unload().",
            PendingDeprecationWarning, stacklevel=2)
        if not self._lora_layers:
            raise ValueError("No adapter loaded. Nothing to merge.")
        from gliner2.training.lora import remove_lora_from_model
        remove_lora_from_model(self)
        self._lora_layers = {}
        self._adapter_config = None
        return self

    def save_adapter(self, save_path: str) -> None:
        """Save only the LoRA adapter (not full model)."""
        warnings.warn(
            "Extractor.save_adapter is deprecated; use PeftModel.save_pretrained().",
            PendingDeprecationWarning, stacklevel=2)
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
        """Save model to directory."""
        if save_adapter_only:
            warnings.warn(
                "save_pretrained(save_adapter_only=True) is deprecated; "
                "use PeftModel.save_pretrained().",
                PendingDeprecationWarning, stacklevel=2)
            if not self._lora_layers:
                raise ValueError("save_adapter_only=True but no adapter loaded")
            self.save_adapter(save_directory)
            return

        if merge_lora and self._lora_layers:
            warnings.warn(
                "save_pretrained(merge_lora=True) is deprecated; "
                "use PeftModel.merge_and_unload() then save_pretrained().",
                PendingDeprecationWarning, stacklevel=2)
            self.merge_lora()

        os.makedirs(save_directory, exist_ok=True)
        # Stamp architecture metadata so AutoExtractor can dispatch on reload.
        self.config.architecture = getattr(self, "architecture", "span")
        self.config.architectures = [type(self).__name__]
        self.config.save_pretrained(save_directory)

        encoder_config_path = os.path.join(save_directory, "encoder_config")
        os.makedirs(encoder_config_path, exist_ok=True)
        self.encoder.config.save_pretrained(encoder_config_path)

        model_path = os.path.join(save_directory, "model.safetensors")
        save_file(self.state_dict(), model_path)

        self.processor.tokenizer.save_pretrained(save_directory)


# Backward-compatible alias. ``Extractor`` remains the public name for the span
# core model.
Extractor = SpanExtractorModel

__all__ = ["Extractor", "SpanExtractorModel", "ExtractorConfig"]
