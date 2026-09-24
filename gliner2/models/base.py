"""Architecture-neutral query metadata, encoded-batch container, and base model.

``QuerySpec``/``QueryLayout`` describe the extractive/classification queries of
a sample independent of architecture. ``EncodedBatch`` is the vectorized
encoder output both architectures can consume. ``BaseExtractorModel`` provides
shared encoder loading and architecture-stamping save; the span model keeps its
own encode path until parity is proven (per the blueprint).
"""

from __future__ import annotations

import importlib.util
import logging
import os
import warnings
from dataclasses import dataclass, field
from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    PretrainedConfig,
    PreTrainedModel,
)

from gliner2.configuration import ExtractorConfig

logger = logging.getLogger(__name__)


def load_extractor_tokenizer(repo_or_dir: str):
    """Load checkpoint tokenizers across Transformers metadata versions.

    Older GLiNER2 checkpoints serialized ``extra_special_tokens`` as a list.
    Newer Transformers releases reserve that field for a name-to-token mapping
    and fail before ``SchemaTransformer`` can register GLiNER2's special tokens.
    Retry only that known incompatibility while leaving all other load failures
    untouched.
    """
    try:
        return AutoTokenizer.from_pretrained(repo_or_dir)
    except AttributeError as exc:
        if "'list' object has no attribute 'keys'" not in str(exc):
            raise
        warnings.warn(
            "Checkpoint uses legacy list-valued extra_special_tokens metadata; "
            "loading with compatibility normalization.",
            UserWarning,
            stacklevel=2,
        )
        return AutoTokenizer.from_pretrained(
            repo_or_dir, extra_special_tokens={}
        )


# =============================================================================
# Query metadata
# =============================================================================

@dataclass(frozen=True)
class QuerySpec:
    """Metadata for one query (an extractive field/entity or a classification)."""
    query_id: int
    task_index: int
    task_type: str
    task_name: str
    role_index: int = 0
    role_name: str = ""
    field_path: Tuple[str, ...] = ()
    extractive: bool = True


@dataclass(frozen=True)
class QueryLayout:
    """Ordered queries for a single sample, with fast id lookup."""
    queries: Tuple[QuerySpec, ...]
    classification_query_ids: Tuple[int, ...] = ()
    extractive_query_ids: Tuple[int, ...] = ()

    def __post_init__(self) -> None:
        # Derive id groupings if not provided.
        if not self.classification_query_ids and not self.extractive_query_ids and self.queries:
            cls_ids = tuple(q.query_id for q in self.queries if not q.extractive)
            ext_ids = tuple(q.query_id for q in self.queries if q.extractive)
            object.__setattr__(self, "classification_query_ids", cls_ids)
            object.__setattr__(self, "extractive_query_ids", ext_ids)

    def __len__(self) -> int:
        return len(self.queries)

    def query(self, query_id: int) -> QuerySpec:
        for q in self.queries:
            if q.query_id == query_id:
                return q
        raise KeyError(f"no query with id {query_id}")

    def extractive_count(self) -> int:
        return len(self.extractive_query_ids)

    def classification_count(self) -> int:
        return len(self.classification_query_ids)


# =============================================================================
# Encoded batch
# =============================================================================

@dataclass
class EncodedBatch:
    """Vectorized encoder output shared by both architectures."""
    text_states: torch.Tensor            # [B, L, H]
    text_mask: torch.BoolTensor          # [B, L]
    text_lengths: torch.LongTensor       # [B]
    query_states: torch.Tensor           # [B, Q, H]
    query_mask: torch.BoolTensor         # [B, Q]
    query_layouts: Tuple[QueryLayout, ...]
    classification_states: Optional[torch.Tensor] = None

    def to(self, device) -> "EncodedBatch":
        return EncodedBatch(
            text_states=self.text_states.to(device),
            text_mask=self.text_mask.to(device),
            text_lengths=self.text_lengths.to(device),
            query_states=self.query_states.to(device),
            query_mask=self.query_mask.to(device),
            query_layouts=self.query_layouts,
            classification_states=(
                self.classification_states.to(device)
                if self.classification_states is not None else None
            ),
        )


# =============================================================================
# Base model
# =============================================================================

def _pad_hidden_states(
    hidden_states: Sequence[torch.Tensor],
    attention_mask: torch.Tensor,
    *,
    hidden_size: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Right-pad per-row encoder states to ``attention_mask``'s shape.

    Args:
        hidden_states: One ``(tokens_i, hidden)`` tensor per batch row.
        attention_mask: The collated ``[B, L]`` mask; rows are right-padded.
        hidden_size: Expected hidden width.
        dtype: Output dtype, matching the model's parameters.

    Returns:
        A ``[B, L, hidden_size]`` tensor on ``attention_mask``'s device.

    Raises:
        ValueError: On a row count, row length or hidden width mismatch.
    """
    rows, width = attention_mask.shape
    if len(hidden_states) != rows:
        raise ValueError(
            f"hidden_states has {len(hidden_states)} rows; the batch has {rows}"
        )
    lengths = attention_mask.sum(dim=-1).tolist()
    padded = torch.zeros(
        (rows, width, hidden_size), dtype=dtype, device=attention_mask.device
    )
    for index, (states, length) in enumerate(zip(hidden_states, lengths, strict=True)):
        if states.dim() != 2 or states.shape != (length, hidden_size):
            raise ValueError(
                f"hidden_states[{index}] has shape {tuple(states.shape)}; "
                f"expected ({length}, {hidden_size})"
            )
        padded[index, :length] = states.to(device=padded.device, dtype=dtype)
    return padded


class BaseExtractorModel(PreTrainedModel):
    """Shared base for extractor architectures.

    Provides encoder construction and architecture-aware serialization. It does
    not impose an ``encode()`` contract on the span model; the boundary model
    uses ``encode()`` while the span model retains its legacy path.
    """
    config_class = ExtractorConfig

    @staticmethod
    def _load_encoder(
        model_name: str,
        encoder_config: Optional[PretrainedConfig] = None,
        attn_implementation: Optional[str] = "sdpa",
        use_flashdeberta: Optional[bool] = None,
    ) -> nn.Module:
        """Load a shared optimized encoder with a safe eager fallback."""
        config = encoder_config
        if config is None:
            config = AutoConfig.from_pretrained(
                model_name, trust_remote_code=True
            )

        if use_flashdeberta is None:
            use_flashdeberta = bool(os.environ.get("USE_FLASHDEBERTA"))
        flashdeberta_available = (
            importlib.util.find_spec("flashdeberta") is not None
        )
        if (
            config.__class__.__name__ == "DebertaV2Config"
            and use_flashdeberta
            and flashdeberta_available
        ):
            try:
                from flashdeberta import FlashDebertaV2Model

                logger.info("Using FlashDeBERTa backend")
                if encoder_config is not None:
                    return FlashDebertaV2Model(config).float()
                return FlashDebertaV2Model.from_pretrained(model_name).float()
            except Exception as error:  # noqa: BLE001 - optional backend boundary
                warnings.warn(
                    "FlashDeBERTa could not initialize; falling back to the "
                    f"standard Transformers encoder ({error})",
                    RuntimeWarning,
                    stacklevel=2,
                )

        def load(implementation: Optional[str]) -> nn.Module:
            kwargs = {"trust_remote_code": True}
            if implementation:
                kwargs["attn_implementation"] = implementation
            if encoder_config is not None:
                encoder = AutoModel.from_config(config, **kwargs)
            else:
                encoder = AutoModel.from_pretrained(model_name, **kwargs)
            # Transformers 5 honors a serialized encoder dtype during
            # ``from_config``. GLiNER2 task heads are initialized in FP32, so a
            # half-precision encoder would emit activations that cannot enter
            # those heads. Build the complete model consistently in FP32;
            # callers can still cast it atomically with ``model.half()`` or
            # ``model.to(dtype=...)`` afterwards.
            return encoder.float()

        try:
            return load(attn_implementation)
        except (TypeError, ValueError, ImportError) as error:
            if not attn_implementation or attn_implementation == "eager":
                raise
            warnings.warn(
                f"Encoder rejected attn_implementation={attn_implementation!r}; "
                f"falling back to 'eager' ({error})",
                RuntimeWarning,
                stacklevel=2,
            )
            return load("eager")

    def encode_tokens(
        self,
        batch,
        hidden_states: Optional[Sequence[torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Return padded ``[B, L, H]`` token states for a collated batch.

        This is the only inference call into ``self.encoder``. A serving engine
        that already ran the encoder passes its per-row output instead.

        Args:
            batch: A ``PreprocessedBatch`` on the model's device.
            hidden_states: Optional precomputed encoder output, one
                ``(tokens_i, hidden)`` tensor per row, where ``tokens_i`` is
                that row's unpadded length in ``batch.attention_mask``.

        Returns:
            Token states shaped like ``batch.input_ids`` plus a hidden axis.

        Raises:
            ValueError: If ``hidden_states`` does not match the batch, or the
                model has no encoder and ``hidden_states`` is ``None``.
        """
        if hidden_states is None:
            if getattr(self, "encoder", None) is None:
                raise ValueError(
                    f"{type(self).__name__} was built without an encoder; "
                    "pass hidden_states"
                )
            return self.encoder(
                input_ids=batch.input_ids, attention_mask=batch.attention_mask
            ).last_hidden_state
        return _pad_hidden_states(
            hidden_states,
            batch.attention_mask,
            hidden_size=self.hidden_size,
            dtype=next(self.parameters()).dtype,
        )

    def task_module_names(self) -> Tuple[str, ...]:
        raise NotImplementedError

    def save_pretrained(self, *args, **kwargs):
        self.config.architecture = getattr(self, "architecture", self.config.architecture)
        self.config.architectures = [type(self).__name__]
        return super().save_pretrained(*args, **kwargs)

    def set_word_splitter(self, word_splitter):
        """Replace the word splitter used at inference and training collation.

        ``word_splitter`` may be a built-in name (``"whitespace"`` or
        ``"char"``) or a callable yielding ``(token, start, end)`` with
        exclusive-end offsets into the original text. The default
        ``"whitespace"`` strategy is the one used to train public checkpoints.

        ``"char"`` is suitable for languages without whitespace-delimited
        words, such as Chinese. Changing a pretrained model's word boundaries
        can affect quality unless training used the same splitter.
        """
        from gliner2.processing.word_splitter import resolve_word_splitter

        processor = getattr(self, "processor", None)
        if processor is None:
            raise AttributeError(
                f"{type(self).__name__} has no processor to attach a word splitter to"
            )
        processor.word_splitter = resolve_word_splitter(word_splitter)
        return self
