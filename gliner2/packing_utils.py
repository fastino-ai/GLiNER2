from typing import Optional, Dict, Any

import torch
from transformers.modeling_outputs import BaseModelOutput


def build_position_ids(packed) -> torch.Tensor:
    """Build position_ids that reset at each packed segment."""
    position_ids = torch.zeros_like(packed.input_ids)
    for row, (offsets, lengths) in enumerate(zip(packed.offsets, packed.lengths)):
        for offset, length in zip(offsets, lengths):
            if length > 0:
                position_ids[row, offset:offset + length] = torch.arange(
                    length, device=position_ids.device, dtype=position_ids.dtype
                )
    return position_ids


def _get_model_dtype(encoder) -> torch.dtype:
    try:
        return next(encoder.parameters()).dtype
    except StopIteration:
        return torch.float32


def _prepare_pair_attention_masks(
        pair_attention_mask: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        input_ids: Optional[torch.Tensor],
        inputs_embeds: Optional[torch.Tensor],
        encoder
) -> dict:
    device = pair_attention_mask.device
    if input_ids is not None:
        device = input_ids.device
    elif inputs_embeds is not None:
        device = inputs_embeds.device

    pair_mask_bool = pair_attention_mask.to(device=device, dtype=torch.bool)

    token_mask_bool = pair_mask_bool.any(dim=-1)
    if attention_mask is not None:
        token_mask_bool = token_mask_bool & attention_mask.to(device=device, dtype=torch.bool)

    seq_len = pair_mask_bool.size(-1)
    if seq_len:
        identity = torch.eye(seq_len, device=device, dtype=torch.bool).unsqueeze(0)
        token_diag = token_mask_bool.unsqueeze(-1)
        pair_mask_bool = pair_mask_bool | (identity & token_diag)

    active = token_mask_bool.unsqueeze(-1) & token_mask_bool.unsqueeze(-2)
    pair_mask_bool = pair_mask_bool & active

    if attention_mask is not None:
        token_mask = token_mask_bool.to(attention_mask.dtype)
    else:
        token_mask = token_mask_bool.to(dtype=torch.float32)

    mask_dtype = _get_model_dtype(encoder)
    neg_inf = torch.finfo(mask_dtype).min
    extended_mask = torch.zeros(
        pair_mask_bool.shape, dtype=mask_dtype, device=device
    ).masked_fill(~pair_mask_bool, neg_inf).unsqueeze(1)

    inactive = ~token_mask_bool
    if inactive.any():
        extended_mask = extended_mask.masked_fill(
            inactive.unsqueeze(1).unsqueeze(-1),
            torch.tensor(0.0, dtype=mask_dtype, device=device),
        )

    return {
        "token_mask": token_mask,
        "token_mask_bool": token_mask_bool,
        "extended_mask": extended_mask,
        "block_mask": pair_mask_bool,
    }


def _forward_deberta(
        encoder,
        input_ids: Optional[torch.Tensor],
        model_kwargs: Dict[str, Any],
        mask_info: dict,
) -> BaseModelOutput:
    inputs_embeds = model_kwargs.pop("inputs_embeds", None)
    token_type_ids = model_kwargs.pop("token_type_ids", None)
    position_ids = model_kwargs.pop("position_ids", None)
    output_attentions = model_kwargs.pop("output_attentions")
    produce_hidden = model_kwargs.pop("output_hidden_states")
    return_dict = model_kwargs.pop("return_dict")

    if input_ids is None and inputs_embeds is None:
        raise ValueError("Either input_ids or inputs_embeds must be provided for packed attention")
    if input_ids is not None and inputs_embeds is not None:
        raise ValueError("Cannot supply both input_ids and inputs_embeds")

    if token_type_ids is None:
        ref = inputs_embeds if inputs_embeds is not None else input_ids
        shape = ref.size()[:-1] if inputs_embeds is not None else ref.size()
        token_type_ids = torch.zeros(shape, dtype=torch.long, device=ref.device)

    embedding_output = encoder.embeddings(
        input_ids=input_ids,
        token_type_ids=token_type_ids,
        position_ids=position_ids,
        mask=mask_info["token_mask"],
        inputs_embeds=inputs_embeds,
    )

    encoder_outputs = encoder.encoder(
        embedding_output,
        mask_info["block_mask"],
        output_hidden_states=True,
        output_attentions=output_attentions,
        return_dict=True,
    )

    encoded_layers = list(encoder_outputs.hidden_states)
    sequence_output = encoded_layers[-1]
    hidden_states_tuple = tuple(encoded_layers) if produce_hidden else None
    attentions = encoder_outputs.attentions if output_attentions else None

    if not return_dict:
        result = (sequence_output,)
        if hidden_states_tuple is not None:
            result += (hidden_states_tuple,)
        if attentions is not None:
            result += (attentions,)
        return result

    return BaseModelOutput(
        last_hidden_state=sequence_output,
        hidden_states=hidden_states_tuple,
        attentions=attentions,
    )


def apply_pair_attention_encoder(
        encoder,
        packed_ids: torch.Tensor,
        pair_mask: torch.Tensor,
        fallback_mask: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        **kwargs,
):
    attn_to_use = pair_mask if pair_mask.numel() else fallback_mask

    model_name = encoder.__class__.__name__.lower()
    model_type = getattr(encoder.config, "model_type", "") or ""

    if pair_mask.numel() and ("deberta" in model_name or "deberta" in model_type):
        mask_info = _prepare_pair_attention_masks(
            pair_mask,
            fallback_mask,
            packed_ids,
            kwargs.get("inputs_embeds"),
            encoder,
        )
        model_kwargs = dict(kwargs)
        model_kwargs.setdefault("output_attentions", False)
        model_kwargs.setdefault("output_hidden_states", False)
        model_kwargs.setdefault("return_dict", True)
        if position_ids is not None:
            model_kwargs.setdefault("position_ids", position_ids)
        return _forward_deberta(
            encoder,
            packed_ids if "input_ids" not in model_kwargs else model_kwargs.get("input_ids"),
            model_kwargs,
            mask_info,
        )

    return encoder(
        input_ids=packed_ids,
        attention_mask=attn_to_use,
        position_ids=position_ids,
        **kwargs,
    )
