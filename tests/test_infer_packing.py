import torch

from gliner2.infer_packing import InferencePackingConfig, pack_requests, unpack_spans


class DummyEncoder(torch.nn.Module):
    def __init__(self, vocab_size: int = 128, hidden_size: int = 16):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, hidden_size)

    def forward(self, input_ids, attention_mask=None):
        # Mask is unused; we only need deterministic embeddings.
        return self.embedding(input_ids)


def _run_baseline(encoder, requests, pad_token_id: int):
    lengths = [len(req["input_ids"]) for req in requests]
    max_len = max(lengths)
    input_ids = torch.full((len(requests), max_len), pad_token_id, dtype=torch.long)
    for i, req in enumerate(requests):
        tokens = torch.tensor(req["input_ids"], dtype=torch.long)
        input_ids[i, : len(tokens)] = tokens
    outputs = encoder(input_ids=input_ids)
    result = []
    for i, length in enumerate(lengths):
        result.append(outputs[i, :length].detach())
    return result


def _run_packed(encoder, requests, cfg: InferencePackingConfig, pad_token_id: int):
    packed = pack_requests(requests, cfg, pad_token_id)
    outputs = encoder(
        input_ids=packed.input_ids,
        attention_mask=packed.pair_attention_mask,
    )
    unpacked = unpack_spans(outputs, packed)
    return [tensor.detach() for tensor in unpacked]


def test_pack_and_unpack_matches_baseline():
    requests = [
        {"input_ids": [1, 2, 3, 4, 5]},
        {"input_ids": [6, 7, 8]},
        {"input_ids": [9, 10, 11, 12]},
    ]
    pad_token_id = 0
    cfg = InferencePackingConfig(max_length=16, sep_token_id=None, streams_per_batch=1)

    encoder = DummyEncoder()
    baseline = _run_baseline(encoder, requests, pad_token_id)
    packed = _run_packed(encoder, requests, cfg, pad_token_id)

    assert len(baseline) == len(packed)
    for base, pack in zip(baseline, packed):
        torch.testing.assert_close(base, pack, atol=1e-5, rtol=1e-4)
