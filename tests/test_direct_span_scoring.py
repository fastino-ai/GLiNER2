import torch

from gliner2.layers import SpanMarkerV0


def _safe_spans(batch_size: int, length: int, max_width: int, device):
    starts = torch.arange(length, device=device).unsqueeze(1).expand(-1, max_width)
    widths = torch.arange(max_width, device=device).unsqueeze(0)
    ends = starts + widths
    valid = ends < length
    starts = starts.reshape(-1).unsqueeze(0).expand(batch_size, -1)
    ends = ends.reshape(-1).unsqueeze(0).expand(batch_size, -1)
    valid_flat = valid.reshape(-1).unsqueeze(0).expand(batch_size, -1)
    spans = torch.stack(
        [
            torch.where(valid_flat, starts, torch.zeros_like(starts)),
            torch.where(valid_flat, ends, torch.zeros_like(ends)),
        ],
        dim=-1,
    )
    return spans, valid


def _reference_scores(layer, hidden, queries):
    batch, length, _ = hidden.shape
    spans, valid = _safe_spans(batch, length, layer.max_width, hidden.device)
    span_rep = layer(hidden, spans)
    scores = torch.einsum("blwh,bqh->bqlw", span_rep, queries)
    return scores, valid


def test_direct_span_scores_match_reference_fp32_eval():
    torch.manual_seed(7)
    layer = SpanMarkerV0(hidden_size=128, max_width=8, dropout=0.1).eval()
    hidden = torch.randn(3, 61, 128)
    queries = torch.randn(3, 9, 128)

    with torch.inference_mode():
        reference, _ = _reference_scores(layer, hidden, queries)
        direct = layer.score_queries(hidden, queries)

    valid = torch.isfinite(direct)
    torch.testing.assert_close(direct[valid], reference[valid], rtol=2e-5, atol=2e-5)


def test_direct_span_scores_respect_variable_lengths():
    torch.manual_seed(11)
    layer = SpanMarkerV0(hidden_size=96, max_width=8, dropout=0.1).eval()
    hidden = torch.randn(3, 64, 96)
    queries = torch.randn(3, 5, 96)
    lengths = torch.tensor([64, 37, 13])

    with torch.inference_mode():
        direct = layer.score_queries(hidden, queries, lengths)
        for i, length in enumerate(lengths.tolist()):
            reference, _ = _reference_scores(
                layer, hidden[i:i + 1, :length], queries[i:i + 1]
            )
            current = direct[i:i + 1, :, :length]
            valid = torch.isfinite(current)
            torch.testing.assert_close(
                current[valid], reference[valid], rtol=2e-5, atol=2e-5
            )
            assert not torch.isfinite(direct[i, :, length:]).any()


def test_direct_span_scores_preserve_training_dropout_location():
    torch.manual_seed(17)
    layer = SpanMarkerV0(hidden_size=64, max_width=4, dropout=0.1).train()
    hidden = torch.randn(2, 23, 64)
    queries = torch.randn(2, 6, 64)

    torch.manual_seed(1234)
    reference, valid_2d = _reference_scores(layer, hidden, queries)
    torch.manual_seed(1234)
    direct = layer.score_queries(hidden, queries, mask_invalid=False)

    valid = valid_2d[None, None].expand_as(reference)
    torch.testing.assert_close(
        direct[valid], reference[valid], rtol=3e-5, atol=3e-5
    )


def test_direct_span_scores_match_training_gradients():
    """Factorized training preserves valid-span gradients with the same dropout mask."""
    torch.manual_seed(23)
    layer = SpanMarkerV0(hidden_size=48, max_width=4, dropout=0.1).train()
    hidden_base = torch.randn(2, 19, 48)
    queries_base = torch.randn(2, 5, 48)

    hidden_ref = hidden_base.detach().clone().requires_grad_(True)
    queries_ref = queries_base.detach().clone().requires_grad_(True)
    torch.manual_seed(4321)
    reference, valid_2d = _reference_scores(layer, hidden_ref, queries_ref)
    valid = valid_2d[None, None].expand_as(reference)
    reference_loss = reference[valid].square().mean()
    reference_loss.backward()

    ref_hidden_grad = hidden_ref.grad.detach().clone()
    ref_query_grad = queries_ref.grad.detach().clone()
    ref_param_grads = {
        name: param.grad.detach().clone()
        for name, param in layer.named_parameters()
        if param.grad is not None
    }

    layer.zero_grad(set_to_none=True)
    hidden_direct = hidden_base.detach().clone().requires_grad_(True)
    queries_direct = queries_base.detach().clone().requires_grad_(True)
    torch.manual_seed(4321)
    direct = layer.score_queries(
        hidden_direct, queries_direct, mask_invalid=False
    )
    direct_loss = direct[valid].square().mean()
    direct_loss.backward()

    torch.testing.assert_close(direct_loss, reference_loss, rtol=5e-5, atol=5e-6)
    torch.testing.assert_close(
        hidden_direct.grad, ref_hidden_grad, rtol=2e-4, atol=2e-6
    )
    torch.testing.assert_close(
        queries_direct.grad, ref_query_grad, rtol=2e-4, atol=2e-6
    )
    for name, param in layer.named_parameters():
        if name not in ref_param_grads:
            continue
        assert param.grad is not None
        torch.testing.assert_close(
            param.grad, ref_param_grads[name], rtol=3e-4, atol=3e-6
        )


def test_direct_structure_loss_matches_legacy_without_dropout(tiny_span_model):
    """Direct raw logits reproduce the legacy structure loss exactly up to FP noise."""
    model = tiny_span_model.eval()
    hidden = model.hidden_size
    token_embeddings = torch.randn(17, hidden)
    schema_emb = torch.randn(4, hidden)  # [P] + three fields
    structure = [
        2,
        [
            [(1, 2), (4, 4), (7, 8)],
            [(2, 3), None, [(9, 9), (11, 12)]],
        ],
    ]

    with torch.no_grad():
        span_info = model.compute_span_rep(token_embeddings)
        legacy_loss = model.compute_struct_loss(
            span_info["span_rep"],
            schema_emb,
            structure,
            span_info["span_mask"],
            masking_rate=0.0,
        )

        gold_count = min(structure[0], 19)
        projected = model.count_embed(schema_emb[1:], gold_count)
        count_dim, field_dim, hidden_dim = projected.shape
        direct = model.compute_span_scores_batched(
            [token_embeddings],
            [projected.reshape(count_dim * field_dim, hidden_dim)],
            mask_invalid=False,
        )[0].reshape(count_dim, field_dim, len(token_embeddings), model.max_width)
        direct_loss = model.compute_struct_loss_from_scores(
            direct, structure, masking_rate=0.0
        )

    torch.testing.assert_close(direct_loss, legacy_loss, rtol=2e-5, atol=2e-5)


def test_training_forward_uses_direct_span_scoring(tiny_span_model, monkeypatch):
    """Training must not fall back to legacy span-representation materialization."""
    from gliner2.training.data import InputExample, Relation, Structure
    from gliner2.training.trainer import ExtractorCollator, ExtractorDataset

    examples = [
        InputExample(
            text="Apple hired Tim Cook in Cupertino.",
            entities={"company": ["Apple"], "person": ["Tim Cook"]},
            structures=[Structure("employment", company="Apple", person="Tim Cook")],
            relations=[Relation("works_for", head="Tim Cook", tail="Apple")],
        ),
        InputExample(
            text="Google hired Jane Smith in London.",
            entities={"company": ["Google"], "person": ["Jane Smith"]},
            structures=[Structure("employment", company="Google", person="Jane Smith")],
            relations=[Relation("works_for", head="Jane Smith", tail="Google")],
        ),
    ]
    dataset = ExtractorDataset.from_examples(examples, shuffle=False, validate=True)
    collator = ExtractorCollator(
        tiny_span_model.processor,
        is_training=True,
        architecture=tiny_span_model.architecture,
    )
    batch = collator([dataset[0], dataset[1]])

    def _legacy_path_forbidden(*args, **kwargs):
        raise AssertionError("legacy span representation path was called")

    monkeypatch.setattr(tiny_span_model, "compute_span_rep_batched", _legacy_path_forbidden)
    monkeypatch.setattr(tiny_span_model, "compute_span_rep", _legacy_path_forbidden)

    tiny_span_model.train()
    out = tiny_span_model(batch)
    assert torch.isfinite(out["total_loss"])
    assert torch.isfinite(out["structure_loss"])
    out["total_loss"].backward()

    span_grads = [
        p.grad for p in tiny_span_model.span_rep.parameters()
        if p.requires_grad
    ]
    assert any(g is not None and torch.isfinite(g).all() and g.abs().sum() > 0 for g in span_grads)


def test_mixed_inference_uses_direct_span_scoring(tiny_span_model, monkeypatch):
    """Entities, structures, relations and classifications share the direct path."""
    model = tiny_span_model.eval()
    schema = model.create_schema()
    schema.entities(["company", "person", "location"])
    schema.classification("sentiment", ["positive", "negative"])
    schema.structure("employment").field("person").field("company")
    schema.relations(["works_for"])

    def _legacy_path_forbidden(*args, **kwargs):
        raise AssertionError("legacy span representation path was called")

    monkeypatch.setattr(model, "compute_span_rep_batched", _legacy_path_forbidden)
    monkeypatch.setattr(model, "compute_span_rep", _legacy_path_forbidden)

    result = model.extract(
        "Tim Cook works for Apple in Cupertino.",
        schema,
        include_confidence=True,
        include_spans=True,
    )
    assert isinstance(result, dict)
