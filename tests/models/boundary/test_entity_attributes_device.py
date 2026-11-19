"""Entity attributes decode with the model on an accelerator (the attribute logits are copied to CPU first)."""

import pytest
import torch

from gliner2.inference.schema import AttributeGroup
from tests.models.boundary.test_record_head_pipeline import _build_tiny_records_model


def _accelerator():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return None


@pytest.mark.skipif(_accelerator() is None, reason="needs CUDA or MPS")
def test_entity_attributes_decode_on_an_accelerator():
    texts = ["Alice bought apples in Paris", "The camera is excellent but the battery is weak"]
    schema_groups = {"sentiment": AttributeGroup(["positive", "negative"], applies_to=["product"])}

    def decode(device):
        model = _build_tiny_records_model().to(device)
        schema = model.create_schema().entities(["product", "person"]).entity_attributes(schema_groups)
        return model.batch_extract(texts, schema, batch_size=2, threshold=0.0)

    on_device = decode(_accelerator())
    products = [item for result in on_device for item in result["entities"]["product"]]
    assert products and all("sentiment" in item for item in products)
    on_cpu = decode("cpu")
    assert [[i["text"] for i in r["entities"]["product"]] for r in on_device] == \
        [[i["text"] for i in r["entities"]["product"]] for r in on_cpu]
