"""Final gate: the spec's safety schema (the canonical adversarial case) decodes
end-to-end on the FakeClsModel, and the iff/any_other_selected coupling is
enforced exactly.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from gliner2.classification import constraints as C
from gliner2.classification.engine import ClassificationConfig, Classifier
from gliner2.classification.schema import ClassificationSchema
from gliner2.models.base import BaseExtractorModel

SAFETY = ["safe", "unsafe"]
TOXICITY = ["violence", "sexual_content", "hate", "self_harm", "pii", "benign"]
JAILBREAK = ["prompt_injection", "instruction_override", "roleplay_bypass", "benign"]


def _safety_schema():
    return (
        ClassificationSchema()
        .single("prompt_safety", SAFETY, instruction="Judge the User turn.")
        .single("response_safety", SAFETY, instruction="Judge the Assistant turn.")
        .single("response_action", ["refusal", "compliance"],
                instruction="Does the Assistant turn refuse or comply?")
        .multi("prompt_toxicity", TOXICITY, threshold=0.4, default="benign",
               instruction="Toxicity in the User turn.")
        .multi("response_toxicity", TOXICITY, threshold=0.4, default="benign",
               instruction="Toxicity in the Assistant turn.")
        .multi("jailbreak", JAILBREAK, threshold=0.4, default="benign",
               instruction="Jailbreak attempts in the User turn.")
        .constrain(
            C.iff(("prompt_safety", "unsafe"),
                  C.any_of(C.any_other_selected("prompt_toxicity"),
                           C.any_other_selected("jailbreak"))),
            C.iff(("response_safety", "unsafe"),
                  C.any_other_selected("response_toxicity")),
        )
    )


# A fake that plants per-task/per-label logits directly, resolving task names by
# boundary-aware match (the prompts carry "task: instruction").
class _Processor:
    def __init__(self, tasks, logits):
        self.tasks, self.logits, self.is_training = tasks, logits, False

    def change_mode(self, is_training):
        self.is_training = is_training

    def collate_fn_inference(self, rows, max_len=None, architecture="span"):
        texts = [t for t, _ in rows]
        b = SimpleNamespace()
        b.input_ids = torch.ones((len(texts), 4), dtype=torch.long)
        b.attention_mask = torch.ones_like(b.input_ids)
        b.task_types = [["classifications"] * len(self.tasks) for _ in texts]
        b.schema_tokens_list = [[
            ["(", "[P]", f"{name}: instr", "("]
            + sum(([["[L]", l][k] for k in (0, 1)] for l in labels), [])
            + [")", ")"]
            for name, labels in self.tasks
        ] for _ in texts]
        b.to = lambda *a, **k: b
        return b

    def extract_embeddings_from_batch(self, encoded, input_ids, batch):
        n = batch.input_ids.shape[0]
        token = [torch.zeros((2, 4)) for _ in range(n)]
        schema = []
        for _ in range(n):
            per_task = []
            for name, labels in self.tasks:
                rows = [torch.zeros(4)]
                rows += [torch.full((4,), self.logits[name][l]) for l in labels]
                per_task.append(rows)
            schema.append(per_task)
        return token, schema


class _Encoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))

    def forward(self, input_ids, attention_mask):
        return SimpleNamespace(last_hidden_state=torch.zeros((*input_ids.shape, 4)))


class _Model(torch.nn.Module):
    encode_tokens = BaseExtractorModel.encode_tokens

    def __init__(self, tasks, logits):
        super().__init__()
        self.encoder = _Encoder()
        self.processor = _Processor(tasks, logits)

    def classifier(self, embeds):
        return embeds[:, :1]


def _tasks_and_logits(prompt_unsafe, tox_hate):
    tasks = [("prompt_safety", SAFETY), ("response_safety", SAFETY),
             ("response_action", ["refusal", "compliance"]),
             ("prompt_toxicity", TOXICITY), ("response_toxicity", TOXICITY),
             ("jailbreak", JAILBREAK)]
    logits = {
        "prompt_safety": {"safe": -3.0 if prompt_unsafe else 3.0,
                          "unsafe": 3.0 if prompt_unsafe else -3.0},
        "response_safety": {"safe": 3.0, "unsafe": -3.0},
        "response_action": {"refusal": 1.0, "compliance": -1.0},
        "prompt_toxicity": {l: (2.0 if (l == "hate" and tox_hate) else -3.0)
                            for l in TOXICITY},
        "response_toxicity": {l: -3.0 for l in TOXICITY},
        "jailbreak": {l: -3.0 for l in JAILBREAK},
    }
    logits["prompt_toxicity"]["benign"] = -3.0
    logits["response_toxicity"]["benign"] = -3.0
    logits["jailbreak"]["benign"] = -3.0
    return tasks, logits


def test_safety_schema_decodes_end_to_end():
    # prompt evidence is unsafe AND toxicity(hate) fires -> iff is consistent.
    tasks, logits = _tasks_and_logits(prompt_unsafe=True, tox_hate=True)
    clf = Classifier(_Model(tasks, logits))
    result = clf.classify("some transcript", _safety_schema(),
                          config=ClassificationConfig(decoder="exact"))
    assert result.feasible and result.exact
    assert result.value("prompt_safety") == "unsafe"
    assert "hate" in result.selected("prompt_toxicity")


def test_iff_forces_consistency_when_evidence_conflicts():
    # prompt_safety wants "unsafe" but NO toxicity/jailbreak fires: the iff
    # forbids unsafe-with-no-signal, so the decoder must flip something.
    tasks, logits = _tasks_and_logits(prompt_unsafe=True, tox_hate=False)
    clf = Classifier(_Model(tasks, logits))
    result = clf.classify("t", _safety_schema(),
                          config=ClassificationConfig(decoder="exact"))
    assert result.feasible
    safety = result.value("prompt_safety")
    has_other = any(l != "benign" for l in result.selected("prompt_toxicity")) \
        or any(l != "benign" for l in result.selected("jailbreak"))
    # the iff is satisfied: unsafe <=> some non-default toxicity/jailbreak selected
    assert (safety == "unsafe") == has_other


def test_active_view_is_score_preserving():
    tasks, logits = _tasks_and_logits(prompt_unsafe=True, tox_hate=True)
    clf = Classifier(_Model(tasks, logits))
    scores = clf.score("t", _safety_schema())
    view = clf.decode(scores, _safety_schema(),
                      active=("prompt_safety", "prompt_toxicity", "jailbreak"),
                      config=ClassificationConfig(decoder="exact"))
    assert set(view.tasks) == {"prompt_safety", "prompt_toxicity", "jailbreak"}
    assert view.feasible
