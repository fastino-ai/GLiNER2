"""
Semantic Relation Verifier for GLiNER2.

Filters false positive relations by verifying semantic validity
using a lightweight MLP trained on relation extraction datasets.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Union, Tuple
from dataclasses import dataclass
from huggingface_hub import hf_hub_download
import os


@dataclass
class VerifierConfig:
    """Configuration for RelationVerifier."""
    input_dim: int = 3105  # 768*4 + 32 + 1
    hidden_dim: int = 768
    dropout: float = 0.3
    threshold: float = 0.55
    encoder_name: str = "microsoft/deberta-v3-base"
    distance_buckets: List[int] = None

    def __post_init__(self):
        if self.distance_buckets is None:
            self.distance_buckets = [0, 5, 10, 20, 50]


class RelationVerifierModel(nn.Module):
    """MLP model for relation verification."""

    def __init__(self, config: VerifierConfig):
        super().__init__()
        self.config = config

        # Distance embedding (matches checkpoint: dist_embedding)
        self.dist_embedding = nn.Embedding(len(config.distance_buckets), 32)

        # MLP classifier (matches checkpoint structure)
        # classifier.0: Linear(3105, 768)
        # classifier.1: GELU
        # classifier.2: Dropout
        # classifier.3: Linear(768, 384)
        # classifier.4: GELU
        # classifier.5: Dropout
        # classifier.6: Linear(384, 1)
        self.classifier = nn.Sequential(
            nn.Linear(config.input_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim // 2, 1)
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Forward pass returning logits."""
        return self.classifier(features).squeeze(-1)


class RelationVerifier:
    """
    Verifies extracted relations using a trained semantic classifier.

    Filters false positives by checking if head-tail entity pairs
    semantically match the claimed relation type.

    Example:
        verifier = RelationVerifier.from_pretrained("oneryalcin/gliner2-relation-verifier")

        # Use with GLiNER2
        model = GLiNER2.from_pretrained("fastino/gliner2-base-v1")
        results = model.extract_relations(text, ["works_for", "founded"])
        verified = verifier.verify(text, results["relation_extraction"])
    """

    def __init__(
        self,
        model: RelationVerifierModel,
        encoder,
        tokenizer,
        config: VerifierConfig,
        device: str = None
    ):
        self.model = model
        self.encoder = encoder
        self.tokenizer = tokenizer
        self.config = config
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model.to(self.device)
        self.model.eval()
        self.encoder.to(self.device)
        self.encoder.eval()

    @classmethod
    def from_pretrained(
        cls,
        model_id: str = "oneryalcin/gliner2-relation-verifier",
        threshold: float = 0.55,
        device: str = None,
        **kwargs
    ) -> "RelationVerifier":
        """
        Load verifier from HuggingFace Hub.

        Args:
            model_id: HuggingFace model ID or local path
            threshold: Verification threshold (0.55 recommended)
            device: Device to load model on

        Returns:
            RelationVerifier instance
        """
        from transformers import AutoModel, AutoTokenizer

        device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Load config
        config = VerifierConfig(threshold=threshold)

        # Try to load from HF Hub or local path
        if os.path.isfile(model_id):
            checkpoint_path = model_id
        else:
            try:
                checkpoint_path = hf_hub_download(
                    repo_id=model_id,
                    filename="verifier.pt"
                )
            except Exception:
                # Try .bin extension
                checkpoint_path = hf_hub_download(
                    repo_id=model_id,
                    filename="pytorch_model.bin"
                )

        # Load encoder
        encoder = AutoModel.from_pretrained(config.encoder_name)
        tokenizer = AutoTokenizer.from_pretrained(config.encoder_name)

        # Load verifier weights
        model = RelationVerifierModel(config)
        state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)

        # Handle different checkpoint formats
        if "model_state_dict" in state_dict:
            model.load_state_dict(state_dict["model_state_dict"])
        else:
            model.load_state_dict(state_dict)

        return cls(model, encoder, tokenizer, config, device)

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        threshold: float = 0.55,
        device: str = None
    ) -> "RelationVerifier":
        """Load verifier from local checkpoint file."""
        return cls.from_pretrained(checkpoint_path, threshold=threshold, device=device)

    def _get_token_embeddings(self, text: str) -> torch.Tensor:
        """Get token embeddings for text."""
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            return_offsets_mapping=True
        )
        offset_mapping = inputs.pop("offset_mapping")[0]
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.encoder(**inputs)
            embeddings = outputs.last_hidden_state[0]  # [seq_len, hidden]

        return embeddings, offset_mapping

    def _char_to_token_idx(
        self,
        char_start: int,
        char_end: int,
        offset_mapping: torch.Tensor
    ) -> Tuple[int, int]:
        """Convert character offsets to token indices."""
        token_start, token_end = None, None

        for idx, (start, end) in enumerate(offset_mapping.tolist()):
            if start == end == 0:  # Special token
                continue
            if token_start is None and start <= char_start < end:
                token_start = idx
            if start < char_end <= end:
                token_end = idx + 1
                break

        # Fallback
        if token_start is None:
            token_start = 1
        if token_end is None:
            token_end = len(offset_mapping) - 1

        return token_start, token_end

    def _get_span_embedding(
        self,
        embeddings: torch.Tensor,
        token_start: int,
        token_end: int
    ) -> torch.Tensor:
        """Get mean pooled embedding for span."""
        if token_start >= token_end:
            token_end = token_start + 1
        if token_end > len(embeddings):
            token_end = len(embeddings)
        return embeddings[token_start:token_end].mean(dim=0)

    def _get_between_embedding(
        self,
        embeddings: torch.Tensor,
        head_end: int,
        tail_start: int
    ) -> torch.Tensor:
        """Get embedding for tokens between head and tail."""
        if head_end >= tail_start:
            # Entities overlap or adjacent - use mean of both
            return embeddings.mean(dim=0)
        return embeddings[head_end:tail_start].mean(dim=0)

    def _bucket_distance(self, distance: int) -> int:
        """Convert token distance to bucket index."""
        buckets = self.config.distance_buckets
        for i, threshold in enumerate(buckets):
            if distance <= threshold:
                return i
        return len(buckets)

    def _get_relation_embedding(self, relation: str) -> torch.Tensor:
        """Encode relation name and return [CLS] embedding."""
        inputs = self.tokenizer(relation, return_tensors="pt", truncation=True, max_length=64)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.encoder(**inputs)
            # Use [CLS] token for relation representation
            return outputs.last_hidden_state[0, 0]

    def _build_features(
        self,
        text: str,
        head_span: Dict,
        tail_span: Dict,
        relation: str,
        embeddings: torch.Tensor,
        offset_mapping: torch.Tensor,
        rel_emb: torch.Tensor = None
    ) -> torch.Tensor:
        """Build feature vector for verification."""
        # Get token indices (first token of span, matching training)
        head_tok = self._char_to_token_idx(
            head_span["start"], head_span["end"], offset_mapping
        )[0]
        tail_tok = self._char_to_token_idx(
            tail_span["start"], tail_span["end"], offset_mapping
        )[0]

        # Use first token of each entity (matching training approach)
        head_emb = embeddings[head_tok]
        tail_emb = embeddings[tail_tok]

        # Between-span embedding
        start_pos = min(head_tok, tail_tok)
        end_pos = max(head_tok, tail_tok)
        if end_pos - start_pos <= 1:
            # Adjacent or same position - use average
            between_emb = (head_emb + tail_emb) / 2
        else:
            # Mean of tokens between head and tail (excluding endpoints)
            between_emb = embeddings[start_pos + 1:end_pos].mean(dim=0)

        # Relation embedding (encode relation name separately)
        if rel_emb is None:
            rel_emb = self._get_relation_embedding(relation)

        # Distance features
        distance = abs(head_tok - tail_tok)
        dist_bucket = self._bucket_distance(distance)
        dist_emb = self.model.dist_embedding(
            torch.tensor([dist_bucket], device=self.device)
        )[0]

        # Order feature
        order = torch.tensor([1.0 if head_tok < tail_tok else 0.0], device=self.device)

        # Concatenate all features
        features = torch.cat([head_emb, tail_emb, between_emb, rel_emb, dist_emb, order])
        return features

    def verify_single(
        self,
        text: str,
        head: Dict,
        tail: Dict,
        relation: str
    ) -> Tuple[bool, float]:
        """
        Verify a single relation instance.

        Args:
            text: Source text
            head: Head entity dict with 'text', 'start', 'end'
            tail: Tail entity dict with 'text', 'start', 'end'
            relation: Relation type name

        Returns:
            (is_valid, score) tuple
        """
        embeddings, offset_mapping = self._get_token_embeddings(text)
        features = self._build_features(
            text, head, tail, relation, embeddings, offset_mapping
        )

        with torch.no_grad():
            logit = self.model(features.unsqueeze(0))
            score = torch.sigmoid(logit).item()

        return score >= self.config.threshold, score

    def verify(
        self,
        text: str,
        relations: Dict[str, List],
        return_scores: bool = True
    ) -> Dict[str, List]:
        """
        Verify all relations extracted from text.

        Args:
            text: Source text
            relations: Dict mapping relation types to list of instances
                       Each instance should have 'head' and 'tail' dicts
            return_scores: If True, add 'verifier_score' to each instance

        Returns:
            Filtered relations dict with only verified instances
        """
        if not relations:
            return {}

        # Get embeddings once for efficiency
        embeddings, offset_mapping = self._get_token_embeddings(text)

        # Cache relation embeddings
        rel_embeddings = {}

        verified = {}
        for rel_type, instances in relations.items():
            verified[rel_type] = []

            # Get or compute relation embedding
            if rel_type not in rel_embeddings:
                rel_embeddings[rel_type] = self._get_relation_embedding(rel_type)
            rel_emb = rel_embeddings[rel_type]

            for instance in instances:
                # Handle both tuple and dict formats
                if isinstance(instance, tuple):
                    head, tail = instance
                    head = {"text": head, "start": text.find(head), "end": text.find(head) + len(head)}
                    tail = {"text": tail, "start": text.find(tail), "end": text.find(tail) + len(tail)}
                else:
                    head = instance.get("head", {})
                    tail = instance.get("tail", {})

                # Skip if missing position info
                if "start" not in head or "start" not in tail:
                    continue

                # Build features and verify
                features = self._build_features(
                    text, head, tail, rel_type, embeddings, offset_mapping, rel_emb=rel_emb
                )

                with torch.no_grad():
                    logit = self.model(features.unsqueeze(0))
                    score = torch.sigmoid(logit).item()

                if score >= self.config.threshold:
                    if return_scores:
                        if isinstance(instance, dict):
                            instance = instance.copy()
                            instance["verifier_score"] = score
                        else:
                            instance = {"head": head, "tail": tail, "verifier_score": score}
                    verified[rel_type].append(instance)

        return verified

    def set_threshold(self, threshold: float):
        """Update verification threshold."""
        self.config.threshold = threshold
