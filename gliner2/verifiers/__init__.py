"""
Verifiers for GLiNER2 relation extraction.

Provides semantic verification to filter false positive relations.
"""

from .relation_verifier import (
    RelationVerifier,
    RelationVerifierModel,
    VerifierConfig,
)

__all__ = [
    "RelationVerifier",
    "RelationVerifierModel",
    "VerifierConfig",
]
