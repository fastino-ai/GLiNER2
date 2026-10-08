"""
Test: SchemaTransformer must pickle, so DataLoader workers can receive it.

`_tokenize_cached` is a `functools.lru_cache` wrapper, and those are never
picklable. It was the only unpicklable attribute on the processor -- the fast
tokenizer itself pickles fine -- which meant a DataLoader could not use workers
anywhere the collator has to be pickled to reach them: `spawn` (the macOS
default) and `forkserver` (Linux's default from Python 3.14).

Dropping the cache on pickle is safe because it memoises a deterministic
function, so a rebuilt cache returns identical tokens.

Run:
    pytest tests/test_processor_pickling.py
"""

import functools
import pickle

from gliner2.processor import SchemaTransformer

TEXTS = [
    "Ada Lovelace wrote the first algorithm.",
    "兰州肉苁蓉为列当科肉苁蓉属下的一个种",
    "Kahramanmaraş merkezli depremler",
]


def _processor(tokenizer):
    return SchemaTransformer(tokenizer=tokenizer, token_pooling="first")


def test_processor_round_trips_through_pickle(tiny_tokenizer):
    processor = _processor(tiny_tokenizer)
    assert isinstance(pickle.loads(pickle.dumps(processor)), SchemaTransformer)


def test_lru_cache_is_excluded_from_the_pickled_state(tiny_tokenizer):
    processor = _processor(tiny_tokenizer)
    assert isinstance(processor._tokenize_cached, functools._lru_cache_wrapper)
    assert "_tokenize_cached" not in processor.__getstate__()


def test_cache_is_rebuilt_after_unpickling(tiny_tokenizer):
    restored = pickle.loads(pickle.dumps(_processor(tiny_tokenizer)))
    assert isinstance(restored._tokenize_cached, functools._lru_cache_wrapper)


def test_tokenisation_is_unchanged_by_a_round_trip(tiny_tokenizer):
    processor = _processor(tiny_tokenizer)
    restored = pickle.loads(pickle.dumps(processor))
    for text in TEXTS:
        assert processor._tokenize_text(text) == restored._tokenize_text(text)
