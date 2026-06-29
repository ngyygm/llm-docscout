"""Centralized token counting for DocScout's read-budget accounting.

WHY THIS EXISTS (round-result.md finding #1): the env originally counted read
cost with `len(text.split())` — i.e. WORDS, not BPE tokens. The paper's headline
metric is "accuracy-per-read-TOKEN", so word counts made it名不副实 (measured
BPE/word ≈ 1.21 on synth). This module makes `token_len` a real BPE count.

Resolution order (cached, lazy, offline-safe):
  1. An explicitly configured HF tokenizer (the policy model's), via
     DOCSCOUT_TOKENIZER (a HF id or local path). On the GPU box this should point
     at /mnt/workspace/zsxdata/local-model/Qwen/... so train/eval agree.
  2. tiktoken `cl100k_base` if available offline (good generic BPE proxy).
  3. Word count × 1.3 fallback (never crashes; keeps CPU smoke tests dependency-free).

`count_tokens(text)` is the single entry point used by Section/SectionRef.token_len
and the snippet cost in search_env. Set the tokenizer once via `set_tokenizer(...)`
(or the env var) before a run so accounting is consistent end-to-end.
"""

from __future__ import annotations

import os
from typing import Callable

# Module-level cached counter (a callable: str -> int). None until first use.
_COUNTER: Callable[[str], int] | None = None
_BACKEND: str = "uninitialized"

_ENV_TOKENIZER = "DOCSCOUT_TOKENIZER"
_WORD_TO_BPE = 1.3  # fallback multiplier (measured ~1.21-1.5 on synth/real text)


def _build_hf_counter(name_or_path: str) -> Callable[[str], int] | None:
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(name_or_path, trust_remote_code=True)

        def _count(text: str) -> int:
            if not text:
                return 0
            return len(tok.encode(text, add_special_tokens=False))

        return _count
    except Exception:
        return None


def _build_tiktoken_counter() -> Callable[[str], int] | None:
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")

        def _count(text: str) -> int:
            if not text:
                return 0
            return len(enc.encode(text))

        return _count
    except Exception:
        return None


def _build_word_counter() -> Callable[[str], int]:
    def _count(text: str) -> int:
        if not text:
            return 0
        return max(1, round(len(text.split()) * _WORD_TO_BPE))

    return _count


def _init() -> None:
    """Resolve a counter once, honoring the env var, with graceful fallback."""
    global _COUNTER, _BACKEND
    if _COUNTER is not None:
        return
    name = os.environ.get(_ENV_TOKENIZER)
    if name:
        c = _build_hf_counter(name)
        if c is not None:
            _COUNTER, _BACKEND = c, f"hf:{name}"
            return
    c = _build_tiktoken_counter()
    if c is not None:
        _COUNTER, _BACKEND = c, "tiktoken:cl100k_base"
        return
    _COUNTER, _BACKEND = _build_word_counter(), "words*1.3"


def set_tokenizer(name_or_path: str | None) -> str:
    """Force a specific tokenizer (HF id/path). None -> re-resolve from env/fallback.

    Returns the resulting backend label. Call this at the start of a run to pin
    the tokenizer (e.g. to the policy model) so cost accounting is consistent.
    """
    global _COUNTER, _BACKEND
    _COUNTER = None
    if name_or_path:
        c = _build_hf_counter(name_or_path)
        if c is not None:
            _COUNTER, _BACKEND = c, f"hf:{name_or_path}"
            return _BACKEND
    _init()
    return _BACKEND


def count_tokens(text: str) -> int:
    """Real (BPE) token count for read-budget accounting. The single entry point."""
    if _COUNTER is None:
        _init()
    assert _COUNTER is not None
    return _COUNTER(text)


def backend() -> str:
    """Which tokenizer backend is active (for logging / reproducibility)."""
    if _COUNTER is None:
        _init()
    return _BACKEND
