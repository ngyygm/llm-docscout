"""LLM clients for DocScout rollouts.

- `StubClient`: deterministic, CPU-only heuristic. Produces formatted ACTION
  text (so the full parse path is exercised) by keyword-searching the question,
  reading top candidates, and extracting a candidate answer. NOT optimal — its
  job is to give a non-trivial trajectory so R0 can verify env/reward/metrics
  plumbing and confirm the reward *distinguishes* efficient vs wasteful behavior.
- `OpenAIClient`: thin wrapper over an OpenAI-compatible endpoint (e.g. vLLM
  serve) for real GRPO rollouts on GPU.
"""

from __future__ import annotations

import re
from typing import Any, Protocol

from docscout.env.search_env import SearchEnv


class LLMClient(Protocol):
    def act(self, env: SearchEnv, messages: list[dict]) -> str: ...


# tokens that signal a numeric answer is expected
_NUMERIC_HINTS = ("how many", "minutes", "days", "retries", "rate limit", "per-minute", "retention", "delay", "limit")
_APPROVER_HINTS = ("who must approve", "approve", "sign-off")
_COMPARE_HINTS = ("which has a higher", "which has a higher")


def _entity_from_question(q: str) -> str:
    """Grab capitalized product/entity names from the question as search keywords."""
    caps = re.findall(r"\b[A-Z][A-Za-z0-9]+(?:[-][A-Za-z0-9]+)?\b", q)
    # drop common sentence starters
    caps = [c for c in caps if c.lower() not in {"which", "reply", "how", "what", "who"}]
    # also add lowercase content words
    words = [w for w in re.findall(r"[a-z]{4,}", q.lower())
             if w not in {"which", "reply", "with", "single", "entity", "name", "must", "after",
                          "approval", "does", "have", "higher", "minutes", "days", "retries"}]
    keys = caps + words
    return " ".join(keys[:4]) if keys else q


def _extract_answer(q: str, env: SearchEnv) -> str:
    """Heuristic answer extraction from read content. Partial credit only."""
    blob = " ".join(f"{e.doc_id}.{e.section_id}" for e in env.read_log)
    # rebuild a text blob of read section contents
    contents = []
    seen = set()
    for e in env.read_log:
        k = (e.doc_id, e.section_id)
        if k in seen:
            continue
        seen.add(k)
        res = env.store.read(e.doc_id, e.section_id)
        if res:
            contents.append(res["content"])
    text = " ".join(contents)
    ql = q.lower()
    if any(h in ql for h in _COMPARE_HINTS):
        caps = re.findall(r"\b[A-Z][A-Za-z0-9-]+\b", q)
        return caps[0] if caps else ""
    if any(h in ql for h in _APPROVER_HINTS):
        m = re.search(r"sign-off from the ([a-z\s]+?)\.", text)
        if m:
            return m.group(1).strip()
        return ""
    if any(h in ql for h in _NUMERIC_HINTS):
        nums = re.findall(r"\b\d+\b", text)
        return nums[0] if nums else ""
    return ""


class StubClient:
    """Heuristic agent. Keyword search → read top candidates → answer."""

    def __init__(self, max_reads: int = 3):
        self.max_reads = max_reads

    def act(self, env: SearchEnv, messages: list[dict]) -> str:
        # decide based on env bookkeeping
        if env.n_search == 0:
            q = _entity_from_question(env.question)
            return f"ACTION: search\nquery: {q}"
        # pick top candidate the agent has NOT committed to read yet (snippets don't count)
        committed = env.committed_read_uids()
        for hit in env.current_candidates:
            uid = (hit["doc_id"], hit["section_id"])
            if uid not in committed and env.n_read < self.max_reads:
                return (f"ACTION: read\ndoc_id: {hit['doc_id']}\nsection_id: {hit['section_id']}")
            # occasionally expand a neighbor to exercise the EXPAND action
        if env.n_expand == 0 and env.n_read > 0 and env.current_candidates:
            hit = env.current_candidates[0]
            res = env.store.read(hit["doc_id"], hit["section_id"])
            if res and res.get("has_next"):
                return (f"ACTION: expand\ndoc_id: {hit['doc_id']}\nsection_id: {hit['section_id']}\ndirection: right")
        # otherwise answer
        ans = _extract_answer(env.question, env)
        ev = ""
        if env.read_uids():
            first = next(iter(env.read_uids()))
            ev = f"{first[0]}:{first[1]}"
        return f"ACTION: answer\ntext: {ans}\nevidence: {ev}"


import re as _re


def _strip_think(text: str) -> str:
    """Remove Qwen3 <think>...</think> blocks (and stray tags) from model output."""
    text = _re.sub(r"<think>.*?</think>", "", text, flags=_re.DOTALL)
    text = _re.sub(r"</?think>", "", text)
    return text.strip()


class OpenAIClient:

    def __init__(self, model: str, base_url: str | None = None, api_key: str = "EMPTY",
                 temperature: float = 0.7, max_tokens: int = 256, seed: int | None = None, **kwargs: Any):
        try:
            from openai import OpenAI  # optional dependency
        except ImportError as e:
            raise ImportError("OpenAIClient requires `pip install openai`") from e
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.seed = seed
        self.client = OpenAI(base_url=base_url, api_key=api_key)

    def _extra(self):
        eb = {"chat_template_kwargs": {"enable_thinking": False}}
        if self.seed is not None:
            eb["seed"] = self.seed
        return eb

    def act(self, env: SearchEnv, messages: list[dict]) -> str:
        resp = self.client.chat.completions.create(
            model=self.model, messages=messages, temperature=self.temperature,
            max_tokens=self.max_tokens, extra_body=self._extra())
        return _strip_think(resp.choices[0].message.content or "")

    def complete(self, prompt: str, max_tokens: int = 64, temperature: float = 0.0) -> str:
        resp = self.client.chat.completions.create(
            model=self.model, messages=[{"role": "user", "content": prompt}],
            temperature=temperature, max_tokens=max_tokens, extra_body=self._extra())
        return _strip_think(resp.choices[0].message.content or "")


class HFClient:
    """Local HuggingFace inference (transformers). Reliable when vLLM serve hangs.

    Implements the LLMClient protocol (act) for the agent loop, plus `complete`
    for direct prompt→text (oracle/RAG baselines). Qwen3 thinking is disabled
    for speed on short-answer evals.
    """

    def __init__(self, model_name: str, device: str = "cuda", dtype: str = "bfloat16",
                 temperature: float = 0.7, max_new_tokens: int = 256, enable_thinking: bool = False):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        assert torch.cuda.is_available(), "HFClient requires CUDA (GPU); CPU is not supported."
        self.model_name = model_name
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.enable_thinking = enable_thinking
        self.tok = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
        # force all weights onto the single visible GPU (CUDA_VISIBLE_DEVICES sets which one)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=getattr(torch, dtype), device_map=device, local_files_only=True)
        # hard verify the model actually landed on GPU, never CPU
        first_dev = str(next(self.model.parameters()).device)
        assert "cuda" in first_dev, f"model loaded on {first_dev}, expected cuda — refusing CPU fallback"
        print(f"[HFClient] {model_name} on {first_dev} ({torch.cuda.get_device_name(0)}, "
              f"vrsm {torch.cuda.max_memory_allocated()//1e6:.0f}MB)", flush=True)
        self.model.eval()

    def _gen(self, prompt: str, max_new_tokens: int, temperature: float) -> str:
        import torch
        enc = self.tok(prompt, return_tensors="pt").to(self.model.device)
        do_sample = temperature > 0
        with torch.no_grad():
            out = self.model.generate(
                **enc, do_sample=do_sample,
                temperature=max(temperature, 1e-3) if do_sample else 1.0,
                max_new_tokens=max_new_tokens)
        new = out[0, enc["input_ids"].shape[1]:]
        return self.tok.decode(new, skip_special_tokens=True)

    def act(self, env: SearchEnv, messages: list[dict]) -> str:
        prompt = self.tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=self.enable_thinking)
        return self._gen(prompt, self.max_new_tokens, self.temperature)

    def complete(self, prompt: str, max_tokens: int = 64, temperature: float = 0.0) -> str:
        # wrap in chat template so an instruct-tuned model follows the instruction
        text = self.tok.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True,
            enable_thinking=self.enable_thinking)
        return self._gen(text, max_tokens, temperature).strip()
