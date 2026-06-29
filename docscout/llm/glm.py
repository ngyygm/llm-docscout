"""GLM (OpenAI-compatible) LLM client for DocScout host-side tasks.

Design goals:
  - **No secrets in the repo.** The API key is read from the environment
    (DOCSCOUT_LLM_TOKEN) or an untracked YAML config — never hard-coded.
  - **Robust for batch jobs.** Timeout, exponential-backoff retry on transient
    errors (429/5xx/network), and a thread-pooled `batch()` for throughput.
  - **JSON mode helper.** `complete_json()` parses a JSON object out of the reply
    (tolerating ```json fences and leading prose) and retries on parse failure —
    essential for synth generation and LLM-as-judge where we need structured out.
  - **Thin + dependency-light.** Uses `requests`; no SDK lock-in. The endpoint is
    OpenAI-compatible (POST /v1/chat/completions).

Config resolution order (first hit wins) for each field:
  1. explicit argument to load_config()
  2. environment variable      (DOCSCOUT_LLM_TOKEN / _BASE_URL / _MODEL)
  3. YAML file                 (configs/llm_api.yaml, untracked; see .example)
  4. built-in default          (base_url / model only; token has NO default)

Example:
    from docscout.llm import load_config, LLMClient
    client = LLMClient(load_config())
    print(client.complete("Reply with exactly: PONG"))

    # structured:
    obj = client.complete_json("Return JSON {\"ok\": true}.")

    # batch (threaded):
    outs = client.batch(["q1", "q2", "q3"], max_workers=8)
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import requests

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

_DEFAULT_BASE_URL = "https://api3.orangeai.cc"
_DEFAULT_MODEL = "glm-5.2"
_DEFAULT_CONFIG_PATH = "configs/llm_api.yaml"

# Env var names (documented in configs/llm_api.example.yaml)
_ENV_TOKEN = "DOCSCOUT_LLM_TOKEN"
_ENV_BASE = "DOCSCOUT_LLM_BASE_URL"
_ENV_MODEL = "DOCSCOUT_LLM_MODEL"


@dataclass
class LLMConfig:
    token: str
    base_url: str = _DEFAULT_BASE_URL
    model: str = _DEFAULT_MODEL
    timeout: float = 120.0
    max_retries: int = 4
    temperature: float = 0.7
    max_tokens: int = 32768  # max OUTPUT tokens/reply; GLM-5.2 total context is 400k

    def masked(self) -> str:
        t = self.token or ""
        tail = t[-4:] if len(t) >= 4 else "????"
        return f"LLMConfig(model={self.model}, base_url={self.base_url}, token=sk-***{tail})"


def _load_yaml(path: str | Path) -> dict:
    p = Path(path)
    if not p.exists():
        return {}
    try:
        import yaml  # lazy; project already depends on yaml for configs
        data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def load_config(
    token: str | None = None,
    base_url: str | None = None,
    model: str | None = None,
    config_path: str | Path = _DEFAULT_CONFIG_PATH,
    **overrides: Any,
) -> LLMConfig:
    """Resolve an LLMConfig from arg > env > yaml > default.

    Raises a clear error if no token can be found, pointing at the two ways to
    supply one (env var or untracked yaml).
    """
    yml = _load_yaml(config_path)

    tok = token or os.environ.get(_ENV_TOKEN) or yml.get("token")
    base = base_url or os.environ.get(_ENV_BASE) or yml.get("base_url") or _DEFAULT_BASE_URL
    mdl = model or os.environ.get(_ENV_MODEL) or yml.get("model") or _DEFAULT_MODEL

    if not tok:
        raise RuntimeError(
            "No LLM API token found. Provide one via:\n"
            f"  1. env var:  export {_ENV_TOKEN}=sk-...\n"
            f"  2. yaml:     cp configs/llm_api.example.yaml {_DEFAULT_CONFIG_PATH}  "
            "(then fill in token; this path is gitignored)\n"
            "  3. argument: load_config(token=...)"
        )

    # numeric knobs may come from yaml too
    def _num(key: str, default: float | int) -> Any:
        if key in overrides:
            return overrides[key]
        if key in yml:
            return yml[key]
        return default

    return LLMConfig(
        token=tok,
        base_url=str(base).rstrip("/"),
        model=mdl,
        timeout=float(_num("timeout", 120.0)),
        max_retries=int(_num("max_retries", 4)),
        temperature=float(_num("temperature", 0.7)),
        max_tokens=int(_num("max_tokens", 32768)),
    )


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

_RETRYABLE_STATUS = {408, 409, 429, 500, 502, 503, 504, 529}


class LLMError(RuntimeError):
    pass


class LLMClient:
    """Thin OpenAI-compatible chat client with retry + JSON helpers."""

    def __init__(self, config: LLMConfig | None = None):
        self.cfg = config or load_config()
        self._session = requests.Session()

    # ------------------------------------------------------------------ core
    def _chat_raw(
        self,
        messages: list[dict],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_format: dict | None = None,
        stop: list[str] | None = None,
    ) -> dict:
        url = f"{self.cfg.base_url}/v1/chat/completions"
        payload: dict[str, Any] = {
            "model": self.cfg.model,
            "messages": messages,
            "temperature": self.cfg.temperature if temperature is None else temperature,
            "max_tokens": self.cfg.max_tokens if max_tokens is None else max_tokens,
        }
        if response_format is not None:
            payload["response_format"] = response_format
        if stop:
            payload["stop"] = stop
        headers = {
            "Authorization": f"Bearer {self.cfg.token}",
            "Content-Type": "application/json",
        }

        last_err: Exception | None = None
        for attempt in range(self.cfg.max_retries + 1):
            try:
                resp = self._session.post(url, headers=headers, json=payload, timeout=self.cfg.timeout)
                if resp.status_code in _RETRYABLE_STATUS:
                    raise LLMError(f"retryable HTTP {resp.status_code}: {resp.text[:200]}")
                if resp.status_code != 200:
                    raise LLMError(f"HTTP {resp.status_code}: {resp.text[:300]}")
                data = resp.json()
                if "choices" not in data or not data["choices"]:
                    raise LLMError(f"malformed response: {json.dumps(data)[:300]}")
                return data
            except (requests.RequestException, LLMError, ValueError) as e:
                last_err = e
                if attempt < self.cfg.max_retries:
                    # exponential backoff with light jitter (jitter from attempt, no RNG)
                    sleep_s = min(2.0 ** attempt, 30.0) + (attempt * 0.13)
                    time.sleep(sleep_s)
                else:
                    break
        raise LLMError(f"LLM call failed after {self.cfg.max_retries + 1} attempts: {last_err}")

    # ------------------------------------------------------------- complete
    def complete(
        self,
        prompt: str | list[dict],
        *,
        system: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        stop: list[str] | None = None,
    ) -> str:
        """Single completion. `prompt` is a user string or a full messages list.

        Returns the assistant message content (stripped).
        """
        if isinstance(prompt, str):
            messages: list[dict] = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})
        else:
            messages = prompt
        data = self._chat_raw(messages, temperature=temperature, max_tokens=max_tokens, stop=stop)
        return (data["choices"][0]["message"]["content"] or "").strip()

    # --------------------------------------------------------- complete_json
    def complete_json(
        self,
        prompt: str | list[dict],
        *,
        system: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        retries_on_parse: int = 2,
    ) -> Any:
        """Completion that must yield a JSON value. Tolerates ```json fences and
        leading/trailing prose; retries (lowering temperature) on parse failure.

        Raises LLMError if no valid JSON could be parsed after retries.
        """
        last_text = ""
        for i in range(retries_on_parse + 1):
            temp = 0.0 if i > 0 else (temperature if temperature is not None else 0.2)
            text = self.complete(prompt, system=system, temperature=temp, max_tokens=max_tokens)
            last_text = text
            parsed = _extract_json(text)
            if parsed is not None:
                return parsed
        raise LLMError(f"could not parse JSON from model output:\n{last_text[:400]}")

    # --------------------------------------------------------------- batch
    def batch(
        self,
        prompts: Sequence[str | list[dict]],
        *,
        system: str | None = None,
        max_workers: int = 8,
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_mode: bool = False,
        on_error: str = "none",  # "none" -> None placeholder | "raise"
    ) -> list[Any]:
        """Threaded batch completion preserving input order.

        Returns a list aligned with `prompts`; failed items are None when
        on_error="none" (default), so callers can .filter / inspect. Threading is
        I/O-bound friendly (HTTP), and the server tolerates concurrency.
        """
        from concurrent.futures import ThreadPoolExecutor

        results: list[Any] = [None] * len(prompts)

        def _one(i_p: tuple[int, Any]) -> None:
            i, p = i_p
            try:
                if json_mode:
                    results[i] = self.complete_json(
                        p, system=system, temperature=temperature, max_tokens=max_tokens
                    )
                else:
                    results[i] = self.complete(
                        p, system=system, temperature=temperature, max_tokens=max_tokens
                    )
            except Exception:
                if on_error == "raise":
                    raise
                results[i] = None

        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            list(ex.map(_one, enumerate(prompts)))
        return results

    # --------------------------------------------------------------- ping
    def ping(self) -> bool:
        """Lightweight connectivity check. Returns True on a 200 with content."""
        try:
            out = self.complete("Reply with exactly: PONG", max_tokens=10, temperature=0.0)
            return "PONG" in out.upper()
        except Exception:
            return False


# ---------------------------------------------------------------------------
# JSON extraction helper
# ---------------------------------------------------------------------------

_FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)


def _extract_json(text: str) -> Any | None:
    """Best-effort: parse a JSON object/array out of a model reply.

    Strategy: (1) try the whole string; (2) try fenced ```json block;
    (3) try the first {...} or [...] span via bracket matching.
    Returns the parsed value, or None if nothing parses.
    """
    text = (text or "").strip()
    if not text:
        return None
    # 1. whole string
    try:
        return json.loads(text)
    except Exception:
        pass
    # 2. fenced block
    m = _FENCE.search(text)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass
    # 3. first balanced {...} or [...]
    for open_ch, close_ch in (("{", "}"), ("[", "]")):
        start = text.find(open_ch)
        if start == -1:
            continue
        depth = 0
        for j in range(start, len(text)):
            if text[j] == open_ch:
                depth += 1
            elif text[j] == close_ch:
                depth -= 1
                if depth == 0:
                    candidate = text[start : j + 1]
                    try:
                        return json.loads(candidate)
                    except Exception:
                        break
    return None
