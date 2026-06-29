"""The DocScout search/read/expand/answer environment.

State held for the policy = the three-tier memory (RESEARCH_BRIEF.md §七):
  1. question            (permanent)
  2. action_log          (compact: "search(q) -> 5 hits", "read(doc_03.4)")
  3. current_candidates  (last search snippets) + read_log (sections read)

The env is the authoritative accountant: it tracks *which sections were read*
and *how many document tokens entered context* — both needed by the read-budget
reward (esp. the efficiency-ratio term, which attributes gold-evidence tokens
among all tokens read).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from docscout.env.docstore import DocStore
from docscout.types import Action, ActionType, QAInstance, StepResult


@dataclass
class EnvConfig:
    max_steps: int = 8
    max_read_tokens: int = 2000
    search_k: int = 5
    snippet_token_cost: bool = True  # snippets enter context too (cheap by design)
    answer_required: bool = True     # if True, step_budget without answer => low reward signal
    rerank: bool = False             # BM25 pool -> cross-encoder rerank (lifts paraphrase R@5)
    dynamic_max_steps: bool = True   # if True, per-instance max_steps from meta.oracle_min_steps

    def effective_max_steps(self, instance) -> int:
        """Compute per-instance max_steps, with static max_steps as fallback.

        If dynamic_max_steps is True and the instance has meta.oracle_min_steps,
        override: max_steps = min(meta.oracle_min_steps + 2, 12).
        Otherwise, fall back to the configured max_steps (default 8).
        """
        if not self.dynamic_max_steps:
            return self.max_steps
        oracle = (instance.meta or {}).get("oracle_min_steps", None)
        if oracle is not None and isinstance(oracle, (int, float)) and oracle > 0:
            return min(int(oracle) + 2, 12)
        return self.max_steps


@dataclass
class ReadEntry:
    doc_id: str
    section_id: str
    token_len: int
    source: str  # "search_snippet" | "read" | "expand"


class SearchEnv:
    def __init__(self, instance: QAInstance, docstore: DocStore, config: EnvConfig | None = None):
        self.instance = instance
        self.store = docstore
        self.cfg = config or EnvConfig()
        self.reset()

    # ------------------------------------------------------------------ reset
    def reset(self):
        self.action_log: list[str] = []
        self.current_candidates: list[dict] = []
        self.read_log: list[ReadEntry] = []
        self.total_read_tokens = 0
        self.n_steps = 0
        self.n_search = self.n_read = self.n_expand = 0
        self.done = False
        self.terminated_by = ""
        self.final_answer = ""
        self.final_evidence: list[tuple[str, str]] = []
        return self

    # ------------------------------------------------------------------ state
    @property
    def question(self) -> str:
        return self.instance.question

    def read_uids(self) -> set[tuple[str, str]]:
        """All sections whose content entered context (snippet OR committed read)."""
        return {(e.doc_id, e.section_id) for e in self.read_log}

    def committed_read_uids(self) -> set[tuple[str, str]]:
        """Sections the agent *committed* to read (READ/EXPAND), not mere snippets."""
        return {(e.doc_id, e.section_id) for e in self.read_log if e.source != "search_snippet"}

    @property
    def committed_read_tokens(self) -> int:
        return sum(e.token_len for e in self.read_log if e.source != "search_snippet")

    def gold_tokens_read(self, committed_only: bool = False) -> int:
        """Tokens of gold-evidence content that entered context.

        committed_only=True restricts to READ/EXPAND (verified evidence); False
        (default) counts snippets too — the anti-context-pollution signal for the
        efficiency ratio.
        """
        gold = self.instance.gold_sections()
        if committed_only:
            entries = [e for e in self.read_log if e.source != "search_snippet"]
        else:
            entries = self.read_log
        return sum(e.token_len for e in entries if (e.doc_id, e.section_id) in gold)

    def return_all_read(self) -> str:
        """Return the full read_log content for analysis / debugging.

        Iterates every entry in `env.read_log` (including search_snippets),
        fetches the actual content from the DocStore, and formats it as
        a flat text block.  Unlike `_recent_reads()` this does **no**
        truncation or deduplication — every byte that entered context is
        surfaced.

        Returns an empty string when nothing has been read yet.
        """
        seen: set[tuple[str, str]] = set()
        parts: list[str] = []
        for e in self.read_log:
            k = (e.doc_id, e.section_id)
            if k in seen:
                continue
            seen.add(k)
            res = self.store.read(e.doc_id, e.section_id)
            if res:
                parts.append(f"[{e.doc_id}.{e.section_id}] {res['content']}")
        return "\n".join(parts)

    # ------------------------------------------------------------------ step
    def step(self, action: Action) -> StepResult:
        if self.done:
            return StepResult(observation="(episode already ended)", done=True)

        self.n_steps += 1
        over_budget = False

        if action.type == ActionType.SEARCH:
            obs, rt = self._do_search(action)
        elif action.type == ActionType.READ:
            obs, rt = self._do_read(action)
        elif action.type == ActionType.EXPAND:
            obs, rt = self._do_expand(action)
        elif action.type == ActionType.ANSWER:
            return self._do_answer(action)
        else:
            obs, rt = f"(unknown action: {action.type})", 0

        self.total_read_tokens += rt
        # budget enforcement
        if self.n_steps >= self.cfg.max_steps:
            self.done = True
            self.terminated_by = "step_budget"
            obs += "\n[step budget exhausted — you must answer now]"
            over_budget = True
        if self.total_read_tokens >= self.cfg.max_read_tokens:
            self.done = True
            if self.terminated_by == "":
                self.terminated_by = "token_budget"
            obs += "\n[read-token budget exhausted — you must answer now]"
            over_budget = True
        return StepResult(observation=obs, done=self.done, read_tokens=rt,
                          info={"over_budget": over_budget})

    # ---------------------------------------------------------- action bodies
    def _do_search(self, action: Action) -> tuple[str, int]:
        query = str(action.args.get("query", "")).strip()
        self.n_search += 1
        self.action_log.append(f"search({query!r}) -> top-{self.cfg.search_k}")
        hits = self.store.search(query, k=self.cfg.search_k) if query else []
        self.current_candidates = hits
        if not hits:
            return "search returned no hits.", 0
        lines = [f"{i+1}. [{h['doc_id']}.{h['section_id']}] {h['section_title']} — {h['snippet']}"
                 for i, h in enumerate(hits)]
        obs = "SEARCH HITS (snippet only; use read to see full text):\n" + "\n".join(lines)
        rt = 0
        if self.cfg.snippet_token_cost:
            seen = self.read_uids()
            for h in hits:
                uid = (h["doc_id"], h["section_id"])
                if uid in seen:
                    continue  # dedup: don't re-charge a snippet already surfaced/read
                self.read_log.append(ReadEntry(h["doc_id"], h["section_id"],
                                               len(h["snippet"].split()), "search_snippet"))
                rt += len(h["snippet"].split())
                seen.add(uid)
        return obs, rt

    def _do_read(self, action: Action) -> tuple[str, int]:
        # support read-by-candidate-index: "read(idx=N)" -> read Nth current candidate
        if "idx" in action.args and action.args["idx"] not in (None, ""):
            try:
                idx = int(action.args["idx"]) - 1
            except (ValueError, TypeError):
                idx = -1
            if not self.current_candidates or not (0 <= idx < len(self.current_candidates)):
                self.n_read += 1
                self.action_log.append(f"read(idx) -> OUT OF RANGE")
                return "read: candidate index out of range.", 0
            h = self.current_candidates[idx]
            action.args["doc_id"] = h["doc_id"]
            action.args["section_id"] = h["section_id"]
        doc_id = str(action.args.get("doc_id", ""))
        sid = str(action.args.get("section_id", ""))
        self.n_read += 1
        res = self.store.read(doc_id, sid)
        if res is None:
            self.action_log.append(f"read({doc_id}.{sid}) -> NOT FOUND")
            return f"read({doc_id}.{sid}): no such section.", 0
        if (doc_id, sid) in self.committed_read_uids():
            self.action_log.append(f"read({doc_id}.{sid}) -> ALREADY READ")
            return (f"read({doc_id}.{sid}): already read — no new content. "
                    f"If you have enough evidence, submit answer.", 0)
        self.action_log.append(f"read({doc_id}.{sid}) -> {res['token_len']} tok")
        self.read_log.append(ReadEntry(doc_id, sid, res["token_len"], "read"))
        nbrs = []
        if res["has_prev"]:
            nbrs.append("left")
        if res["has_next"]:
            nbrs.append("right")
        obs = (f"READ {doc_id}.{sid} [{res['section_title']}] ({res['token_len']} tok):\n"
               f"{res['content']}")
        if nbrs:
            obs += f"\n(neighbors available: {', '.join(nbrs)} — use expand)"
        return obs, res["token_len"]

    def _do_expand(self, action: Action) -> tuple[str, int]:
        doc_id = str(action.args.get("doc_id", ""))
        sid = str(action.args.get("section_id", ""))
        direction = str(action.args.get("direction", ""))
        self.n_expand += 1
        res = self.store.expand(doc_id, sid, direction)
        if res is None:
            self.action_log.append(f"expand({doc_id}.{sid},{direction}) -> none")
            return f"expand({doc_id}.{sid},{direction}): no neighbor in that direction.", 0
        if (doc_id, res["section_id"]) in self.committed_read_uids():
            self.action_log.append(f"expand({doc_id}.{sid},{direction}) -> ALREADY READ")
            return (f"expand: section {doc_id}.{res['section_id']} already read — no new content.", 0)
        self.action_log.append(f"expand({doc_id}.{sid},{direction}) -> {res['section_id']} ({res['token_len']} tok)")
        self.read_log.append(ReadEntry(doc_id, res["section_id"], res["token_len"], "expand"))
        obs = (f"EXPAND {direction} -> {doc_id}.{res['section_id']} [{res['section_title']}] "
               f"({res['token_len']} tok):\n{res['content']}")
        return obs, res["token_len"]

    def _do_answer(self, action: Action) -> StepResult:
        self.final_answer = str(action.args.get("text", "")).strip()
        ev = action.args.get("evidence") or []
        self.final_evidence = [(str(e.get("doc_id", "")), str(e.get("section_id", ""))) for e in ev if isinstance(e, dict)]
        self.action_log.append(f"answer({self.final_answer[:40]!r})")
        self.done = True
        self.terminated_by = "answer"
        return StepResult(observation=f"ANSWER submitted: {self.final_answer}", done=True)
