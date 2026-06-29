"""diagnostic_oracle_path.py — Oracle path analysis for v4_eval300.

For each instance in v4_eval300.json, this script:

1. Manually computes the shortest oracle path (minimum search+read+answer
   steps) that a perfect agent would follow.
2. Verifies the oracle path length ≤ max_steps (flags violations).
3. Verifies each intermediate step in the oracle path is reachable (no
   semantic dead-ends — i.e. the entity used for searching at each step
   can be found by BM25).
4. Checks for unintended shortcuts: does the final answer appear in any
   snippet from the FIRST search step?  Does any single search return
   two or more gold sections at once?  Can the answer be derived from
   a single read?
5. Outputs a detailed per-instance report + summary to
   results/diagnostic/oracle_path_report.json.

Usage:
    python -m scripts.diagnostic_oracle_path
    # or
    python scripts/diagnostic_oracle_path.py
"""

from __future__ import annotations

import json
import math
import re
import statistics as st
from collections import Counter
from pathlib import Path

# ---- BM25 (lightweight copy to avoid env import issues) --------------------

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text)]


def _snippet(text: str, max_words: int = 40) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text.strip()
    return " ".join(words[:max_words]).strip() + " ..."


class _BM25:
    """Minimal BM25 for computing snippet + gold-section retrievability offline."""

    def __init__(self, sections: list[dict]):
        # sections: list of {doc_id, section_id, title, text}
        self.refs = sections
        self.doc_tokens = [_tokenize(f"{s['title']} {s['text']}") for s in sections]
        self.N = len(sections)
        self.avgdl = sum(len(d) for d in self.doc_tokens) / self.N if self.N else 0.0
        self.df: dict[str, int] = {}
        for toks in self.doc_tokens:
            for t in set(toks):
                self.df[t] = self.df.get(t, 0) + 1
        self.idf = {t: math.log(1 + (self.N - n + 0.5) / (n + 0.5)) for t, n in self.df.items()}
        self.k1, self.b = 1.5, 0.75

    def _score(self, q_terms: list[str], idx: int) -> float:
        toks = self.doc_tokens[idx]
        dl = len(toks) or 1
        tf: dict[str, int] = {}
        for t in toks:
            tf[t] = tf.get(t, 0) + 1
        s = 0.0
        for q in q_terms:
            if q not in self.idf:
                continue
            f = tf.get(q, 0)
            s += self.idf[q] * (f * (self.k1 + 1)) / (f + self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1)))
        return s

    def search(self, query: str, k: int = 5) -> list[dict]:
        q_terms = _tokenize(query)
        if not q_terms:
            return []
        scored = [(self._score(q_terms, i), i) for i in range(self.N)]
        scored.sort(reverse=True)
        results = []
        for sc, idx in scored[:k]:
            ref = self.refs[idx]
            results.append({
                "doc_id": ref["doc_id"],
                "section_id": ref["section_id"],
                "title": ref["title"],
                "text": ref["text"],
                "snippet": _snippet(ref["text"], 40),
                "score": round(float(sc), 4),
            })
        return results


# ---- Oracle path analysis ------------------------------------------------


def _build_section_index(instance: dict) -> dict:
    """Build (doc_id, section_id) -> section dict for fast lookup."""
    idx = {}
    for doc in instance["docs"]:
        for sec in doc["sections"]:
            idx[(doc["doc_id"], sec["section_id"])] = {
                "doc_id": doc["doc_id"],
                "section_id": sec["section_id"],
                "title": sec["title"],
                "text": sec["text"],
            }
    return idx


def _build_doc_entity_map(instance: dict) -> dict:
    """Map doc_id -> entity name from question context (first entity mentioned)."""
    # The entity names are from ENTITY_NAMES in the generator. We can detect them
    # by checking which entity names appear in the question or doc titles.
    # Simpler: just use the meta chain_ents to figure out who's where.
    meta = instance["meta"]
    ents = meta.get("chain_ents", [])
    # We need to figure out which doc goes with which entity.
    # In the v4 generator, entity order matches doc order: entities[i] -> doc_{i+1:02d}
    # But the corpus docs may not all be in the dataset (8 docs per instance from 30 total).
    # Let's instead read the question for entity names.
    question = instance["question"]
    entity_names = [
        "AuroraPay", "BrightShip", "CobaltHR", "DeltaSync", "EchoVault", "FluxAPI",
        "GarnetMail", "HelioAuth", "IrisCache", "JunoQueue", "KestrelDB", "LumenCRM",
        "MiraDocs", "NexusBI", "OrcaLog", "PulseFax", "QuartzLedger", "RavenOCR",
        "SableNet", "TideForm", "UmbraSign", "VegaBatch", "WillowFeed", "XenonGate",
        "YonderSync", "ZephyrPay", "AeroKey", "BoltMail", "CrestAPI", "DuneLog",
    ]
    doc_entity: dict[str, str] = {}
    for doc in instance["docs"]:
        title = doc["title"]
        for ent in entity_names:
            if ent in title:
                doc_entity[doc["doc_id"]] = ent
                break
    return doc_entity


def _find_entity_doc(entity: str, instance: dict, doc_entity: dict) -> str | None:
    """Inverse of doc_entity: find which doc_id belongs to an entity."""
    for doc_id, ent in doc_entity.items():
        if ent == entity:
            return doc_id
    return None


def _get_gold_text(instance: dict, sec_idx: dict, ev: dict) -> str | None:
    sec = sec_idx.get((ev["doc_id"], ev["section_id"]))
    return sec["text"] if sec else None


def analyze_instance(instance: dict, env_max_steps: int = 8) -> dict:
    """Compute the oracle path and checks for one v4 chain instance.

    Returns a dict with detailed diagnostics for this instance.
    """
    iid = instance["instance_id"]
    meta = instance["meta"]
    kind = meta["kind"]
    K = meta["K"]  # number of hops (link steps)
    n_gold = meta["n_gold"]  # = K + 1
    oracle_min_steps = meta.get("oracle_min_steps", 2 * n_gold + 1)

    gold_evidence = instance["gold_evidence"]
    question = instance["question"]
    gold_answer = str(instance["gold_answer"])

    # ---- build indexes ----
    sec_idx = _build_section_index(instance)
    doc_entity = _build_doc_entity_map(instance)

    # a flat list of all section refs for BM25
    all_sections = []
    for doc in instance["docs"]:
        for sec in doc["sections"]:
            all_sections.append({
                "doc_id": doc["doc_id"],
                "section_id": sec["section_id"],
                "title": sec["title"],
                "text": sec["text"],
            })

    bm25 = _BM25(all_sections)

    # ---- Chain structure from meta ----
    chain_ents = meta.get("chain_ents", [])
    chain_val_attrs = meta.get("chain_val_attrs", [])
    chain_link_attrs = meta.get("chain_link_attrs", [])
    chain_values = meta.get("chain_values", [])

    result: dict = {
        "instance_id": iid,
        "kind": kind,
        "n_gold": n_gold,
        "oracle_min_steps": oracle_min_steps,
        "gold_answer": gold_answer,
        "chain_ents": chain_ents,
        "chain_val_attrs": chain_val_attrs,
        "chain_link_attrs": chain_link_attrs,
        "chain_values": chain_values,
        "gold_evidence": gold_evidence,
    }

    # ---- Check 1: Oracle path length vs max_steps ----
    result["env_max_steps"] = env_max_steps
    dynamic_max_steps = min(oracle_min_steps + 2, 12)
    result["dynamic_max_steps"] = dynamic_max_steps
    result["violates_max_steps"] = oracle_min_steps > dynamic_max_steps
    result["violates_static_max_steps"] = oracle_min_steps > env_max_steps

    # ---- Construct the oracle path ----
    # Oracle path for a K-hop chain:
    # Step 1:  search(start_entity_name) -> find e0's attribute section as snippet
    # Step 2:  read(e0's attribute section) -> get v0
    # Step 3:  search(v0 as value) -> find e1's link section (the one containing v0)
    # Step 4:  read(e1's value attribute section) -> get v1
    # ...repeat for each hop...
    # Step 2K+1: answer(v_K, unit)   [only 1 step for K+1 searches + K+1 reads + 1 answer]

    oracle_path: list[dict] = []
    search_queries: list[str] = []

    # Helper: get the correct search query for each step
    # For e0: search by entity name
    # For e_i (i>=1): search by the link value from previous step

    # Step 1: First search for start entity + the attribute
    step_num = 1
    first_query = f"{chain_ents[0]}"
    search_queries.append(first_query)

    # The first read needs to get e0's attribute section which has the value
    step_info = _simulate_search_read(
        bm25, sec_idx, step_num, first_query, gold_evidence[0],
        "search start entity", chain_ents[0], chain_val_attrs[0]
    )
    oracle_path.append(step_info)
    step_num += 1

    # Now loop through each hop
    for hop_idx in range(K):
        # The link value is chain_values[hop_idx]
        # Search for that value
        link_val = chain_values[hop_idx]
        link_attr = chain_link_attrs[hop_idx]
        next_entity = chain_ents[hop_idx + 1]

        # Search query for the next entity's LINK section
        # The link section is the section for attr `link_attr` of next_entity
        # But we don't know next_entity yet. We search for the VALUE + attribute name
        search_query = f"{link_val}"
        search_queries.append(search_query)

        # The search should return the section of next_entity that has the link attr value
        # That section contains text like "X enforces a rate limit of 123 requests..."
        # We need to find it: it's the section where link_attr == link_val for entity next_entity
        # But this section is NOT necessarily the gold evidence for this hop.
        # The gold evidence is the VALUE section of next_entity, not the LINK section.
        #
        # Wait — let me re-read the generator:
        # gold_evidence[0] = e0's val_attr section (the one containing its value)
        # gold_evidence[i] (i>=1) = e_i's val_attr section
        # So the oracle uses:
        #   search(entity_0_name) -> find ANY section of entity_0 (to know which doc)
        #   read(entity_0's val_attr section) -> get entity_0's value v0
        #
        # Actually, let me think more carefully about what the agent sees.
        # The oracle path is:
        # 1. search(start_entity_name) -> snippets of start_entity's sections
        #    (returns snippets from ALL sections of start_entity + maybe others)
        # 2. pick the gold evidence section (read) -> get v0
        # 3. search(v0 as number) -> find section of next_entity that contains this number
        #    (this is the link section, which has the form "entity's link_attr = v0")
        # 4. read the next_entity's gold_evidence (value attr) section -> get v1
        # 5. ... continue ...
        #
        # But wait — the agent reads multiple sections of the same entity to get values.
        # Actually, each gold_evidence section is its OWN entity's value attribute section.
        # So:
        # - gold_evidence[0] has chain_ents[0]'s chain_val_attrs[0] value
        # - gold_evidence[1] has chain_ents[1]'s chain_val_attrs[1] value
        # etc.
        #
        # The trick is that to FIND chain_ents[1], we need to search for chain_values[0]
        # (the value obtained from reading chain_ents[0]'s val_attr section).
        # That search returns the section of chain_ents[1] that has chain_link_attrs[0] = chain_values[0].
        # But that section is NOT the gold_evidence[1] section.
        # The gold_evidence[1] section is chain_ents[1]'s chain_val_attrs[1] section.
        #
        # So the oracle needs:
        # search(link_value) -> find the LINK section of next_entity
        # -> identifies which doc has next_entity
        # -> then read the VALUE section of next_entity

        # Let's figure out what the search for link_val returns
        search_result = bm25.search(search_query, k=5)
        # Check if any gold_evidence[hop_idx+1] section is in the results, or any
        # section of the next_entity's doc.

        # Find the next entity's doc
        next_doc_id = _find_entity_doc(next_entity, instance, doc_entity)

        # The link section: next_entity's section that has attr=link_attr, which
        # should contain the text with the value chain_values[hop_idx].
        # In the v4 generator: section_id = attr_index + 1
        link_attr_idx = list(_ATTR_NAMES).index(link_attr) + 1
        link_section_id = str(link_attr_idx)
        link_section = sec_idx.get((next_doc_id, link_section_id))

        # The value section: next_entity's section that has attr=next_val_attr
        next_val_attr = chain_val_attrs[hop_idx + 1]
        val_attr_idx = list(_ATTR_NAMES).index(next_val_attr) + 1
        val_section_id = str(val_attr_idx)
        val_section = sec_idx.get((next_doc_id, val_section_id))

        # Check if search for link_val returns the link section in top-k
        found_link = any(
            h["doc_id"] == next_doc_id and h["section_id"] == link_section_id
            for h in search_result
        )

        # Better yet, check if search for link_val returns ANY section of next_entity
        found_any_next = any(h["doc_id"] == next_doc_id for h in search_result)

        step_info = {
            "step": step_num,
            "action": "search",
            "query": search_query,
            "search_context": f"find entity where {link_attr} = {link_val} (linking from {chain_ents[hop_idx]})",
            "next_entity": next_entity,
            "next_doc_id": next_doc_id,
            "target_link_section": f"{next_doc_id}.{link_section_id}",
            "target_value_section": f"{next_doc_id}.{val_section_id}",
            "found_link_section": found_link,
            "found_any_section_of_target_entity": found_any_next,
            "top5_results": [
                {"doc_id": h["doc_id"], "section_id": h["section_id"],
                 "title": h["title"], "snippet": h["snippet"][:60]}
                for h in search_result
            ],
        }
        oracle_path.append(step_info)
        step_num += 1

        # Now the read step for the value section
        gold_ev = gold_evidence[hop_idx + 1]
        gold_text = _get_gold_text(instance, sec_idx, gold_ev)
        read_info = {
            "step": step_num,
            "action": "read",
            "target": f"{gold_ev['doc_id']}.{gold_ev['section_id']}",
            "entity": next_entity,
            "attribute": next_val_attr,
            "expected_value": chain_values[hop_idx + 1],
            "gold_text_excerpt": (gold_text[:100] if gold_text else None),
        }
        oracle_path.append(read_info)
        step_num += 1

    # Final step: answer
    oracle_path.append({
        "step": step_num,
        "action": "answer",
        "expected_answer": gold_answer,
    })

    result["oracle_path"] = oracle_path
    result["search_queries"] = search_queries

    # ---- Check 2: Is the oracle path reachable? ----
    reachable = True
    unreachable_reasons = []
    for s in oracle_path:
        if s["action"] == "search":
            if "found_any_section_of_target_entity" in s and not s["found_any_section_of_target_entity"]:
                # The BM25 search didn't return any section of the target entity
                unreachable_reasons.append(f"Step {s['step']}: search '{s['query']}' did not retrieve "
                                           f"target entity {s.get('next_entity')}")
                reachable = False
        elif s["action"] == "read":
            target = s["target"]
            if target not in _all_uids(instance):
                unreachable_reasons.append(f"Step {s['step']}: read target {target} not found in corpus")
                reachable = False

    result["oracle_reachable"] = reachable
    result["unreachable_reasons"] = unreachable_reasons

    # ---- Check 3: Shortcut analysis ----
    # 3a: Does the first search's snippets contain the final answer directly?
    first_search = bm25.search(first_query, k=5)
    answer_in_first_snippets = False
    answer_in_first_snippets_detail = []
    for h in first_search:
        if gold_answer in h["text"]:
            answer_in_first_snippets = True
            answer_in_first_snippets_detail.append({
                "doc_id": h["doc_id"],
                "section_id": h["section_id"],
                "snippet": h["snippet"],
            })

    # 3b: Can the final answer be derived from a single read of some section in the corpus?
    gold_answer_num = "".join(re.findall(r"\d+", gold_answer))
    shortcut_single_read = []
    for doc in instance["docs"]:
        for sec in doc["sections"]:
            if gold_answer_num and gold_answer_num in sec["text"]:
                shortcut_single_read.append({
                    "doc_id": doc["doc_id"],
                    "section_id": sec["section_id"],
                    "section_title": sec["title"],
                    "text_excerpt": sec["text"][:80],
                    "is_gold_evidence": any(
                        ev["doc_id"] == doc["doc_id"] and ev["section_id"] == sec["section_id"]
                        for ev in gold_evidence
                    ),
                })

    # 3c: Does any single search return multiple gold evidence sections? (over-retrieval)
    multi_gold_search = False
    for q in search_queries + [first_query]:
        hits = bm25.search(q, k=5)
        gold_in_hits = 0
        for h in hits:
            uid = (h["doc_id"], h["section_id"])
            if any(ev["doc_id"] == uid[0] and ev["section_id"] == uid[1] for ev in gold_evidence):
                gold_in_hits += 1
        if gold_in_hits >= 2:
            multi_gold_search = True

    # 3d: Does the answer value appear in the first entity's doc (before reading all gold)?
    e0_doc_id = _find_entity_doc(chain_ents[0], instance, doc_entity)
    answer_in_e0_doc = False
    if e0_doc_id:
        for doc in instance["docs"]:
            if doc["doc_id"] == e0_doc_id:
                for sec in doc["sections"]:
                    if gold_answer_num and gold_answer_num in sec["text"]:
                        answer_in_e0_doc = True
                        break

    result["shortcut_analysis"] = {
        "answer_in_first_search_snippets": {
            "present": answer_in_first_snippets,
            "details": answer_in_first_snippets_detail if answer_in_first_snippets else [],
        },
        "answer_in_single_read": {
            "count": len(shortcut_single_read),
            "details": shortcut_single_read,
        },
        "any_single_search_returns_multiple_gold": multi_gold_search,
        "answer_in_start_entity_doc": answer_in_e0_doc,
    }

    # ---- Summary verdict ----
    leak_level = "none"
    if answer_in_first_snippets:
        leak_level = "severe"  # answer visible from first search
    elif answer_in_e0_doc:
        leak_level = "moderate"  # answer in same doc as start entity
    elif len(shortcut_single_read) > 0:
        leak_level = "minor"  # answer visible somewhere in corpus in one read

    result["leak_level"] = leak_level
    result["verdict"] = "PASS" if (reachable and leak_level == "none" and not multi_gold_search) else "FLAG"

    return result


# ---- ATTR_NAMES from generator (copy for lookup) ----
_ATTR_NAMES = ["sync_delay", "retry_limit", "retention", "rate_limit", "backup_freq", "session_timeout"]


def _all_uids(instance: dict) -> set:
    uids = set()
    for doc in instance["docs"]:
        for sec in doc["sections"]:
            uids.add((doc["doc_id"], sec["section_id"]))
    return uids


# ---- Main ----

def main():
    data_path = Path("data/synth/v4_eval300.json")
    report_dir = Path("results/diagnostic")
    report_dir.mkdir(parents=True, exist_ok=True)

    instances = json.load(open(data_path))
    print(f"Loaded {len(instances)} instances from {data_path}")

    env_max_steps = 8  # from configs/env_default.yaml

    results = []
    summaries = {
        "total": len(instances),
        "violates_static_max_steps": 0,
        "violates_dynamic_max_steps": 0,
        "unreachable": 0,
        "answer_in_first_snippets": 0,
        "answer_in_single_read": {"0": 0, "1": 0, "2+": 0},
        "multi_gold_search": 0,
        "answer_in_start_entity_doc": 0,
        "leak_levels": Counter(),
        "verdicts": Counter(),
        "by_kind": {},
    }

    for i, inst in enumerate(instances):
        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(instances)}]")
        r = analyze_instance(inst, env_max_steps)
        results.append(r)

        # accumulate summaries
        if r["violates_static_max_steps"]:
            summaries["violates_static_max_steps"] += 1
        if r["violates_dynamic_max_steps"]:
            summaries["violates_dynamic_max_steps"] += 1
        if not r["oracle_reachable"]:
            summaries["unreachable"] += 1
        if r["shortcut_analysis"]["answer_in_first_search_snippets"]["present"]:
            summaries["answer_in_first_snippets"] += 1
        n_single = r["shortcut_analysis"]["answer_in_single_read"]["count"]
        if n_single == 0:
            summaries["answer_in_single_read"]["0"] += 1
        elif n_single == 1:
            summaries["answer_in_single_read"]["1"] += 1
        else:
            summaries["answer_in_single_read"]["2+"] += 1
        if r["shortcut_analysis"]["any_single_search_returns_multiple_gold"]:
            summaries["multi_gold_search"] += 1
        if r["shortcut_analysis"]["answer_in_start_entity_doc"]:
            summaries["answer_in_start_entity_doc"] += 1
        summaries["leak_levels"][r["leak_level"]] += 1
        summaries["verdicts"][r["verdict"]] += 1

        kind = r["kind"]
        if kind not in summaries["by_kind"]:
            summaries["by_kind"][kind] = {"n": 0, "violates": 0, "unreachable": 0, "answer_in_first_snippets": 0}
        summaries["by_kind"][kind]["n"] += 1
        if r["violates_static_max_steps"]:
            summaries["by_kind"][kind]["violates"] += 1
        if not r["oracle_reachable"]:
            summaries["by_kind"][kind]["unreachable"] += 1
        if r["shortcut_analysis"]["answer_in_first_search_snippets"]["present"]:
            summaries["by_kind"][kind]["answer_in_first_snippets"] += 1

    report = {
        "summary": summaries,
        "detailed": results,
        "config": {
            "data_path": str(data_path),
            "env_max_steps": env_max_steps,
            "bm25_k": 5,
            "snippet_words": 40,
        },
    }

    out_path = report_dir / "oracle_path_report.json"
    json.dump(report, open(out_path, "w"), indent=2, ensure_ascii=False)
    print(f"\nReport saved -> {out_path}")

    # Print summary table
    print(f"\n{'='*60}")
    print(f"ORACLE PATH REPORT SUMMARY")
    print(f"{'='*60}")
    print(f"Total instances:              {summaries['total']}")
    print(f"Violates static max_steps (8): {summaries['violates_static_max_steps']}")
    print(f"Violates dynamic max_steps:    {summaries['violates_dynamic_max_steps']}")
    print(f"Unreachable oracle paths:      {summaries['unreachable']}")
    print(f"Answer in first search snip:   {summaries['answer_in_first_snippets']}")
    print(f"Single-read answer shortcut:   0:{summaries['answer_in_single_read']['0']}  1:{summaries['answer_in_single_read']['1']}  2+:{summaries['answer_in_single_read']['2+']}")
    print(f"Multi-gold in one search:      {summaries['multi_gold_search']}")
    print(f"Answer in start entity doc:    {summaries['answer_in_start_entity_doc']}")
    print(f"\nLeak levels: {dict(summaries['leak_levels'])}")
    print(f"Verdicts:    {dict(summaries['verdicts'])}")
    print(f"\nBy kind:")
    for kind, ks in sorted(summaries["by_kind"].items()):
        print(f"  {kind:15s}: n={ks['n']:3d}  violates={ks['violates']}  unreachable={ks['unreachable']}  ans_in_first_snip={ks['answer_in_first_snippets']}")

    return report


if __name__ == "__main__":
    main()
