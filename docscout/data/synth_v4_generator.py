"""synth-v4: HARD multi-hop chain QA substrate for DocScout.

Motivation (user): synth-v3's RAG reaches 0.75-0.82 because questions are 1-2 hop
and retrievable by name. To showcase DocScout's multi-step reading advantage we
need questions that require CHAINING across documents, where one-shot RAG cannot
follow the chain (the intermediate values are not in the original query).

A K-hop chain question:
  "What is {a_K} of the system whose {L_{K-1}} equals the {a_{K-1}} of ... the
   system whose {L_0} equals the {a_0} of {entity_0}?"
To answer, an agent must: read entity_0's {a_0} section -> value v_0; search for
the system whose {L_0}=v_0 -> entity_1; read entity_1's {a_1} -> v_1; ... ; read
entity_K's {a_K} -> answer. RAG (one-shot, original query) retrieves entity_0's
section but cannot discover v_0/v_1/... (they are read, not named), so it fails.

Construction guarantees: 6 numeric attributes share the SAME set S of N unique
values, each assigned to the N entities via a per-attribute permutation. Hence a
value v in S appears exactly once per attribute -> "the system whose {L}=v" has a
unique solution. Section text embeds the value with its unit; gold evidence is the
chain's K+1 sections. Deterministic by seed; plugs into the existing env/reward.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from docscout.types import Document, EvidenceSpan, QAInstance, Section

# Numeric attributes (shared value space). (human_label, unit, prose_template)
ATTRS: dict[str, tuple[str, str, str]] = {
    "sync_delay": ("synchronization delay, in minutes", "minutes",
        "Once approved, {e} completes synchronization within {v} minutes."),
    "retry_limit": ("retry limit", "retries",
        "On failure, {e} retries the operation up to {v} times before escalating."),
    "retention": ("retention period, in days", "days",
        "Operational records of {e} are retained for {v} days."),
    "rate_limit": ("rate limit, in requests per minute", "requests per minute",
        "{e} enforces a rate limit of {v} requests per minute per tenant."),
    "backup_freq": ("backup frequency, in hours", "hours",
        "{e} captures a full backup every {v} hours."),
    "session_timeout": ("session timeout, in minutes", "minutes",
        "A {e} session expires after {v} minutes of inactivity."),
}
ATTR_NAMES = list(ATTRS.keys())

FILLER = [
    "This section describes the general operating context for the system.",
    "All terms follow the standard organizational glossary.",
    "Stakeholders should review this policy quarterly.",
    "Audit logs are collected centrally and reviewed monthly.",
    "Contact the operations team for environment-specific guidance.",
    "This policy supersedes all prior versions on the same topic.",
    "Implementation details may vary across deployment environments.",
]
SECTION_TITLES = ["Overview", "Synchronization and Propagation", "Failure Handling and Retries",
                  "Data Retention and Lifecycle", "Rate Limiting and Throttling",
                  "Backup and Recovery", "Session Management", "Monitoring",
                  "Incident Response", "Revision History"]
# which section title holds each attribute
ATTR_SECTION_TITLE = {
    "sync_delay": "Synchronization and Propagation", "retry_limit": "Failure Handling and Retries",
    "retention": "Data Retention and Lifecycle", "rate_limit": "Rate Limiting and Throttling",
    "backup_freq": "Backup and Recovery", "session_timeout": "Session Management",
}

ENTITY_NAMES = [
    "AuroraPay", "BrightShip", "CobaltHR", "DeltaSync", "EchoVault", "FluxAPI",
    "GarnetMail", "HelioAuth", "IrisCache", "JunoQueue", "KestrelDB", "LumenCRM",
    "MiraDocs", "NexusBI", "OrcaLog", "PulseFax", "QuartzLedger", "RavenOCR",
    "SableNet", "TideForm", "UmbraSign", "VegaBatch", "WillowFeed", "XenonGate",
    "YonderSync", "ZephyrPay", "AeroKey", "BoltMail", "CrestAPI", "DuneLog",
]


def _section_text(entity: str, attr: str | None, value: int | None, rng: random.Random) -> str:
    """Build a section. If attr is set, embed the value; else filler."""
    if attr is None:
        return ". ".join(rng.sample(FILLER, k=3)) + "."
    label, unit, prose = ATTRS[attr]
    body = prose.format(e=entity, v=value)
    # pad with realistic context so sections aren't trivially short
    extra = ". ".join(rng.sample(FILLER, k=2)) + "."
    return f"{body} {extra}"


def _build_doc(rng, doc_id, entity, attr_values: dict[str, int]) -> Document:
    """attr_values: attr->value for this entity. Build a doc with one section per attribute + filler."""
    sections = []
    # section per attribute (in SECTION_TITLES order) + a couple filler sections
    placed = set()
    # attribute sections
    for attr in ATTR_NAMES:
        title = ATTR_SECTION_TITLE[attr]
        sid = str(ATTR_NAMES.index(attr) + 1)
        sections.append(Section(section_id=sid, title=title,
                                text=_section_text(entity, attr, attr_values[attr], rng)))
        placed.add(title)
    # a couple filler sections
    sid = len(ATTR_NAMES) + 1
    for title in ["Overview", "Revision History"]:
        sections.append(Section(section_id=str(sid), title=title, text=_section_text(entity, None, None, rng)))
        sid += 1
    return Document(doc_id=doc_id, title=f"{entity} Operations Policy", sections=sections)


def generate_corpus_v4(n_docs: int = 30, seed: int = 0):
    """Build the corpus + the value lookup tables for chain construction.

    Returns (corpus, entity_attr_value, value_to_entity_per_attr).
      entity_attr_value[entity][attr] = value
      value_to_entity_per_attr[attr][value] = entity   (unique inverse)
    All attributes share the same value set S (n_docs unique values), each mapped
    to entities via an independent permutation -> every link is uniquely solvable.
    """
    rng = random.Random(seed)
    entities = ENTITY_NAMES[:n_docs]
    S = rng.sample(range(1, 1000), k=n_docs)  # shared unique value set
    entity_attr_value: dict[str, dict[str, int]] = {}
    val2ent: dict[str, dict[int, str]] = {a: {} for a in ATTR_NAMES}
    for ent in entities:
        entity_attr_value[ent] = {}
    for attr in ATTR_NAMES:
        perm = S[:]  # copy
        rng.shuffle(perm)  # independent permutation per attribute
        for ent, v in zip(entities, perm):
            entity_attr_value[ent][attr] = v
            val2ent[attr][v] = ent
    # build docs
    corpus = []
    for i, ent in enumerate(entities):
        corpus.append(_build_doc(rng, f"doc_{i+1:02d}", ent, entity_attr_value[ent]))
    return corpus, entities, entity_attr_value, val2ent, S


def _attr_human(attr):
    return ATTRS[attr][0]


def _chain_question(e0, val_attrs, link_attrs):
    """Flat step-by-step chain question (easier to parse/follow than nested)."""
    K = len(link_attrs)
    steps = [f"Step 1: find the {_attr_human(val_attrs[0])} of {e0}."]
    for i in range(K):
        steps.append(f"Step {i+2}: find the system whose {_attr_human(link_attrs[i])} EQUALS the value from step {i+1}; read that system's {_attr_human(val_attrs[i+1])}.")
    steps.append(f"Final answer: the value from step {K+1}'s {_attr_human(val_attrs[-1])}.")
    return " ".join(steps)


def build_chain(rng, entities, eav, val2ent, K):
    """Construct one K-hop chain. Returns (entity_path, val_attrs, link_attrs, values, gold_sections, answer)."""
    e0 = rng.choice(entities)
    val_attrs = [rng.choice(ATTR_NAMES)]
    link_attrs = []
    ents = [e0]
    values = [eav[e0][val_attrs[0]]]
    used_pairs = {(e0, val_attrs[0])}
    gold_attrs_per_entity = {e0: val_attrs[0]}  # which attr section to read per entity
    for i in range(K):
        v = values[-1]
        # link attribute L_i != current value attr, and the value v must map to some entity under L_i
        candidates = [a for a in ATTR_NAMES if a != val_attrs[-1]]
        rng.shuffle(candidates)
        e_next, L = None, None
        for L in candidates:
            if v in val2ent[L] and val2ent[L][v] not in {e for e in ents}:
                e_next = val2ent[L][v]
                break
        if e_next is None:
            return None  # retry
        # value attribute for e_next
        a_next_candidates = [a for a in ATTR_NAMES if a != L and (e_next, a) not in used_pairs]
        if not a_next_candidates:
            return None
        a_next = rng.choice(a_next_candidates)
        ents.append(e_next); link_attrs.append(L); val_attrs.append(a_next)
        values.append(eav[e_next][a_next])
        used_pairs.add((e_next, a_next))
        gold_attrs_per_entity[e_next] = (L, a_next)  # read L section (to confirm match) + a_next? we read a_next
    answer = values[-1]
    return ents, val_attrs, link_attrs, values, gold_attrs_per_entity, answer


def generate_instances(corpus, entities, eav, val2ent, n_instances, hop_distribution, seed):
    rng = random.Random(seed + 1)
    instances = []
    attempts = 0
    while len(instances) < n_instances and attempts < n_instances * 50:
        attempts += 1
        K = rng.choices(list(hop_distribution.keys()), weights=list(hop_distribution.values()))[0]
        chain = build_chain(rng, entities, eav, val2ent, K)
        if chain is None:
            continue
        ents, val_attrs, link_attrs, values, gold_attrs_per_entity, answer = chain
        q = _chain_question(ents[0], val_attrs, link_attrs)
        # gold evidence: for e0 read a_0 section; for each e_i (i>=1) read its a_i section
        gold = []
        doc_idx = {d.doc_id: d for d in corpus}
        ent_to_docid = {e: f"doc_{i+1:02d}" for i, e in enumerate(entities)}
        # e0's value section
        sid0 = str(ATTR_NAMES.index(val_attrs[0]) + 1)
        gold.append(EvidenceSpan(doc_id=ent_to_docid[ents[0]], section_id=sid0))
        for i in range(1, len(ents)):
            sid = str(ATTR_NAMES.index(val_attrs[i]) + 1)
            gold.append(EvidenceSpan(doc_id=ent_to_docid[ents[i]], section_id=sid))
        units = ATTRS[val_attrs[-1]][1]
        instances.append(QAInstance(
            instance_id=f"qa_{len(instances)+1:04d}",
            question=q, gold_answer=f"{answer} {units}",
            gold_evidence=gold, docs=corpus,
            meta={"kind": f"chain_{K+1}hop", "K": K, "n_gold": len(gold),
                  "start_entity": ents[0], "answer_attr": val_attrs[-1],
                  # chain structure for demo generation (doc-relative indices):
                  "chain_ents": ents, "chain_val_attrs": val_attrs,
                  "chain_link_attrs": link_attrs, "chain_values": values},
        ))
    return instances


def serialize(instances, path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    json.dump([{
        "instance_id": ins.instance_id, "question": ins.question, "gold_answer": ins.gold_answer,
        "gold_evidence": [{"doc_id": e.doc_id, "section_id": e.section_id} for e in ins.gold_evidence],
        "meta": ins.meta,
        "docs": [{"doc_id": d.doc_id, "title": d.title,
                  "sections": [{"section_id": s.section_id, "title": s.title, "text": s.text} for s in d.sections]}
                 for d in ins.docs],
    } for ins in instances], open(path, "w"), ensure_ascii=False, indent=2)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-docs", type=int, default=30)
    p.add_argument("--n-instances", type=int, default=300)
    p.add_argument("--seed", type=int, default=77)
    p.add_argument("--out", default="data/synth/v4_eval300.json")
    args = p.parse_args()
    corpus, entities, eav, val2ent, S = generate_corpus_v4(args.n_docs, args.seed)
    # hop distribution: K=1 (2-hop q),2 (3-hop),3 (4-hop),4 (5-hop)
    hop_dist = {1: 0.3, 2: 0.4, 3: 0.2, 4: 0.1}
    insts = generate_instances(corpus, entities, eav, val2ent, args.n_instances, hop_dist, args.seed)
    serialize(insts, args.out)
    print(f"Generated {len(insts)} chain instances -> {args.out}")
    from collections import Counter
    c = Counter(i.meta["kind"] for i in insts)
    print("  hop distribution:", dict(c))
    ex = insts[0]
    print(f"  example ({ex.meta['kind']}, n_gold={ex.meta['n_gold']}): {ex.question}")
    print(f"  answer: {ex.gold_answer}")


if __name__ == "__main__":
    main()
