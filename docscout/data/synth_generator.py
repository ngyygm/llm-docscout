"""Synthetic NL-document corpus generator for DocScout.

Produces multi-section natural-language documents (policy / technical-manual
style) with planted facts, plus QA instances whose answers are section-locatable
— so reward can attribute *which section* the evidence lives in (essential for
the section-F1 and efficiency-ratio reward terms).

Deterministic given a seed. Controllable: corpus size, doc length, distractor
density, single vs. multi-hop. This is the Stage-2 "synthetic first" corpus per
EXPERIMENT_PLAN.md; real benchmarks (MuSiQue/HotpotQA) plug in later via the
same QAInstance schema.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from docscout.types import Document, EvidenceSpan, QAInstance, Section

# ---------------------------------------------------------------------------
# Template pools. Values are chosen so answers are unambiguous & checkable (EM).
# ---------------------------------------------------------------------------

# Each attribute: (question_template, prose_template, value_pool).
# prose_template embeds {entity}, {value} (and extra context) into a section.
ATTRIBUTES: dict[str, dict[str, Any]] = {
    "sync_delay_minutes": {
        "q": "How many minutes after approval does {entity} synchronize?",
        "q_para": "Once a change to {entity} is signed off, how long before the update takes effect everywhere?",
        "prose": (
            "Once approval is granted, {entity} completes synchronization "
            "within {value} minutes. The synchronization job runs on a fixed "
            "schedule and retries automatically if a transient failure occurs."
        ),
        "values": [5, 10, 15, 30, 60],
    },
    "approver_role": {
        "q": "Who must approve changes to {entity}?",
        "q_para": "Whose sign-off is required before a modification to {entity} can proceed?",
        "prose": (
            "Any change to {entity} requires sign-off from the {value}. "
            "Requests submitted without this approval are automatically "
            "rejected and returned to the submitter for correction."
        ),
        "values": ["department head", "security officer", "product owner", "compliance lead"],
    },
    "retry_limit": {
        "q": "How many retries does {entity} attempt on failure?",
        "q_para": "If {entity} hits an error, how many times will it try again before escalating?",
        "prose": (
            "When {entity} encounters a failure, the system retries the "
            "operation up to {value} times before escalating to the on-call "
            "engineer. Each retry is spaced with exponential backoff."
        ),
        "values": [3, 5, 8, 10],
    },
    "retention_days": {
        "q": "How many days are {entity} records retained?",
        "q_para": "How long does {entity} keep its records on file before they are removed?",
        "prose": (
            "Records produced by {entity} are retained for {value} days, "
            "after which they are archived to cold storage and eventually "
            "purged in accordance with the data governance policy."
        ),
        "values": [30, 90, 180, 365],
    },
    "rate_limit_per_min": {
        "q": "What is the per-minute rate limit for {entity}?",
        "q_para": "How many calls to {entity} are allowed each minute before throttling kicks in?",
        "prose": (
            "To protect downstream services, {entity} enforces a rate limit "
            "of {value} requests per minute per tenant. Requests exceeding "
            "this threshold receive a 429 response."
        ),
        "values": [60, 120, 300, 600],
    },
}

# Distractor section titles + filler sentences (no planted facts).
SECTION_TITLES = [
    "Overview", "Scope", "Definitions", "Roles and Responsibilities",
    "Process Description", "Exceptions", "Compliance", "Monitoring",
    "Incident Response", "Revision History", "References", "Appendix",
]
FILLER = [
    "This section describes the general context and intended audience.",
    "All terms used here follow the standard organizational glossary.",
    "Stakeholders should review this document on a quarterly basis.",
    "Implementation details may vary across deployment environments.",
    "Contact the operations team for environment-specific guidance.",
    "This policy supersedes all prior versions on the same topic.",
    "Audit logs are collected centrally and reviewed monthly.",
    "No action is required from end users under normal operation.",
]

ENTITY_NAMES = [
    "AuroraPay", "BrightShip", "CobaltHR", "DeltaSync", "EchoVault", "FluxAPI",
    "GarnetMail", "HelioAuth", "IrisCache", "JunoQueue", "KestrelDB", "LumenCRM",
    "MiraDocs", "NexusBI", "OrcaLog", "PulseFax", "QuartzLedger", "RavenOCR",
]


def _make_doc(
    rng: random.Random,
    doc_id: str,
    entity: str,
    n_sections: int,
    facts: dict[str, Any],
) -> tuple[Document, dict[str, EvidenceSpan]]:
    """Build one document. `facts` maps attribute -> value to plant.

    Returns the document and a map attribute -> EvidenceSpan(section holding it).
    """
    titles = rng.sample(SECTION_TITLES, k=min(n_sections, len(SECTION_TITLES)))
    while len(titles) < n_sections:
        titles.append(rng.choice(SECTION_TITLES))

    # Pick distinct section ids to plant facts into (1 fact per section max).
    plant_slots = rng.sample(range(n_sections), k=len(facts))
    attr_by_slot = {slot: attr for slot, attr in zip(plant_slots, facts.keys())}

    sections: list[Section] = []
    evidence: dict[str, EvidenceSpan] = {}
    for i in range(n_sections):
        sid = str(i + 1)
        title = titles[i] if i < len(titles) else f"Section {sid}"
        if i in attr_by_slot:
            attr = attr_by_slot[i]
            spec = ATTRIBUTES[attr]
            value = facts[attr]
            text = spec["prose"].format(entity=entity, value=value)
            text += " " + rng.choice(FILLER)
            evidence[attr] = EvidenceSpan(doc_id=doc_id, section_id=sid)
        else:
            text = rng.choice(FILLER) + " " + rng.choice(FILLER)
        # Vary length so reading granularity matters: pad some distractors.
        if rng.random() < 0.3:
            text += " " + " ".join(rng.sample(FILLER, k=2))
        sections.append(Section(section_id=sid, title=title, text=text))

    doc = Document(doc_id=doc_id, title=f"{entity} Operations Policy", sections=sections)
    return doc, evidence


def generate_corpus(
    n_docs: int = 8,
    sections_per_doc: int = 8,
    n_instances: int = 200,
    multi_hop_frac: float = 0.2,
    paraphrase_frac: float = 0.0,
    seed: int = 0,
) -> list[QAInstance]:
    """Generate a corpus + QA instances.

    Each QA instance carries the *full corpus* (so the agent must search, not
    guess). Single-hop: ask one attribute of one entity. Multi-hop: compare an
    attribute across two entities (answer is a comparison result).
    """
    rng = random.Random(seed)
    entities = rng.sample(ENTITY_NAMES, k=min(n_docs, len(ENTITY_NAMES)))
    while len(entities) < n_docs:
        entities.append(f"{rng.choice(ENTITY_NAMES)}-{rng.randint(2,9)}")

    corpus: list[Document] = []
    # entity -> {attr: (value, EvidenceSpan)}
    lookup: dict[str, dict[str, tuple[Any, EvidenceSpan]]] = {}
    for i, ent in enumerate(entities):
        doc_id = f"doc_{i+1:02d}"
        # plant 1-3 facts per doc
        k = rng.randint(1, 3)
        attrs = rng.sample(list(ATTRIBUTES.keys()), k=k)
        facts = {a: rng.choice(ATTRIBUTES[a]["values"]) for a in attrs}
        doc, evi = _make_doc(rng, doc_id, ent, sections_per_doc, facts)
        corpus.append(doc)
        lookup[ent] = {a: (v, evi[a]) for a, v in facts.items()}

    instances: list[QAInstance] = []
    ent_list = list(lookup.keys())
    for j in range(n_instances):
        if rng.random() < multi_hop_frac and len(ent_list) >= 2:
            # multi-hop: compare numeric attribute across two entities
            attr = rng.choice([a for a in ATTRIBUTES if a != "approver_role"])
            e1, e2 = rng.sample(ent_list, 2)
            # ensure both have the attr; else fallback to single-hop
            if attr in lookup[e1] and attr in lookup[e2]:
                v1, sp1 = lookup[e1][attr]
                v2, sp2 = lookup[e2][attr]
                q = (
                    f"Which has a higher {attr.replace('_',' ')}, {e1} or {e2}? "
                    f"Reply with the single entity name."
                )
                ans = e1 if v1 > v2 else (e2 if v2 > v1 else e1)
                instances.append(
                    QAInstance(
                        instance_id=f"qa_{j+1:04d}",
                        question=q,
                        gold_answer=ans,
                        gold_evidence=[sp1, sp2],
                        docs=corpus,
                        meta={"kind": "multi_hop", "attr": attr},
                    )
                )
                continue
        # single-hop
        ent = rng.choice(ent_list)
        attrs_present = list(lookup[ent].keys())
        attr = rng.choice(attrs_present)
        value, span = lookup[ent][attr]
        para = rng.random() < paraphrase_frac and "q_para" in ATTRIBUTES[attr]
        q = (ATTRIBUTES[attr]["q_para"] if para else ATTRIBUTES[attr]["q"]).format(entity=ent)
        instances.append(
            QAInstance(
                instance_id=f"qa_{j+1:04d}",
                question=q,
                gold_answer=str(value),
                gold_evidence=[span],
                docs=corpus,
                meta={"kind": "single_hop", "attr": attr, "entity": ent,
                      "paraphrased": para, "num_gold_sections": 1},
            )
        )

    rng.shuffle(instances)
    return instances


def serialize(instances: list[QAInstance], path: str | Path) -> None:
    """Save instances to JSON. Corpus docs are duplicated per instance (simple, self-contained)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = []
    for ins in instances:
        out.append(
            {
                "instance_id": ins.instance_id,
                "question": ins.question,
                "gold_answer": ins.gold_answer,
                "gold_evidence": [{"doc_id": e.doc_id, "section_id": e.section_id} for e in ins.gold_evidence],
                "meta": ins.meta,
                "docs": [
                    {
                        "doc_id": d.doc_id,
                        "title": d.title,
                        "sections": [{"section_id": s.section_id, "title": s.title, "text": s.text} for s in d.sections],
                    }
                    for d in ins.docs
                ],
            }
        )
    json.dump(out, open(path, "w"), ensure_ascii=False, indent=2)


def main():
    p = argparse.ArgumentParser(description="Generate DocScout synthetic corpus.")
    p.add_argument("--n-docs", type=int, default=8)
    p.add_argument("--sections-per-doc", type=int, default=8)
    p.add_argument("--n-instances", type=int, default=200)
    p.add_argument("--multi-hop-frac", type=float, default=0.2)
    p.add_argument("--paraphrase-frac", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, default="data/synth/train.json")
    args = p.parse_args()
    insts = generate_corpus(
        n_docs=args.n_docs,
        sections_per_doc=args.sections_per_doc,
        n_instances=args.n_instances,
        multi_hop_frac=args.multi_hop_frac,
        paraphrase_frac=args.paraphrase_frac,
        seed=args.seed,
    )
    serialize(insts, args.out)
    print(f"Generated {len(insts)} instances -> {args.out}")
    # quick sanity print
    ex = insts[0]
    print("Example:", ex.instance_id, "|", ex.question, "|=>", ex.gold_answer)
    print("  gold evidence:", [(e.doc_id, e.section_id) for e in ex.gold_evidence])


if __name__ == "__main__":
    main()
