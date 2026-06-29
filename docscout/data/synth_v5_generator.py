"""synth-v5: Version-conflict QA substrate for DocScout.

Motivation: synth-v4 tests *chain-of-reading* across entities. But a common
real-world failure is *version confusion*: the same entity has multiple policy
documents in the corpus (current + deprecated), and an agent that only skims
snippets can grab the wrong value because the deprecated doc's answer also
matches the query surface-level.

Design:
- Every entity has 1 **current** document (the "operations policy").
- 20 % of entities ALSO get 1-2 **old-version** documents in the same corpus.
- Old-version sections have identical titles but carry a deprecation marker
  in the first paragraph (e.g. "[DEPRECATED — superseded by revision 3.2]")
  and contain *different* answer values for the queried attributes.
- 20 % of questions are **version queries**: they mention "current policy"
  or "latest version" so the gold doc is the current one, but the deprecated
  doc's value surfaces in a search snippet as a distractor.
- The agent MUST read past the snippet to check the deprecation marker --
  reading the section title alone is not enough.

Construction guarantees:
- Uses the v4 permutation logic for unique attribute values (1 value per attr
  per entity, shared value set per attribute).
- Old versions are assigned values for a SUBSET of attributes (the ones the
  version-conflict question targets), drawn from the shared value set BUT
  from a different pool slot so they differ from the current value.
- Deterministic by seed; same QAInstance/Document/Section schema.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from docscout.types import Document, EvidenceSpan, QAInstance, Section

# ---------------------------------------------------------------------------
# Numeric attributes (shared value space) — same structure as v4
# ---------------------------------------------------------------------------
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

# Filler paragraphs for padding sections
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

ATTR_SECTION_TITLE = {
    "sync_delay": "Synchronization and Propagation",
    "retry_limit": "Failure Handling and Retries",
    "retention": "Data Retention and Lifecycle",
    "rate_limit": "Rate Limiting and Throttling",
    "backup_freq": "Backup and Recovery",
    "session_timeout": "Session Management",
}

ENTITY_NAMES = [
    "AuroraPay", "BrightShip", "CobaltHR", "DeltaSync", "EchoVault", "FluxAPI",
    "GarnetMail", "HelioAuth", "IrisCache", "JunoQueue", "KestrelDB", "LumenCRM",
    "MiraDocs", "NexusBI", "OrcaLog", "PulseFax", "QuartzLedger", "RavenOCR",
    "SableNet", "TideForm", "UmbraSign", "VegaBatch", "WillowFeed", "XenonGate",
    "YonderSync", "ZephyrPay", "AeroKey", "BoltMail", "CrestAPI", "DuneLog",
]

# ---------------------------------------------------------------------------
# Deprecation markers — each old-version document carries one in its Overview
# section (and optionally in other sections) so any read reveals the version.
# ---------------------------------------------------------------------------
DEPRECATION_MARKERS = [
    "[IMPORTANT -- This document has been SUPERSEDED. Refer to the latest revision of the {e} Operations Policy for current values.]",
    "[DEPRECATED -- This revision is no longer active. The current {e} policy overrides all prior versions.]",
    "[OBSOLETE -- This document is retained for audit reference only. Do NOT use these values for operational decisions. See the active {e} Operations Policy.]",
    "[SUPERSEDED -- This version was replaced on {date}. Consult the latest {e} Operations Policy for authoritative figures.]",
    "[ARCHIVED -- This policy is no longer in effect. The values stated here are historical and may not reflect current operations of {e}.]",
]

DEPRECATION_DATES = [
    "2024-03-15", "2024-06-01", "2024-09-10", "2024-11-20",
    "2025-01-05", "2025-04-18", "2025-07-22", "2025-10-30",
]

# ---------------------------------------------------------------------------
# Version-indicating phrases for section titles in old docs
# ---------------------------------------------------------------------------
OLD_VERSION_SUFFIXES = [" (v2.0)", " (v1.3)", " (archived)", " (superseded)", ""]


def _section_text(entity: str, attr: str | None, value: int | None,
                  rng: random.Random, is_old_version: bool = False,
                  old_version_suffix: str = "") -> str:
    """Build a section. If attr is set, embed the value; else filler.

    For old-version docs, the deprecation notice is NOT embedded here — it goes
    in the Overview section only, so the agent must read the Overview (or any
    section) to discover the version status.
    """
    if attr is None:
        return ". ".join(rng.sample(FILLER, k=3)) + "."
    label, unit, prose = ATTRS[attr]
    body = prose.format(e=entity, v=value)
    if is_old_version and old_version_suffix:
        # Append a subtle version indicator in the text body for old docs
        body += f" [Note: this value reflects {old_version_suffix.strip()} of this policy.]"
    extra = ". ".join(rng.sample(FILLER, k=2)) + "."
    return f"{body} {extra}"


def _build_version_doc(
    rng: random.Random,
    doc_id: str,
    entity: str,
    version_label: str,  # e.g. " (v2.0)", " (v1.3)"
    deprecation_marker: str,
    attr_values: dict[str, int],
    is_current: bool,
) -> Document:
    """Build a document for an entity at a specific version.

    If is_current=True, no deprecation markers. If is_current=False, the Overview
    and Revision History sections carry the deprecation marker.
    """
    sections = []
    placed: set[str] = set()

    # Attribute sections
    for attr in ATTR_NAMES:
        title = ATTR_SECTION_TITLE[attr]
        if not is_current:
            title = title + version_label  # e.g. "Rate Limiting and Throttling (v2.0)"
        sid = str(ATTR_NAMES.index(attr) + 1)
        text = _section_text(entity, attr, attr_values.get(attr), rng,
                             is_old_version=not is_current,
                             old_version_suffix=version_label if not is_current else "")
        sections.append(Section(section_id=sid, title=title, text=text))
        placed.add(title)

    # Filler sections: Overview + Revision History
    sid = len(ATTR_NAMES) + 1

    # Overview
    overview_title = "Overview"
    if not is_current:
        overview_title = "Overview" + version_label
    if is_current:
        overview_text = ". ".join(rng.sample(FILLER, k=4)) + "."
    else:
        overview_text = deprecation_marker + " " + ". ".join(rng.sample(FILLER, k=3)) + "."
    sections.append(Section(section_id=str(sid), title=overview_title, text=overview_text))
    sid += 1

    # Revision History
    rev_title = "Revision History"
    if not is_current:
        rev_title = "Revision History" + version_label
    if is_current:
        rev_text = "This document is the active revision. All updates are tracked in the version control system."
    else:
        rev_text = deprecation_marker + " This revision is archived. The active revision is maintained separately."
    sections.append(Section(section_id=str(sid), title=rev_title, text=rev_text))

    doc_title = f"{entity} Operations Policy"
    if not is_current:
        doc_title = f"{entity} Operations Policy {version_label.strip()}"
    return Document(doc_id=doc_id, title=doc_title, sections=sections)


def generate_corpus_v5(
    n_docs: int = 30,
    version_conflict_frac: float = 0.2,
    old_versions_per_entity: tuple[int, int] = (1, 2),
    seed: int = 0,
):
    """Build the corpus with version conflict capability.

    Returns:
        corpus: list[Document] — all docs (current + old versions)
        entities: list[str]
        entity_attr_value: dict[entity][attr] = value (CURRENT version)
        old_attr_values: dict[entity][{attr: value, ...}] — old-version attrs
        val2ent: dict[attr][value] = entity  (for current versions; unique inverse)
        S: list[int] — shared unique value set for current version
        old_version_info: dict[entity -> list of dicts with old-version metadata]
        version_conflict_entities: set[str] — entities that have old-version docs
    """
    rng = random.Random(seed)
    entities = ENTITY_NAMES[:n_docs]

    # Shared unique value set for current versions
    S = rng.sample(range(1, 1000), k=n_docs)
    entity_attr_value: dict[str, dict[str, int]] = {e: {} for e in entities}
    val2ent: dict[str, dict[int, str]] = {a: {} for a in ATTR_NAMES}
    for attr in ATTR_NAMES:
        perm = S[:]
        rng.shuffle(perm)
        for ent, v in zip(entities, perm):
            entity_attr_value[ent][attr] = v
            val2ent[attr][v] = ent

    # Select entities that get old versions
    n_conflict = max(1, int(n_docs * version_conflict_frac))
    conflict_entities = set(rng.sample(entities, n_conflict))

    old_attr_values: dict[str, dict[str, int]] = {}
    old_version_info: dict[str, list[dict[str, Any]]] = {}
    old_doc_id_counter = [n_docs + 1]  # mutable counter for unique old doc IDs

    for ent in conflict_entities:
        n_old = rng.randint(old_versions_per_entity[0], old_versions_per_entity[1])
        old_version_info[ent] = []
        old_attr_values[ent] = {}
        for _ in range(n_old):
            # Each old version gets its own set of attribute values
            # drawn from a DIFFERENT subset of the shared value pool:
            # we pick from the complement of S (overshoot range) to ensure
            # old values differ from current values
            old_vals: dict[str, int] = {}
            for attr in ATTR_NAMES:
                # Pick a value not equal to the current one
                available = [x for x in range(1, 1000) if x != entity_attr_value[ent][attr]]
                old_vals[attr] = rng.choice(available)
            old_attr_values[ent].update(old_vals)

            # Build old-version document
            old_doc_id = f"doc_{old_doc_id_counter[0]:02d}"
            old_doc_id_counter[0] += 1
            version_label = rng.choice(OLD_VERSION_SUFFIXES)
            if not version_label:
                version_label = f" (r{rng.randint(1,3)}.0)"
            dep_marker = rng.choice(DEPRECATION_MARKERS).format(
                e=ent, date=rng.choice(DEPRECATION_DATES)
            )
            old_doc = _build_version_doc(
                rng, old_doc_id, ent, version_label, dep_marker, old_vals, is_current=False
            )
            old_version_info[ent].append({
                "doc_id": old_doc_id,
                "version_label": version_label,
                "deprecation_marker": dep_marker,
                "attr_values": old_vals,
            })

    # Build current-version docs
    corpus = []
    for i, ent in enumerate(entities):
        doc_id = f"doc_{i+1:02d}"
        is_conflict = ent in conflict_entities
        doc = _build_version_doc(
            rng, doc_id, ent, "", "", entity_attr_value[ent], is_current=True
        )
        corpus.append(doc)

    # Append old-version docs to corpus
    for ent in conflict_entities:
        for old_info in old_version_info[ent]:
            # Find the old doc we already built — reconstruct it
            old_doc_id = old_info["doc_id"]
            dep_marker = old_info["deprecation_marker"]
            version_label = old_info["version_label"]
            old_vals = old_info["attr_values"]
            old_doc = _build_version_doc(
                rng, old_doc_id, ent, version_label, dep_marker, old_vals, is_current=False
            )
            corpus.append(old_doc)

    return corpus, entities, entity_attr_value, old_attr_values, val2ent, S, conflict_entities, old_version_info


def _attr_human(attr: str) -> str:
    return ATTRS[attr][0]


def generate_instances_v5(
    corpus: list[Document],
    entities: list[str],
    eav: dict[str, dict[str, int]],
    old_av: dict[str, dict[str, int]],
    val2ent: dict[str, dict[int, str]],
    conflict_entities: set[str],
    old_version_info: dict[str, list[dict[str, Any]]],
    n_instances: int,
    version_conflict_frac: float = 0.2,
    seed: int = 0,
) -> list[QAInstance]:
    """Generate instances, ~20% of which are version-conflict questions.

    Version-conflict questions:
    - Ask about an entity's attribute, specifying "current policy" / "latest version"
    - Gold answer comes from the CURRENT doc
    - The old version's value for the same attr also appears in the corpus (distractor)
    - Gold evidence is the current doc's section
    - Meta tags the question as "version_conflict"

    Non-conflict questions:
    - Simple single-hop: "What is the {attr} of {entity}?"
    - Gold is the current version
    """
    rng = random.Random(seed + 1)
    instances: list[QAInstance] = []

    ent_to_docid = {e: f"doc_{i+1:02d}" for i, e in enumerate(entities)}

    # Build doc_id lookup for old versions
    old_doc_map: dict[str, list[str]] = {}  # entity -> list of old doc_ids
    for ent in conflict_entities:
        old_doc_map[ent] = [info["doc_id"] for info in old_version_info.get(ent, [])]

    attempts = 0
    max_attempts = n_instances * 50

    while len(instances) < n_instances and attempts < max_attempts:
        attempts += 1
        is_conflict = rng.random() < version_conflict_frac and conflict_entities

        if is_conflict:
            ent = rng.choice(list(conflict_entities))
        else:
            ent = rng.choice(entities)

        attr = rng.choice(ATTR_NAMES)
        current_value = eav[ent][attr]

        if is_conflict:
            # Version-conflict question
            old_values_for_ent = old_version_info[ent]
            # Pick a random old version
            old_info = rng.choice(old_values_for_ent)
            old_value = old_info["attr_values"][attr]
            old_doc_id = old_info["doc_id"]

            # Vary phrasing:
            phrasings = [
                "What is the {attr_human} of {ent} under the current policy?",
                "According to the latest active policy, what is the {attr_human} of {ent}?",
                "What is {ent}'s {attr_human} in the most recent revision of its operations policy?",
                "Per the current operations policy, what is the {attr_human} of {ent}?",
                "What does the active policy say is the {attr_human} of {ent}?",
            ]
            q = rng.choice(phrasings).format(attr_human=_attr_human(attr), ent=ent)

            units = ATTRS[attr][1]
            doc_id = ent_to_docid[ent]
            sid = str(ATTR_NAMES.index(attr) + 1)

            gold = [EvidenceSpan(doc_id=doc_id, section_id=sid)]

            instances.append(QAInstance(
                instance_id=f"qa_{len(instances)+1:04d}",
                question=q,
                gold_answer=f"{current_value} {units}",
                gold_evidence=gold,
                docs=corpus,
                meta={
                    "kind": "version_conflict",
                    "entity": ent,
                    "attr": attr,
                    "current_value": current_value,
                    "old_value": old_value,
                    "old_doc_id": old_doc_id,
                    "current_doc_id": doc_id,
                },
            ))
        else:
            # Simple question on current value
            phr = rng.choice([
                "What is the {attr_human} of {ent}?",
                "What is {ent}'s {attr_human}?",
                "What value does {ent} use for {attr_human}?",
            ])
            q = phr.format(attr_human=_attr_human(attr), ent=ent)
            units = ATTRS[attr][1]
            doc_id = ent_to_docid[ent]
            sid = str(ATTR_NAMES.index(attr) + 1)
            gold = [EvidenceSpan(doc_id=doc_id, section_id=sid)]

            instances.append(QAInstance(
                instance_id=f"qa_{len(instances)+1:04d}",
                question=q,
                gold_answer=f"{current_value} {units}",
                gold_evidence=gold,
                docs=corpus,
                meta={
                    "kind": "single",
                    "entity": ent,
                    "attr": attr,
                    "current_value": current_value,
                },
            ))

    if len(instances) < n_instances:
        print(f"Warning: only generated {len(instances)} / {n_instances} instances (attempts exhausted)")

    rng.shuffle(instances)
    return instances


def serialize(instances: list[QAInstance], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = []
    for ins in instances:
        out.append({
            "instance_id": ins.instance_id,
            "question": ins.question,
            "gold_answer": ins.gold_answer,
            "gold_evidence": [{"doc_id": e.doc_id, "section_id": e.section_id} for e in ins.gold_evidence],
            "meta": ins.meta,
            "docs": [{
                "doc_id": d.doc_id, "title": d.title,
                "sections": [{"section_id": s.section_id, "title": s.title, "text": s.text} for s in d.sections]
            } for d in ins.docs],
        })
    json.dump(out, open(path, "w"), ensure_ascii=False, indent=2)


def main():
    p = argparse.ArgumentParser(description="Generate DocScout synth-v5 version-conflict corpus.")
    p.add_argument("--n-docs", type=int, default=30)
    p.add_argument("--n-instances", type=int, default=300)
    p.add_argument("--version-conflict-frac", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=77)
    p.add_argument("--out", default="data/synth/v5_eval300.json")
    args = p.parse_args()

    corpus, entities, eav, old_av, val2ent, S, conflict_entities, old_version_info = generate_corpus_v5(
        n_docs=args.n_docs,
        version_conflict_frac=args.version_conflict_frac,
        seed=args.seed,
    )

    n_conflict = len(conflict_entities)
    n_old_docs = sum(len(v) for v in old_version_info.values())
    print(f"Corpus: {len(corpus)} docs ({args.n_docs} current + {n_old_docs} old), "
          f"{n_conflict} entities with old versions")

    insts = generate_instances_v5(
        corpus, entities, eav, old_av, val2ent, conflict_entities, old_version_info,
        n_instances=args.n_instances,
        version_conflict_frac=args.version_conflict_frac,
        seed=args.seed,
    )

    serialize(insts, args.out)
    print(f"Generated {len(insts)} instances -> {args.out}")

    # Summary stats
    from collections import Counter
    kinds = Counter(i.meta["kind"] for i in insts)
    print(f"  Kind distribution: {dict(kinds)}")

    # Show one of each kind
    for kind in ["single", "version_conflict"]:
        examples = [i for i in insts if i.meta["kind"] == kind]
        if examples:
            ex = examples[0]
            print(f"\n  Example ({kind}):")
            print(f"    Q: {ex.question}")
            print(f"    A: {ex.gold_answer}")
            print(f"    Gold evidence: {[(e.doc_id, e.section_id) for e in ex.gold_evidence]}")
            if kind == "version_conflict":
                print(f"    Old value (distractor): {ex.meta['old_value']} in doc {ex.meta['old_doc_id']}")

    # Check: in conflict instances, does the old doc section actually contain the old value?
    doc_lookup = {d.doc_id: d for d in corpus}
    if "version_conflict" in kinds:
        cex = [i for i in insts if i.meta["kind"] == "version_conflict"][0]
        old_doc = doc_lookup[cex.meta["old_doc_id"]]
        attr = cex.meta["attr"]
        sid = str(ATTR_NAMES.index(attr) + 1)
        old_section = next(s for s in old_doc.sections if s.section_id == sid)
        print(f"\n  Verification: old doc '{old_doc.doc_id}' section {sid} contains "
              f"old value {cex.meta['old_value']}: "
              f"{str(cex.meta['old_value']) in old_section.text}")
        overview = next(s for s in old_doc.sections if "Overview" in s.title)
        print(f"  Old doc Overview (first 200 chars): {overview.text[:200]}")

    # Section length stats
    import statistics as st
    allwl = [len(s.text.split()) for ins in insts for d in ins.docs[:3] for s in d.sections]
    if allwl:
        print(f"\n  Section words (sample): mean={st.mean(allwl):.0f} min={min(allwl)} max={max(allwl)}")


if __name__ == "__main__":
    main()
