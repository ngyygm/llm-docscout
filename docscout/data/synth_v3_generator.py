"""Realistic-scale synthetic NL-document corpus for DocScout (synth-v3).

Why this exists: the v1/v2 synth sections averaged ~26 words, so a one-shot RAG
top-5 dumped ~the whole relevant corpus into context (saturation, RAG~0.96), and
"reading less" directly cost accuracy — making the read-budget thesis untestable
(see refine-logs/round-result.md, Round 5 audit). synth-v3 fixes the dominant
failure: realistic-length (150-400 word) multi-paragraph sections where:

  - the gold value is planted AFTER the first ~60 words, so the 40-word snippet
    a search returns does NOT contain it (RAG must now read, not skim) -> RAG
    desaturates and an oracle-vs-RAG headroom appears;
  - confounder values sit in the SAME section (e.g. "standard tier 90 days,
    audit 365 days, cache 7 days") -> careful reading is required, not keyword
    matching -> addresses the "reads but can't understand" MuSiQue-style failure;
  - sections are long enough that reading a wrong one is a real token cost ->
    the efficiency-ratio reward finally has leverage.

Plugs into the existing DocStore/SearchEnv/reward unchanged via the same
QAInstance/Document/Section schema. Deterministic by seed.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from docscout.types import Document, EvidenceSpan, QAInstance, Section

# ---------------------------------------------------------------------------
# Attribute specs. Each carries: direct + paraphrased question templates, a
# multi-sentence prose block that embeds {value} after an intro (so the value is
# NOT in the first ~40 words), confounder sentences, and a value pool.
# The prose block is written so the value is unambiguous once the section is read.
# ---------------------------------------------------------------------------

ATTRIBUTES_V3: dict[str, dict[str, Any]] = {
    "sync_delay_minutes": {
        "section": "Synchronization and Propagation",
        "q": "How many minutes after approval does {entity} synchronize changes?",
        "q_para": "Once a change to {entity} is signed off, how long before it takes effect across all environments?",
        "intro": (
            "This section governs how approved modifications to {entity} are "
            "propagated from the authoring environment to every downstream "
            "consumer. Propagation is performed by a scheduled reconciliation "
            "job that runs continuously and reconciles the authoritative store "
            "against each consumer's local cache. The job is idempotent, so "
            "partial network disruptions do not produce inconsistent states."
        ),
        "fact": (
            "For the standard deployment tier, {entity} completes synchronization "
            "within {value} minutes of approval. This window is measured from "
            "the moment the change record is marked approved until the change "
            "is observable on every read replica. Operators should treat this "
            "interval as a hard guarantee for status reporting and on-call paging."
        ),
        "confounders": [
            "The premium tier, available only under an enterprise contract, "
            "synchronizes within 2 minutes via a dedicated fast-path channel.",
            "Cache invalidation for edge nodes may lag synchronization by up to "
            "5 additional minutes and is billed separately under the edge SLA.",
        ],
        "values": [5, 10, 15, 30, 60],
        "ans_unit": "minutes",
    },
    "approver_role": {
        "section": "Change Approval and Governance",
        "q": "Whose sign-off is required to approve changes to {entity}?",
        "q_para": "Before a modification to {entity} can proceed, which role must give approval?",
        "intro": (
            "All modifications to {entity} are subject to a mandatory review and "
            "approval workflow before they may be merged or deployed. The "
            "workflow exists to enforce separation of duties and to ensure that "
            "each change is vetted by a party with the appropriate authority and "
            "context. Requests that bypass this workflow are automatically "
            "rejected by the change-management system and returned to the author."
        ),
        "fact": (
            "Approval authority for changes to {entity} rests exclusively with "
            "the {value}. No other role, including the system architect or the "
            "submitting engineer, is permitted to mark a change record as "
            "approved. The approver's identity is recorded immutably in the "
            "audit trail alongside the change for later review."
        ),
        "confounders": [
            "The security officer must additionally co-sign any change that "
            "touches authentication or encryption configuration.",
            "Routine documentation-only edits may be self-approved by the author "
            "but still require the {value}'s acknowledgment within 48 hours.",
        ],
        "values": ["department head", "security officer", "product owner", "compliance lead"],
    },
    "retry_limit": {
        "section": "Failure Handling and Retries",
        "q": "How many retries does {entity} attempt before escalating a failure?",
        "q_para": "When {entity} encounters an error, how many times does it retry before escalating?",
        "intro": (
            "Transient failures are an expected operating condition for {entity}, "
            "and the system is designed to absorb them without operator "
            "intervention where possible. When an operation fails, the runtime "
            "classifies the failure as transient or terminal and schedules "
            "retry attempts according to a bounded exponential-backoff policy. "
            "Each retry is logged with its attempt number and the failure code "
            "that triggered it."
        ),
        "fact": (
            "Before escalating a failure to the on-call engineer, {entity} "
            "will retry the operation up to {value} times. The backoff interval "
            "doubles between successive attempts, starting at one second, so the "
            "total retry window is bounded and predictable. After the final "
            "attempt exhausts without success, the operation is marked failed "
            "and an escalation ticket is opened automatically."
        ),
        "confounders": [
            "Idempotent write operations may be retried up to 8 times because "
            "duplicate application is harmless.",
            "Authentication token refresh failures bypass the retry counter "
            "entirely and trigger immediate re-authentication.",
        ],
        "values": [3, 5, 8, 10],
    },
    "retention_days": {
        "section": "Data Retention and Lifecycle",
        "q": "How many days are operational records produced by {entity} retained?",
        "q_para": "How long does {entity} keep its operational records before they are removed?",
        "intro": (
            "{entity} emits several classes of records during normal operation, "
            "and each class is governed by a distinct retention policy aligned "
            "with the organization's data-governance framework. Retention "
            "windows are enforced automatically by a nightly lifecycle job that "
            "transitions records between storage tiers and ultimately purges "
            "them. The policies described here supersede any earlier drafts and "
            "apply uniformly across all deployment regions."
        ),
        "fact": (
            "Operational records produced by {entity} are retained for {value} "
            "days, counted from the record's creation timestamp. After this "
            "window elapses, the records are moved to cold archival storage and "
            "are no longer queryable through the standard API. A separate "
            "governance review must approve any extension of this window for "
            "litigation or compliance hold purposes."
        ),
        "confounders": [
            "Audit-trail records are retained for 365 days and are never moved "
            "to cold storage during that window.",
            "Temporary cache entries expire after 7 days and are not considered "
            "operational records for retention purposes.",
        ],
        "values": [30, 90, 180, 365],
        "ans_unit": "days",
    },
    "rate_limit_per_min": {
        "section": "Rate Limiting and Throttling",
        "q": "What is the per-minute request rate limit for {entity}?",
        "q_para": "How many calls to {entity} are allowed each minute before throttling begins?",
        "intro": (
            "To protect downstream dependencies from runaway load, {entity} "
            "enforces a per-tenant rate limit on incoming requests. The limiter "
            "is a sliding-window counter applied at the ingress gateway before "
            "any request reaches the application layer. Requests that arrive "
            "within the allowed budget are processed normally; those that exceed "
            "the budget receive an immediate 429 response with a Retry-After "
            "header indicating when capacity will free up."
        ),
        "fact": (
            "The default per-minute rate limit for {entity} is {value} requests "
            "per minute per tenant. This limit applies to the aggregate of all "
            "authenticated requests from a tenant regardless of the specific "
            "endpoint invoked. Tenants requiring higher throughput may negotiate "
            "a raised limit through their account representative."
        ),
        "confounders": [
            "Bulk export endpoints are subject to a separate, lower limit of 60 "
            "requests per minute to protect batch infrastructure.",
            "Internal service-to-service calls bypass the tenant limiter entirely "
            "and are governed by a distinct internal quota.",
        ],
        "values": [60, 120, 300, 600],
        "ans_unit": "requests per minute",
    },
    "backup_frequency_hours": {
        "section": "Backup and Recovery",
        "q": "How often, in hours, does {entity} create a backup?",
        "q_para": "At what hourly interval does {entity} take backups?",
        "intro": (
            "{entity} maintains point-in-time recoverability through a scheduled "
            "backup regimen. Backups are written to geo-redundant object storage "
            "and are integrity-checked on write by comparing a stored checksum. "
            "The recovery point objective documented here reflects the maximum "
            "data loss a tenant should plan for in a disaster scenario."
        ),
        "fact": (
            "Under the standard policy, {entity} captures a full backup every "
            "{value} hours. Incremental snapshots are taken more frequently but "
            "are only retained for the most recent 24-hour window. Restoration "
            "from a full backup is the supported path for disaster recovery; "
            "incremental snapshots are intended for finer-grained undo only."
        ),
        "confounders": [
            "Critical configuration stores are backed up every 1 hour under a "
            "separate high-availability policy.",
            "Weekly archives are written every 168 hours and retained for seven "
            "years for long-term compliance.",
        ],
        "values": [4, 6, 12, 24],
        "ans_unit": "hours",
    },
    "session_timeout_minutes": {
        "section": "Session Management",
        "q": "After how many minutes of inactivity does a {entity} session expire?",
        "q_para": "How long can a {entity} session sit idle before it times out?",
        "intro": (
            "{entity} controls interactive access through expiring sessions that "
            "are invalidated automatically after a period of inactivity. This "
            "inactivity timeout is the primary control against abandoned-session "
            "abuse and is enforced server-side so that tampering with client "
            "clocks cannot extend a session. Active use of the session resets "
            "the inactivity timer."
        ),
        "fact": (
            "A {entity} session expires after {value} minutes of continuous "
            "inactivity. Once expired, the session token is revoked and the user "
            "must re-authenticate to establish a new session. Administrators "
            "cannot raise this limit beyond the documented value without an "
            "approved security exception."
        ),
        "confounders": [
            "Sessions established with hardware-token step-up authentication are "
            "granted a 480-minute inactivity window.",
            "Absolute session lifetime is capped at 720 minutes regardless of "
            "activity, after which re-authentication is mandatory.",
        ],
        "values": [15, 30, 60, 120],
        "ans_unit": "minutes",
    },
}

# Topic sections used as filler (no planted facts). Each is a realistic
# multi-paragraph block so documents have plausible density and length.
FILLER_TOPICS: dict[str, str] = {
    "Overview": (
        "This document specifies the operational policy for {entity} and is the "
        "authoritative source for day-to-day administration. It is intended for "
        "site reliability engineers, platform operators, and on-call responders "
        "who interact with {entity} directly. The policy is reviewed annually and "
        "updated whenever a material change to the system's operating model "
        "occurs. Where this document conflicts with a vendor whitepaper, this "
        "document takes precedence for internal operations.\n\n"
        "Readers should familiarize themselves with the definitions section "
        "before consulting topic-specific sections, because several terms are "
        "used in a sense narrower than their common meaning. Cross-references "
        "throughout the document point to related policies that govern "
        "adjacent systems; those are out of scope here but are linked for "
        "convenience."
    ),
    "Scope and Applicability": (
        "The provisions of this policy apply to all production and pre-production "
        "instances of {entity} operated by the organization, including instances "
        "provisioned on behalf of external tenants under a managed-service "
        "agreement. Development instances used solely for engineering "
        "experimentation are exempt, provided they contain no production data.\n\n"
        "Third-party contractors who operate {entity} on the organization's "
        "behalf are bound by this policy through their service agreement and are "
        "subject to the same audit and review obligations as internal staff."
    ),
    "Definitions": (
        "For the purposes of this document, an operational record is any "
        "machine-generated artifact produced by {entity} in the course of "
        "servicing a request, including logs, metrics, and emitted events. A "
        "tenant is an isolated administrative unit to which requests are "
        "attributed for quota and billing purposes. The authoritative store is "
        "the single source of truth from which all replicas are derived.\n\n"
        "Where this document refers to a business day, it means a calendar day "
        "excluding recognized organizational holidays. All time windows are "
        "expressed in Coordinated Universal Time unless a local timezone is "
        "explicitly stated."
    ),
    "Roles and Responsibilities": (
        "The platform operations team owns the day-to-day health of {entity} "
        "and is the first responder for alerts. The service owner is "
        "accountable for the policy contents and for approving material "
        "changes. The security team retains override authority for any matter "
        "touching authentication, authorization, or encryption.\n\n"
        "On-call engineers are empowered to take corrective action within the "
        "bounds of this policy but must escalate any action that would violate "
        "a stated guarantee. Every escalation is reviewed in the next "
        "operations review meeting."
    ),
    "Monitoring and Alerting": (
        "{entity} exposes health and performance metrics through a standard "
        "scrape endpoint consumed by the centralized monitoring platform. "
        "Alerting rules are version-controlled alongside the service code and "
        "are reviewed each quarter to retire noisy or stale conditions.\n\n"
        "Every alert routes to a runbook that documents the diagnostic steps and "
        "the escalation path. Alert fatigue is treated as an operational defect: "
        "any alert that fires more than three times without actionable response "
        "is candidates for tuning or suppression."
    ),
    "Incident Response": (
        "Incidents affecting {entity} are classified by severity according to "
        "the organization's shared severity matrix. Severity-one incidents "
        "require an incident commander within fifteen minutes and a customer-"
        "facing statement within one hour. Lower severities follow a relaxed "
        "but still bounded cadence.\n\n"
        "Post-incident reviews are mandatory for severity-one and severity-two "
        "incidents and are completed within five business days. The review "
        "focuses on systemic contributing factors rather than individual blame."
    ),
    "Revision History": (
        "This policy is maintained under version control and every change is "
        "attributed to a named owner with a recorded rationale. The current "
        "revision supersedes all prior revisions in full; partial applicability "
        "of older revisions is not supported.\n\n"
        "A changelog summarizing material revisions is published alongside the "
        "document and is the recommended starting point for readers familiar "
        "with a previous version who want to understand what has changed."
    ),
}

ENTITY_NAMES = [
    "AuroraPay", "BrightShip", "CobaltHR", "DeltaSync", "EchoVault", "FluxAPI",
    "GarnetMail", "HelioAuth", "IrisCache", "JunoQueue", "KestrelDB", "LumenCRM",
    "MiraDocs", "NexusBI", "OrcaLog", "PulseFax", "QuartzLedger", "RavenOCR",
    "SableNet", "TideForm", "UmbraSign", "VegaBatch", "WillowFeed", "XenonGate",
]

SECTION_ORDER = [  # plausible document skeleton; attribute sections slotted in
    "Overview", "Scope and Applicability", "Definitions", "Change Approval and Governance",
    "Synchronization and Propagation", "Failure Handling and Retries", "Data Retention and Lifecycle",
    "Rate Limiting and Throttling", "Backup and Recovery", "Session Management",
    "Roles and Responsibilities", "Monitoring and Alerting", "Incident Response",
    "Revision History",
]

# Map attribute -> its natural section title.
_ATTR_SECTION = {a: spec["section"] for a, spec in ATTRIBUTES_V3.items()}


def _build_section(rng: random.Random, title: str, entity: str) -> str:
    """Build a filler (non-fact) section from its topic template, padded to a
    realistic length with occasional extra filler sentences."""
    base = FILLER_TOPICS.get(title, FILLER_TOPICS["Overview"]).format(entity=entity)
    # ~70% of filler sections get an extra paragraph for length variation.
    if rng.random() < 0.7:
        extra = rng.choice(list(FILLER_TOPICS.values())).format(entity=entity)
        base += "\n\n" + extra.split("\n\n")[0]
    return base


def _build_fact_section(rng: random.Random, attr: str, entity: str, value: Any) -> str:
    """Build the section containing the gold fact. The value is placed AFTER an
    intro paragraph so it is not in the snippet (first ~40 words)."""
    spec = ATTRIBUTES_V3[attr]
    parts = [
        spec["intro"].format(entity=entity, value=value),
        spec["fact"].format(entity=entity, value=value),
    ]
    # confounders after the fact (so the reader must distinguish the true value)
    for c in spec["confounders"]:
        parts.append(c.format(entity=entity, value=value))
    rng.shuffle(parts[2:])  # randomize confounder order, keep intro+fact first
    return "\n\n".join(parts)


def _make_doc(rng: random.Random, doc_id: str, entity: str, n_sections: int,
              facts: dict[str, Any]) -> tuple[Document, dict[str, EvidenceSpan]]:
    """facts maps attribute -> value to plant (each in its natural section)."""
    # choose section titles: guarantee fact sections are present, fill the rest
    chosen: list[str] = []
    fact_sections = [_ATTR_SECTION[a] for a in facts]
    # start from the skeleton order, ensure fact sections included
    pool = [t for t in SECTION_ORDER if t not in fact_sections]
    for fs in fact_sections:
        chosen.append(fs)
    # fill remaining slots from pool (cycling if needed)
    rng.shuffle(pool)
    while len(chosen) < n_sections:
        chosen.append(pool[len(chosen) % len(pool)] if pool else "Overview")
    rng.shuffle(chosen)

    sections: list[Section] = []
    evidence: dict[str, EvidenceSpan] = {}
    # map attribute -> the section title that will hold it (== its natural title)
    attr_by_title = {_ATTR_SECTION[a]: a for a in facts}
    used_titles: set[str] = set()
    sid = 0
    for title in chosen:
        # ensure unique section_ids even if a title repeats
        sid += 1
        if title in attr_by_title and title not in used_titles:
            attr = attr_by_title[title]
            text = _build_fact_section(rng, attr, entity, facts[attr])
            evidence[attr] = EvidenceSpan(doc_id=doc_id, section_id=str(sid))
            used_titles.add(title)
        else:
            text = _build_section(rng, title, entity)
        sections.append(Section(section_id=str(sid), title=title, text=text))

    doc = Document(doc_id=doc_id, title=f"{entity} Operations Policy", sections=sections)
    return doc, evidence


def generate_corpus_v3(
    n_docs: int = 24,
    sections_per_doc: int = 10,
    n_instances: int = 200,
    multi_hop_frac: float = 0.3,
    paraphrase_frac: float = 0.6,
    seed: int = 0,
) -> list[QAInstance]:
    """Generate a realistic-scale corpus + QA instances.

    Single-hop: ask one attribute of one entity. Multi-hop: compare a numeric
    attribute across two entities (answer is the entity name with the higher value).
    """
    rng = random.Random(seed)
    entities = rng.sample(ENTITY_NAMES, k=min(n_docs, len(ENTITY_NAMES)))
    while len(entities) < n_docs:
        entities.append(f"{rng.choice(ENTITY_NAMES)}-{rng.randint(2,9)}")

    corpus: list[Document] = []
    lookup: dict[str, dict[str, tuple[Any, EvidenceSpan]]] = {}
    for i, ent in enumerate(entities):
        doc_id = f"doc_{i+1:02d}"
        k = rng.randint(2, 4)  # 2-4 planted facts per doc (more than v2's 1-3)
        attrs = rng.sample(list(ATTRIBUTES_V3.keys()), k=k)
        facts = {a: rng.choice(ATTRIBUTES_V3[a]["values"]) for a in attrs}
        doc, evi = _make_doc(rng, doc_id, ent, sections_per_doc, facts)
        corpus.append(doc)
        lookup[ent] = {a: (v, evi[a]) for a, v in facts.items()}

    numeric_attrs = [a for a in ATTRIBUTES_V3 if ATTRIBUTES_V3[a]["values"][0] not in ("",) and isinstance(ATTRIBUTES_V3[a]["values"][0], (int, float))]
    instances: list[QAInstance] = []
    ent_list = list(lookup.keys())
    for j in range(n_instances):
        if rng.random() < multi_hop_frac and len(ent_list) >= 2:
            attr = rng.choice(numeric_attrs)
            # find two entities that both have this attr
            cand = [e for e in ent_list if attr in lookup[e]]
            if len(cand) >= 2:
                e1, e2 = rng.sample(cand, 2)
                v1, sp1 = lookup[e1][attr]
                v2, sp2 = lookup[e2][attr]
                human = attr.replace("_", " ")
                q = (f"Between {e1} and {e2}, which has the higher {human}? "
                     f"Reply with only that entity's name.")
                ans = e1 if v1 >= v2 else e2
                instances.append(QAInstance(
                    instance_id=f"qa_{j+1:04d}",
                    question=q, gold_answer=ans,
                    gold_evidence=[sp1, sp2], docs=corpus,
                    meta={"kind": "multi_hop", "attr": attr, "num_gold_sections": 2},
                ))
                continue
        # single-hop
        ent = rng.choice(ent_list)
        attrs_present = list(lookup[ent].keys())
        attr = rng.choice(attrs_present)
        value, span = lookup[ent][attr]
        spec = ATTRIBUTES_V3[attr]
        para = rng.random() < paraphrase_frac
        q = (spec["q_para"] if para else spec["q"]).format(entity=ent)
        instances.append(QAInstance(
            instance_id=f"qa_{j+1:04d}",
            question=q, gold_answer=str(value),
            gold_evidence=[span], docs=corpus,
            meta={"kind": "single_hop", "attr": attr, "entity": ent,
                  "paraphrased": para, "num_gold_sections": 1},
        ))

    rng.shuffle(instances)
    return instances


def serialize(instances: list[QAInstance], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = []
    for ins in instances:
        out.append({
            "instance_id": ins.instance_id, "question": ins.question,
            "gold_answer": ins.gold_answer,
            "gold_evidence": [{"doc_id": e.doc_id, "section_id": e.section_id} for e in ins.gold_evidence],
            "meta": ins.meta,
            "docs": [{"doc_id": d.doc_id, "title": d.title,
                      "sections": [{"section_id": s.section_id, "title": s.title, "text": s.text} for s in d.sections]}
                     for d in ins.docs],
        })
    json.dump(out, open(path, "w"), ensure_ascii=False, indent=2)


def main():
    p = argparse.ArgumentParser(description="Generate DocScout synth-v3 realistic-scale corpus.")
    p.add_argument("--n-docs", type=int, default=24)
    p.add_argument("--sections-per-doc", type=int, default=10)
    p.add_argument("--n-instances", type=int, default=200)
    p.add_argument("--multi-hop-frac", type=float, default=0.3)
    p.add_argument("--paraphrase-frac", type=float, default=0.6)
    p.add_argument("--seed", type=int, default=77)
    p.add_argument("--out", type=str, default="data/synth/v3_eval200.json")
    args = p.parse_args()
    insts = generate_corpus_v3(
        n_docs=args.n_docs, sections_per_doc=args.sections_per_doc,
        n_instances=args.n_instances, multi_hop_frac=args.multi_hop_frac,
        paraphrase_frac=args.paraphrase_frac, seed=args.seed,
    )
    serialize(insts, args.out)
    ex = insts[0]
    print(f"Generated {len(insts)} instances -> {args.out}")
    print(f"  docs={args.n_docs} sections/doc={args.sections_per_doc}")
    # report section length stats
    import statistics as st
    wlens = [len(s.text.split()) for d in [insts[0].docs[0]] for s in d.sections]
    allwl = [len(s.text.split()) for ins in insts for d in ins.docs[:1] for s in d.sections]
    print(f"  section WORDS (one corpus): mean={st.mean(allwl):.0f} min={min(allwl)} max={max(allwl)}")
    print(f"  example: {ex.question[:90]} |=> {ex.gold_answer} (kind={ex.meta.get('kind')})")


if __name__ == "__main__":
    main()
