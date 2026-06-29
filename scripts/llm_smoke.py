"""Smoke-test the DocScout GLM API wrapper.

Usage:
    python -m scripts.llm_smoke                # ping + 1 completion + JSON + batch
    python -m scripts.llm_smoke --judge        # also demo LLM-as-judge scoring

Reads config from $DOCSCOUT_LLM_TOKEN or configs/llm_api.yaml (see
configs/llm_api.example.yaml). Run on the HOST laptop (GPU box has no internet).
"""

from __future__ import annotations

import argparse
import json
import time

from docscout.llm import LLMClient, load_config


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", action="store_true", help="also demo LLM-as-judge")
    args = ap.parse_args()

    cfg = load_config()
    print(f"[config] {cfg.masked()}")
    client = LLMClient(cfg)

    # 1. ping
    t0 = time.time()
    ok = client.ping()
    print(f"[ping] {'OK' if ok else 'FAILED'}  ({time.time()-t0:.1f}s)")
    if not ok:
        raise SystemExit("ping failed — check token / base_url / network")

    # 2. plain completion
    out = client.complete("In one sentence, what is a multi-hop QA dataset?", max_tokens=80)
    print(f"[complete] {out}")

    # 3. JSON mode
    obj = client.complete_json(
        'Return a JSON object: {"capital_of_france": "<city>", "n": <1+1>}.',
        max_tokens=60,
    )
    print(f"[complete_json] {json.dumps(obj, ensure_ascii=False)}")

    # 4. batch (threaded)
    t0 = time.time()
    qs = [f"Reply with exactly the number {i}." for i in range(1, 6)]
    outs = client.batch(qs, max_workers=5, max_tokens=10, temperature=0.0)
    print(f"[batch] {outs}  ({time.time()-t0:.1f}s for {len(qs)} calls)")

    # 5. optional LLM-as-judge demo
    if args.judge:
        prompt = (
            "You are grading a short-answer QA prediction.\n"
            'Question: "How long after approval does AuroraPay sync permission changes?"\n'
            'Gold answer: "5 minutes"\n'
            'Prediction: "within five min"\n'
            "Is the prediction correct? Reply JSON: "
            '{"correct": true/false, "reason": "<short>"}.'
        )
        verdict = client.complete_json(prompt, max_tokens=80)
        print(f"[judge] {json.dumps(verdict, ensure_ascii=False)}")

    print("\nAll smoke checks passed.")


if __name__ == "__main__":
    main()
