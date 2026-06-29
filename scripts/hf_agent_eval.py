"""HF-based DocScout agent eval — reports the accuracy-per-read-token frontier point
for any checkpoint (SFT or RL). Uses HFClient (transformers direct, GPU) so it works
without a vLLM server. Runs the full agent loop (search/read/expand/answer).

Metric of record: answer accuracy vs committed read-tokens (READ/EXPAND content only,
NOT snippet skims — this is the read-budget cost the policy controls).

  python -m scripts.hf_agent_eval --model ckpts/docscout-sft11-1b5-musique \
      --split data/grounded/musique_test.json -n 100 --tag sft11_musique
"""
from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path

from docscout.agent.client import HFClient
from docscout.agent.rollout import rollout
from docscout.env.search_env import EnvConfig
from docscout.reward.answer_scoring import score_answer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--split", default="data/grounded/musique_test.json")
    p.add_argument("-n", type=int, default=100)
    p.add_argument("--max-steps", type=int, default=6)
    p.add_argument("--search-k", type=int, default=5)
    p.add_argument("--temp", type=float, default=0.3, help="low temp = deterministic policy point")
    p.add_argument("--rerank", action="store_true")
    p.add_argument("--tag", default="eval")
    p.add_argument("--out", default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--adapter", default=None, help="LoRA adapter to load onto --model (for RL eval)")
    args = p.parse_args()

    from scripts.run_retriever_eval import load_split
    insts = load_split(args.split)[: args.n]
    client = HFClient(args.model, device=args.device, temperature=args.temp, max_new_tokens=96)
    if args.adapter:
        from peft import PeftModel
        client.model = PeftModel.from_pretrained(client.model, args.adapter).merge_and_unload()
        client.model.eval()
        print(f"[eval] merged LoRA adapter {args.adapter} into base", flush=True)
    env_cfg = EnvConfig(max_steps=args.max_steps, search_k=args.search_k, rerank=args.rerank)
    # Honest BPE accounting: re-tokenize the text of each COMMITTED read (READ/EXPAND).
    from transformers import AutoTokenizer
    btok = AutoTokenizer.from_pretrained(args.model, local_files_only=True)

    def committed_bpe(env):
        tot = 0
        for (d, s) in env.committed_read_uids():
            r = env.store.read(d, s)
            if r:
                tot += len(btok(r["content"])["input_ids"])
        return tot

    rows = []
    n_ans = 0
    for i, ins in enumerate(insts):
        res, env = rollout(ins, client, env_config=env_cfg, return_env=True)
        sc = score_answer(res.trajectory.final_answer, str(ins.gold_answer))
        committed_tok = env.committed_read_tokens
        committed_bpe_tok = committed_bpe(env)
        if res.trajectory.terminated_by == "answer":
            n_ans += 1
        rows.append({
            "id": ins.instance_id, "gold": str(ins.gold_answer),
            "pred": res.trajectory.final_answer[:80], "score": sc,
            "committed_tok": committed_tok, "committed_bpe": committed_bpe_tok,
            "total_tok": res.trajectory.total_read_tokens,
            "n_read": res.trajectory.n_read, "n_steps": res.trajectory.n_steps,
            "terminated": res.trajectory.terminated_by,
        })
        if (i + 1) % 20 == 0:
            acc = st.mean(r["score"] for r in rows)
            print(f"  [{i+1}/{len(insts)}] running acc={acc:.3f} ans_rate={n_ans/(i+1):.2f}", flush=True)

    acc = st.mean(r["score"] for r in rows)
    ctok = st.mean(r["committed_tok"] for r in rows)
    cbpe = st.mean(r["committed_bpe"] for r in rows)
    summary = {
        "model": args.model, "split": args.split, "n": len(rows), "tag": args.tag,
        "temp": args.temp, "max_steps": args.max_steps,
        "answer_accuracy": acc,
        "mean_committed_read_tokens": ctok,
        "mean_committed_read_bpe": cbpe,
        "mean_total_read_tokens": st.mean(r["total_tok"] for r in rows),
        "mean_n_read": st.mean(r["n_read"] for r in rows),
        "answer_rate": n_ans / len(rows),
        "acc_per_committed_bpe": acc / max(cbpe, 1.0),
        "acc_per_committed_tok": acc / max(ctok, 1.0),
    }
    print("\n=== HF AGENT EVAL ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    out = args.out or f"results/hf_agent_{args.tag}.json"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"summary": summary, "rows": rows}, open(out, "w"), indent=2, ensure_ascii=False)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
