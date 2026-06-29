"""Round-9 failure-classification diagnostic (自动化实验迭代方案.md §四/§八Step6).

Runs an agent checkpoint on synth-v3 and breaks down failures into:
  - selection_fail : agent never read the gold section(s)  (retrieval/selection)
  - read_fail      : agent read gold but answered wrong    (extraction/answering)
  - correct        : read gold AND answered right
Also reports evidence recall/precision and the conditional accuracy
P(correct | read gold) vs P(correct | not read gold) — this isolates whether
the bottleneck is SELECTING the right section or EXTRACTING the answer from it.

  python -m scripts.v3_failure --model <base> --adapter ckpts/docscout-sft-v3-lora -n 150
"""
from __future__ import annotations

import argparse
import json
import statistics as st

from docscout.agent.client import HFClient
from docscout.agent.rollout import rollout
from docscout.env.search_env import EnvConfig
from docscout.reward.answer_scoring import score_answer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="/home/linkco/.cache/modelscope/hub/models/Qwen/Qwen3-1___7B")
    p.add_argument("--adapter", default="ckpts/docscout-sft-v3-lora")
    p.add_argument("--split", default="data/synth/v3_eval300.json")
    p.add_argument("-n", type=int, default=150)
    p.add_argument("--temp", type=float, default=0.0)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--out", default="results/v3_failure.json")
    args = p.parse_args()

    from scripts.run_retriever_eval import load_split
    insts = load_split(args.split)[: args.n]
    client = HFClient(args.model, device=args.device, temperature=args.temp, max_new_tokens=96)
    if args.adapter:
        from peft import PeftModel
        client.model = PeftModel.from_pretrained(client.model, args.adapter).merge_and_unload()
        client.model.eval()

    n = len(insts)
    sel_fail = read_fail = correct = 0
    read_gold_acc = []   # accuracy on instances where agent DID read gold
    no_gold_acc = []     # accuracy on instances where agent did NOT read gold
    ev_recall = []
    for i, ins in enumerate(insts):
        res, env = rollout(ins, client, env_config=EnvConfig(max_steps=6, search_k=5), return_env=True)
        gold = ins.gold_sections()
        read = env.committed_read_uids()
        hit = len(read & gold)
        rec = hit / len(gold) if gold else 1.0
        ev_recall.append(rec)
        acc = score_answer(res.trajectory.final_answer, str(ins.gold_answer))
        read_all_gold = (hit == len(gold))
        if read_all_gold:
            read_gold_acc.append(acc)
            if acc >= 0.5:
                correct += 1
            else:
                read_fail += 1
        else:
            no_gold_acc.append(acc)
            sel_fail += 1
        if (i + 1) % 30 == 0:
            print(f"  [{i+1}/{n}] sel_fail={sel_fail} read_fail={read_fail} correct={correct}", flush=True)

    ans_acc = (correct + read_fail * 0) / n  # correct uses acc>=0.5 threshold; recompute properly
    # proper answer accuracy
    # (re-derive: correct = read gold & right; read_fail = read gold & wrong; sel_fail = didn't read gold)
    summary = {
        "model": args.model, "adapter": args.adapter, "n": n,
        "answer_accuracy": round(correct / n, 4),
        "evidence_recall": round(st.mean(ev_recall), 4),
        "selection_fail_pct": round(sel_fail / n, 4),
        "read_fail_pct": round(read_fail / n, 4),
        "correct_pct": round(correct / n, 4),
        "acc_given_read_gold": round(st.mean(read_gold_acc), 4) if read_gold_acc else None,
        "acc_given_no_gold": round(st.mean(no_gold_acc), 4) if no_gold_acc else None,
        "n_read_gold": len(read_gold_acc), "n_no_gold": len(no_gold_acc),
    }
    print("\n===== FAILURE CLASSIFICATION =====")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print(f"\nDiagnosis: bottleneck is {'SELECTION (did not read gold)' if sel_fail > read_fail else 'READING/ANSWERING (read gold but wrong)'}")
    if read_gold_acc:
        print(f"  When agent reads gold, acc={st.mean(read_gold_acc):.3f} "
              f"(cf oracle 0.827 — gap = answer-generation quality)")
    json.dump(summary, open(args.out, "w"), indent=2)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
