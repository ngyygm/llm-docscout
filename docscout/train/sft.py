"""SFT trainer for DocScout demonstration trajectories (自动化实验迭代方案.md §第三轮).

Supervises ONLY the assistant action tokens (masks system/user/tool). Builds
(prompt=messages[:i] with generation prompt, target=assistant_text) pairs from the
demo JSONL, trains next-token on the action tokens. RL (grpo.py) is then run from
this SFT checkpoint.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_pairs(jsonl_path: str, tok, max_per_traj: int = 8, answer_upweight: int = 4):
    """Yield (input_ids, labels) per assistant turn. Answer turns are duplicated
    `answer_upweight`x so the model actually learns to STOP and answer (otherwise
    read/search turns dominate and the model never emits ANSWER)."""
    pairs = []
    for line in open(jsonl_path):
        traj = json.loads(line)
        msgs = traj["messages"]
        ai = [k for k, m in enumerate(msgs) if m["role"] == "assistant"]
        for k in ai[:max_per_traj]:
            prefix = msgs[:k]
            target = msgs[k]["content"]
            is_answer = target.lstrip().lower().startswith("action: answer")
            inp = tok.apply_chat_template(prefix, tokenize=False, add_generation_prompt=True,
                                          enable_thinking=False)
            inp_ids = tok(inp, add_special_tokens=False)["input_ids"]
            tgt_ids = tok(target, add_special_tokens=False)["input_ids"] + [tok.eos_token_id or 151645]
            ids = inp_ids + tgt_ids
            labels = [-100] * len(inp_ids) + tgt_ids
            reps = answer_upweight if is_answer else 1
            for _ in range(reps):
                pairs.append((ids, labels))
    return pairs


def main():
    import torch
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/sft/v2_demo.jsonl")
    p.add_argument("--base", required=True)
    p.add_argument("--out", default="ckpts/docscout-sft")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--bsz", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--max-len", type=int, default=2048)
    p.add_argument("--lora", action="store_true", help="LoRA fine-tune (for 7B on 24GB)")
    args = p.parse_args()

    tok = AutoTokenizer.from_pretrained(args.base, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.base, dtype=torch.bfloat16, device_map="cuda", local_files_only=True)
    if args.lora:
        from peft import LoraConfig, get_peft_model
        lc = LoraConfig(r=32, lora_alpha=64, lora_dropout=0.05, bias="none",
                        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                        task_type="CAUSAL_LM")
        model = get_peft_model(model, lc)
        model.print_trainable_parameters()
    model.gradient_checkpointing_enable()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    model.train()
    pairs = load_pairs(args.data, tok)
    pairs = [(i, l) for i, l in pairs if len(i) <= args.max_len]
    print(f"SFT pairs: {len(pairs)}", flush=True)

    class DS(Dataset):
        def __len__(self): return len(pairs)
        def __getitem__(self, k): return {"ids": pairs[k][0], "labels": pairs[k][1]}

    pad = tok.pad_token_id or tok.eos_token_id
    def coll(b):
        m = max(len(x["ids"]) for x in b)
        ids = torch.tensor([[*x["ids"]] + [pad] * (m - len(x["ids"])) for x in b])
        lab = torch.tensor([[*x["labels"]] + [-100] * (m - len(x["labels"])) for x in b])
        return {"input_ids": ids, "labels": lab, "attention_mask": (ids != pad).long()}

    dl = DataLoader(DS(), batch_size=args.bsz, shuffle=True, collate_fn=coll)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    accum, step = 0, 0
    for ep in range(args.epochs):
        for b in dl:
            b = {k: v.to("cuda") for k, v in b.items()}
            out = model(**b)
            (out.loss / args.grad_accum).backward()
            accum += 1
            if accum % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); opt.zero_grad(); step += 1
                if step % 10 == 0:
                    print(f"[sft ep{ep} step{step}] loss={out.loss.item():.4f}", flush=True)
    Path(args.out).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.out); tok.save_pretrained(args.out)
    print(f"saved SFT model -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
