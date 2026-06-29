"""Correct GRPO trainer for DocScout (GPU only).

Fixes vs. the earlier draft:
  - NO gradient_checkpointing (it silently zeroed grads when inputs had no requires_grad).
  - Rollout stores the ACTUAL per-step (prompt_ids, gen_ids) used during generation;
    logp is computed per-step against that exact context (not a re-derived prompt),
    so the policy gradient is correct.

DR.GRPO-style (group-relative advantage). For the CodeScout-faithful path, mirror
`reference/codescout/src/train.py` (SkyRL). This module is the framework-agnostic
fallback (HF transformers + vLLM-free).
"""

from __future__ import annotations

from dataclasses import dataclass

from docscout.agent.parsing import parse_action
from docscout.agent.rollout import build_messages, _force_answer_messages
from docscout.env.docstore import DocStore
from docscout.env.search_env import EnvConfig, SearchEnv
from docscout.reward.reward import RewardConfig, compute_reward
from docscout.types import QAInstance


@dataclass
class TrainConfig:
    base_model: str = "Qwen/Qwen3-1.7B"
    reward_name: str = "ratio"
    env: EnvConfig = None
    reward: RewardConfig = None
    lr: float = 1e-6
    group_size: int = 4
    batch_instances: int = 2
    max_steps_per_episode: int = 6
    recent_reads_kept: int = 2
    temperature: float = 0.8
    max_new_tokens: int = 96
    max_train_steps: int = 500
    save_dir: str = "ckpts/docscout"
    lora_adapter: str = None   # if set, load this LoRA adapter onto base_model (for RL from a LoRA SFT)
    train_lora: bool = False   # if True, add fresh LoRA to base (for RL from base without SFT)


def _one_episode(model, tok, instance: QAInstance, tcfg, reward_name, reward_cfg, device):
    """Run one episode; return (reward, [(prompt_ids, gen_ids), ...]) with the
    ACTUAL prompt ids used at each generation step (for correct logp)."""
    import torch
    store = DocStore(instance.docs, rerank=getattr(tcfg.env, "rerank", False) if tcfg.env else False)
    env = SearchEnv(instance, store, tcfg.env or EnvConfig(max_steps=tcfg.max_steps_per_episode))
    pairs = []
    while not env.done:
        max_steps = (tcfg.env or EnvConfig(max_steps=tcfg.max_steps_per_episode)).max_steps
        if env.n_steps >= max_steps - 1:
            messages = _force_answer_messages(env, tcfg.recent_reads_kept)
        else:
            messages = build_messages(env, tcfg.recent_reads_kept)
        prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                         enable_thinking=False)
        prompt_ids = tok(prompt, return_tensors="pt")["input_ids"][0]
        enc = prompt_ids.unsqueeze(0).to(device)
        model.eval()  # generate in eval mode (dropout off) for stable rollouts
        with torch.no_grad():
            out = model.generate(input_ids=enc, do_sample=True, temperature=max(tcfg.temperature, 1e-3),
                                 top_p=0.95, max_new_tokens=tcfg.max_new_tokens,
                                 pad_token_id=tok.eos_token_id or 151645)
        model.train()
        gen_ids = out[0, enc.shape[1]:]
        if gen_ids.numel() == 0:
            break
        pairs.append((prompt_ids, gen_ids.detach().cpu()))
        text = tok.decode(gen_ids, skip_special_tokens=True)
        env.step(parse_action(text))
    reward, _ = compute_reward(reward_name, env, instance, reward_cfg)
    return reward, pairs


def _logp(model, tok, pairs, device):
    """Sum logprob of generated tokens, each computed against its ACTUAL prompt
    context (grad-enabled forward per step)."""
    import torch
    import torch.nn.functional as F
    total = torch.zeros((), device=device)
    for prompt_ids, gen_ids in pairs:
        full = torch.cat([prompt_ids.to(device), gen_ids.to(device)]).unsqueeze(0)
        logits = model(full).logits[0]            # (T, V); grads flow to model weights
        lp = F.log_softmax(logits, dim=-1)
        n = prompt_ids.shape[0]
        for j, tid in enumerate(gen_ids.tolist()):
            total = total + lp[n + j - 1, tid]
    return total


def train(instances: list[QAInstance], tcfg: TrainConfig, reward_name: str, reward_cfg: RewardConfig):
    """GRPO training loop. Requires GPU + torch/transformers."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda"
    tok = AutoTokenizer.from_pretrained(tcfg.base_model, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        tcfg.base_model, dtype=torch.bfloat16, device_map=device, local_files_only=True)
    if getattr(tcfg, "lora_adapter", None):
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, tcfg.lora_adapter, is_trainable=True)
    elif getattr(tcfg, "train_lora", False):
        from peft import LoraConfig, get_peft_model
        lc = LoraConfig(r=32, lora_alpha=64, lora_dropout=0.05, bias="none",
                        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
                        task_type="CAUSAL_LM")
        model = get_peft_model(model, lc)
    # gradient checkpointing with use_reentrant=False: saves activation memory AND
    # does NOT require inputs to have requires_grad (the earlier reentrant default
    # silently produced "Gradients will be None").
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    model.config.use_cache = False
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=tcfg.lr)
    tcfg.env = tcfg.env or EnvConfig(max_steps=tcfg.max_steps_per_episode)
    pad_id = tok.eos_token_id or 151645

    step = 0
    G = max(tcfg.group_size, 1)  # rollouts per instance (GRPO group)
    REINFORCE = (G == 1)  # group==1 -> REINFORCE-with-baseline (across-instance)
    n_var_groups = 0  # diagnostic: groups with within-group reward variance
    baseline = 0.0  # EMA baseline for REINFORCE-with-baseline
    while step < tcfg.max_train_steps:
        bi = (step * tcfg.batch_instances) % len(instances)
        batch = (instances[bi:] + instances[:bi])[: tcfg.batch_instances]
        opt.zero_grad()
        rewards_log = []
        loss_acc = torch.zeros((), device=device)
        for inst in batch:
            rolls = [_one_episode(model, tok, inst, tcfg, reward_name, reward_cfg, device)
                     for _ in range(G)]
            rs = [r for r, _ in rolls]
            rewards_log.extend(rs)
            if REINFORCE:
                # Across-instance advantage: every instance contributes a gradient
                # (correct instances r>baseline reinforced, wrong r<baseline discouraged).
                # Does NOT depend on within-group variance, so it works even when the
                # SFT policy is peaked (where GRPO sees zero advantage).
                r, pairs = rolls[0]
                adv = r - baseline
                logp = _logp(model, tok, pairs, device)
                loss_acc = loss_acc + (-adv * logp)
                if abs(adv) > 1e-6:
                    n_var_groups += 1
            else:
                # GRPO: group-relative advantage (DR.GRPO: no std). Meaningful only
                # when rollouts diverge (within-group variance); degenerate groups
                # (all-same reward) contribute zero advantage honestly.
                gmean = sum(rs) / len(rs)
                if max(rs) - min(rs) > 1e-6:
                    n_var_groups += 1
                for r, pairs in rolls:
                    adv = r - gmean
                    if abs(adv) < 1e-6:
                        continue
                    logp = _logp(model, tok, pairs, device)
                    loss_acc = loss_acc + (-adv * logp)
        if REINFORCE:
            bm = sum(rewards_log) / len(rewards_log)
            baseline = 0.9 * baseline + 0.1 * bm  # update EMA baseline
        denom = max(len(batch) * G, 1)
        # Guard: if NO group had within-group variance this step, loss_acc is a bare
        # zeros tensor with no grad_fn -> backward would crash. Skip the update
        # honestly (the policy is too peaked to learn from at this temp).
        if loss_acc.requires_grad:
            (loss_acc / len(batch)).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        step += 1
        if step % 1 == 0:
            print(f"[step {step}] loss={float(loss_acc)/denom:.4f} "
                  f"mean_reward={sum(rewards_log)/len(rewards_log):+.3f} "
                  f"min/max={min(rewards_log):.2f}/{max(rewards_log):.2f} "
                  f"var_groups={n_var_groups}", flush=True)
        # intermediate checkpoint every 20 steps (so we can eval progress)
        if step % 20 == 0 or step == tcfg.max_train_steps:
            import pathlib
            pathlib.Path(tcfg.save_dir).mkdir(parents=True, exist_ok=True)
            model.save_pretrained(tcfg.save_dir); tok.save_pretrained(tcfg.save_dir)
            print(f"  [ckpt @ step {step}] saved -> {tcfg.save_dir}", flush=True)
    print(f"GRPO done. var_groups (w/ within-group signal) = {n_var_groups}/{step}", flush=True)
