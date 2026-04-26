"""End-to-end smoke test for ``official_hf_training.ipynb``.

Loads Qwen3-0.6B, collects 2 rollouts in-process, scores them with
``proxy_reward_one``, and runs a single GRPO step. Exits non-zero on any
failure. Designed to run on a single consumer GPU in <5 min, so we can
verify the notebook's pipeline before paying for HF Spaces compute.

Run from the repo root:
    python -m counterfeint.training.smoke_official_hf
"""

from __future__ import annotations

import sys
import time
import traceback
from pathlib import Path


def main() -> int:
    print("=" * 70)
    print("CounterFeint - official_hf_training.ipynb smoke test")
    print("=" * 70)

    # ---------------------------------------------------------------- #
    # 0. Imports
    # ---------------------------------------------------------------- #
    print("\n[0/5] Importing dependencies ...")
    import torch
    print(f"  torch={torch.__version__} cuda={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    from counterfeint.agents import HFInvestigator
    from counterfeint.scripted import HeuristicAuditor, ReactiveFraudster
    from counterfeint.training import (
        build_gold_lookup,
        collect_dataset_in_process,
        make_proxy_reward_fn,
        proxy_reward_one,
        samples_to_hf_dataset,
    )

    # ---------------------------------------------------------------- #
    # 1. Load model
    # ---------------------------------------------------------------- #
    BASE_MODEL = "Qwen/Qwen3-0.6B"
    LOAD_IN_4BIT = torch.cuda.is_available()  # 4-bit needs a GPU

    print(f"\n[1/5] Loading {BASE_MODEL} (load_in_4bit={LOAD_IN_4BIT}) ...")
    t0 = time.perf_counter()
    hf_inv = HFInvestigator.from_pretrained(
        BASE_MODEL,
        load_in_4bit=LOAD_IN_4BIT,
        max_new_tokens=128,
        temperature=0.3,
        do_sample=True,
        enable_thinking=False,
    )
    print(f"  Loaded in {time.perf_counter() - t0:.1f}s")

    # Quick chat-template smoke
    probe = hf_inv._call_chat([
        {"role": "system", "content": "You output one line of JSON only."},
        {"role": "user",   "content": "Reply with {\"ok\": true}"},
    ])
    print(f"  Probe completion: {probe[:160]!r}")

    # ---------------------------------------------------------------- #
    # 2. Collect 2 rollouts in-process
    # ---------------------------------------------------------------- #
    print("\n[2/5] Collecting 2 rollouts (task_1 seeds 11, 13, in-process) ...")
    t0 = time.perf_counter()
    samples = collect_dataset_in_process(
        hf_investigator=hf_inv,
        seeds_by_task={"task_1": [11, 13]},
        fraudster_factory=lambda: ReactiveFraudster(seed=42),
        auditor_factory=lambda: HeuristicAuditor(),
        max_steps=80,
        show_trace=False,
    )
    print(f"  Collected {len(samples)} rows in {time.perf_counter() - t0:.1f}s")
    print(f"  Investigator fallback={hf_inv.fallback_count}/{hf_inv.call_count}")

    if not samples:
        print("\n  FAIL: no usable samples — every step fell back to scripted.")
        return 1

    # ---------------------------------------------------------------- #
    # 3. Score completions with proxy_reward
    # ---------------------------------------------------------------- #
    print("\n[3/5] Scoring recorded completions with proxy_reward_one ...")
    gold_lookup = build_gold_lookup(samples)
    print(f"  gold_lookup size: {len(gold_lookup)}")

    def _score(sample):
        gold = gold_lookup.get(sample.prompt, {"fields": {}, "episode_score": 0.0})
        return proxy_reward_one(
            sample.prompt,
            sample.completion,
            gold=gold["fields"],
            gold_episode_score=float(gold["episode_score"]),
        )

    rewards = [_score(s) for s in samples[:5]]
    print(f"  First 5 rewards: {[round(r, 3) for r in rewards]}")

    fn = make_proxy_reward_fn(gold_lookup=gold_lookup)
    batched = fn([s.prompt for s in samples[:3]], [s.completion for s in samples[:3]])
    print(f"  Vectorized reward (3 rows): {[round(r, 3) for r in batched]}")

    # ---------------------------------------------------------------- #
    # 4. Build HF dataset
    # ---------------------------------------------------------------- #
    print("\n[4/5] Converting to HF Dataset ...")
    from counterfeint.agents.prompts import INVESTIGATOR_SYSTEM_PROMPT
    ds = samples_to_hf_dataset(samples, system_prompt=INVESTIGATOR_SYSTEM_PROMPT)
    print(f"  Dataset: {ds}")
    print(f"  Columns: {list(ds.column_names)}")

    # ---------------------------------------------------------------- #
    # 5. Wire up TRL GRPOConfig + GRPOTrainer (we don't run train(), just
    #    construct it — that's what catches version-incompat issues)
    # ---------------------------------------------------------------- #
    print("\n[5/5] Constructing GRPOTrainer ...")
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    if LOAD_IN_4BIT:
        hf_inv.model = prepare_model_for_kbit_training(hf_inv.model)
    lora_cfg = LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none", task_type="CAUSAL_LM",
    )
    hf_inv.model = get_peft_model(hf_inv.model, lora_cfg)
    hf_inv.model.print_trainable_parameters()

    from trl import GRPOConfig, GRPOTrainer
    out_dir = Path("outputs/smoke")
    out_dir.mkdir(parents=True, exist_ok=True)
    import inspect
    _cfg_kwargs = dict(
        output_dir=str(out_dir),
        learning_rate=5e-6,
        num_generations=2,
        beta=0.01,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=2,
        max_completion_length=256,
        num_train_epochs=1,
        save_steps=10000,
        logging_steps=1,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to="none",
        seed=7,
        remove_unused_columns=False,
        max_steps=3,
    )
    _grpo_params = set(inspect.signature(GRPOConfig.__init__).parameters)
    if "temperature" in _grpo_params:
        _cfg_kwargs["temperature"] = 0.7
    if "max_prompt_length" in _grpo_params:
        _cfg_kwargs["max_prompt_length"] = 1024
    cfg = GRPOConfig(**_cfg_kwargs)
    trainer = GRPOTrainer(
        model=hf_inv.model,
        args=cfg,
        train_dataset=ds,
        reward_funcs=[fn],
        processing_class=hf_inv.tokenizer,
    )
    if hasattr(trainer, "generation_config"):
        trainer.generation_config.temperature = 0.9
        trainer.generation_config.do_sample = True
    print("  GRPOTrainer ready.")

    print("\n[6/6] Running 1 GRPO training step ...")
    t0 = time.perf_counter()
    result = trainer.train()
    elapsed = time.perf_counter() - t0
    print(f"  Step took {elapsed:.1f}s")

    log = trainer.state.log_history
    if log:
        last = log[-1]
        loss = last.get("loss", last.get("train_loss", None))
        print(f"  Last log entry: {last}")
        if loss is not None and loss > 0.0:
            print(f"  loss={loss:.6f} — NON-ZERO — GRPO is learning!")
        else:
            print(f"  loss={loss} — WARNING: still zero, check reward variance")
    else:
        print("  No log entries recorded.")

    print("\n" + "=" * 70)
    print("SMOKE TEST PASSED")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(1)
