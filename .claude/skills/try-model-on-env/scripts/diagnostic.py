#!/usr/bin/env python3
"""
diagnostic.py — Smoke-test a model on one sample from a reasoning-gym environment.

Reuses the project's own `environments` package so that reward functions,
answer extraction, and normalization are tested exactly as they run during
GRPO training — not reimplemented here.

Usage (run from project root):
    python .claude/skills/try-model-on-env/scripts/diagnostic.py <env_name> [model_name]

Examples:
    python .claude/skills/try-model-on-env/scripts/diagnostic.py countdown
    python .claude/skills/try-model-on-env/scripts/diagnostic.py maze
    python .claude/skills/try-model-on-env/scripts/diagnostic.py countdown outputs/my-checkpoint
"""

import os
import sys

# Resolve project root: scripts/ → try-model-on-env/ → skills/ → .claude/ → project root
_HERE        = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(_HERE))))
sys.path.insert(0, PROJECT_ROOT)

DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful assistant. "
    "Think step by step inside <think>...</think> tags, "
    "then provide your final answer inside <answer>...</answer> tags."
)

REQUIRED_TAGS = ["<think>", "<answer>"]


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------

def _bar():
    print("─" * 64)

def _section(title: str) -> None:
    print(f"\n{'─' * 64}")
    print(f"  {title}")
    print(f"{'─' * 64}")

def _ok(msg):   print(f"  [OK]   {msg}")
def _warn(msg): print(f"  [WARN] {msg}")
def _info(msg): print(f"         {msg}")


# ---------------------------------------------------------------------------
# Build a minimal config dict that mirrors what train_grpo.py uses.
# ReasoningGymEnvironment reads environment.task_name, environment.samples,
# environment.task_params, and the top-level seed.
# ---------------------------------------------------------------------------

def make_config(env_name: str, model_name: str) -> dict:
    return {
        "model": {
            "name_or_path": model_name,
            "torch_dtype": "auto",
            "device_map": "auto",
        },
        "environment": {
            "name": "reasoning_gym",
            "task_name": env_name,
            "samples": 1,
        },
        "system_prompt": DEFAULT_SYSTEM_PROMPT,
        "seed": 42,
    }


# ---------------------------------------------------------------------------
# Step 1 — Load environment and get one sample.
# Uses ReasoningGymEnvironment directly (same class as train_grpo.py uses).
# ---------------------------------------------------------------------------

def load_env_and_sample(config: dict):
    from environments.reasoning_gym_env import ReasoningGymEnvironment  # project code

    env_name = config["environment"]["task_name"]
    _section(f"Loading environment: {env_name}")

    env = ReasoningGymEnvironment(config)
    dataset = env.get_dataset(config)
    sample = dataset[0]
    _ok(f"Sample loaded  — columns: {list(sample.keys())}")
    return env, sample


# ---------------------------------------------------------------------------
# Step 2 — System prompt check.
# Ensures the prompt instructs <think> / <answer> format before we commit
# to running inference.
# ---------------------------------------------------------------------------

def check_system_prompt(system_prompt: str) -> str:
    _section("System prompt check")
    print(f"  Content: {system_prompt!r}\n")

    missing = [tag for tag in REQUIRED_TAGS if tag not in system_prompt]
    if not missing:
        _ok("System prompt contains <think> and <answer> tag instructions.")
    else:
        for tag in missing:
            _warn(f"'{tag}' instruction missing — appending default system prompt.")
        system_prompt = system_prompt.rstrip() + "\n" + DEFAULT_SYSTEM_PROMPT
        print(f"\n  Augmented: {system_prompt!r}")

    return system_prompt


# ---------------------------------------------------------------------------
# Step 3 — Build the prompt string exactly as GRPOTrainer does:
#   1. Start with sample["prompt"]  (already a list of chat messages)
#   2. Prepend system message if none present  (mirrors train_grpo.py logic)
#   3. Apply chat template with add_generation_prompt=True
# ---------------------------------------------------------------------------

def build_prompt_string(sample: dict, system_prompt: str, tokenizer) -> str:
    messages = list(sample["prompt"])  # copy — don't mutate dataset row

    if not any(m.get("role") == "system" for m in messages):
        messages = [{"role": "system", "content": system_prompt}] + messages

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


# ---------------------------------------------------------------------------
# Step 4 — Load model and tokenizer.
# ---------------------------------------------------------------------------

def load_model_and_tokenizer(model_name: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    _section(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto",
    )
    model.eval()
    _ok(f"Loaded — device: {next(model.parameters()).device}")
    return model, tokenizer


# ---------------------------------------------------------------------------
# Step 5 — Run inference.
# ---------------------------------------------------------------------------

def run_inference(model, tokenizer, prompt_str: str, max_new_tokens: int = 512) -> str:
    import torch

    _section("Running inference")
    inputs = tokenizer(prompt_str, return_tensors="pt").to(model.device)
    _info(f"Input tokens : {inputs.input_ids.shape[1]}")
    _info(f"Max new      : {max_new_tokens}")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id,
        )

    completion = tokenizer.decode(
        outputs[0][inputs.input_ids.shape[1]:],
        skip_special_tokens=True,
    )
    _info(f"Output tokens: {outputs.shape[1] - inputs.input_ids.shape[1]}")
    return completion


# ---------------------------------------------------------------------------
# Step 6 — Evaluate rewards using the environment's own reward functions.
#
# Wraps the completion in TRL's format:
#   completions = [[{"role": "assistant", "content": text}]]
# so that env._unwrap_completion() handles it exactly as during training.
#
# Passes ground_truth and metadata as kwargs, matching how GRPOTrainer
# builds the reward call signature.
# ---------------------------------------------------------------------------

def eval_rewards(env, completion: str, sample: dict) -> dict:
    # TRL passes completions as a list-of-lists-of-dicts
    trl_completions = [[{"role": "assistant", "content": completion}]]
    kwargs = {
        "ground_truth": [sample["ground_truth"]],
        "metadata":     [sample.get("metadata", {})],
    }

    reward_funcs = env.get_reward_functions()
    results = {}
    for fn in reward_funcs:
        try:
            scores = fn(trl_completions, **kwargs)
            results[fn.__name__] = scores[0]
        except Exception as exc:
            results[fn.__name__] = f"ERROR: {exc}"

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("env_name", help="reasoning-gym task name, e.g. countdown, maze, gsm8k")
    parser.add_argument("model_name", nargs="?", default=DEFAULT_MODEL, help="HuggingFace ID or local checkpoint path")
    parser.add_argument("--system-prompt", default=None, dest="system_prompt",
                        help="Override the default system prompt (wrap in quotes)")
    args = parser.parse_args()

    env_name   = args.env_name
    model_name = args.model_name

    print("=" * 64)
    print("  try-model-on-env  ·  diagnostic run")
    print("=" * 64)
    print(f"  Env   : {env_name}")
    print(f"  Model : {model_name}")

    config = make_config(env_name, model_name)
    if args.system_prompt:
        config["system_prompt"] = args.system_prompt

    # 1 — Environment + sample
    env, sample = load_env_and_sample(config)

    # 2 — System prompt
    system_prompt = check_system_prompt(config["system_prompt"])

    # 3 — Prompt preview (tokenizer only — don't load the full model yet)
    from transformers import AutoTokenizer
    tok_preview = AutoTokenizer.from_pretrained(model_name)
    if tok_preview.pad_token is None:
        tok_preview.pad_token = tok_preview.eos_token

    prompt_str = build_prompt_string(sample, system_prompt, tok_preview)

    _section("Prompt the model will see")
    print(prompt_str)

    _section("Ground truth")
    print(f"  {sample['ground_truth']!r}")

    # 4 — Full model load
    model, tokenizer = load_model_and_tokenizer(model_name)

    # 5 — Inference
    completion = run_inference(model, tokenizer, prompt_str)

    _section("Model response")
    print(completion)

    # 6 — Rewards
    _section("Reward diagnostics")

    has_think  = "<think>"  in completion and "</think>"  in completion
    has_answer = "<answer>" in completion and "</answer>" in completion
    print(f"  <think>...</think>   present : {'YES ✓' if has_think  else 'NO  ✗'}")
    print(f"  <answer>...</answer> present : {'YES ✓' if has_answer else 'NO  ✗'}")

    # Reuse env._extract_answer() — the same function used by correctness_reward
    extracted = env._extract_answer(completion)
    print(f"\n  Extracted answer : {extracted!r}")
    print(f"  Ground truth     : {sample['ground_truth']!r}")

    rewards = eval_rewards(env, completion, sample)

    print()
    for name, score in rewards.items():
        if isinstance(score, float):
            if score >= 1.0:
                tag = "PASS"
            elif score > 0.0:
                tag = "PARTIAL"
            else:
                tag = "FAIL"
            print(f"  {name:35s}: {score:.1f}  [{tag}]")
        else:
            print(f"  {name:35s}: {score}")

    _section("Summary")
    numeric = {k: v for k, v in rewards.items() if isinstance(v, float)}
    total   = sum(numeric.values())
    maximum = float(len(numeric))
    for name, score in numeric.items():
        print(f"  {name:35s}: {score:.1f} / 1.0")
    print(f"  {'combined':35s}: {total:.1f} / {maximum:.1f}")


if __name__ == "__main__":
    main()
