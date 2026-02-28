#!/usr/bin/env python3
"""
optimize_prompt_local.py — Same as optimize_prompt.py but with pre-planned
prompts supplied directly (no ANTHROPIC_API_KEY required).

Loads model once, evaluates each candidate prompt on N samples, reports
round-by-round diagnostics, and prints a RESULT block at the end.

Usage (run from project root):
    python .claude/skills/get-sys-prompt/scripts/optimize_prompt_local.py countdown
"""

import os, sys, textwrap

_HERE        = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(_HERE))))
sys.path.insert(0, PROJECT_ROOT)
_DIAG_DIR = os.path.join(PROJECT_ROOT, ".claude", "skills", "try-model-on-env", "scripts")
sys.path.insert(0, _DIAG_DIR)

DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
N_SAMPLES     = 3

# ---------------------------------------------------------------------------
# Pre-planned candidate prompts (Claude Code acting as the optimizer)
# ---------------------------------------------------------------------------

CANDIDATE_PROMPTS = [
    # Round 1 — seed (minimal, permissive)
    (
        "You are a helpful assistant. "
        "Think step by step inside <think>...</think> tags, "
        "then provide your final answer inside <answer>...</answer> tags."
    ),

    # Round 2 — explicit format block, imperative tone
    (
        "You MUST format every response like this:\n"
        "<think>\n"
        "[your step-by-step reasoning]\n"
        "</think>\n"
        "<answer>\n"
        "[your final answer only]\n"
        "</answer>\n"
        "Do not write anything before <think> or after </answer>."
    ),

    # Round 3 — role + numbered rules + short example skeleton
    (
        "You are a mathematical reasoning assistant.\n"
        "Rules:\n"
        "1. Always start with <think> and close with </think>.\n"
        "2. After </think>, write <answer> then your answer then </answer>.\n"
        "3. Never place text outside these two tag pairs.\n"
        "Format example:\n"
        "<think>\n"
        "Step-by-step work here.\n"
        "</think>\n"
        "<answer>\n"
        "Final answer here.\n"
        "</answer>"
    ),

    # Round 4 — ultra-terse single-line format spec
    (
        "Respond ONLY in this format: "
        "<think>your reasoning</think><answer>your final answer</answer>"
    ),

    # Round 5 — countdown-specific + format rules embedded in context
    (
        "You solve arithmetic countdown puzzles.\n"
        "For every puzzle, respond in this exact structure:\n"
        "<think>\n"
        "Explore combinations step by step until you find an expression that equals the target.\n"
        "</think>\n"
        "<answer>\n"
        "The arithmetic expression (no '=' sign, no target number, just the expression).\n"
        "</answer>"
    ),
]

# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------

def _section(title):
    print(f"\n{'─' * 64}")
    print(f"  {title}")
    print(f"{'─' * 64}")

def _ok(msg):   print(f"  [OK]   {msg}")
def _warn(msg): print(f"  [WARN] {msg}")

# ---------------------------------------------------------------------------
# Evaluate one prompt on all samples
# ---------------------------------------------------------------------------

def evaluate_prompt(system_prompt, env, samples, model, tokenizer):
    from diagnostic import build_prompt_string, run_inference, eval_rewards

    per_sample = []
    for sample in samples:
        prompt_str = build_prompt_string(sample, system_prompt, tokenizer)
        completion = run_inference(model, tokenizer, prompt_str, max_new_tokens=512)
        rewards    = eval_rewards(env, completion, sample)
        per_sample.append({
            "completion":  completion,
            "rewards":     rewards,
            "has_think":   "<think>"  in completion and "</think>"  in completion,
            "has_answer":  "<answer>" in completion and "</answer>" in completion,
        })

    def _avg(key):
        vals = [r["rewards"].get(key, 0.0) for r in per_sample
                if isinstance(r["rewards"].get(key), float)]
        return sum(vals) / len(vals) if vals else 0.0

    return {"per_sample": per_sample, "avg_format": _avg("format_reward"),
            "avg_correct": _avg("correctness_reward")}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("env_name",   default="countdown", nargs="?")
    parser.add_argument("model_name", default=DEFAULT_MODEL, nargs="?")
    parser.add_argument("--samples",  type=int, default=N_SAMPLES)
    args = parser.parse_args()

    print("=" * 64)
    print("  get-sys-prompt  ·  local optimizer (no API key needed)")
    print("=" * 64)
    print(f"  Env     : {args.env_name}")
    print(f"  Model   : {args.model_name}")
    print(f"  Rounds  : {len(CANDIDATE_PROMPTS)}  |  Samples/round: {args.samples}")

    # Load env + samples (once)
    _section("Loading environment and samples")
    from diagnostic import make_config
    from environments.reasoning_gym_env import ReasoningGymEnvironment

    config = make_config(args.env_name, args.model_name)
    config["environment"]["samples"] = args.samples
    env = ReasoningGymEnvironment(config)
    dataset = env.get_dataset(config)
    samples = [dataset[i] for i in range(min(args.samples, len(dataset)))]
    _ok(f"Loaded {len(samples)} samples from '{args.env_name}'")

    # Load model + tokenizer (once)
    from diagnostic import load_model_and_tokenizer
    _section(f"Loading model: {args.model_name}")
    model, tokenizer = load_model_and_tokenizer(args.model_name)

    best = {"system_prompt": CANDIDATE_PROMPTS[0], "avg_format": -1.0, "avg_correct": 0.0}

    for round_num, prompt in enumerate(CANDIDATE_PROMPTS, 1):
        _section(f"Round {round_num} / {len(CANDIDATE_PROMPTS)}")
        display = textwrap.shorten(prompt.replace('\n', ' '), width=100, placeholder="…")
        print(f"  Prompt: {display!r}\n")

        result  = evaluate_prompt(prompt, env, samples, model, tokenizer)
        avg_fmt = result["avg_format"]
        avg_cor = result["avg_correct"]
        print(f"  Avg format reward    : {avg_fmt:.2f}")
        print(f"  Avg correctness      : {avg_cor:.2f}")
        for i, s in enumerate(result["per_sample"]):
            tag_str = (f"think={'✓' if s['has_think'] else '✗'}  "
                       f"answer={'✓' if s['has_answer'] else '✗'}")
            snippet = s["completion"][:120].replace("\n", " ")
            print(f"    [{i+1}] {tag_str}  |  {snippet!r}")

        if avg_fmt > best["avg_format"] or (
                avg_fmt == best["avg_format"] and avg_cor > best["avg_correct"]):
            best = {"system_prompt": prompt, "avg_format": avg_fmt, "avg_correct": avg_cor}

        if avg_fmt >= 1.0:
            _ok("Format reward = 1.0 — stopping early.")
            break

    # Final report
    _section("RESULT — Best system prompt found")
    print(f"\n  Format reward   : {best['avg_format']:.2f}")
    print(f"  Correct reward  : {best['avg_correct']:.2f}")
    print(f"\n  Prompt text:\n")
    for line in best["system_prompt"].splitlines():
        print(f"    {line}")
    print()

    if best["avg_format"] >= 1.0:
        _ok("SUCCESS: format reward 1.0 — model reliably uses <think> and <answer> tags.")
    elif best["avg_format"] > 0.0:
        _warn(f"Partial: best format reward = {best['avg_format']:.2f}. "
              "Consider more rounds or a stronger base model.")
    else:
        _warn("Format reward stayed at 0.0. SFT warm-up likely needed before prompt-tuning helps.")


if __name__ == "__main__":
    main()
