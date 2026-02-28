#!/usr/bin/env python3
"""
optimize_prompt.py — Iteratively refine a system prompt so that a model
generates the correct <think>...</think><answer>...</answer> format.

Loads the model once, then loops:
  1. Evaluate current system prompt on N samples (format + correctness reward)
  2. If format reward is already 1.0, stop.
  3. Otherwise, call Claude API to propose a refined prompt.
  4. Repeat up to MAX_ROUNDS times.

Requires ANTHROPIC_API_KEY in the environment.

Usage (run from project root):
    python .claude/skills/get-sys-prompt/scripts/optimize_prompt.py <env_name> [model_name]

Examples:
    python .claude/skills/get-sys-prompt/scripts/optimize_prompt.py countdown
    python .claude/skills/get-sys-prompt/scripts/optimize_prompt.py maze outputs/my-checkpoint
"""

import os
import sys
import argparse

# Resolve project root: scripts/ → get-sys-prompt/ → skills/ → .claude/ → project root
_HERE        = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(_HERE))))
sys.path.insert(0, PROJECT_ROOT)

# Also expose diagnostic helpers from the sibling skill
_DIAG_DIR = os.path.join(PROJECT_ROOT, ".claude", "skills", "try-model-on-env", "scripts")
sys.path.insert(0, _DIAG_DIR)

DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
MAX_ROUNDS    = 5
N_SAMPLES     = 3

SEED_PROMPT = (
    "You are a helpful assistant. "
    "Think step by step inside <think>...</think> tags, "
    "then provide your final answer inside <answer>...</answer> tags."
)


# ---------------------------------------------------------------------------
# UI helpers (mirrors diagnostic.py style)
# ---------------------------------------------------------------------------

def _section(title: str) -> None:
    print(f"\n{'─' * 64}")
    print(f"  {title}")
    print(f"{'─' * 64}")

def _ok(msg):   print(f"  [OK]   {msg}")
def _warn(msg): print(f"  [WARN] {msg}")
def _info(msg): print(f"         {msg}")


# ---------------------------------------------------------------------------
# Load environment samples
# ---------------------------------------------------------------------------

def load_env_and_samples(env_name: str, model_name: str, n: int):
    from diagnostic import make_config
    from environments.reasoning_gym_env import ReasoningGymEnvironment

    config = make_config(env_name, model_name)
    config["environment"]["samples"] = n

    env = ReasoningGymEnvironment(config)
    dataset = env.get_dataset(config)
    samples = [dataset[i] for i in range(min(n, len(dataset)))]
    _ok(f"Loaded {len(samples)} samples from '{env_name}'")
    return env, samples


# ---------------------------------------------------------------------------
# Evaluate one system prompt across all samples
# ---------------------------------------------------------------------------

def evaluate_prompt(system_prompt: str, env, samples: list, model, tokenizer) -> dict:
    """Run inference on every sample; return aggregate + per-sample results."""
    from diagnostic import build_prompt_string, run_inference, eval_rewards

    per_sample = []
    for sample in samples:
        prompt_str  = build_prompt_string(sample, system_prompt, tokenizer)
        completion  = run_inference(model, tokenizer, prompt_str, max_new_tokens=512)
        rewards     = eval_rewards(env, completion, sample)
        per_sample.append({
            "completion": completion,
            "rewards":    rewards,
            "has_think":  "<think>"  in completion and "</think>"  in completion,
            "has_answer": "<answer>" in completion and "</answer>" in completion,
        })

    def _avg(key):
        vals = [r["rewards"].get(key, 0.0) for r in per_sample
                if isinstance(r["rewards"].get(key), float)]
        return sum(vals) / len(vals) if vals else 0.0

    return {
        "per_sample":   per_sample,
        "avg_format":   _avg("format_reward"),
        "avg_correct":  _avg("correctness_reward"),
    }


# ---------------------------------------------------------------------------
# Ask Claude to propose a refined system prompt
# ---------------------------------------------------------------------------

def propose_via_claude(env_name: str, history: list) -> str:
    import anthropic

    history_lines = []
    for i, entry in enumerate(history):
        history_lines.append(f"\n--- Round {i + 1} ---")
        history_lines.append(f"Prompt tried:\n  {entry['system_prompt']!r}")
        history_lines.append(f"Avg format reward    : {entry['avg_format']:.2f}")
        history_lines.append(f"Avg correctness      : {entry['avg_correct']:.2f}")
        history_lines.append("Sample outputs (first 2):")
        for j, s in enumerate(entry["per_sample"][:2]):
            history_lines.append(
                f"  [{j+1}] has_think={s['has_think']}  has_answer={s['has_answer']}"
            )
            snippet = s["completion"][:300].replace("\n", " ")
            history_lines.append(f"       output: {snippet!r}")

    history_text = "\n".join(history_lines)

    client = anthropic.Anthropic()
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=512,
        messages=[{
            "role": "user",
            "content": (
                f"You are refining a system prompt for a small language model "
                f"(Qwen2.5-0.5B-Instruct) so that it always wraps its reasoning "
                f"in <think>...</think> tags and its final answer in <answer>...</answer> tags.\n\n"
                f"Task: {env_name}\n\n"
                f"Format requirements the model MUST satisfy:\n"
                f"  • Every response must contain <think>...</think> (reasoning section)\n"
                f"  • Every response must contain <answer>...</answer> (final answer)\n"
                f"  • Tags must appear in that order, with no text before <think>.\n\n"
                f"History of attempts:\n{history_text}\n\n"
                f"A format_reward of 1.0 means the model used both tag pairs correctly.\n"
                f"The goal is 1.0. Based on what failed, propose a better system prompt.\n\n"
                f"Output ONLY the system prompt text — no explanation, no quotes, no markdown."
            ),
        }],
    )
    return response.content[0].text.strip()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("env_name",    help="reasoning-gym task name, e.g. countdown, maze")
    parser.add_argument("model_name",  nargs="?", default=DEFAULT_MODEL,
                        help="HuggingFace ID or local checkpoint path")
    parser.add_argument("--rounds",    type=int, default=MAX_ROUNDS,
                        help=f"Max optimisation rounds (default {MAX_ROUNDS})")
    parser.add_argument("--samples",   type=int, default=N_SAMPLES,
                        help=f"Samples per round (default {N_SAMPLES})")
    parser.add_argument("--seed-prompt", default=None, dest="seed_prompt",
                        help="Starting system prompt (default: built-in)")
    args = parser.parse_args()

    env_name    = args.env_name
    model_name  = args.model_name
    max_rounds  = args.rounds
    n_samples   = args.samples
    seed_prompt = args.seed_prompt or SEED_PROMPT

    print("=" * 64)
    print("  get-sys-prompt  ·  system prompt optimizer")
    print("=" * 64)
    print(f"  Env     : {env_name}")
    print(f"  Model   : {model_name}")
    print(f"  Rounds  : {max_rounds}  |  Samples/round: {n_samples}")

    # Load environment + samples (once)
    _section("Loading environment and samples")
    env, samples = load_env_and_samples(env_name, model_name, n_samples)

    # Load model + tokenizer (once)
    from diagnostic import load_model_and_tokenizer
    _section(f"Loading model: {model_name}")
    model, tokenizer = load_model_and_tokenizer(model_name)

    # Check Anthropic API key early so we fail fast
    if not os.environ.get("ANTHROPIC_API_KEY"):
        _warn("ANTHROPIC_API_KEY not set — Claude API calls will fail.")
        _warn("Set the key and retry, or pass --seed-prompt to test a single prompt.")
        sys.exit(1)

    current_prompt = seed_prompt
    history        = []
    best           = {"system_prompt": current_prompt, "avg_format": -1.0, "avg_correct": 0.0}

    for round_num in range(1, max_rounds + 1):
        _section(f"Round {round_num} / {max_rounds}")
        print(f"  Prompt: {current_prompt!r}\n")

        result = evaluate_prompt(current_prompt, env, samples, model, tokenizer)

        avg_fmt = result["avg_format"]
        avg_cor = result["avg_correct"]
        print(f"  Avg format reward    : {avg_fmt:.2f}")
        print(f"  Avg correctness      : {avg_cor:.2f}")
        for i, s in enumerate(result["per_sample"]):
            tag_str = (
                f"think={'✓' if s['has_think'] else '✗'}  "
                f"answer={'✓' if s['has_answer'] else '✗'}"
            )
            snippet = s["completion"][:120].replace("\n", " ")
            print(f"    [{i+1}] {tag_str}  |  {snippet!r}")

        history.append({
            "system_prompt": current_prompt,
            "avg_format":    avg_fmt,
            "avg_correct":   avg_cor,
            "per_sample":    result["per_sample"],
        })

        if avg_fmt > best["avg_format"] or (
            avg_fmt == best["avg_format"] and avg_cor > best["avg_correct"]
        ):
            best = {
                "system_prompt": current_prompt,
                "avg_format":    avg_fmt,
                "avg_correct":   avg_cor,
            }

        if avg_fmt >= 1.0:
            _ok("Format reward = 1.0 — stopping early.")
            break

        if round_num < max_rounds:
            _section("Proposing refined prompt via Claude API")
            try:
                current_prompt = propose_via_claude(env_name, history)
                print(f"  New prompt: {current_prompt!r}")
            except Exception as exc:
                _warn(f"Claude API call failed: {exc}")
                break

    # -----------------------------------------------------------------------
    # Final report
    # -----------------------------------------------------------------------
    _section("RESULT — Best system prompt found")
    print(f"\n  Format reward   : {best['avg_format']:.2f}")
    print(f"  Correct reward  : {best['avg_correct']:.2f}")
    print(f"\n  Prompt text:\n")
    print(f"    {best['system_prompt']}")
    print()

    if best["avg_format"] >= 1.0:
        _ok("SUCCESS: format reward 1.0 — model reliably uses <think> and <answer> tags.")
    elif best["avg_format"] > 0.0:
        _warn(f"Partial: best format reward = {best['avg_format']:.2f}. "
              "Consider more rounds or a stronger base model.")
    else:
        _warn("Format reward stayed at 0.0. "
              "The base model may need SFT before system-prompt tuning can help.")


if __name__ == "__main__":
    main()
