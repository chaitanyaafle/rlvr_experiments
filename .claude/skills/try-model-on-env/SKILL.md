---
name: try-model-on-env
description: Smoke-test a model on one sample from a reasoning-gym environment. Shows the exact prompt the model sees, runs inference, then evaluates the environment's own format and correctness reward functions. Use when you want to quickly verify a model or checkpoint works correctly on a specific task.
argument-hint: <env_name> [model_name]
disable-model-invocation: true
allowed-tools: Bash(python *)
---

Smoke-test a model against one sample from a reasoning-gym environment.

## Arguments

- `$ARGUMENTS[0]` — **env_name** (required): reasoning-gym task name, e.g. `countdown`, `maze`, `gsm8k`.
  If missing, ask the user before proceeding.
- `$ARGUMENTS[1]` — **model_name** (optional): HuggingFace ID or local checkpoint path.
  Default: `Qwen/Qwen2.5-0.5B-Instruct`

## Steps

1. Run the diagnostic script from the project root with a 10-minute timeout:
   ```bash
   python .claude/skills/try-model-on-env/scripts/diagnostic.py $ARGUMENTS
   ```

2. Show the full output, calling out each phase clearly:
   - **System prompt check** — does it include `<think>` and `<answer>` tag instructions?
   - **Prompt** — exact text the model receives after `apply_chat_template`
   - **Ground truth** — what the correct answer is
   - **Model response** — raw generated text
   - **Reward diagnostics** — format score (0.0–1.0) and correctness score (0.0–1.0)
   - **Summary table**

3. Give a brief plain-English interpretation:
   - Is the format reward functioning? (1.0 = both `<think>...</think>` and `<answer>...</answer>` present)
   - Did the model answer correctly?
   - Any issues worth flagging (truncated output, wrong answer format, missing tags)?

## Supporting files

- [scripts/diagnostic.py](scripts/diagnostic.py) — the diagnostic runner; imports directly from the project's `environments` package to reuse reward functions, answer extraction, and normalization logic.
