---
name: get-sys-prompt
description: Iteratively optimise a system prompt so that a model generates the correct <think>...</think> and <answer>...</answer> format on a given reasoning-gym task. Loads the model once, evaluates each candidate prompt on N real samples (no API key needed), and reports the best one found. Use when try-model-on-env shows format_reward = 0.0.
argument-hint: <env_name> [model_name]
disable-model-invocation: true
allowed-tools: Bash(python *)
---

Evaluate a pre-planned set of candidate system prompts so a model reliably outputs
`<think>...</think><answer>...</answer>` format on a reasoning task. No API key required.

## Arguments

- `$ARGUMENTS[0]` — **env_name** (required): reasoning-gym task name, e.g. `countdown`, `maze`, `gsm8k`.
  If missing, ask the user before proceeding.
- `$ARGUMENTS[1]` — **model_name** (optional): HuggingFace ID or local checkpoint path.
  Default: `Qwen/Qwen2.5-0.5B-Instruct`

## Steps

1. Run the local optimizer script from the project root with a 15-minute timeout:
   ```bash
   python .claude/skills/get-sys-prompt/scripts/optimize_prompt_local.py $ARGUMENTS
   ```

2. Show the full output, calling out each phase clearly:
   - **Round N** — system prompt tried, per-sample tag presence (`think=✓/✗ answer=✓/✗`), avg format + correctness reward
   - **RESULT block** — the winning prompt and its scores

3. Give a brief plain-English summary:
   - What format reward was achieved? Did it reach 1.0?
   - Which system prompt worked best? Quote it.
   - Which prompt style won (e.g. explicit tag rules, role-playing framing, inline example)?
   - **Recommendation**: if format reward reached 1.0, suggest updating `DEFAULT_SYSTEM_PROMPT`
     in [try-model-on-env/scripts/diagnostic.py](../try-model-on-env/scripts/diagnostic.py).
     If it stayed at 0.0, note that SFT warm-up is likely needed before prompt-tuning helps.

## Optional flags (pass after model_name)

| Flag | Default | Purpose |
|------|---------|---------|
| `--samples N` | 3 | Samples evaluated per round |

Example with flags:
```bash
python .claude/skills/get-sys-prompt/scripts/optimize_prompt_local.py countdown Qwen/Qwen2.5-0.5B-Instruct --samples 5
```

## Supporting files

- [scripts/optimize_prompt_local.py](scripts/optimize_prompt_local.py) — the local optimizer; loads model + env once, iterates through pre-planned candidate prompts, prints round-by-round diagnostics and a final RESULT block. No API key needed.
- [scripts/optimize_prompt.py](scripts/optimize_prompt.py) — alternative that calls Claude API (`claude-sonnet-4-6`) to generate new prompts dynamically. Requires `ANTHROPIC_API_KEY`.
- [../try-model-on-env/scripts/diagnostic.py](../try-model-on-env/scripts/diagnostic.py) — sibling skill whose helpers (`build_prompt_string`, `run_inference`, `eval_rewards`, `load_model_and_tokenizer`) are imported directly. Also accepts `--system-prompt` flag for one-off manual tests.
