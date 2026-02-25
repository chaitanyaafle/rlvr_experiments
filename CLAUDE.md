# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Research Plan

See [./research_plan/research_plan.md](research_plan.md) for the full research agenda: *SkillThink — Compositional Latent Reasoning in Small LMs via Hierarchical RLVR*.

## Overview

This is an RLVR (Reinforcement Learning with Verifiable Rewards) research framework for training small language models (primarily Qwen2-0.5B) on reasoning tasks. Models output reasoning in `<think>...</think><answer>...</answer>` format. Training supports two paradigms: SFT (supervised fine-tuning on generated traces) and GRPO (Group Relative Policy Optimization with task reward signals).

## Commands

### Data Generation
```bash
python data_gen/generate_sft_data.py configs/data_gen_maze.yaml
```
Produces a JSONL file with prompt + reasoning-augmented answer pairs using an Instruct model as teacher.

### Training
```bash
# Supervised Fine-Tuning
python train_sft.py configs/sft_maze.yaml

# GRPO (RL training)
python train_grpo.py configs/config_battleship.yaml

# GRPO with latent thoughts (Coconut-style)
python train_grpo_latent.py configs/train_latent_gsm8k.yaml
```

### Evaluation
```bash
# Unified multi-environment evaluator (preferred)
python evaluate_unified.py configs/eval_unified.yaml
python evaluate_unified.py configs/eval_unified.yaml --checkpoint path/to/ckpt
python evaluate_unified.py configs/eval_unified.yaml --output my_results/

# Legacy single-environment evaluators
python evaluate_sft.py configs/eval.yaml [optional_model_path] [num_samples]
python evaluate.py configs/eval.yaml
```

### Verification / Smoke Tests
```bash
python verify_battleship.py   # Validate Battleship environment mechanics
python verify_latent.py       # Validate latent model forward pass and logits shape
```

## Architecture

### Environment Interface (`environments/base.py`)
All task environments implement `BaseEnvironment` with three abstract methods:
- `get_dataset(config)` — returns a HuggingFace `Dataset`
- `get_reward_functions()` — returns a list of callables `(completions, **kwargs) -> List[float]`
- `get_system_prompt()` — returns the system prompt string

**Unified environment:** `reasoning_gym_env.py` (`ReasoningGymEnvironment`) wraps any of the 20 tasks registered in `reasoning_gym` (maze, countdown, knights_knaves, syllogism, …). The task is selected via `environment.name` in the config; all extra keys are forwarded to `reasoning_gym.create_dataset()`. Use `environments/__init__.py:load_environment()` to instantiate the right class.

Legacy environments: `maze_env.py`, `battleship_env.py`, `gsm8k.py`, `syllogism_env.py`.

External dependencies per environment:
- `reasoning_gym` tasks: `reasoning_gym` library
- GSM8K: HF `openai/gsm8k` dataset + `math_verify` for answer checking
- Battleship: self-contained (`battleship_logic.py`)

### Training Scripts
- `train_sft.py` — uses HF `SFTTrainer`; loads JSONL data, trains on full `<think>...<answer>` responses
- `train_grpo.py` — uses TRL `GRPOTrainer`; passes environment's reward functions directly as `reward_funcs`
- `train_grpo_latent.py` — same GRPO loop but with the custom `LatentQwen2ForCausalLM` model

All training scripts load hyperparameters from YAML configs.

### Configuration
All experiments are driven by YAML configs in `configs/`. Key fields shared across configs:
- `model.name_or_path` — base model or checkpoint to load
- `training.output_dir` — where checkpoints are saved
- `environment.name` — which task/env to use (e.g. `maze`, `countdown`, `gsm8k`)
- Training hyperparameters (lr, batch size, epochs, etc.)
- Generation parameters (max_new_tokens, temperature, etc.)

**Evaluation config** (`configs/eval_unified.yaml`) adds:
- `environments` — list of env entries to evaluate sequentially (each is an `environment` block)
- `evaluation.num_samples` — samples per environment
- `evaluation.save_samples` — whether to dump full completions to `samples.json`
- `--checkpoint` / `--output` CLI flags override config values at runtime

### Reward Functions (GRPO)
Each environment's `get_reward_functions()` returns a list of reward functions composed additively. Typical pattern: one function rewards correct format (`<think>` + `<answer>` tags present), another rewards correctness of the extracted answer content.
