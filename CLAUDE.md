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

Environments: `maze_env.py`, `battleship_env.py`, `gsm8k.py`, `syllogism_env.py`.

External dependencies per environment:
- Maze / Syllogism: `reasoning_gym` library
- GSM8K: HF `openai/gsm8k` dataset + `math_verify` for answer checking
- Battleship: self-contained (`battleship_logic.py`)

### Training Scripts
- `train_sft.py` — uses HF `SFTTrainer`; loads JSONL data, trains on full `<think>...<answer>` responses
- `train_grpo.py` — uses TRL `GRPOTrainer`; passes environment's reward functions directly as `reward_funcs`
- `train_grpo_latent.py` — same GRPO loop but with the custom `LatentQwen2ForCausalLM` model

All training scripts load hyperparameters from YAML configs.

### Latent Thought Model (`latent_qwen.py`)
`LatentQwen2ForCausalLM` extends `Qwen2ForCausalLM` to implement Coconut-style latent reasoning:
- During `forward()`, it detects `<think>` tokens and expands them to `num_latent_thoughts` placeholder tokens
- Runs the model chunk-by-chunk, replacing each latent token's embedding with the previous chunk's final hidden state
- This allows iterative "thinking" in embedding space before the visible answer is generated
- During `generate()`, a `LatentControlLogitsProcessor` forces `</think>` immediately after the latent expansion phase
- Key attributes: `num_latent_thoughts`, `think_token_id`, `close_think_token_id`

### Configuration
All experiments are driven by YAML configs in `configs/`. Key fields shared across configs:
- `model_name_or_path` — base model to load
- `output_dir` — where checkpoints are saved
- `environment` — which env class to use
- Training hyperparameters (lr, batch size, epochs, etc.)
- Generation parameters (max_new_tokens, temperature, etc.)

### Reward Functions (GRPO)
Each environment's `get_reward_functions()` returns a list of reward functions composed additively. Typical pattern: one function rewards correct format (`<think>` + `<answer>` tags present), another rewards correctness of the extracted answer content.
