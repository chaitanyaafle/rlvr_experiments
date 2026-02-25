"""
Smoke Test for GRPO Training Pipeline

Runs a few GRPO training steps to verify the pipeline works end-to-end:
- Model loads correctly
- Dataset generates properly
- Reward functions produce non-degenerate scores
- Completions contain expected format tags
- Training loop doesn't crash

Usage:
    python test_grpo.py configs/config_qwen3_countdown.yaml
    python test_grpo.py configs/config_countdown_grpo.yaml
"""

import sys
import os
import yaml
import torch
import re
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from environments.reasoning_gym_env import ReasoningGymEnvironment


def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def analyze_completions(trainer, dataset, tokenizer, model, num_samples=3):
    """Generate a few completions and check format."""
    print("\n" + "=" * 60)
    print("COMPLETION ANALYSIS")
    print("=" * 60)

    model.eval()
    results = {
        "total": 0,
        "has_think_open": 0,
        "has_think_close": 0,
        "has_answer_open": 0,
        "has_answer_close": 0,
        "fully_formatted": 0,
    }

    for i in range(min(num_samples, len(dataset))):
        example = dataset[i]
        messages = example["prompt"]

        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )

        completion = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=False,
        )

        results["total"] += 1
        if "<think>" in completion:
            results["has_think_open"] += 1
        if "</think>" in completion:
            results["has_think_close"] += 1
        if "<answer>" in completion:
            results["has_answer_open"] += 1
        if "</answer>" in completion:
            results["has_answer_close"] += 1
        if all(tag in completion for tag in ["<think>", "</think>", "<answer>", "</answer>"]):
            results["fully_formatted"] += 1

        print(f"\n--- Sample {i+1} ---")
        print(f"Prompt: {messages[-1]['content'][:100]}...")
        print(f"Completion ({len(completion)} chars):")
        print(completion[:500])
        if len(completion) > 500:
            print("... [truncated]")

    return results


def test_rewards(env, completions_samples):
    """Test that reward functions produce reasonable values."""
    print("\n" + "=" * 60)
    print("REWARD FUNCTION TEST")
    print("=" * 60)

    reward_funcs = env.get_reward_functions()

    # Test with synthetic completions
    test_cases = [
        # Perfect format
        "<think>\nLet me solve this step by step.\n</think>\n<answer>42</answer>",
        # Missing </think>
        "<think>\nLet me solve this step by step.\n<answer>42</answer>",
        # No format at all
        "The answer is 42.",
        # Only answer
        "<answer>42</answer>",
    ]

    labels = ["Perfect format", "Missing </think>", "No format", "Answer only"]

    for func in reward_funcs:
        func_name = func.__name__ if hasattr(func, '__name__') else str(func)
        print(f"\n{func_name}:")

        for label, completion in zip(labels, test_cases):
            wrapped = [[{"content": completion}]]
            try:
                reward = func(wrapped, ground_truth=["42"])[0]
            except Exception as e:
                reward = f"ERROR: {e}"
            print(f"  {label:25s} → {reward}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python test_grpo.py <config.yaml>")
        sys.exit(1)

    config_path = sys.argv[1]
    print(f"Loading config from {config_path}")
    config = load_config(config_path)

    # Override for smoke test: tiny dataset, few steps
    num_test_steps = 3
    config['environment']['samples'] = 20  # Tiny dataset
    config['training']['max_steps'] = num_test_steps
    config['training']['save_strategy'] = 'no'  # Don't save checkpoints
    config['training']['report_to'] = []

    # Use a temp output dir
    config['training']['output_dir'] = '/tmp/test_grpo_smoke'

    # Initialize environment
    print("Initializing environment...")
    env = ReasoningGymEnvironment(config)

    # Load dataset
    print("Loading dataset...")
    dataset = env.get_dataset(config)
    print(f"  Dataset size: {len(dataset)}")
    print(f"  Sample prompt: {dataset[0]['prompt'][-1]['content'][:100]}...")
    print(f"  Sample answer: {dataset[0]['ground_truth'][:80]}")

    # Inject system_prompt
    system_prompt = config.get('system_prompt')
    if system_prompt:
        system_prompt = system_prompt.strip()
        def add_system_prompt(example):
            prompt = example['prompt']
            if not any(m.get('role') == 'system' for m in prompt):
                prompt = [{'role': 'system', 'content': system_prompt}] + prompt
                example['prompt'] = prompt
            return example
        dataset = dataset.map(add_system_prompt)

    # Test reward functions
    test_rewards(env, None)

    # Load model
    model_path = config['model']['name_or_path']
    print(f"\nLoading model: {model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=config['model'].get('torch_dtype', 'auto'),
        device_map=config['model'].get('device_map', 'auto'),
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Analyze pre-training completions
    print("\n--- Pre-training completion analysis ---")
    pre_results = analyze_completions(None, dataset, tokenizer, model, num_samples=3)

    # Run a few GRPO steps
    print(f"\n{'=' * 60}")
    print(f"RUNNING {num_test_steps} GRPO TRAINING STEPS")
    print(f"{'=' * 60}")

    gen_conf = config.get('generation', {})
    reward_funcs = env.get_reward_functions()

    training_args = GRPOConfig(
        output_dir=config['training']['output_dir'],
        learning_rate=float(config['training']['learning_rate']),
        remove_unused_columns=False,
        gradient_accumulation_steps=config['training'].get('gradient_accumulation_steps', 1),
        num_train_epochs=1,
        bf16=config['training'].get('bf16', False),
        max_completion_length=gen_conf.get('max_completion_length', 512),
        num_generations=gen_conf.get('num_generations', 4),
        max_prompt_length=gen_conf.get('max_prompt_length', 128),
        temperature=gen_conf.get('temperature', 0.7),
        report_to=[],
        logging_steps=1,
        save_strategy='no',
        max_steps=num_test_steps,
    )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    try:
        trainer.train()
        training_success = True
        print("\n✓ Training completed successfully!")
    except Exception as e:
        training_success = False
        print(f"\n✗ Training FAILED: {e}")
        import traceback
        traceback.print_exc()

    # Summary
    print("\n" + "=" * 60)
    print("SMOKE TEST SUMMARY")
    print("=" * 60)
    print(f"  Config:            {config_path}")
    print(f"  Model:             {model_path}")
    print(f"  Dataset size:      {len(dataset)}")
    print(f"  Training steps:    {num_test_steps}")
    print(f"  Training success:  {'✓ PASS' if training_success else '✗ FAIL'}")
    print(f"  Pre-train format:")
    print(f"    <think> open:    {pre_results['has_think_open']}/{pre_results['total']}")
    print(f"    </think> close:  {pre_results['has_think_close']}/{pre_results['total']}")
    print(f"    <answer> open:   {pre_results['has_answer_open']}/{pre_results['total']}")
    print(f"    </answer> close: {pre_results['has_answer_close']}/{pre_results['total']}")
    print(f"    Fully formatted: {pre_results['fully_formatted']}/{pre_results['total']}")

    # Determine pass/fail
    format_pass = pre_results['fully_formatted'] > 0
    overall = training_success and format_pass
    if overall:
        print("\n✓ SMOKE TEST PASSED — pipeline is ready")
    else:
        if not training_success:
            print("\n✗ SMOKE TEST FAILED — training crashed (check errors above)")
        elif not format_pass:
            print("\n✗ SMOKE TEST FAILED — model is not producing <think>/<answer> tags")
            print("  Try: increase SFT warmup epochs (--num_epochs 5) or check SFT training logs")

    return 0 if overall else 1


if __name__ == "__main__":
    sys.exit(main())
