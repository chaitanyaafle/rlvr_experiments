"""
Generate Reasoning Gym Data for Arithmetic Transfer Learning

This script generates SFT and RLHF data from reasoning-gym tasks:
- Training (Siblings): basic_arithmetic, chain_sum, leg_counting, fraction_simplification
- Test (Held-out): countdown

Output formats:
- SFT: {"messages": [{"role": "user", ...}, {"role": "assistant", ...}]}
- RLHF: {"prompt": [...], "ground_truth": "..."}
"""

import pandas as pd
import os
from pathlib import Path
from typing import Optional

try:
    import reasoning_gym
    REASONING_GYM_AVAILABLE = True
except ImportError:
    REASONING_GYM_AVAILABLE = False
    print("Warning: reasoning-gym not installed. Install with: pip install reasoning-gym")

OUTPUT_DIR = Path(__file__).parent / "data" / "siblings"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Task configurations with difficulty settings
TASK_CONFIGS = {
    "basic_arithmetic": {
        "min_terms": 2,
        "max_terms": 4,
        "min_digits": 1,
        "max_digits": 3,
        "operators": ('+', '-', '*'),  # Exclude division for cleaner examples
        "allow_parentheses": False,
        "allow_negation": True,
    },
    "chain_sum": {
        "min_terms": 2,
        "max_terms": 5,
        "min_digits": 1,
        "max_digits": 3,
    },
    "leg_counting": {
        "min_animals": 2,
        "max_animals": 5,
        "min_instances": 1,
        "max_instances": 10,
    },
    "fraction_simplification": {
        "min_value": 1,
        "max_value": 100,
        "min_factor": 2,
        "max_factor": 20,
        "styles": ('plain',),  # Keep it simple, no latex
    },
    "countdown": {
        "min_numbers": 4,
        "max_numbers": 5,
        "min_value": 1,
        "max_value": 50,
        "min_target": 50,
        "max_target": 200,
    },
}


def format_for_sft(item: dict, include_reasoning: bool = True) -> dict:
    """
    Format reasoning-gym item for SFT training.
    
    Uses <think>...</think> tags for chain-of-thought reasoning.
    """
    question = item['question']
    answer = str(item['answer'])
    
    # Generate simple reasoning based on task type
    metadata = item.get('metadata', {})
    source = metadata.get('source_dataset', 'unknown')
    
    if include_reasoning:
        reasoning = generate_reasoning(item, source)
        response = f"<think>\n{reasoning}\n</think>\n\n#### {answer}"
    else:
        response = f"#### {answer}"
    
    return {
        "messages": [
            {"role": "user", "content": question},
            {"role": "assistant", "content": response}
        ]
    }


def generate_reasoning(item: dict, source: str) -> str:
    """Generate chain-of-thought reasoning for an item."""
    metadata = item.get('metadata', {})
    answer = str(item['answer'])
    
    if source == "basic_arithmetic":
        expr = metadata.get('expression', '')
        return f"Let me calculate step by step.\nExpression: {expr}\nResult: {answer}"
    
    elif source == "chain_sum":
        expr = metadata.get('expression', '')
        return f"Adding/subtracting from left to right.\n{expr} = {answer}"
    
    elif source == "leg_counting":
        animals = metadata.get('animals', {})
        steps = []
        for animal, count in animals.items():
            # Common leg counts
            leg_map = {'human': 2, 'dog': 4, 'cat': 4, 'bird': 2, 'spider': 8, 
                       'insect': 6, 'snake': 0, 'fish': 0, 'elephant': 4, 
                       'cow': 4, 'horse': 4, 'chicken': 2, 'duck': 2}
            legs = leg_map.get(animal.lower(), 4)  # Default to 4
            steps.append(f"{count} {animal}(s) × {legs} legs = {count * legs}")
        return "Counting legs:\n" + "\n".join(steps) + f"\nTotal: {answer}"
    
    elif source == "fraction_simplification":
        num = metadata.get('numerator', '')
        den = metadata.get('denominator', '')
        factor = metadata.get('reduction_factor', '')
        return f"Simplifying {num}/{den}.\nGCD is {factor}.\nDividing both by {factor}: {answer}"
    
    elif source == "countdown":
        expr = metadata.get('expression', answer)
        target = metadata.get('target', '')
        return f"Need to reach {target} using the given numbers.\nTrying combinations...\nFound: {expr} = {target}"
    
    else:
        return f"Working through the problem.\nAnswer: {answer}"


def format_for_rlhf(item: dict) -> dict:
    """
    Format reasoning-gym item for RLHF/GRPO training.
    
    Returns prompt and ground truth for reward computation.
    """
    return {
        "prompt": [{"role": "user", "content": item['question']}],
        "ground_truth": str(item['answer'])
    }


def generate_dataset(
    task_name: str,
    num_samples: int = 2000,
    mode: str = "sft",
    prefix: str = "train",
    seed: int = 42,
    config_overrides: Optional[dict] = None
) -> None:
    """
    Generate dataset for a specific task.
    
    Args:
        task_name: Name of the reasoning-gym task
        num_samples: Number of samples to generate
        mode: "sft" or "rlhf"
        prefix: Filename prefix (e.g., "train", "test")
        seed: Random seed
        config_overrides: Override default task config
    """
    if not REASONING_GYM_AVAILABLE:
        print(f"Skipping {task_name}: reasoning-gym not available")
        return
    
    print(f"Generating {num_samples} {mode} samples for {task_name}...")
    
    # Build config
    config = {"seed": seed, "size": num_samples}
    if task_name in TASK_CONFIGS:
        config.update(TASK_CONFIGS[task_name])
    if config_overrides:
        config.update(config_overrides)
    
    # Create dataset
    try:
        dataset = reasoning_gym.create_dataset(task_name, **config)
    except Exception as e:
        print(f"Error creating {task_name}: {e}")
        return
    
    # Format data
    if mode == "sft":
        data = [format_for_sft(item) for item in dataset]
    else:
        data = [format_for_rlhf(item) for item in dataset]
    
    # Save
    filename = f"{prefix}_{task_name}_{mode}.parquet"
    filepath = OUTPUT_DIR / filename
    pd.DataFrame(data).to_parquet(filepath)
    print(f"Saved {len(data)} rows to {filepath}")


def main():
    print("=" * 60)
    print("Generating Reasoning Gym Data for Arithmetic Transfer")
    print("=" * 60)
    
    if not REASONING_GYM_AVAILABLE:
        print("\nERROR: reasoning-gym not installed!")
        print("Install with: pip install reasoning-gym")
        return
    
    # Training data (Sibling Tasks)
    print("\n--- Training Data (Siblings) ---")
    training_tasks = ["basic_arithmetic", "chain_sum", "leg_counting", "fraction_simplification"]
    
    for task in training_tasks:
        generate_dataset(task, num_samples=2000, mode="sft", prefix="train_sibling", seed=42)
        generate_dataset(task, num_samples=2000, mode="rlhf", prefix="train_sibling", seed=43)
    
    # Test data (Held-out Target)
    print("\n--- Test Data (Held-out: countdown) ---")
    generate_dataset("countdown", num_samples=500, mode="rlhf", prefix="test_target", seed=44)
    
    # Also generate a small SFT set for countdown (for comparison experiments)
    generate_dataset("countdown", num_samples=200, mode="sft", prefix="comparison", seed=45)
    
    print("\n" + "=" * 60)
    print(f"All data saved to {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
