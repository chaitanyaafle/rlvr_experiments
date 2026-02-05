"""
Evaluate Transfer Learning on Held-Out Task (Countdown)

This script evaluates how well a model trained on sibling arithmetic tasks
transfers to the held-out countdown task.

Usage:
    python evaluate_transfer.py configs/eval_countdown.yaml
    python evaluate_transfer.py configs/eval_countdown.yaml --model outputs/stage3_siblings_grpo
"""

import sys
import os
import yaml
import json
import torch
import re
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
from tqdm import tqdm

try:
    import reasoning_gym
    REASONING_GYM_AVAILABLE = True
except ImportError:
    REASONING_GYM_AVAILABLE = False

# Try Unsloth first
try:
    from unsloth import FastLanguageModel
    UNSLOTH_AVAILABLE = True
except ImportError:
    UNSLOTH_AVAILABLE = False
    from transformers import AutoModelForCausalLM, AutoTokenizer


def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML config file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_model(model_path: str, max_seq_length: int = 2048):
    """Load model for inference."""
    print(f"Loading model from {model_path}...")
    
    if UNSLOTH_AVAILABLE:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_path,
            max_seq_length=max_seq_length,
            load_in_4bit=True,
        )
        FastLanguageModel.for_inference(model)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    return model, tokenizer


def generate_response(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 512,
    temperature: float = 0.1,
) -> str:
    """Generate model response for a prompt."""
    messages = [{"role": "user", "content": prompt}]
    
    input_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    
    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    return response


def extract_answer(text: str) -> Optional[str]:
    """Extract answer from model response."""
    # Try #### marker
    if '####' in text:
        parts = text.split('####')
        if len(parts) > 1:
            answer = parts[-1].strip().split()[0] if parts[-1].strip().split() else parts[-1].strip()
            return answer
    
    # For countdown, look for arithmetic expression
    # The answer format is an expression like "15 - 4 + 95"
    lines = text.strip().split('\n')
    for line in reversed(lines):
        line = line.strip()
        # Check if line looks like an arithmetic expression
        if re.match(r'^[\d\s\+\-\*\/\(\)]+$', line) and any(c.isdigit() for c in line):
            return line
    
    return None


def evaluate_countdown(
    model,
    tokenizer,
    num_samples: int = 100,
    seed: int = 42,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Evaluate model on countdown task.
    
    Returns metrics:
    - accuracy: Fraction of correct answers
    - valid_format: Fraction of responses with extractable answers
    - avg_reasoning_length: Average length of reasoning traces
    """
    if not REASONING_GYM_AVAILABLE:
        raise ImportError("reasoning-gym required for evaluation")
    
    # Generate test set
    print(f"\nGenerating {num_samples} countdown problems...")
    dataset = reasoning_gym.create_dataset(
        "countdown",
        size=num_samples,
        seed=seed,
        min_numbers=4,
        max_numbers=5,
        min_target=50,
        max_target=200,
    )
    
    results = []
    correct = 0
    valid_format = 0
    reasoning_lengths = []
    
    for item in tqdm(dataset, desc="Evaluating"):
        question = item['question']
        target = item['metadata']['target']
        numbers = item['metadata']['numbers']
        gold_answer = item['answer']
        
        # Generate response
        response = generate_response(model, tokenizer, question)
        
        # Extract answer
        extracted = extract_answer(response)
        
        # Check format
        if extracted:
            valid_format += 1
        
        # Check correctness
        is_correct = False
        if extracted:
            try:
                # Evaluate the expression
                result = eval(extracted)
                if result == target:
                    is_correct = True
                    correct += 1
            except:
                pass
        
        # Track reasoning length
        if '<think>' in response and '</think>' in response:
            think_match = re.search(r'<think>(.*?)</think>', response, re.DOTALL)
            if think_match:
                reasoning_lengths.append(len(think_match.group(1)))
        
        result = {
            "question": question,
            "target": target,
            "numbers": numbers,
            "gold_answer": gold_answer,
            "response": response,
            "extracted": extracted,
            "correct": is_correct,
        }
        results.append(result)
        
        if verbose and len(results) <= 3:
            print(f"\n--- Example {len(results)} ---")
            print(f"Question: {question[:100]}...")
            print(f"Target: {target}")
            print(f"Response: {response[:200]}...")
            print(f"Extracted: {extracted}")
            print(f"Correct: {is_correct}")
    
    metrics = {
        "accuracy": correct / num_samples,
        "valid_format_rate": valid_format / num_samples,
        "avg_reasoning_length": sum(reasoning_lengths) / len(reasoning_lengths) if reasoning_lengths else 0,
        "num_samples": num_samples,
        "num_correct": correct,
    }
    
    return metrics, results


def main():
    if len(sys.argv) < 2:
        print("Usage: python evaluate_transfer.py <config.yaml> [--model <model_path>]")
        sys.exit(1)
    
    config_path = sys.argv[1]
    config = load_config(config_path)
    
    # Check for model override
    model_path = config['model']['name_or_path']
    if '--model' in sys.argv:
        idx = sys.argv.index('--model')
        model_path = sys.argv[idx + 1]
    
    # Load model
    model, tokenizer = load_model(
        model_path,
        max_seq_length=config['model'].get('max_seq_length', 2048)
    )
    
    # Evaluate
    eval_config = config.get('evaluation', {})
    metrics, results = evaluate_countdown(
        model,
        tokenizer,
        num_samples=eval_config.get('num_samples', 100),
        seed=eval_config.get('seed', 42),
        verbose=eval_config.get('verbose', True),
    )
    
    # Print results
    print("\n" + "=" * 50)
    print("EVALUATION RESULTS: Countdown (Held-Out)")
    print("=" * 50)
    print(f"Model: {model_path}")
    print(f"Samples: {metrics['num_samples']}")
    print(f"Accuracy: {metrics['accuracy']:.2%}")
    print(f"Valid Format Rate: {metrics['valid_format_rate']:.2%}")
    print(f"Avg Reasoning Length: {metrics['avg_reasoning_length']:.1f} chars")
    print("=" * 50)
    
    # Save results
    output_dir = Path(config.get('output_dir', 'outputs/evaluation'))
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save metrics
    metrics_file = output_dir / f"metrics_{timestamp}.json"
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved to {metrics_file}")
    
    # Save detailed results
    results_file = output_dir / f"results_{timestamp}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Detailed results saved to {results_file}")


if __name__ == "__main__":
    main()
