import sys
import argparse
import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer
from environments.gsm8k import GSM8KEnvironment
from environments.maze_env import MazeEnvironment
from environments.syllogism_env import SyllogismEnvironment
from environments.battleship_env import BattleshipEnvironment
from environments.arithmetic_env import ArithmeticEnvironment
from environments.reasoning_gym_env import ReasoningGymEnvironment

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def get_environment(config):
    env_config = config.get('environment', {})
    if not env_config:
        # Fallback for legacy config or direct dataset inference
        if config.get('data', {}).get('dataset_name') == 'openai/gsm8k':
            return GSM8KEnvironment(config)
        else:
            raise ValueError("Could not determine environment from config.")
            
    name = env_config.get('name')
    if name == 'gsm8k':
        return GSM8KEnvironment(config)
    elif name == 'maze':
        return MazeEnvironment(config)
    elif name == 'syllogism':
        return SyllogismEnvironment(config)
    elif name == 'battleship':
        return BattleshipEnvironment(config)
    elif name == 'arithmetic_multi':
        return ArithmeticEnvironment(config)
    elif name == 'reasoning_gym':
        return ReasoningGymEnvironment(config)
    else:
        raise ValueError(f"Unknown environment: {name}")


def parse_args():
    parser = argparse.ArgumentParser(description="GRPO Training")
    parser.add_argument("config", nargs="?", default="config.yaml", help="Path to YAML config")
    parser.add_argument("--max_steps", type=int, default=None, help="Override max_steps (for smoke testing)")
    return parser.parse_args()


def main():
    args = parse_args()
    config_path = args.config

    print(f"Loading config from {config_path}")
    config = load_config(config_path)

    # Initialize Environment
    print("Initializing Environment...")
    env = get_environment(config)
    
    # Load and Process Dataset
    dataset = env.get_dataset(config)
    
    # Inject system_prompt into dataset rows if specified in config
    system_prompt = config.get('system_prompt')
    if system_prompt:
        system_prompt = system_prompt.strip()
        def add_system_prompt(example):
            prompt = example['prompt']
            # Only add if not already present
            if not any(m.get('role') == 'system' for m in prompt):
                prompt = [{'role': 'system', 'content': system_prompt}] + prompt
                example['prompt'] = prompt
            return example
        dataset = dataset.map(add_system_prompt)
    
    # Reward Functions
    reward_funcs = env.get_reward_functions()
    
    # Load Model
    model_path = config['model']['name_or_path']
    print(f"Loading model: {model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=config['model'].get('torch_dtype', 'auto'),
        device_map=config['model'].get('device_map', 'auto')
    )
    
    # Load tokenizer (needed for stop_strings and reward decoding)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Training Arguments
    print("Configuring training arguments...")
    training_conf = config['training']
    gen_conf = config.get('generation', {})
    
    # Allow CLI override of max_steps (for smoke testing)
    max_steps = args.max_steps if args.max_steps is not None else training_conf.get('max_steps', -1)
    
    training_args = GRPOConfig(
        output_dir=training_conf['output_dir'],
        learning_rate=float(training_conf['learning_rate']),
        remove_unused_columns=False,
        gradient_accumulation_steps=training_conf.get('gradient_accumulation_steps', 1),
        num_train_epochs=training_conf.get('num_train_epochs', 1),
        bf16=training_conf.get('bf16', False),
        max_completion_length=gen_conf.get('max_completion_length', 1024),
        num_generations=gen_conf.get('num_generations', 4),
        max_prompt_length=gen_conf.get('max_prompt_length', 128),
        temperature=gen_conf.get('temperature', 0.7),
        report_to=training_conf.get('report_to', []),
        logging_steps=training_conf.get('logging_steps', 10),
        push_to_hub=training_conf.get('push_to_hub', False),
        save_strategy=training_conf.get('save_strategy', 'steps'),
        save_steps=training_conf.get('save_steps', 10),
        max_steps=max_steps,
        warmup_ratio=training_conf.get('warmup_ratio', 0.0),
    )

    # Trainer
    print("Initializing GRPOTrainer...")
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    # Train
    print("Starting training...")
    trainer.train()
    
    # Save Model
    trainer.save_model(config['training']['output_dir'])
    print(f"Model saved to {config['training']['output_dir']}")

if __name__ == "__main__":
    main()
