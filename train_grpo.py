import sys
import torch
import yaml
from transformers import AutoModelForCausalLM
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


def main():
    if len(sys.argv) < 2:
        config_path = "config.yaml"
    else:
        config_path = sys.argv[1]

    print(f"Loading config from {config_path}")
    config = load_config(config_path)

    # Initialize Environment
    print("Initializing Environment...")
    env = get_environment(config)
    
    # Load and Process Dataset
    dataset = env.get_dataset(config)
    
    # Reward Functions
    reward_funcs = env.get_reward_functions()
    
    # Load Model
    print(f"Loading model: {config['model']['name_or_path']}")
    model = AutoModelForCausalLM.from_pretrained(
        config['model']['name_or_path'],
        torch_dtype=config['model'].get('torch_dtype', 'auto'),
        device_map=config['model'].get('device_map', 'auto')
    )
    
    # Training Arguments
    print("Configuring training arguments...")
    training_conf = config['training']
    
    training_args = GRPOConfig(
        output_dir=training_conf['output_dir'],
        learning_rate=float(training_conf['learning_rate']),
        remove_unused_columns=False,
        gradient_accumulation_steps=training_conf.get('gradient_accumulation_steps', 1),
        num_train_epochs=training_conf.get('num_train_epochs', 1),
        bf16=training_conf.get('bf16', False),
        max_completion_length=config['generation']['max_completion_length'],
        num_generations=config['generation']['num_generations'],
        max_prompt_length=config['generation'].get('max_prompt_length', 128),
        report_to=training_conf.get('report_to', []),
        logging_steps=training_conf.get('logging_steps', 10),
        push_to_hub=training_conf.get('push_to_hub', False),
        save_strategy=training_conf.get('save_strategy', 'steps'),
        save_steps=training_conf.get('save_steps', 10),
        max_steps=training_conf.get('max_steps', -1),
    )

    # Trainer
    print("Initializing GRPOTrainer...")
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset,
    )

    # Train
    print("Starting training...")
    trainer.train()
    
    # Save Model
    trainer.save_model(config['training']['output_dir'])
    print(f"Model saved to {config['training']['output_dir']}")

if __name__ == "__main__":
    main()
