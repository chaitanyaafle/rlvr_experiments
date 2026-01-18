
import sys
import yaml
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer
from datasets import load_dataset, Dataset

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    if len(sys.argv) < 2:
        config_path = "config.yaml"
    else:
        config_path = sys.argv[1]

    print(f"Loading config from {config_path}")
    config = load_config(config_path)

    # Load SFT Data
    sft_file = config.get('sft', {}).get('output_file', 'sft_data.jsonl')
    print(f"Loading SFT data from {sft_file}")
    
    # helper to load jsonl
    dataset = load_dataset('json', data_files=sft_file, split='train')
    
    # Model
    model_path = config['model']['name_or_path']
    print(f"Loading model: {model_path}")
    
    # Config
    sft_conf = config.get('sft', {})
    training_conf = config.get('training', {})
    
    # If specific SFT training args are valid, use them, else fallback or use 'sft' key
    # For simplicity, we might reuse 'training' section or a new 'sft_training' section
    # Let's say we look for 'sft_training' in config, else use 'training' but override output_dir
    
    train_args_conf = config.get('sft_training', training_conf)
    output_dir = train_args_conf.get('output_dir', sft_conf.get('output_dir', 'sft_output'))
    
    training_args = SFTConfig(
        output_dir=output_dir,
        # dataset_text_field="text", # Not needed for messages format
        learning_rate=float(train_args_conf.get('learning_rate', 2e-5)),
        num_train_epochs=train_args_conf.get('num_train_epochs', 1),
        bf16=train_args_conf.get('bf16', False),
        logging_steps=train_args_conf.get('logging_steps', 10),
        per_device_train_batch_size=train_args_conf.get('batch_size', 4),
        gradient_accumulation_steps=train_args_conf.get('gradient_accumulation_steps', 1),
        save_strategy=train_args_conf.get('save_strategy', 'steps'),
        save_steps=train_args_conf.get('save_steps', 50),
        report_to=train_args_conf.get('report_to', []),
    )
    
    # Load model & tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=config['model'].get('torch_dtype', 'auto'),
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    print("Initializing SFTTrainer...")
    
    # SFTTrainer automatically masks the user prompt when using 'messages' format in the dataset.
    # We rely on this built-in behavior to ensure only the assistant response is trained.

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=training_args,
        processing_class=tokenizer,
    )
    
    print("Starting SFT...")
    trainer.train()
    
    print(f"Saving model to {output_dir}")
    trainer.save_model(output_dir)

if __name__ == "__main__":
    main()
