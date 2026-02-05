"""
Unified SFT Training Script with Unsloth

Supports:
- Unsloth FastLanguageModel (2x faster, 50% less memory)
- Fallback to standard Transformers if Unsloth unavailable
- QLoRA (4-bit quantization) for memory efficiency
- Multi-dataset training (parquet files)

Usage:
    python train_unsloth_sft.py configs/atoms_sft.yaml
    python train_unsloth_sft.py configs/siblings_sft.yaml
"""

import sys
import os
import yaml
import torch
from pathlib import Path
from typing import Optional, List, Dict, Any
from datasets import load_dataset, Dataset, concatenate_datasets

# Try to import Unsloth, fallback to standard transformers
try:
    from unsloth import FastLanguageModel
    UNSLOTH_AVAILABLE = True
    print("✓ Unsloth available - using optimized training")
except ImportError:
    UNSLOTH_AVAILABLE = False
    print("⚠ Unsloth not available - using standard Transformers")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

from transformers import BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer


def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_datasets(data_config: Dict[str, Any]) -> Dataset:
    """
    Load and concatenate multiple parquet files.
    
    Supports:
    - Single file: data_files: "path/to/file.parquet"
    - Multiple files: data_files: ["path1.parquet", "path2.parquet"]
    - Directory glob: data_files: "path/to/dir/*.parquet"
    """
    data_files = data_config.get('data_files', [])
    
    if isinstance(data_files, str):
        # Handle glob patterns
        if '*' in data_files:
            from glob import glob
            data_files = sorted(glob(data_files))
        else:
            data_files = [data_files]
    
    if not data_files:
        raise ValueError("No data files specified in config")
    
    print(f"Loading {len(data_files)} data file(s)...")
    
    datasets = []
    for filepath in data_files:
        print(f"  - {filepath}")
        ds = load_dataset('parquet', data_files=filepath, split='train')
        datasets.append(ds)
    
    if len(datasets) == 1:
        return datasets[0]
    
    return concatenate_datasets(datasets)


def load_model_unsloth(model_config: Dict[str, Any]):
    """Load model using Unsloth (optimized)."""
    model_name = model_config['name_or_path']
    max_seq_length = model_config.get('max_seq_length', 2048)
    load_in_4bit = model_config.get('load_in_4bit', True)
    
    print(f"Loading model with Unsloth: {model_name}")
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=load_in_4bit,
        dtype=None,  # Auto-detect
    )
    
    # Add LoRA adapters
    lora_config = model_config.get('lora', {})
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_config.get('r', 16),
        lora_alpha=lora_config.get('alpha', 32),
        lora_dropout=lora_config.get('dropout', 0.05),
        target_modules=lora_config.get('target_modules', [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ]),
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )
    
    return model, tokenizer


def load_model_standard(model_config: Dict[str, Any]):
    """Load model using standard Transformers + PEFT."""
    model_name = model_config['name_or_path']
    load_in_4bit = model_config.get('load_in_4bit', True)
    
    print(f"Loading model with Transformers: {model_name}")
    
    # Quantization config
    if load_in_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    else:
        bnb_config = None
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Prepare for k-bit training
    if load_in_4bit:
        model = prepare_model_for_kbit_training(model)
    
    # Add LoRA
    lora_config = model_config.get('lora', {})
    peft_config = LoraConfig(
        r=lora_config.get('r', 16),
        lora_alpha=lora_config.get('alpha', 32),
        lora_dropout=lora_config.get('dropout', 0.05),
        target_modules=lora_config.get('target_modules', [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ]),
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    return model, tokenizer


def main():
    # Parse arguments
    if len(sys.argv) < 2:
        print("Usage: python train_unsloth_sft.py <config.yaml> [--max_steps N]")
        sys.exit(1)
    
    config_path = sys.argv[1]
    
    # Check for max_steps override (for testing)
    max_steps_override = None
    if '--max_steps' in sys.argv:
        idx = sys.argv.index('--max_steps')
        max_steps_override = int(sys.argv[idx + 1])
    
    print(f"Loading config from {config_path}")
    config = load_config(config_path)
    
    # Load dataset
    print("\n--- Loading Dataset ---")
    dataset = load_datasets(config['data'])
    print(f"Total samples: {len(dataset)}")
    
    # Load model
    print("\n--- Loading Model ---")
    if UNSLOTH_AVAILABLE:
        model, tokenizer = load_model_unsloth(config['model'])
    else:
        model, tokenizer = load_model_standard(config['model'])
    
    # Ensure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create formatting function for messages format
    # Unsloth tests with single example first, then uses batched
    def formatting_func(example):
        """Convert messages format to chat template string."""
        # Handle single example (dict with 'messages' key containing a list of message dicts)
        messages = example['messages']
        
        # Apply chat template to convert messages to string
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
        return text
    
    # Training config
    print("\n--- Configuring Training ---")
    train_config = config['training']
    
    # Handle max_steps override
    max_steps = max_steps_override if max_steps_override else train_config.get('max_steps', -1)
    
    training_args = SFTConfig(
        output_dir=train_config['output_dir'],
        learning_rate=float(train_config.get('learning_rate', 2e-4)),
        num_train_epochs=train_config.get('num_train_epochs', 1),
        max_steps=max_steps,
        per_device_train_batch_size=train_config.get('batch_size', 4),
        gradient_accumulation_steps=train_config.get('gradient_accumulation_steps', 4),
        warmup_ratio=train_config.get('warmup_ratio', 0.03),
        logging_steps=train_config.get('logging_steps', 10),
        save_strategy=train_config.get('save_strategy', 'steps'),
        save_steps=train_config.get('save_steps', 100),
        bf16=train_config.get('bf16', True),
        fp16=train_config.get('fp16', False),
        optim=train_config.get('optim', 'adamw_8bit'),
        max_seq_length=config['model'].get('max_seq_length', 2048),
        dataset_num_proc=train_config.get('dataset_num_proc', 4),
        report_to=train_config.get('report_to', []),
        seed=train_config.get('seed', 42),
    )
    
    # Create trainer with formatting function
    print("\n--- Initializing Trainer ---")
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=training_args,
        processing_class=tokenizer,
        formatting_func=formatting_func,
    )
    
    # Train
    print("\n--- Starting Training ---")
    trainer.train()
    
    # Save
    output_dir = train_config['output_dir']
    print(f"\n--- Saving Model to {output_dir} ---")
    trainer.save_model(output_dir)
    
    # Save tokenizer
    tokenizer.save_pretrained(output_dir)
    
    print("\n✓ Training complete!")


if __name__ == "__main__":
    main()

