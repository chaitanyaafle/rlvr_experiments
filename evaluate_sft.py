import sys
import os
import torch
import tqdm
import re
from typing import List, Dict, Any
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

from dataclasses import dataclass, field
# Reuse configuration and environment loading from generation script
# Assuming generate_sft_data.py is in data_gen/ or we need to add path
sys.path.append(os.path.join(os.getcwd(), 'data_gen'))
try:
    from generate_sft_data import load_config, DataGenConfig, get_environment, ModelConfig, EnvironmentConfig
except ImportError:
    # If strictly running from root and file structure is different, handle it.
    # But user is in /home/rlvr_experiments, so data_gen.generate_sft_data might work if __init__ exists
    # explicitly adding path is safer given the structure
    sys.path.append(os.getcwd())
    from data_gen.generate_sft_data import load_config, DataGenConfig, get_environment, ModelConfig, EnvironmentConfig

@dataclass
class EvalParameters:
    num_samples: int = 50
    batch_size: int = 1

@dataclass
class SFTFormatConfig:
    think_start_token: str = "<think>"
    think_end_token: str = "</think>"
    answer_start_token: str = "<answer>"
    answer_end_token: str = "</answer>"
    system_prompt: str = "You are a helpful assistant."

@dataclass
class EvalConfig:
    model: ModelConfig
    environment: EnvironmentConfig
    evaluation: EvalParameters
    sft_format: SFTFormatConfig

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "EvalConfig":
        model_cfg = ModelConfig(
            name_or_path=config_dict['model']['name_or_path'],
            dtype=config_dict['model'].get('torch_dtype', 'auto'),
            device_map=config_dict['model'].get('device_map', 'auto')
        )
        
        env_dict = config_dict.get('environment', {})
        env_name = env_dict.get('name', 'unknown')
        env_params = {k: v for k, v in env_dict.items() if k != 'name'}
        environment_cfg = EnvironmentConfig(name=env_name, params=env_params)
        
        eval_dict = config_dict.get('evaluation', {})
        eval_params = EvalParameters(
            num_samples=eval_dict.get('num_samples', 50),
            batch_size=eval_dict.get('batch_size', 1)
        )
        
        sft_dict = config_dict.get('sft_format', config_dict.get('sft', {})) # Fallback to sft key if sft_format not present
        sft_fmt = SFTFormatConfig(
            think_start_token=sft_dict.get('think_start_token', '<think>'),
            think_end_token=sft_dict.get('think_end_token', '</think>'),
            answer_start_token=sft_dict.get('answer_start_token', '<answer>'),
            answer_end_token=sft_dict.get('answer_end_token', '</answer>'),
            system_prompt=sft_dict.get('system_prompt', "You are a helpful assistant.")
        )

        return cls(model=model_cfg, environment=environment_cfg, evaluation=eval_params, sft_format=sft_fmt)

def extract_answer(text: str, start_token: str, end_token: str) -> str:
    """Extracts the content between start and end tokens."""
    pattern = re.escape(start_token) + r"(.*?)" + re.escape(end_token)
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""

def evaluate(config_path: str, model_path_override: str = None, num_samples_override: int = None):
    print(f"Loading config from {config_path}")
    raw_config = load_config(config_path)
    config = EvalConfig.from_dict(raw_config)

    # Override model path if provided
    if model_path_override:
        print(f"Overriding model path: {model_path_override}")
        config.model.name_or_path = model_path_override
        
    num_samples = num_samples_override if num_samples_override is not None else config.evaluation.num_samples

    # Initialize Environment
    # Generating fresh data for evaluation to avoid memorization
    env = get_environment(raw_config)
    # We use the dataset generation method from the env
    dataset = env.get_dataset(raw_config)
    
    # Load Model
    print(f"Loading model: {config.model.name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(config.model.name_or_path)
    # Ensure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForCausalLM.from_pretrained(
        config.model.name_or_path,
        torch_dtype=config.model.dtype,
        device_map=config.model.device_map
    )
    model.eval()

    print(f"Evaluating on {num_samples} samples...")
    
    correct_count = 0
    format_compliant_count = 0
    total = 0
    
    results = []

    # Get tokens
    think_start = config.sft_format.think_start_token
    think_end = config.sft_format.think_end_token
    answer_start = config.sft_format.answer_start_token
    answer_end = config.sft_format.answer_end_token
    
    # System prompt
    system_prompt_tmpl = config.sft_format.system_prompt
    try:
         system_msg = system_prompt_tmpl.format(think_start=think_start, think_end=think_end, answer_start=answer_start, answer_end=answer_end)
    except KeyError:
         system_msg = system_prompt_tmpl

    for i, item in tqdm.tqdm(enumerate(dataset), total=min(len(dataset), num_samples)):
        if i >= num_samples:
            break
            
        total += 1
        
        user_content = next((msg['content'] for msg in item['prompt'] if msg['role'] == 'user'), None)
        ground_truth = str(item['answer']).strip()
        
        # Prepare input
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": f"Problem:\n{user_content}\n\nCorrect Solution:\n{ground_truth}\n\nPlease explain the reasoning step-by-step to arrive at this solution."}
        ]
        
        # We want the model to generate the REST. The prompt in training included the system and user and part of the assistant?
        # No, in training we provided the full sequence. Here we provide System + User and expect Assistant content.
        # But wait, in the SFT data gen script, the logic was:
        # prompt = system + user
        # We force the model to start with <think> if we want, but for evaluation of a trained model, 
        # we generally let it generate naturally or prompt it exactly as during inference.
        # Standard SFT inference prompt:
        inference_messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": f"Problem:\n{user_content}\n\nWhat is the Minimum number of steps?"} 
            # Note: The SFT training data generator constructed a specific user prompt:
            # "Problem:\n{user_content}\n\nCorrect Solution:\n{ground_truth}\n\nPlease explain..." 
            # This included the solution! That was for "Data Generation" (Distillation) or was it?
            # Let's re-read generate_sft_data.py carefully.
            # It uses "Problem:\n{user_content}\n\nCorrect Solution:\n{ground_truth}\n\nPlease explain..."
            # This means the "Teacher" model (Qwen-0.5B-Instruct in the config??) was given the answer to generate thoughts.
            # Wait, the config uses Qwen-0.5B-Instruct. Is it acting as a teacher or is this the SFT target?
            # The USER REQUEST says: "generate sft data ... I want to use this for sft". 
            # And "training should be on the assistant reply".
            # If the SFT data includes the solution in the USER prompt, then at test time, we won't have the solution.
            # This seems like a "Reasoning Trace Generation" step where we cheat and give the answer to get the trace.
            # BUT for EVALUATION, we must NOT give the answer.
            # We must prompt the model with just the problem.
        ]
        
        # Wait, if the model was TRAINED on prompts containing the solution (as per generate_sft_data.py), 
        # then it expects the solution in the prompt?
        # Let's check generate_sft_data.py again.
        # Line 173: {"role": "user", "content": f"Problem:\n{user_content}\n\nCorrect Solution:\n{ground_truth}\n\nPlease explain..."}
        # Line 218: {"role": "user", "content": user_content},
        # Line 219: {"role": "assistant", "content": final_assistant_content}
        #
        # AHA! The 'meta_prompt' (lines 171-174) used to GENERATE the thought includes the solution.
        # BUT the 'sft_example' (lines 215-221) saves `user_content` (the original problem) as the user message.
        # So the SFT training data consists of:
        # User: [Problem]
        # Assistant: <think>[Reasoning]</think><answer>[Answer]</answer>
        #
        # So my evaluation prompt should JUST be the User content (Problem).
        
        eval_messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_content}
        ]

        text = tokenizer.apply_chat_template(eval_messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=1024,
                do_sample=False, # Deterministic for eval
                pad_token_id=tokenizer.pad_token_id
            )
            
        completion = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        
        # Check format
        has_think = think_start in completion and think_end in completion
        has_answer = answer_start in completion and answer_end in completion
        
        if has_think and has_answer:
            format_compliant_count += 1
            
        # Check correctness
        # Extract answer
        pred_answer = extract_answer(completion, answer_start, answer_end)
        
        # Simple string matching for now (assuming ground truth is just the number/string)
        is_correct = pred_answer == ground_truth
        if is_correct:
            correct_count += 1
            
        results.append({
            "problem": user_content,
            "ground_truth": ground_truth,
            "prediction": pred_answer,
            "full_completion": completion,
            "correct": is_correct,
            "compliant": has_think and has_answer
        })

    accuracy = correct_count / total if total > 0 else 0
    compliance = format_compliant_count / total if total > 0 else 0
    
    print("-" * 40)
    print(f"Results for {config.model.name_or_path}")
    print(f"Samples: {total}")
    print(f"Accuracy: {accuracy:.2%}")
    print(f"Format Compliance: {compliance:.2%}")
    print("-" * 40)
    
    return accuracy, compliance

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python evaluate_sft.py config.yaml [model_path_override] [num_samples]")
        sys.exit(1)
        
    config_file = sys.argv[1]
    model_override = sys.argv[2] if len(sys.argv) > 2 else None
    
    # Optional 3rd arg for num_samples
    samples = 50
    if len(sys.argv) > 3:
        try:
             samples = int(sys.argv[3])
        except ValueError:
             pass
             
    evaluate(config_file, model_override, samples)
