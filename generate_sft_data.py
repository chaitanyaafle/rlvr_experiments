
import sys
import yaml
import torch
import json
import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from environments.maze_env import MazeEnvironment
from environments.gsm8k import GSM8KEnvironment

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def get_environment(config):
    env_config = config.get('environment', {})
    if not env_config:
        if config.get('data', {}).get('dataset_name') == 'openai/gsm8k':
            return GSM8KEnvironment(config)
    name = env_config.get('name')
    if name == 'gsm8k':
        return GSM8KEnvironment(config)
    elif name == 'maze':
        return MazeEnvironment(config)
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
    env = get_environment(config)
    dataset = env.get_dataset(config)

    # Load Model
    model_path = config['model']['name_or_path']
    print(f"Loading model: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=config['model'].get('torch_dtype', 'auto'),
        device_map="auto"
    )

    output_file = config.get('sft', {}).get('output_file', 'sft_data.jsonl')
    print(f"Generating SFT data to {output_file}")

    results = []
    
    # We will limit the number of generations if specified, or do a subset
    max_samples = config.get('sft', {}).get('max_samples', len(dataset))
    
    # Create the generation prompt
    # We want to ask the model to explain the solution.
    
    for i, item in tqdm.tqdm(enumerate(dataset), total=min(len(dataset), max_samples)):
        if i >= max_samples:
            break
            
        # item['prompt'] is a list of {'role':..., 'content':...}
        # item['answer'] is the ground truth
        
        # Extract the original user query
        user_content = next((msg['content'] for msg in item['prompt'] if msg['role'] == 'user'), None)
        system_content = next((msg['content'] for msg in item['prompt'] if msg['role'] == 'system'), None)
        
        ground_truth = item['answer']
        
        # Construct a meta-prompt to get the reasoning
        # We present the problem and solution, and ask for the thought process.
        
        meta_prompt = [
            {"role": "system", "content": "You are a helpful assistant. You are given a problem and its correct solution. Your task is to generate the step-by-step reasoning that leads to this solution. Output the reasoning inside <think></think> tags and the final answer inside <answer></answer> tags."},
            {"role": "user", "content": f"Problem:\n{user_content}\n\nCorrect Solution:\n{ground_truth}\n\nPlease explain the reasoning step-by-step to arrive at this solution."}
        ]
        
        text = tokenizer.apply_chat_template(meta_prompt, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=1024,
                do_sample=True,
                temperature=0.7
            )
        
        generated_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        
        # Post-process:
        # The generated text is the "Assistant" response for our SFT dataset.
        # However, we want the SFT dataset to look like: User: Problem -> Assistant: <think>...</think><answer>Solution</answer>
        # So we use the GENERATED reasoning as the target for the original problem.
        
        # Ideally, we should check if the generated text actually contains the solution or valid format.
        # But for now, we assume the model follows instructions reasonably well or we filter later.
        
        sft_example = {
            "messages": [
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": generated_text}
            ]
        }
        results.append(sft_example)
        
        # Periodically save
        if i % 10 == 0:
            with open(output_file, 'w') as f:
                for line in results:
                    f.write(json.dumps(line) + "\n")

    # Final save
    with open(output_file, 'w') as f:
        for line in results:
            f.write(json.dumps(line) + "\n")
            
    print(f"Finished. Saved {len(results)} examples to {output_file}")

if __name__ == "__main__":
    main()
