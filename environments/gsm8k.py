import re
import yaml
from datasets import load_dataset
from math_verify import LatexExtractionConfig, parse, verify, StringExtractionConfig, ExprExtractionConfig
from .base import BaseEnvironment

class GSM8KEnvironment(BaseEnvironment):
    def __init__(self, config):
        self.config = config

    def get_system_prompt(self):
        return self.config.get('system_prompt', "")

    def make_conversation(self, example):
        system_prompt = self.get_system_prompt()
        prompt_column = self.config['data']['prompt_column']
        return {
            "prompt": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": example[prompt_column]},
            ],
        }

    def get_dataset(self, config):
        print(f"Loading dataset: {config['data']['dataset_name']}")
        dataset = load_dataset(
            config['data']['dataset_name'], 
            config['data'].get('subset', None),
            split=config['data'].get('split', 'train')
        )
        
        # Use lambda to pass self.make_conversation
        dataset = dataset.map(lambda x: self.make_conversation(x))
        return dataset

    def format_reward(self, completions, **kwargs):
        """Reward function that checks if the completion has a specific format."""
        pattern = r"^<think>.*?</think>\s*<answer>.*?</answer>$"
        completion_contents = [completion[0]["content"] for completion in completions]
        matches = [re.match(pattern, content, re.DOTALL) for content in completion_contents]
        return [1.0 if match else 0.0 for match in matches]

    def accuracy_reward(self, completions, **kwargs):
        """Reward function that checks if the completion is the same as the ground truth (GSM8K specific)."""
        solutions = kwargs['answer'] # In GSM8K, the ground truth is in the 'answer' column
        completion_contents = [completion[0]["content"] for completion in completions]
        rewards = []
        for content, solution in zip(completion_contents, solutions):
            # Extract the ground truth value from GSM8K format (after ####)
            if isinstance(solution, str):
                 gold_answer_match = solution.split("####")
                 if len(gold_answer_match) > 1:
                     gold_answer = gold_answer_match[1].strip()
                 else:
                     gold_answer = solution.strip() # Fallback
            else:
                 gold_answer = str(solution)
    
            gold_parsed = parse(gold_answer, extraction_mode="first_match", extraction_config=[LatexExtractionConfig(),ExprExtractionConfig(),StringExtractionConfig()])
    
            # Extract answer from the model completion (inside <answer> tags)
            answer_match = re.search(r"<answer>(.*?)</answer>", content, re.DOTALL)
    
            if answer_match:
                answer_content = answer_match.group(1).strip()
                answer_parsed = parse(answer_content, extraction_mode="first_match", extraction_config=[LatexExtractionConfig(),ExprExtractionConfig(),StringExtractionConfig()])
    
                if len(gold_parsed) != 0:
                    try:
                        rewards.append(float(verify(answer_parsed, gold_parsed)))
                    except Exception:
                        rewards.append(0.0)
                else:
                    rewards.append(1.0) 
            else:
                rewards.append(0.0) # No answer tag found
                
        return rewards

    def get_reward_functions(self):
        return [self.format_reward, self.accuracy_reward]
