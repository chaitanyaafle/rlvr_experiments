import re
import random
from datasets import Dataset
from .base import BaseEnvironment
import reasoning_gym.games.maze as maze_game

class MazeEnvironment(BaseEnvironment):
    def __init__(self, config):
        self.config = config
        env_args = config.get('environment', config.get('env', {}))
        self.maze_config = maze_game.MazeConfig(
            min_dist=env_args.get('min_dist', 5),
            max_dist=env_args.get('max_dist', 10),
            min_grid_size=env_args.get('min_grid_size', 5),
            max_grid_size=env_args.get('max_grid_size', 10),
            seed=env_args.get('seed', 42),
            size=env_args.get('size', 500)
        )

    def get_system_prompt(self):
        return self.config.get('system_prompt', 
            """You are a helpful assistant. You must output your reasoning steps within <think></think> tags and your final answer within <answer></answer> tags.
            You goal is to find the shortest path from start to end, you can move steps in passage, while avoiding the Walls."""
        )

    def make_conversation(self, example):
        system_prompt = self.get_system_prompt()
        # The maze dataset provides 'question' and 'answer'.
        # We wrap the question to ensure the model knows to use the tags if not already implicit.
        # But since we set the system prompt, that should be enough.
        
        return {
            "prompt": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": example['question']},
            ],
            # tracking ground truth in 'answer' column
            "answer": example['answer'] 
        }

    def get_dataset(self, config):
        print(f"Generating Maze dataset with config: {self.maze_config}")
        dataset_generator = maze_game.MazeDataset(self.maze_config)
        
        # reasoning_gym datasets are iterable. We convert to list to make a HF Dataset.
        # This might be slow for very large datasets, but for RLVR 500-1000 is usually fine for a start.
        data_list = []
        for i in range(self.maze_config.size):
            item = dataset_generator[i]
            data_list.append(item)
            
        hf_dataset = Dataset.from_list(data_list)
        
        # Reuse mapping logic
        hf_dataset = hf_dataset.map(lambda x: self.make_conversation(x))
        return hf_dataset

    def format_reward(self, completions, **kwargs):
        """Reward function that checks if the completion has a specific format."""
        pattern = r"^<think>.*?</think>\s*<answer>.*?</answer>$"
        completion_contents = [completion[0]["content"] for completion in completions]
        matches = [re.match(pattern, content, re.DOTALL) for content in completion_contents]
        return [1.0 if match else 0.0 for match in matches]

    def accuracy_reward(self, completions, **kwargs):
        """Reward function that checks if the completion is correct."""
        solutions = kwargs['answer']
        completion_contents = [completion[0]["content"] for completion in completions]
        rewards = []
        for content, solution in zip(completion_contents, solutions):
             # Extract answer from the model completion (inside <answer> tags)
            answer_match = re.search(r"<answer>(.*?)</answer>", content, re.DOTALL)
            
            if answer_match:
                try:
                    predicted_val = int(answer_match.group(1).strip())
                    gold_val = int(solution)
                    if predicted_val == gold_val:
                        rewards.append(1.0)
                    else:
                        rewards.append(0.0)
                except ValueError:
                    rewards.append(0.0) # Not an integer
            else:
                rewards.append(0.0)
        return rewards

    def get_reward_functions(self):
        return [self.format_reward, self.accuracy_reward]
