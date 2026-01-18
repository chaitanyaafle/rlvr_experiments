import re
from datasets import Dataset
from .base import BaseEnvironment
from .battleship_logic import BattleshipConfig, BattleshipDataset
import numpy as np

class BattleshipEnvironment(BaseEnvironment):
    def __init__(self, config):
        self.config = config
        env_args = config.get('environment', config.get('env', {}))
        
        # Parse fleet config if valid, else use default logic in BattleshipConfig
        fleet_spec = env_args.get('fleet_spec', None) 
        
        self.bs_config = BattleshipConfig(
            min_grid_size=env_args.get('min_grid_size', 6),
            max_grid_size=env_args.get('max_grid_size', 10),
            seed=env_args.get('seed', 42),
            size=env_args.get('size', 500)
        )
        if fleet_spec:
            self.bs_config.fleet_spec = fleet_spec

    def get_system_prompt(self):
        return self.config.get('system_prompt', 
            """You are a helpful assistant. You must output your reasoning steps within <think></think> tags and the ship placement solution within <answer></answer> tags.
            The answer must be a list of XML-like tags: <ship row=".." col=".." size=".." dir=".." />"""
        )

    def make_conversation(self, example):
        system_prompt = self.get_system_prompt()
        return {
            "prompt": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": example['question']},
            ],
            "answer": example['answer'],
            "metadata": example['metadata']
        }

    def get_dataset(self, config):
        print(f"Generating Battleship dataset with config: {self.bs_config}")
        dataset_generator = BattleshipDataset(self.bs_config)
        
        data_list = list(dataset_generator)
        hf_dataset = Dataset.from_list(data_list)
        
        hf_dataset = hf_dataset.map(lambda x: self.make_conversation(x))
        return hf_dataset

    def format_reward(self, completions, **kwargs):
        """Checks if the completion has the correct XML tag structure for answer."""
        pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
        completion_contents = [completion[0]["content"] for completion in completions]
        matches = [re.search(pattern, content, re.DOTALL) for content in completion_contents]
        return [1.0 if match else 0.0 for match in matches]

    def completeness_reward(self, completions, **kwargs):
        """Checks if all ships are accounted for and validly placed (no overlap, within bounds). 
        Matches row/col counts."""
        metadata = kwargs['metadata'] # list of metadata dicts
        completion_contents = [completion[0]["content"] for completion in completions]
        
        rewards = []
        for content, meta in zip(completion_contents, metadata):
            answer_match = re.search(r"<answer>(.*?)</answer>", content, re.DOTALL)
            if not answer_match:
                rewards.append(0.0)
                continue
                
            answer_text = answer_match.group(1)
            # Parse ships
            # Regex for <ship row="1" col="2" size="3" dir="H" />
            # Allowing flexible whitespace and quotes
            ship_pattern = r'<ship\s+row=[\'"](\d+)[\'"]\s+col=[\'"](\d+)[\'"]\s+size=[\'"](\d+)[\'"]\s+dir=[\'"]([HV])[\'"]\s*/>'
            ships_found = re.findall(ship_pattern, answer_text)
            
            if not ships_found:
                 rewards.append(0.0)
                 continue
            
            # Reconstruct board
            w, h = meta['width'], meta['height']
            board = np.zeros((h, w), dtype=int)
            valid_placement = True
            
            # Track fleet found
            found_fleet = {}
            
            for r, c, size, orient in ships_found:
                r, c, size = int(r), int(c), int(size)
                
                # Update found fleet count
                found_fleet[size] = found_fleet.get(size, 0) + 1
                
                if orient == 'H':
                    if c + size > w:
                        valid_placement = False; break
                    # Overlap check (strict logic puzzle: usually NO touching, but let's just check raw visual overlap first)
                    # We will enforce the generate rules. Logic Generator used strict no-touching?
                    # Generator used "_can_place" which checks neighborhood.
                    # For reward, we minimally expect non-overlapping ships that satisfy counts.
                    # Stricter: enforce no touching.
                    
                    # Check bounds
                    if r < 0 or r >= h: 
                         valid_placement = False; break
                    
                    # Check overlap
                    if np.sum(board[r, c:c+size]) > 0:
                        valid_placement = False; break
                        
                    board[r, c:c+size] = 1
                    
                else: # V
                    if r + size > h:
                        valid_placement = False; break
                    
                    if c < 0 or c >= w:
                         valid_placement = False; break
                         
                    if np.sum(board[r:r+size, c]) > 0:
                        valid_placement = False; break
                        
                    board[r:r+size, c] = 1
            
            if not valid_placement:
                rewards.append(0.0)
                continue
                
            # Check row/col counts
            row_counts = np.sum(board, axis=1).tolist()
            col_counts = np.sum(board, axis=0).tolist()
            
            if row_counts == meta['row_counts'] and col_counts == meta['col_counts']:
                 # Also check if fleet matches?
                 # If counts match, it's usually correct, but theoretically you could have wrong fleet with same counts (maybe?).
                 # Let's simple check counts first.
                 rewards.append(1.0)
            else:
                rewards.append(0.0)
                
        return rewards

    def get_reward_functions(self):
        return [self.format_reward, self.completeness_reward]
