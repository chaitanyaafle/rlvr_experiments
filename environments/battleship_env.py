
import re
import ast
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
            """You are a helpful assistant playing Battleship. You must output your reasoning steps within <think></think> tags and the next shot coordinate within <answer></answer> tags.
            The answer must be a single tuple (row, col)."""
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
        """Checks if the completion has the correct XML tag structure for answer and contains a tuple."""
        pattern = r"<think>.*?</think>\s*<answer>\s*\(.*?\)\s*</answer>"
        completion_contents = [completion[0]["content"] for completion in completions]
        matches = [re.search(pattern, content, re.DOTALL) for content in completion_contents]
        return [1.0 if match else 0.0 for match in matches]

    def completeness_reward(self, completions, **kwargs):
        """
        Rewards:
        +1.0 if Hit.
        +2.0 if Hit on a ship that already has a hit.
        +10.0 if all ships sunk (winning move).
        0.0 if Miss.
        -1.0 if shot was already taken or invalid.
        """
        metadata = kwargs['metadata'] # list of metadata dicts
        completion_contents = [completion[0]["content"] for completion in completions]
        
        rewards = []
        for content, meta in zip(completion_contents, metadata):
            answer_match = re.search(r"<answer>(.*?)</answer>", content, re.DOTALL)
            if not answer_match:
                rewards.append(0.0)
                continue
                
            answer_text = answer_match.group(1).strip()
            
            # Parse tuple (r, c)
            try:
                # Use ast.literal_eval for safe parsing
                coords = ast.literal_eval(answer_text)
                if not isinstance(coords, (list, tuple)) or len(coords) != 2:
                    raise ValueError
                r, c = int(coords[0]), int(coords[1])
            except:
                rewards.append(0.0) 
                continue
            
            # Validate bounds
            w, h = meta['width'], meta['height']
            if not (0 <= r < h and 0 <= c < w):
                rewards.append(-1.0) # Invalid move
                continue
                
            # Check if already shot
            shots_grid = np.array(meta['shots_grid'])
            if shots_grid[r, c] != 0:
                rewards.append(-1.0) # Already shot
                continue
            
            # Check Hit or Miss
            ship_board = np.array(meta['ship_board'])
            
            if ship_board[r, c] == 1:
                # HIT Logic
                reward = 1.0 # Base Hit
                
                # Check for +2 (Hit on ship that already has a hit) and +10 (All ships sunk)
                # To do this, we need ship ownership.
                # Assuming ship_id_grid is in metadata (added in update)
                if 'ship_id_grid' in meta:
                    ship_id_grid = np.array(meta['ship_id_grid'])
                    hit_ship_id = ship_id_grid[r, c]
                    
                    # 1. Check if this ship ID was already partially hit elsewhere
                    # Find all cells belonging to this ship
                    ship_cells = (ship_id_grid == hit_ship_id)
                    # Check if any of these cells are already in shots_grid as Hit (2)
                    already_hit_mask = (shots_grid == 2) & ship_cells
                    if np.any(already_hit_mask):
                        reward = 2.0 # Upgrade to +2
                
                    # 2. Check if this shot SINKS the LAST ship (Winning Move)
                    # Simulate the shot
                    shots_grid[r, c] = 2 # Apply hit locally for check
                    
                    # Check if ALL ships are sunk
                    # Iterate over all unique ship IDs > 0
                    all_sunk = True
                    unique_ships = np.unique(ship_id_grid)
                    for s_id in unique_ships:
                        if s_id == 0: continue
                        # Check if all cells for s_id are hit
                        s_cells = (ship_id_grid == s_id)
                        if not np.all(shots_grid[s_cells] == 2):
                            all_sunk = False
                            break
                    
                    if all_sunk:
                        reward = 10.0 # Override with +10 for win
                
                rewards.append(reward)
            else:
                rewards.append(0.0) # Miss
                
        return rewards

    def get_reward_functions(self):
        return [self.format_reward, self.completeness_reward]
