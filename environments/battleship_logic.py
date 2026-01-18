import random
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Any, Iterator
import numpy as np

# Mocking the base class if we were outside the package, 
# but assuming we can import from reasoning_gym since it is installed.
try:
    from reasoning_gym.dataset import ProceduralDataset
except ImportError:
    # Fallback/Mock for standalone usage if reasoning_gym not in path context correctly
    # But in this environment we saw it is installed.
    class ProceduralDataset:
        def __init__(self, config, seed=None, size=500):
            self.config = config
            self.seed = seed
            self.size = size

@dataclass
class BattleshipConfig:
    min_grid_size: int = 6
    max_grid_size: int = 10
    seed: Optional[int] = None
    size: int = 500
    # Fleet spec: {size: count}
    # Using field default_factory for mutable defaults
    fleet_spec: Dict[int, int] = field(default_factory=lambda: {4: 1, 3: 2, 2: 3, 1: 4})

    def validate(self):
        """Validate configuration parameters"""
        assert self.min_grid_size > 0, "min_grid_size must be positive"
        assert self.max_grid_size >= self.min_grid_size, "max_grid_size must be >= min_grid_size"
        assert self.size > 0, "size must be possible"

class BattleshipGame:
    def __init__(self, width: int, height: int, fleet_spec: Dict[int, int], seed: Optional[int] = None):
        self.width = width
        self.height = height
        self.fleet_spec = fleet_spec
        self.rng = random.Random(seed)
        self.board = np.zeros((height, width), dtype=int)
        self.ships = []  # List of dicts
        
        self._place_ships()
        
    def _place_ships(self):
        self.board.fill(0)
        self.ships = []
        
        ships_to_place = []
        for size, count in self.fleet_spec.items():
            ships_to_place.extend([size] * count)
        ships_to_place.sort(reverse=True)
        
        for size in ships_to_place:
            placed = False
            attempts = 0
            while not placed and attempts < 1000:
                attempts += 1
                orientation = self.rng.choice(['H', 'V'])
                if orientation == 'H':
                    r = self.rng.randint(0, self.height - 1)
                    c = self.rng.randint(0, self.width - size)
                    if self._can_place_h(r, c, size):
                        self._place_h(r, c, size)
                        self.ships.append({'row': r, 'col': c, 'size': size, 'orientation': 'H'})
                        placed = True
                else:
                    r = self.rng.randint(0, self.height - size)
                    c = self.rng.randint(0, self.width - 1)
                    if self._can_place_v(r, c, size):
                        self._place_v(r, c, size)
                        self.ships.append({'row': r, 'col': c, 'size': size, 'orientation': 'V'})
                        placed = True

    def _can_place_h(self, r, c, size):
        r_min = max(0, r - 1)
        r_max = min(self.height, r + 2)
        c_min = max(0, c - 1)
        c_max = min(self.width, c + size + 1)
        return np.sum(self.board[r_min:r_max, c_min:c_max]) == 0

    def _can_place_v(self, r, c, size):
        r_min = max(0, r - 1)
        r_max = min(self.height, r + size + 1)
        c_min = max(0, c - 1)
        c_max = min(self.width, c + 2)
        return np.sum(self.board[r_min:r_max, c_min:c_max]) == 0

    def _place_h(self, r, c, size):
        self.board[r, c:c+size] = 1
        
    def _place_v(self, r, c, size):
        self.board[r:r+size, c] = 1

    def get_hints(self):
        row_counts = np.sum(self.board, axis=1).tolist()
        col_counts = np.sum(self.board, axis=0).tolist()
        return row_counts, col_counts

class BattleshipDataset(ProceduralDataset):
    def __init__(self, config: BattleshipConfig):
        super().__init__(config=config, seed=config.seed, size=config.size)
        
    def __getitem__(self, idx: int) -> dict:
        rng = random.Random(self.seed + idx)
        
        if self.config.min_grid_size == self.config.max_grid_size:
            w = h = self.config.min_grid_size
        else:
            w = h = rng.randint(self.config.min_grid_size, self.config.max_grid_size)
            
        fleet = self.config.fleet_spec.copy()
        # Adjust fleet for small boards
        if w < 10:
            if 4 in fleet: del fleet[4]
            if w < 8:
                if 3 in fleet: fleet[3] = 1
        
        game = BattleshipGame(w, h, fleet, seed=rng.randint(0, 2**32-1))
        row_counts, col_counts = game.get_hints()
        
        prompt = f"Solve the Solitaire Battleship puzzle.\nGrid Size: {w}x{h}\n"
        prompt += f"Row Counts: {','.join(map(str, row_counts))}\n"
        prompt += f"Column Counts: {','.join(map(str, col_counts))}\n"
        prompt += "Fleet: " + ", ".join([f"{k}x{v}" for k,v in fleet.items() if v > 0]) + "\n"
        prompt += "Output the ship placements in the format: <ship row=\"...\" col=\"...\" size=\"...\" dir=\"...\" />\n"
        prompt += "Where row and col are 0-indexed, dir is H or V.\n"
        
        solution_str = ""
        ships_sorted = sorted(game.ships, key=lambda x: (x['row'], x['col']))
        for s in ships_sorted:
            solution_str += f"<ship row=\"{s['row']}\" col=\"{s['col']}\" size=\"{s['size']}\" dir=\"{s['orientation']}\" />\n"
        
        return {
            'question': prompt,
            'answer': solution_str,
            'metadata': {
                'width': w,
                'height': h,
                'row_counts': row_counts,
                'col_counts': col_counts,
                'ships': game.ships,
                'fleet': fleet,
                'difficulty': {
                    'grid_size': (w, h)
                }
            }
        }
