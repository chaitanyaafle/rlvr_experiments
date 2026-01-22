import random
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Any, Iterator
import numpy as np

# Mocking the base class if we were outside the package
try:
    from reasoning_gym.dataset import ProceduralDataset
except ImportError:
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
        # 0=Unknown, 1=Miss, 2=Hit
        self.shots_grid = np.zeros((height, width), dtype=int)
        
        # Store ship ID on board: 0=Water, 1..N=Ship IDs
        self.ship_id_grid = np.zeros((height, width), dtype=int)
        
        # Track which ships from the fleet were actually placed
        self.placed_fleet = {}
        self._place_ships()
        
    def _place_ships(self):
        self.board.fill(0)
        self.ship_id_grid.fill(0)
        self.ships = []
        self.placed_fleet = {}
        
        ships_to_place = []
        for size, count in self.fleet_spec.items():
            ships_to_place.extend([size] * count)
        ships_to_place.sort(reverse=True)
        
        ship_id_counter = 1
        for size in ships_to_place:
            placed = False
            attempts = 0
            while not placed and attempts < 2000:
                attempts += 1
                orientation = self.rng.choice(['H', 'V'])
                if orientation == 'H':
                    r = self.rng.randint(0, self.height - 1)
                    c = self.rng.randint(0, self.width - size)
                    if self._can_place_h(r, c, size):
                        self._place_h(r, c, size, ship_id_counter)
                        self.ships.append({'row': r, 'col': c, 'size': size, 'orientation': 'H', 'id': ship_id_counter})
                        placed = True
                        self.placed_fleet[size] = self.placed_fleet.get(size, 0) + 1
                        ship_id_counter += 1
                else:
                    r = self.rng.randint(0, self.height - size)
                    c = self.rng.randint(0, self.width - 1)
                    if self._can_place_v(r, c, size):
                        self._place_v(r, c, size, ship_id_counter)
                        self.ships.append({'row': r, 'col': c, 'size': size, 'orientation': 'V', 'id': ship_id_counter})
                        placed = True
                        self.placed_fleet[size] = self.placed_fleet.get(size, 0) + 1
                        ship_id_counter += 1
            
            if not placed:
                 pass

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

    def _place_h(self, r, c, size, ship_id):
        self.board[r, c:c+size] = 1
        self.ship_id_grid[r, c:c+size] = ship_id
        
    def _place_v(self, r, c, size, ship_id):
        self.board[r:r+size, c] = 1
        self.ship_id_grid[r:r+size, c] = ship_id

    def simulate_random_shots(self, n_shots):
        """Simulate random unique shots on the board."""
        available_coords = [(r, c) for r in range(self.height) for c in range(self.width)]
        self.rng.shuffle(available_coords)
        
        shots_taken = available_coords[:n_shots]
        for r, c in shots_taken:
            if self.board[r, c] == 1:
                self.shots_grid[r, c] = 2 # Hit
            else:
                self.shots_grid[r, c] = 1 # Miss
        return shots_taken

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
        
        # Use the ACTUALLY placed fleet for the prompt, to ensure truthfulness
        actual_fleet = game.placed_fleet
        
        # Simulate partial game state
        total_cells = w * h
        # Allow up to total_cells - 1 shots (leaving at least one potentially valid move)
        # We also want to ensure we sometimes see winning states? No, we need to make the WINNING move.
        # So max shots should probably leave at least 1 hit if possible, or just be random.
        # The user wants "random state" "it can be anything".
        n_shots = rng.randint(0, max(0, total_cells - 1))
        game.simulate_random_shots(n_shots)
        
        # Construct Prompt
        # M = Miss, H = Hit, . = Unknown
        prompt = f"Play Battleship. Grid Size: {w}x{h}\n"
        prompt += "Board State (M=Miss, H=Hit, .=Unknown):\n"
        
        # Add column headers
        col_header = "  " + " ".join(str(c % 10) for c in range(w))
        prompt += col_header + "\n"
        
        for r in range(h):
            row_str = f"{r % 10} "
            for c in range(w):
                val = game.shots_grid[r, c]
                if val == 0: char = "."
                elif val == 1: char = "M"
                elif val == 2: char = "H"
                row_str += char + " "
            prompt += row_str + "\n"
            
        prompt += "\nFleet sizes: " + ", ".join([f"{v}x size {k}" for k,v in actual_fleet.items() if v > 0]) + "\n"
        prompt += "Output the next shot coordinate as a tuple (row, col).\n"
        
        # Determine Reference Correct Answer
        # Prioritize finding a HIT not yet taken
        # Get all unshot coordinates
        unshot_hits = []
        unshot_misses = []
        
        for r in range(h):
            for c in range(w):
                if game.shots_grid[r, c] == 0:
                    if game.board[r, c] == 1:
                        unshot_hits.append((r, c))
                    else:
                        unshot_misses.append((r, c))
        
        if unshot_hits:
            target = rng.choice(unshot_hits)
        elif unshot_misses:
            target = rng.choice(unshot_misses)
        else:
            # Game Over state (all shot), shouldn't happen with total_cells - 1 logic usually
            target = (0, 0) 
            
        answer_str = f"<answer>{target}</answer>"
        
        return {
            'question': prompt,
            'answer': answer_str,
            'metadata': {
                'width': w,
                'height': h,
                'ship_board': game.board.tolist(),
                'ship_id_grid': game.ship_id_grid.tolist(),
                'shots_grid': game.shots_grid.tolist(),
                'fleet': {str(k): v for k, v in actual_fleet.items()},
                'ships': game.ships,
                'valid_targets': unshot_hits,
                'difficulty': {
                    'grid_size': (w, h)
                }
            }
        }
