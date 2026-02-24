import re
import json
import reasoning_gym
from datasets import Dataset
from .base import BaseEnvironment


class _SafeEncoder(json.JSONEncoder):
    """Serialize anything reasoning_gym puts in metadata (numpy scalars, tuples, etc.)."""
    def default(self, obj):
        try:
            return super().default(obj)
        except TypeError:
            return str(obj)

    def encode(self, obj):
        # Convert tuples to lists so they round-trip cleanly.
        if isinstance(obj, tuple):
            obj = list(obj)
        return super().encode(obj)


def _metadata_to_str(meta: dict) -> str:
    return json.dumps(meta, cls=_SafeEncoder, default=str)

# Correct registered names in reasoning_gym for the 20-task research plan suite.
# Use these in the config's environment.name field.
REASONING_GYM_TASKS = {
    # Games
    "maze", "mini_sudoku", "tower_of_hanoi", "n_queens", "countdown",
    # Logic
    "knights_knaves", "syllogism", "zebra_puzzles", "propositional_logic", "circuit_logic",
    # Algorithmic
    "graph_color", "word_ladder", "caesar_cipher", "number_sorting", "cryptarithm",
    # Graphs
    "shortest_path", "family_relationships",
    # Arithmetic
    "basic_arithmetic", "prime_factorization", "gcd",
}

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful assistant. Think through the problem carefully and show your "
    "reasoning inside <think></think> tags. Then give your final answer inside "
    "<answer></answer> tags."
)


class ReasoningGymEnvironment(BaseEnvironment):
    """Generic environment for any reasoning_gym task.

    Config example (YAML):
        environment:
          name: maze          # any registered reasoning_gym task name
          size: 500           # number of examples to generate
          seed: 42
          # any other kwargs are forwarded to create_dataset as-is
          min_dist: 2
          max_dist: 5
        system_prompt: "..."  # optional override
    """

    def __init__(self, config):
        self.config = config
        env_cfg = config.get("environment", {})
        self.task_name = env_cfg["name"]

        # Pull well-known kwargs; everything else in env_cfg (besides 'name') is
        # forwarded to create_dataset so task-specific params just work.
        self.size = env_cfg.get("size", 500)
        self.seed = env_cfg.get("seed", 42)
        self._extra_kwargs = {
            k: v for k, v in env_cfg.items() if k not in ("name", "size", "seed")
        }

        # Build the dataset once so score_answer is available on self._rg_dataset.
        self._rg_dataset = reasoning_gym.create_dataset(
            self.task_name, size=self.size, seed=self.seed, **self._extra_kwargs
        )

    # ------------------------------------------------------------------
    # BaseEnvironment interface
    # ------------------------------------------------------------------

    def get_system_prompt(self):
        return self.config.get("system_prompt", DEFAULT_SYSTEM_PROMPT)

    def get_dataset(self, config):
        size = self._rg_dataset.size
        # Normalize each item: answer → str, metadata → JSON string.
        # reasoning_gym can return numpy scalars / tuples which PyArrow
        # cannot serialize directly.
        data_list = []
        for i in range(size):
            item = dict(self._rg_dataset[i])
            item["answer"] = str(item.get("answer", ""))
            item["metadata"] = _metadata_to_str(item.get("metadata", {}))
            data_list.append(item)
        hf_dataset = Dataset.from_list(data_list)
        hf_dataset = hf_dataset.map(self._make_conversation)
        return hf_dataset

    def get_reward_functions(self):
        return [self.format_reward, self.accuracy_reward]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_conversation(self, example):
        return {
            "prompt": [
                {"role": "system", "content": self.get_system_prompt()},
                {"role": "user", "content": example["question"]},
            ],
            "answer": example["answer"],
            # Keep metadata for downstream analysis (ARS probing etc.)
            "metadata": example.get("metadata", {}),
        }

    def format_reward(self, completions, **kwargs):
        """Binary reward: 1.0 if completion has <think>...</think><answer>...</answer>."""
        pattern = r"^<think>.*?</think>\s*<answer>.*?</answer>$"
        contents = [c[0]["content"] for c in completions]
        return [1.0 if re.match(pattern, c, re.DOTALL) else 0.0 for c in contents]

    def accuracy_reward(self, completions, **kwargs):
        """Score extracted <answer> content using reasoning_gym's own score_answer."""
        solutions = kwargs["answer"]
        # kwargs also carries the full dataset rows; we need the original entry to
        # call score_answer correctly (it may use metadata).  We reconstruct a
        # minimal entry dict from what the dataset columns provide.
        entries = kwargs.get("metadata", [{}] * len(solutions))

        contents = [c[0]["content"] for c in completions]
        rewards = []
        for content, solution, entry_meta in zip(contents, solutions, entries):
            answer_match = re.search(r"<answer>(.*?)</answer>", content, re.DOTALL)
            if not answer_match:
                rewards.append(0.0)
                continue

            predicted = answer_match.group(1).strip()
            # Reconstruct a minimal entry that score_answer expects.
            entry = {"answer": solution, "metadata": entry_meta}
            try:
                score = self._rg_dataset.score_answer(answer=predicted, entry=entry)
                rewards.append(float(score))
            except Exception:
                # Fallback: exact-match normalised string comparison
                rewards.append(1.0 if predicted.lower() == str(solution).lower() else 0.0)

        return rewards
