"""
Generic Reasoning Gym Environment for GRPO Training

Handles ANY reasoning-gym task via config. Instead of one environment file
per task, this single file works for all tasks. The config specifies which
task to use and any task-specific parameters.

Config example:
    environment:
      name: reasoning_gym
      task_name: countdown
      samples: 5000
      task_params:
        min_numbers: 4
        max_numbers: 5

For use with the existing train_grpo.py infrastructure.
"""

import re
from typing import List, Callable, Dict, Any, Optional
from datasets import Dataset

try:
    import reasoning_gym
    REASONING_GYM_AVAILABLE = True
except ImportError:
    REASONING_GYM_AVAILABLE = False


class ReasoningGymEnvironment:
    """
    Generic single-task environment for any reasoning-gym task.
    
    Takes task_name from config, generates data via reasoning-gym,
    and provides reward functions that use <answer> tag extraction.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.env_config = config.get('environment', {})
        self.task_name = self.env_config.get('task_name', 'countdown')
        self.num_samples = self.env_config.get('samples', 5000)
        self.task_params = self.env_config.get('task_params', {})
        
    def get_dataset(self, config: Dict[str, Any]) -> Dataset:
        """Generate dataset with prompts and ground truth answers."""
        if not REASONING_GYM_AVAILABLE:
            raise ImportError("reasoning-gym required. Install with: pip install reasoning-gym")
        
        seed = config.get('seed', 42)
        
        print(f"Generating {self.num_samples} samples for '{self.task_name}'...")
        print(f"  Task params: {self.task_params}")
        
        try:
            dataset = reasoning_gym.create_dataset(
                self.task_name,
                size=self.num_samples,
                seed=seed,
                **self.task_params,
            )
        except TypeError:
            # Some tasks don't accept extra params — fall back to defaults
            print(f"  [WARN] Task '{self.task_name}' rejected task_params, using defaults")
            dataset = reasoning_gym.create_dataset(
                self.task_name,
                size=self.num_samples,
                seed=seed,
            )
        
        all_data = []
        for item in dataset:
            all_data.append({
                "prompt": [{"role": "user", "content": item['question']}],
                "ground_truth": str(item['answer']),
                "task": self.task_name,
                "metadata": item.get('metadata', {}),
            })
        
        print(f"  Generated {len(all_data)} samples")
        if all_data:
            print(f"  Sample answer: {all_data[0]['ground_truth'][:80]}")
        
        return Dataset.from_list(all_data)
    
    def get_reward_functions(self) -> List[Callable]:
        """Return list of reward functions for GRPO."""
        return [
            self.correctness_reward,
            self.format_reward,
        ]
    
    # ------------------------------------------------------------------
    # Reward functions
    # ------------------------------------------------------------------
    
    def correctness_reward(
        self,
        completions,
        ground_truth,
        **kwargs
    ) -> List[float]:
        """
        Reward for correct answers.
        
        Extracts answer from <answer>...</answer> tags (preferred) or ####.
        Uses task-specific verification for countdown (eval expression),
        falls back to normalized string comparison for other tasks.
        """
        rewards = []
        
        # Get metadata if available (needed for countdown target)
        metadata_list = kwargs.get('metadata', [None] * len(completions))
        
        for i, (completion, truth) in enumerate(zip(completions, ground_truth)):
            completion = self._unwrap_completion(completion)
            extracted = self._extract_answer(completion)
            
            if extracted is None:
                rewards.append(0.0)
                continue
            
            metadata = metadata_list[i] if i < len(metadata_list) else None
            
            # Task-specific verification
            if self.task_name == 'countdown':
                rewards.append(self._verify_countdown(extracted, truth, metadata))
            else:
                rewards.append(self._verify_generic(extracted, truth))
        
        return rewards
    
    def format_reward(
        self,
        completions,
        **kwargs
    ) -> List[float]:
        """
        Reward for proper output format.
        
        Graduated scoring to create gradient signal toward proper tags:
        - <think> present but no </think>: 0.1 (started thinking but didn't close)
        - <think>...</think> complete: 0.3
        - <answer> present but no </answer>: 0.1
        - <answer>...</answer> complete: 0.3
        - Both complete (<think>...</think> + <answer>...</answer>): 1.0 (bonus)
        """
        rewards = []
        
        for completion in completions:
            completion = self._unwrap_completion(completion)
            score = 0.0
            
            has_think_open = '<think>' in completion
            has_think_close = '</think>' in completion
            has_answer_open = '<answer>' in completion
            has_answer_close = '</answer>' in completion
            
            # Thinking tags
            if has_think_open and has_think_close:
                score += 0.3
            elif has_think_open:
                score += 0.1  # partial credit — encourages closing the tag
            
            # Answer tags
            if has_answer_open and has_answer_close:
                score += 0.3
            elif has_answer_open:
                score += 0.1
            
            # Bonus for fully correct format
            if has_think_open and has_think_close and has_answer_open and has_answer_close:
                score = 1.0
            
            rewards.append(score)
        
        return rewards
    
    # ------------------------------------------------------------------
    # Answer extraction
    # ------------------------------------------------------------------
    
    def _extract_answer(self, text: str) -> Optional[str]:
        """
        Extract answer from model response.
        
        Priority order:
        1. <answer>...</answer> tags (DeepSeek-R1 format)
        2. #### marker (GSM8K format)
        3. Last non-empty line (fallback)
        """
        # 1. Try <answer> tags
        match = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL)
        if match:
            return match.group(1).strip()
        
        # 2. Try #### marker
        if '####' in text:
            parts = text.split('####')
            if len(parts) > 1:
                answer = parts[-1].strip()
                # Take first line of answer (in case of trailing text)
                answer = answer.split('\n')[0].strip()
                return answer
        
        # 3. Fallback: last non-empty line
        lines = text.strip().split('\n')
        for line in reversed(lines):
            line = line.strip()
            if line and not line.startswith('<'):  # skip stray tags
                return line
        
        return None
    
    # ------------------------------------------------------------------
    # Task-specific verification
    # ------------------------------------------------------------------
    
    def _verify_countdown(
        self, extracted: str, truth: str, metadata: Any
    ) -> float:
        """
        Verify countdown answer by evaluating the expression.
        
        The answer should be an arithmetic expression that equals the target.
        """
        # Get target from metadata if available
        target = None
        if isinstance(metadata, dict):
            target = metadata.get('target')
        
        # Try evaluating the expression
        try:
            # Safety: only allow digits, operators, parens, spaces
            cleaned = extracted.strip()
            if re.match(r'^[\d\s\+\-\*\/\(\)\.]+$', cleaned):
                result = eval(cleaned)
                if target is not None and result == target:
                    return 1.0
                # Fallback: if no target in metadata, compare to truth expression
                if target is None:
                    truth_result = eval(truth)
                    if result == truth_result:
                        return 1.0
        except Exception:
            pass
        
        # Fallback: exact string match with truth
        if self._normalize_answer(extracted) == self._normalize_answer(truth):
            return 1.0
        
        return 0.0
    
    def _verify_generic(self, extracted: str, truth: str) -> float:
        """
        Generic verification via normalized string comparison.
        
        Works for most tasks. Tries numeric comparison first,
        falls back to string comparison.
        """
        ext_norm = self._normalize_answer(extracted)
        truth_norm = self._normalize_answer(truth)
        
        # Exact match after normalization
        if ext_norm == truth_norm:
            return 1.0
        
        # Try numeric comparison
        try:
            ext_num = float(ext_norm.replace(',', ''))
            truth_num = float(truth_norm.replace(',', ''))
            if abs(ext_num - truth_num) < 1e-6:
                return 1.0
        except (ValueError, TypeError):
            pass
        
        return 0.0
    
    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    
    def _unwrap_completion(self, completion) -> str:
        """Handle different completion formats from TRL."""
        if isinstance(completion, list):
            if len(completion) > 0 and isinstance(completion[0], dict):
                completion = completion[-1].get('content', '')
            else:
                completion = ' '.join(str(x) for x in completion)
        return str(completion) if completion is not None else ''
    
    def _normalize_answer(self, answer: str) -> str:
        """Normalize answer for comparison."""
        answer = answer.strip().lower()
        answer = answer.replace('$', '').replace('%', '')
        answer = answer.rstrip('.')
        # Collapse whitespace
        answer = re.sub(r'\s+', ' ', answer)
        
        # Try to parse as number
        try:
            num = float(answer.replace(',', ''))
            if num == int(num):
                return str(int(num))
            return f"{num:.4f}"
        except (ValueError, TypeError):
            return answer
