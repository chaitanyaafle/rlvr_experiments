"""
Arithmetic Multi-Task Environment for GRPO Training

Supports multiple arithmetic tasks from reasoning-gym:
- basic_arithmetic
- chain_sum
- leg_counting
- fraction_simplification

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


class ArithmeticEnvironment:
    """
    Multi-task arithmetic environment for GRPO training.
    
    Generates prompts and provides reward functions for arithmetic tasks.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.env_config = config.get('environment', {})
        self.tasks = self.env_config.get('tasks', ['basic_arithmetic', 'chain_sum'])
        self.samples_per_task = self.env_config.get('samples_per_task', 500)
        
    def get_dataset(self, config: Dict[str, Any]) -> Dataset:
        """Generate dataset with prompts and ground truth answers."""
        if not REASONING_GYM_AVAILABLE:
            raise ImportError("reasoning-gym required. Install with: pip install reasoning-gym")
        
        all_data = []
        
        for task_name in self.tasks:
            print(f"Generating {self.samples_per_task} samples for {task_name}...")
            
            dataset = reasoning_gym.create_dataset(
                task_name,
                size=self.samples_per_task,
                seed=config.get('seed', 42)
            )
            
            for item in dataset:
                all_data.append({
                    "prompt": [{"role": "user", "content": item['question']}],
                    "ground_truth": str(item['answer']),
                    "task": task_name,
                    "metadata": item.get('metadata', {})
                })
        
        print(f"Total samples: {len(all_data)}")
        return Dataset.from_list(all_data)
    
    def get_reward_functions(self) -> List[Callable]:
        """Return list of reward functions for GRPO."""
        return [
            self.correctness_reward,
            self.format_reward,
        ]
    
    def correctness_reward(
        self,
        completions,
        ground_truth,
        **kwargs
    ) -> List[float]:
        """
        Reward for correct answers.
        
        Extracts answer after "####" and compares to ground truth.
        TRL may pass completions as strings or lists - handle both.
        """
        rewards = []
        
        for completion, truth in zip(completions, ground_truth):
            # Handle different completion formats from TRL
            if isinstance(completion, list):
                # If it's a list of dicts (messages format), extract content
                if len(completion) > 0 and isinstance(completion[0], dict):
                    completion = completion[-1].get('content', '')
                else:
                    # It's a list of something else, try to join
                    completion = ' '.join(str(x) for x in completion)
            
            # Ensure completion is a string
            completion = str(completion) if completion is not None else ''
            
            # Extract answer after #### marker
            extracted = self._extract_answer(completion)
            
            if extracted is None:
                rewards.append(0.0)
                continue
            
            # Normalize and compare
            extracted_norm = self._normalize_answer(extracted)
            truth_norm = self._normalize_answer(str(truth))
            
            if extracted_norm == truth_norm:
                rewards.append(1.0)
            else:
                # Partial credit for close numerical answers
                try:
                    ext_num = float(extracted_norm.replace(',', ''))
                    truth_num = float(truth_norm.replace(',', ''))
                    rel_error = abs(ext_num - truth_num) / max(abs(truth_num), 1)
                    if rel_error < 0.01:
                        rewards.append(0.5)  # Within 1% - partial credit
                    else:
                        rewards.append(0.0)
                except (ValueError, TypeError):
                    rewards.append(0.0)
        
        return rewards
    
    def format_reward(
        self,
        completions,
        **kwargs
    ) -> List[float]:
        """
        Reward for proper output format.
        
        Checks for <think>...</think> tags and #### answer marker.
        """
        rewards = []
        
        for completion in completions:
            # Handle different completion formats from TRL
            if isinstance(completion, list):
                if len(completion) > 0 and isinstance(completion[0], dict):
                    completion = completion[-1].get('content', '')
                else:
                    completion = ' '.join(str(x) for x in completion)
            
            completion = str(completion) if completion is not None else ''
            
            score = 0.0
            
            # Check for thinking tags
            if '<think>' in completion and '</think>' in completion:
                score += 0.5
            
            # Check for answer marker
            if '####' in completion:
                score += 0.5
            
            rewards.append(score)
        
        return rewards
    
    def _extract_answer(self, text: str) -> Optional[str]:
        """Extract answer after #### marker."""
        # Try to find #### marker
        if '####' in text:
            parts = text.split('####')
            if len(parts) > 1:
                answer = parts[-1].strip()
                # Clean up - take first word/number
                answer = answer.split()[0] if answer.split() else answer
                return answer
        
        # Fallback: try to find last number
        numbers = re.findall(r'-?\d+\.?\d*', text)
        if numbers:
            return numbers[-1]
        
        return None
    
    def _normalize_answer(self, answer: str) -> str:
        """Normalize answer for comparison."""
        # Remove common formatting
        answer = answer.strip().lower()
        answer = answer.replace('$', '').replace('%', '')
        answer = answer.rstrip('.')
        
        # Try to parse as number and format consistently
        try:
            num = float(answer.replace(',', ''))
            if num == int(num):
                return str(int(num))
            return f"{num:.4f}"
        except (ValueError, TypeError):
            return answer
