from abc import ABC, abstractmethod

class BaseEnvironment(ABC):
    @abstractmethod
    def get_dataset(self, config):
        """
        Loads and processes the dataset.
        Returns a huggingface Dataset object.
        """
        pass

    @abstractmethod
    def get_reward_functions(self):
        """
        Returns a list of reward functions.
        Each reward function should take (completions, **kwargs) and return a list of scores.
        """
        pass
    
    @abstractmethod
    def get_system_prompt(self):
        """
        Returns the system prompt to be used for this environment.
        """
        pass
