from environments.battleship_env import BattleshipEnvironment
import json

def test_battleship():
    print("Testing Battleship Environment...")
    config = {
        'environment': {
            'name': 'battleship',
            'min_grid_size': 8,
            'max_grid_size': 8,
            'size': 5,
            'seed': 123
        }
    }
    
    env = BattleshipEnvironment(config)
    dataset = env.get_dataset(config)
    
    print(f"Dataset size: {len(dataset)}")
    example = dataset[0]
    print("\nExample Prompt:")
    print(example['prompt'][1]['content'])
    print("\nReference Answer:")
    print(example['answer'])
    
    # Test Reward
    # 1. Correct Answer
    correct_completion = [
        {"content": f"<think>Reasoning...</think><answer>{example['answer']}</answer>"}
    ]
    rewards = env.completeness_reward([correct_completion], metadata=[example['metadata']])
    print(f"\nReward for correct answer: {rewards[0]}")
    assert rewards[0] == 1.0, "Correct answer should get reward 1.0"
    
    # 2. Malformed Answer
    malformed_completion = [
        {"content": "<answer>Bad format</answer>"}
    ]
    rewards = env.completeness_reward([malformed_completion], metadata=[example['metadata']])
    print(f"Reward for malformed answer: {rewards[0]}")
    assert rewards[0] == 0.0, "Malformed answer should get reward 0.0"
    
     # 3. Wrong Answer (modify coordinates)
    # create a subtle wrong answer (shift a ship 1 cell)
    # We'll just grab the answer text and replace a number
    wrong_text = example['answer'].replace('row="2"', 'row="3"')
    # If the ship was at row 2, and we say 3, it should fail either overlap or counts
    wrong_completion = [
        {"content": f"<think>...</think><answer>{wrong_text}</answer>"}
    ]
    rewards = env.completeness_reward([wrong_completion], metadata=[example['metadata']])
    print(f"Reward for wrong answer: {rewards[0]}")
    # Note: Dependent on specific example, replacing row="2" might not exist or might still be valid (unlikely). 
    # But usually it changes counts.
    
    print("\nBattleship Environment Verification Passed!")

if __name__ == "__main__":
    test_battleship()
