
import subprocess
import sys
import yaml
import os

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def run_command(cmd):
    print(f"Running: {cmd}")
    ret = subprocess.call(cmd, shell=True)
    if ret != 0:
        print(f"Error running command: {cmd}")
        sys.exit(ret)

def main():
    if len(sys.argv) < 2:
        print("Usage: python run_chain.py <config_path>")
        sys.exit(1)
        
    config_path = sys.argv[1]
    config = load_config(config_path)
    
    # 1. Generate SFT Data
    print("=== Step 1: Generate SFT Data ===")
    run_command(f"python generate_sft_data.py {config_path}")
    
    # 2. Run SFT
    print("=== Step 2: Run SFT ===")
    run_command(f"python train_sft.py {config_path}")
    
    # 3. Run GRPO (RL)
    # We need to update the model path to point to the SFT output
    # But we can't easily modify the config file on the fly without writing a temp one.
    # Alternatively, we can assume train_grpo.py takes a flag, but it currently only takes config.
    # So we will create a temporary config for the RL stage.
    
    sft_output_dir = config['sft']['output_dir']
    
    # Read config again to be safe
    rl_config = config.copy()
    rl_config['model']['name_or_path'] = sft_output_dir
    
    rl_config_path = config_path.replace(".yaml", "_rl_stage.yaml")
    with open(rl_config_path, 'w') as f:
        yaml.dump(rl_config, f)
        
    print(f"=== Step 3: Run GRPO (RL) starting from {sft_output_dir} ===")
    run_command(f"python train_grpo.py {rl_config_path}")
    
    print("=== Configuration Chain Complete ===")

if __name__ == "__main__":
    main()
