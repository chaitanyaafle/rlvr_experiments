#!/usr/bin/env bash
# ============================================================================
# HP Sweep Launcher for Lightning AI Studios
# ============================================================================
# Usage:
#   ./run_sweep.sh setup              # First-time setup (install deps)
#   ./run_sweep.sh A1                 # Run single config (canonical baseline)
#   ./run_sweep.sh all                # Run all 11 configs sequentially
#   ./run_sweep.sh tensorboard        # Start TensorBoard
#
# Configs live in configs/hp_sweep/ and outputs go to results/hp_sweep/
# ============================================================================

set -euo pipefail

SWEEP_DIR="configs/hp_sweep"
RESULTS_DIR="results/hp_sweep"

# Map short names to config files
declare -A CONFIGS=(
    [A1]="A1_grpo_baseline"
    [A2]="A2_dr_grpo"
    [A3]="A3_dapo"
    [A4]="A4_cispo"
    [B1]="B1_lr_5e5"
    [B3]="B3_lr_5e7"
    [C1]="C1_numgen_4"
    [C3]="C3_numgen_16"
    [D1]="D1_beta_0"
    [D2]="D2_beta_001"
    [D3]="D3_beta_01"
)

run_config() {
    local name="$1"
    local config_file="${SWEEP_DIR}/${CONFIGS[$name]}.yaml"

    if [[ ! -f "$config_file" ]]; then
        echo "ERROR: Config not found: $config_file"
        exit 1
    fi

    echo ""
    echo "================================================================"
    echo "  RUNNING: $name (${CONFIGS[$name]})"
    echo "  Config:  $config_file"
    echo "  Output:  ${RESULTS_DIR}/${CONFIGS[$name]}/"
    echo "  Started: $(date)"
    echo "================================================================"
    echo ""

    python train_grpo.py "$config_file"

    echo ""
    echo "  FINISHED: $name at $(date)"
    echo "================================================================"
}

case "${1:-help}" in
    setup)
        echo "=== Installing dependencies ==="
        pip install -r requirements.txt
        echo ""
        echo "=== Pre-downloading model (cached for all runs) ==="
        python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
model_name = 'Qwen/Qwen3-4B-Instruct-2507'
print(f'Downloading {model_name}...')
AutoTokenizer.from_pretrained(model_name)
AutoModelForCausalLM.from_pretrained(model_name, torch_dtype='bfloat16')
print('Model cached successfully.')
"
        echo ""
        echo "=== Setup complete ==="
        ;;

    tensorboard)
        echo "Starting TensorBoard on ${RESULTS_DIR}..."
        tensorboard --logdir="${RESULTS_DIR}" --bind_all --port 6006 &
        echo "TensorBoard running at http://localhost:6006"
        echo "Lightning AI will auto-detect and show it in the UI."
        ;;

    all)
        echo "=== Running all 11 HP sweep configs ==="
        echo "Estimated total time: ~5-6 hours on H100"
        echo ""
        for name in A1 A2 A3 A4 B1 B3 C1 C3 D1 D2 D3; do
            run_config "$name"
        done
        echo ""
        echo "=== ALL RUNS COMPLETE ==="
        ;;

    A1|A2|A3|A4|B1|B3|C1|C3|D1|D2|D3)
        run_config "$1"
        ;;

    help|*)
        echo "HP Sweep Launcher"
        echo ""
        echo "Usage: ./run_sweep.sh <command>"
        echo ""
        echo "Commands:"
        echo "  setup        Install deps + pre-download model"
        echo "  tensorboard  Start TensorBoard for results"
        echo "  A1           Canonical baseline (GRPO, lr=5e-6, ng=8)"
        echo "  A2           DR-GRPO loss"
        echo "  A3           DAPO loss"
        echo "  A4           CISPO loss"
        echo "  B1           LR = 5e-5"
        echo "  B3           LR = 5e-7"
        echo "  C1           num_generations = 4"
        echo "  C3           num_generations = 16"
        echo "  D1           beta = 0.0 (no KL)"
        echo "  D2           beta = 0.01"
        echo "  D3           beta = 0.1"
        echo "  all          Run all 11 configs sequentially"
        ;;
esac
