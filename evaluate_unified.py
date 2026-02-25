"""Unified evaluation framework for RLVR experiments.

Evaluates a model checkpoint against one or more environments defined in a
config file, then prints a summary table and saves JSON results.

Usage:
    python evaluate_unified.py configs/eval_unified.yaml
    python evaluate_unified.py configs/eval_unified.yaml --checkpoint path/to/ckpt
    python evaluate_unified.py configs/eval_unified.yaml --output my_results/
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml
from tqdm import tqdm

from environments import load_environment


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Model runner
# ---------------------------------------------------------------------------

class ModelRunner:
    """Loads a model + tokenizer once and runs greedy/sampled generation."""

    def __init__(self, model_path: str, model_cfg: dict, gen_cfg: dict):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.gen_cfg = gen_cfg

        print(f"Loading tokenizer: {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        print(f"Loading model:     {model_path}")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=model_cfg.get("torch_dtype", "auto"),
            device_map=model_cfg.get("device_map", "auto"),
        )
        self.model.eval()

    def generate(self, messages: list) -> str:
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)

        gen_kwargs: dict = {
            "max_new_tokens": self.gen_cfg.get("max_new_tokens", 1024),
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        temperature = self.gen_cfg.get("temperature")
        if temperature is not None:
            gen_kwargs["temperature"] = float(temperature)
            gen_kwargs["do_sample"] = True

        with torch.no_grad():
            output_ids = self.model.generate(**inputs, **gen_kwargs)

        return self.tokenizer.decode(
            output_ids[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
        )


# ---------------------------------------------------------------------------
# Per-environment evaluation
# ---------------------------------------------------------------------------

def _build_env_config(env_entry: dict) -> dict:
    """Wrap a single environments-list entry into the full config dict that
    load_environment() and get_dataset() expect.

    For reasoning_gym tasks the entry maps directly to config['environment'].
    For 'gsm8k' we also fill in config['data'] from optional keys in the entry
    so the legacy GSM8KEnvironment keeps working.
    """
    name = env_entry.get("name", "")

    full_config: dict = {"environment": dict(env_entry)}

    if name == "gsm8k":
        full_config["data"] = {
            "dataset_name": env_entry.get("dataset_name", "openai/gsm8k"),
            "subset":        env_entry.get("subset", "main"),
            "split":         env_entry.get("split", "test"),
            "prompt_column": env_entry.get("prompt_column", "question"),
            "answer_column": env_entry.get("answer_column", "answer"),
        }
        if "system_prompt" in env_entry:
            full_config["system_prompt"] = env_entry["system_prompt"]

    return full_config


def evaluate_environment(
    runner: ModelRunner,
    env_entry: dict,
    eval_cfg: dict,
) -> tuple[dict, list]:
    """Evaluate a single environment; returns (metrics_dict, sample_list)."""

    env_name = env_entry["name"]
    full_config = _build_env_config(env_entry)

    env = load_environment(full_config)
    dataset = env.get_dataset(full_config)
    reward_fns = env.get_reward_functions()

    num_samples = eval_cfg.get("num_samples", len(dataset))
    num_samples = min(num_samples, len(dataset))
    dataset = dataset.select(range(num_samples))

    totals = {"format": 0.0, "accuracy": 0.0, "n": 0}
    samples = []

    for example in tqdm(dataset, desc=env_name, total=num_samples):
        completion = runner.generate(example["prompt"])
        formatted = [[{"content": completion}]]

        kwargs = {
            "answer":   [example["answer"]],
            "metadata": [example.get("metadata", {})],
        }

        fmt_score = reward_fns[0](formatted, **kwargs)[0] if len(reward_fns) > 0 else 0.0
        acc_score = reward_fns[1](formatted, **kwargs)[0] if len(reward_fns) > 1 else 0.0

        totals["format"]    += fmt_score
        totals["accuracy"]  += acc_score
        totals["n"]         += 1

        samples.append({
            "prompt":       example["prompt"],
            "completion":   completion,
            "ground_truth": example["answer"],
            "format_score": fmt_score,
            "accuracy_score": acc_score,
        })

    n = totals["n"]
    metrics = {
        "env":              env_name,
        "num_samples":      n,
        "accuracy":         totals["accuracy"] / n if n else 0.0,
        "format_compliance": totals["format"]  / n if n else 0.0,
    }
    return metrics, samples


# ---------------------------------------------------------------------------
# Summary printing
# ---------------------------------------------------------------------------

def print_summary(model_path: str, all_metrics: list[dict]) -> None:
    col_env = 28
    col_num = 9
    col_acc = 10
    col_fmt = 10
    header = (
        f"{'Environment':<{col_env}}"
        f"{'Samples':>{col_num}}"
        f"{'Accuracy':>{col_acc}}"
        f"{'Format':>{col_fmt}}"
    )
    sep = "-" * len(header)
    print(f"\n{'='*len(header)}")
    print(f"EVALUATION SUMMARY  —  {model_path}")
    print(f"{'='*len(header)}")
    print(header)
    print(sep)
    for m in all_metrics:
        print(
            f"{m['env']:<{col_env}}"
            f"{m['num_samples']:>{col_num}}"
            f"{m['accuracy']:>{col_acc}.4f}"
            f"{m['format_compliance']:>{col_fmt}.4f}"
        )
    print(sep)
    if len(all_metrics) > 1:
        avg_acc = sum(m["accuracy"] for m in all_metrics) / len(all_metrics)
        avg_fmt = sum(m["format_compliance"] for m in all_metrics) / len(all_metrics)
        total_n = sum(m["num_samples"] for m in all_metrics)
        print(
            f"{'AVERAGE':<{col_env}}"
            f"{total_n:>{col_num}}"
            f"{avg_acc:>{col_acc}.4f}"
            f"{avg_fmt:>{col_fmt}.4f}"
        )
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Unified RLVR evaluator")
    parser.add_argument("config", help="Path to eval config YAML")
    parser.add_argument(
        "--checkpoint", help="Override model checkpoint (model.name_or_path in config)"
    )
    parser.add_argument(
        "--output", help="Override output directory (evaluation.output_dir in config)"
    )
    args = parser.parse_args()

    config = load_config(args.config)

    model_cfg   = config["model"]
    model_path  = args.checkpoint or model_cfg["name_or_path"]
    gen_cfg     = config.get("generation", {})
    eval_cfg    = config.get("evaluation", {})
    env_list    = config.get("environments", [])

    if not env_list:
        print("ERROR: 'environments' list is empty or missing in config.", file=sys.stderr)
        sys.exit(1)

    output_dir = Path(args.output or eval_cfg.get("output_dir", "eval_results"))
    output_dir.mkdir(parents=True, exist_ok=True)

    runner = ModelRunner(model_path, model_cfg, gen_cfg)

    all_metrics: list[dict] = []
    all_samples: dict[str, list] = {}

    for env_entry in env_list:
        env_name = env_entry.get("name", "unknown")
        print(f"\n{'='*60}")
        print(f"Evaluating: {env_name}")
        print(f"{'='*60}")
        try:
            metrics, samples = evaluate_environment(runner, env_entry, eval_cfg)
        except Exception as exc:
            print(f"  ERROR evaluating {env_name}: {exc}", file=sys.stderr)
            continue

        all_metrics.append(metrics)
        all_samples[env_name] = samples

        print(f"  accuracy:          {metrics['accuracy']:.4f}")
        print(f"  format_compliance: {metrics['format_compliance']:.4f}")

    if not all_metrics:
        print("No environments were successfully evaluated.", file=sys.stderr)
        sys.exit(1)

    print_summary(model_path, all_metrics)

    # Always save metrics
    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump({"model": model_path, "metrics": all_metrics}, f, indent=2)
    print(f"Metrics saved to {metrics_path}")

    # Optionally save full sample details
    if eval_cfg.get("save_samples", False):
        samples_path = output_dir / "samples.json"
        with open(samples_path, "w") as f:
            json.dump(all_samples, f, indent=2, default=str)
        print(f"Samples saved to {samples_path}")


if __name__ == "__main__":
    main()
