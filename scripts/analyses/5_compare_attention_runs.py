#!/usr/bin/env python3

import os
import logging
import torch
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer

# ------------------------------------------------------------------------
# 1. Configuration and Setup
# ------------------------------------------------------------------------
RUN_NAMES = globals().get("RUN_NAMES", ["nice", "mean"])
BASE_OUTPUT_DIR = globals().get("BASE_OUTPUT_DIR", "output/extractions")
ANALYSIS_OUTPUT_DIR = globals().get("ANALYSIS_OUTPUT_DIR", "analysis/compare_runs")
MODEL_NAME_FOR_DECODING = globals().get("MODEL_NAME_FOR_DECODING", "google/gemma-2-2b")

LAYERS_TO_ANALYZE = globals().get("LAYERS_TO_ANALYZE", [0, 5, 10, 15, 20, 25])
HEADS_TO_ANALYZE = globals().get("HEADS_TO_ANALYZE", None)
TOKEN_OF_INTERESTS = globals().get("TOKEN_OF_INTERESTS", ["please", "thank"])  # multiple tokens

os.makedirs(ANALYSIS_OUTPUT_DIR, exist_ok=True)

LOG_FILE = globals().get("LOG_FILE", "logs/compare_runs.log")
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

# ------------------------------------------------------------------------
# 1a. Set up Logging
# ------------------------------------------------------------------------
logger = logging.getLogger("CompareLogger")
logger.setLevel(logging.DEBUG)

file_handler = logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8")
file_handler.setLevel(logging.DEBUG)
file_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
file_handler.setFormatter(file_formatter)
logger.addHandler(file_handler)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter("[%(levelname)s] %(message)s")
console_handler.setFormatter(console_formatter)
logger.addHandler(console_handler)

logger.info("=== Starting side-by-side run comparison ===")
logger.info(f"Run names: {RUN_NAMES}")
logger.info(f"Base output dir: {BASE_OUTPUT_DIR}")
logger.info(f"Analysis output dir: {ANALYSIS_OUTPUT_DIR}")
logger.info(f"Layers to analyze: {LAYERS_TO_ANALYZE}")
logger.info(f"Tokens of interest: {TOKEN_OF_INTERESTS}")

# ------------------------------------------------------------------------
# 2. Load Tokenizer
# ------------------------------------------------------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME_FOR_DECODING)
if tokenizer.pad_token is None and tokenizer.eos_token is not None:
    tokenizer.pad_token = tokenizer.eos_token

# ------------------------------------------------------------------------
# 3. Helper functions
# ------------------------------------------------------------------------
def load_all_samples_for_run(run_name):
    """
    Loads all `.pt` activation dictionaries from the subdir.
    Returns a list of sample dicts:
      {
        'input_ids': Tensor(seq_len),
        'attentions': {
           'layer_0': Tensor(num_heads, seq_len, seq_len),
           ...
        },
        'final_prediction': str
      }
    """
    run_dir = os.path.join(BASE_OUTPUT_DIR, run_name)
    pt_files = sorted(glob.glob(os.path.join(run_dir, "*.pt")))
    all_samples = []
    for pt_file in pt_files:
        data = torch.load(pt_file)  # dictionary with batch data
        # Might contain multiple samples in one dictionary
        if "input_ids" not in data:
            logger.warning(f"Skipping corrupted file: {pt_file}")
            continue
        batch_size = data["input_ids"].size(0)
        for i in range(batch_size):
            all_samples.append({
                "input_ids": data["input_ids"][i],
                "attentions": {k: v[i] for k, v in data["attentions"].items()},
                "final_prediction": data["final_predictions"][i]
            })
    return all_samples

def positions_of_token(input_ids, token_str):
    """Find positions of a given string in the single-sample input_ids."""
    decoded_tokens = [tokenizer.decode([tid], skip_special_tokens=True) for tid in input_ids]
    return [i for i, dtok in enumerate(decoded_tokens) if token_str.lower() in dtok.lower()]

def compute_average_attention_to_positions(attn_matrix, positions):
    """
    attn_matrix: shape [num_heads, seq_len, seq_len],
                 where attn[h, q, k] is how much token q attends to token k.
    positions: list of key positions to measure how much attention is paid to them.
    We'll average over all heads, over all query tokens, and over all positions in `positions`.
    """
    if len(positions) == 0:
        return 0.0
    num_heads, seq_len, _ = attn_matrix.shape
    if HEADS_TO_ANALYZE is not None:
        heads_list = HEADS_TO_ANALYZE
    else:
        heads_list = range(num_heads)

    sums = 0.0
    count = 0
    for h in heads_list:
        for pos_i in positions:
            # average over all query tokens
            # attn_matrix[h, q, pos_i] for q in 0..seq_len
            sums += attn_matrix[h, :, pos_i].sum().item()
            count += seq_len
    if count == 0:
        return 0.0
    return sums / count

# ------------------------------------------------------------------------
# 4. Load and aggregate each run
# ------------------------------------------------------------------------
all_run_results = []

for run_name in RUN_NAMES:
    samples = load_all_samples_for_run(run_name)
    logger.info(f"Loaded {len(samples)} samples for run: {run_name}")

    # We create a DataFrame or list of analysis results for each sample
    run_rows = []
    for idx, sample in enumerate(samples):
        input_ids = sample["input_ids"]
        attn_dict = sample["attentions"]
        final_pred = sample["final_prediction"]

        # Optionally, you can loop over each layer we care about, and accumulate stats.
        # We'll just do a single "mean across the LAYERS_TO_ANALYZE" for demonstration.
        layer_attn_vals_by_token = {tok: [] for tok in TOKEN_OF_INTERESTS}

        for layer_key, attn_matrix in attn_dict.items():
            # layer_key = "layer_0", "layer_5", ...
            layer_idx = int(layer_key.split("_")[1])
            if layer_idx not in LAYERS_TO_ANALYZE:
                continue
            # For each token of interest
            for tok in TOKEN_OF_INTERESTS:
                pos_list = positions_of_token(input_ids, tok)
                avg_attn_this_layer = compute_average_attention_to_positions(attn_matrix, pos_list)
                layer_attn_vals_by_token[tok].append(avg_attn_this_layer)

        # Now average across layers
        row_data = {"run_name": run_name, "sample_idx": idx, "final_pred": final_pred}
        for tok in TOKEN_OF_INTERESTS:
            if layer_attn_vals_by_token[tok]:
                row_data[f"avg_attention_{tok}"] = float(
                    sum(layer_attn_vals_by_token[tok]) / len(layer_attn_vals_by_token[tok])
                )
            else:
                row_data[f"avg_attention_{tok}"] = 0.0
        run_rows.append(row_data)

    df_run = pd.DataFrame(run_rows)
    all_run_results.append(df_run)

df_combined = pd.concat(all_run_results, ignore_index=True)
analysis_csv_path = os.path.join(ANALYSIS_OUTPUT_DIR, "combined_attention_analysis.csv")
df_combined.to_csv(analysis_csv_path, index=False)
logger.info(f"Saved combined run analysis to {analysis_csv_path}")

# ------------------------------------------------------------------------
# 5. Simple Comparison and a Quick Plot
# ------------------------------------------------------------------------
# As a simple demonstration, let's group by run_name and compute the mean for each token_of_interest
grouped_means = df_combined.groupby("run_name").mean(numeric_only=True)

logger.info("=== Mean attention to tokens of interest, by run ===")
logger.info(f"\n{grouped_means}\n")

# We can also do a quick bar plot for each token_of_interest
for tok in TOKEN_OF_INTERESTS:
    plt.figure()
    sub_df = grouped_means[f"avg_attention_{tok}"]
    sub_df.plot(kind="bar", title=f"Mean Attention to '{tok}' by Run")
    plot_file = os.path.join(ANALYSIS_OUTPUT_DIR, f"mean_attention_{tok}.png")
    plt.savefig(plot_file, bbox_inches="tight")
    logger.info(f"Saved bar plot for '{tok}' to {plot_file}")

logger.info("Comparison script complete.")

