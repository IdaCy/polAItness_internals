# file: functions/hpc_attention_analysis.py

import os
import re
import torch
import glob
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from transformers import AutoTokenizer

##############################################################################
# Helper functions & configuration defaults
##############################################################################
def get_default_config(prep_model):
    # Defaults based on prep_model
    prep_dir = os.path.join("output", "extractions", prep_model)
    all_prompt_types = "explained insulting nicer1k normal1k pure reduced1k shortest1k urgent1k"
    all_types_list = [d.strip() for d in re.split(r"[, \n]+", all_prompt_types) if d.strip()]
    
    # Build list of directories from prep_dir and each prompt type
    all_prompt_dirs = " ".join([os.path.join(prep_dir, d) for d in all_types_list])
    all_dirs_list = [d.strip() for d in re.split(r"[, \n]+", all_prompt_dirs) if d.strip()]
    
    # Create underscore-separated string for output naming
    all_dirs_joined = "_".join(all_types_list)
    model_name = "google/" + prep_model
    output_dir = os.path.join("output", "attention", prep_model, all_dirs_joined)
    os.makedirs(output_dir, exist_ok=True)
    
    # Read NICE_WORDS from CSV (assumes CSV exists)
    nice_words_csv = os.path.join("analyses", "nice_words.csv")
    if os.path.exists(nice_words_csv):
        with open(nice_words_csv, "r", encoding="utf-8") as f:
            nice_words = [line.strip() for line in f if line.strip()]
    else:
        nice_words = []
    
    # Default: use robust detection
    robust_nice_detection = True
    
    return {
        "prep_model": prep_model,
        "prep_dir": prep_dir,
        "all_prompt_types": all_prompt_types,
        "all_types_list": all_types_list,
        "all_dirs_list": all_dirs_list,
        "all_dirs_joined": all_dirs_joined,
        "model_name": model_name,
        "output_dir": output_dir,
        "nice_words": nice_words,
        "robust_nice_detection": robust_nice_detection,
    }

def get_pt_files(directory):
    """Return sorted list of .pt files in a directory."""
    files = glob.glob(os.path.join(directory, "activations_*.pt"))
    return sorted(files)

def load_activations(pt_file):
    """Loads a single .pt file."""
    return torch.load(pt_file)

# --- "Nice" token detection ---
def find_nice_tokens_subword(decoded_tokens, nice_words):
    nice_mask = [False] * len(decoded_tokens)
    word_buffer = []
    word_start_idx = 0

    def flush_word(buffer, start_idx, end_idx):
        merged = "".join(buffer).replace("▁", "").replace("Ġ", "").lower()
        merged_clean = re.sub(r"\W+", "", merged)
        for w in nice_words:
            w_clean = re.sub(r"\W+", "", w.lower())
            if merged_clean == w_clean:
                for idx in range(start_idx, end_idx):
                    nice_mask[idx] = True

    def is_wordish(tok):
        return len(re.sub(r"\W+", "", tok)) > 0

    for i, tok in enumerate(decoded_tokens):
        if is_wordish(tok):
            word_buffer.append(tok)
        else:
            if word_buffer:
                flush_word(word_buffer, word_start_idx, i)
                word_buffer = []
            word_start_idx = i + 1

    if word_buffer:
        flush_word(word_buffer, word_start_idx, len(decoded_tokens))
    return nice_mask

def identify_token_ranges_robust(token_ids, tokenizer, nice_words):
    decoded_tokens = tokenizer.convert_ids_to_tokens(token_ids, skip_special_tokens=False)
    colon_positions = [i for i, tok in enumerate(decoded_tokens) if ":" in tok]
    first_colon = colon_positions[0] if colon_positions else None
    nice_mask = find_nice_tokens_subword(decoded_tokens, nice_words)
    math_mask = [False] * len(decoded_tokens)
    if first_colon is not None and first_colon < len(decoded_tokens) - 1:
        for idx in range(first_colon+1, len(decoded_tokens)):
            math_mask[idx] = True
    return {
        "decoded_tokens": decoded_tokens,
        "nice_mask": nice_mask,
        "math_mask": math_mask,
        "colon_pos": first_colon
    }

def identify_token_ranges_naive(token_ids, tokenizer, nice_words):
    decoded_tokens = tokenizer.convert_ids_to_tokens(token_ids, skip_special_tokens=False)
    decoded_tokens_lower = [t.lower() for t in decoded_tokens]
    colon_positions = [i for i, tok in enumerate(decoded_tokens) if ":" in tok]
    first_colon_pos = colon_positions[0] if colon_positions else None
    nice_mask = [False] * len(decoded_tokens)
    for i, tok_l in enumerate(decoded_tokens_lower):
        for w in nice_words:
            if w in tok_l:
                nice_mask[i] = True
                break
    math_mask = [False] * len(decoded_tokens)
    if first_colon_pos is not None:
        for i in range(first_colon_pos+1, len(decoded_tokens)):
            math_mask[i] = True
    return {
        "decoded_tokens": decoded_tokens,
        "nice_mask": nice_mask,
        "math_mask": math_mask,
        "colon_pos": first_colon_pos
    }

def make_prompt_task_masks(colon_pos, seq_len):
    prompt_mask = np.zeros(seq_len, dtype=bool)
    task_mask = np.zeros(seq_len, dtype=bool)
    if colon_pos is not None and colon_pos < seq_len:
        prompt_mask[:colon_pos+1] = True
        if colon_pos+1 < seq_len:
            task_mask[colon_pos+1:] = True
    else:
        task_mask[:] = True
    return prompt_mask, task_mask

def extract_attention_stats(attentions, input_ids):
    stats_per_layer = {}
    for layer_name, attn_tensor in attentions.items():
        attn_sum = attn_tensor.sum(dim=2)
        attn_mean = attn_sum.mean(dim=1)
        stats_per_layer[layer_name] = attn_mean
    layer_list = list(stats_per_layer.values())
    all_layers = torch.stack(layer_list, dim=0)
    avg_across_layers = all_layers.mean(dim=0)
    return {
        "per_layer": stats_per_layer,
        "avg_layers": avg_across_layers,
        "input_ids": input_ids
    }

def extract_per_layer_head_stats(attentions, input_ids):
    results = []
    sorted_layers = sorted(attentions.keys(), key=lambda x: int(x.split('_')[1]) if '_' in x else 0)
    B, S = input_ids.shape
    for layer_name in sorted_layers:
        attn_tensor = attentions[layer_name]
        attn_sum = attn_tensor.sum(dim=3)
        attn_mean = attn_tensor.mean(dim=3)
        layer_idx = int(layer_name.split('_')[1]) if '_' in layer_name else layer_name
        b_size, h_size, s_len, s_len2 = attn_tensor.shape
        if b_size != B or s_len != S or s_len2 != S:
            raise ValueError("Mismatch shape...")
        for b_idx in range(B):
            for h_idx in range(h_size):
                row = {
                    "layer": layer_idx,
                    "head": h_idx,
                    "batch_idx": b_idx,
                    "attn_sum": attn_sum[b_idx, h_idx].cpu().tolist(),
                    "attn_mean": attn_mean[b_idx, h_idx].cpu().tolist()
                }
                results.append(row)
    return results

##############################################################################
# Main function: run_attention_extraction
##############################################################################
def run_attention_extraction(prep_model=None,
                             prep_dir=None,
                             all_prompt_types=None,
                             output_dir=None,
                             nice_words_csv=None,
                             robust_nice_detection=True):
    """
    Runs the full attention extraction analysis. Configuration parameters can be overridden.
    
    Parameters:
      - prep_model: string, e.g. "gemma-2-9b-it". Default from env or "gemma-2-9b-it".
      - prep_dir: directory where extraction .pt files are located. Default: "output/extractions/<prep_model>".
      - all_prompt_types: space-separated string of prompt type names. Default:
           "explained insulting nicer1k normal1k pure reduced1k shortest1k urgent1k"
      - output_dir: directory to save attention analysis outputs. Default:
           "output/attention/<prep_model>/<joined_prompt_types>"
      - nice_words_csv: path to CSV file with nice words. Default: "analyses/nice_words.csv"
      - robust_nice_detection: bool. If True, use robust subword detection.
    
    The function processes all .pt files found in each prompt type directory, computes attention stats,
    saves CSV summaries and boxplots, and prints summary statistics.
    """
    # Set defaults if not provided
    if prep_model is None:
        prep_model = "gemma-2-9b-it"
    if all_prompt_types is None:
        all_prompt_types = "explained insulting nicer1k normal1k pure reduced1k shortest1k urgent1k"
    
    config = get_default_config(prep_model)
    # Override defaults if user supplied prep_dir or output_dir
    if prep_dir is not None:
        config["prep_dir"] = prep_dir
    if output_dir is not None:
        config["output_dir"] = output_dir
    if nice_words_csv is not None:
        config["nice_words_csv"] = nice_words_csv
        if os.path.exists(nice_words_csv):
            with open(nice_words_csv, "r", encoding="utf-8") as f:
                config["nice_words"] = [line.strip() for line in f if line.strip()]
    
    config["robust_nice_detection"] = robust_nice_detection
    
    # Create a tokenizer from MODEL_NAME (using local_files_only so it doesn't re-download)
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"], local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Choose detection function
    identify_fn = (lambda token_ids: identify_token_ranges_robust(token_ids, tokenizer, config["nice_words"])) \
                  if robust_nice_detection else \
                  (lambda token_ids: identify_token_ranges_naive(token_ids, tokenizer, config["nice_words"]))
    
    # Prepare output directory
    os.makedirs(config["output_dir"], exist_ok=True)
    
    # Process each prompt type directory
    all_attention_records = []
    all_plh_records = []
    for ddir in config["all_dirs_list"]:
        pt_files = get_pt_files(ddir)
        prompt_type = os.path.basename(ddir.rstrip("/"))
        print(f"Found {len(pt_files)} .pt files in: {ddir} (prompt_type='{prompt_type}')")
        
        # Process files in this directory
        for pt_file in pt_files:
            data = load_activations(pt_file)
            # AGGREGATE attention stats
            batch_stats = extract_attention_stats(data["attentions"], data["input_ids"])
            B, S = batch_stats["avg_layers"].shape
            for i in range(B):
                attn_vals = batch_stats["avg_layers"][i].float().cpu().numpy()  # shape [S]
                masks_info = identify_fn(data["input_ids"][i].tolist())
                colon_pos = masks_info["colon_pos"]
                prompt_mask, task_mask = make_prompt_task_masks(colon_pos, S)
                total_sum = attn_vals.sum()
                prompt_sum = attn_vals[prompt_mask].sum() if prompt_mask.any() else 0.0
                task_sum = attn_vals[task_mask].sum() if task_mask.any() else 0.0
                frac_prompt = (prompt_sum / total_sum) if total_sum > 1e-10 else 0.0
                frac_task = (task_sum / total_sum) if total_sum > 1e-10 else 0.0
                all_attention_records.append({
                    "batch_file": pt_file,
                    "prompt_type": prompt_type,
                    "avg_attn_all": float(attn_vals.mean()),
                    "prompt_sum": float(prompt_sum),
                    "task_sum": float(task_sum),
                    "frac_prompt": float(frac_prompt),
                    "frac_task": float(frac_task),
                })
            # PER-LAYER-HEAD stats
            plh_data = extract_per_layer_head_stats(data["attentions"], data["input_ids"])
            for row in plh_data:
                b_idx = row["batch_idx"]
                colon_pos = identify_fn(data["input_ids"][b_idx].tolist())["colon_pos"]
                prompt_mask, task_mask = make_prompt_task_masks(colon_pos, S)
                attn_sum_array = np.array(row["attn_sum"], dtype=np.float32)
                total_sum = attn_sum_array.sum()
                prompt_sum = attn_sum_array[prompt_mask].sum() if prompt_mask.any() else 0.0
                task_sum = attn_sum_array[task_mask].sum() if task_mask.any() else 0.0
                frac_prompt = (prompt_sum / total_sum) if total_sum > 1e-10 else 0.0
                frac_task = (task_sum / total_sum) if total_sum > 1e-10 else 0.0
                all_plh_records.append({
                    "batch_file": pt_file,
                    "prompt_type": prompt_type,
                    "layer": row["layer"],
                    "head": row["head"],
                    "batch_idx": b_idx,
                    "sum_all": float(total_sum),
                    "sum_prompt": float(prompt_sum),
                    "sum_task": float(task_sum),
                    "frac_prompt": float(frac_prompt),
                    "frac_task": float(frac_task),
                })
    
    # Build a DataFrame from aggregated records
    df_all = pd.DataFrame(all_attention_records)
    agg_mean = df_all.groupby("prompt_type")[["frac_prompt", "frac_task"]].mean()
    print("=== Average fraction of attention on prompt vs. task portion ===")
    print(agg_mean)
    agg_csv = os.path.join(config["output_dir"], "prompt_task_fraction_aggregate.csv")
    agg_mean.to_csv(agg_csv)
    print(f"Saved fraction summary CSV to: {agg_csv}")
    
    # Boxplots for aggregated stats
    plt.figure()
    df_all.boxplot(column="frac_prompt", by="prompt_type", grid=False)
    plt.suptitle("")
    plt.title("Fraction of Attention on Prompt Portion")
    plt.ylabel("Fraction (0~1)")
    plt.xlabel("Prompt Type")
    boxplot_prompt = os.path.join(config["output_dir"], "boxplot_frac_prompt.png")
    plt.savefig(boxplot_prompt)
    plt.close()
    
    plt.figure()
    df_all.boxplot(column="frac_task", by="prompt_type", grid=False)
    plt.suptitle("")
    plt.title("Fraction of Attention on Task Portion")
    plt.ylabel("Fraction (0~1)")
    plt.xlabel("Prompt Type")
    boxplot_task = os.path.join(config["output_dir"], "boxplot_frac_task.png")
    plt.savefig(boxplot_task)
    plt.close()
    
    # Per-layer-head stats
    df_plh = pd.DataFrame(all_plh_records).drop_duplicates()
    plh_mean = df_plh.groupby(["prompt_type", "layer"])[["frac_prompt", "frac_task"]].mean()
    print("=== Per-layer average fraction of attention on prompt vs. task ===")
    print(plh_mean)
    plh_csv = os.path.join(config["output_dir"], "prompt_task_fraction_perlayer.csv")
    plh_mean.to_csv(plh_csv)
    print(f"Saved per-layer fraction CSV to: {plh_csv}")
    
    # For each prompt_type, boxplot of frac_prompt by layer
    for pt in df_plh["prompt_type"].unique():
        sub = df_plh[df_plh["prompt_type"] == pt]
        if not sub.empty:
            plt.figure()
            sub.boxplot(column="frac_prompt", by="layer", grid=False)
            plt.suptitle("")
            plt.title(f"Fraction of Attention on Prompt vs. Layer - {pt}")
            plt.ylabel("Fraction (0~1)")
            plt.xlabel("Layer")
            boxplot_pl = os.path.join(config["output_dir"], f"boxplot_perlayer_frac_prompt_{pt}.png")
            plt.savefig(boxplot_pl)
            plt.close()
    
    print("All plots and CSVs saved to:", config["output_dir"])
    print("Done!")

# For command-line usage, this block will run if the module is executed directly.
if __name__ == "__main__":
    run_attention_extraction()
