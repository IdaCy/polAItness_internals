#!/usr/bin/env python3

import os
import re
import torch
import glob
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from transformers import AutoTokenizer

# --------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------
MODEL_NAME = (
    os.environ.get("MODEL_NAME")
    or globals().get("MODEL_NAME")
    or "google/gemma-2-2b-it"
)
DATA_NORMAL_DIR = (
    os.environ.get("DATA_NORMAL_DIR")
    or globals().get("DATA_NORMAL_DIR")
    or "output/extractions/gemma2bit/normal"
)
DATA_URGENT_DIR = (
    os.environ.get("DATA_URGENT_DIR")
    or globals().get("DATA_URGENT_DIR")
    or "output/extractions/gemma2bit/urgent"
)
OUTPUT_DIR = (
    os.environ.get("OUTPUT_DIR")
    or globals().get("OUTPUT_DIR")
    or "output/attention/extra_normal_urgent"
)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# If you want a CSV of “urgent” words:
URGENT_WORDS_CSV = "analyses/urgent_words.csv"
with open(URGENT_WORDS_CSV, "r", encoding="utf-8") as f:
    URGENT_WORDS = [line.strip() for line in f if line.strip()]

# If you want robust subword merging:
ROBUST_URGENT_DETECTION = True

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# --------------------------------------------------------------------
# Helper Functions
# --------------------------------------------------------------------
def get_pt_files(directory):
    """Return sorted list of .pt files in a directory."""
    files = glob.glob(os.path.join(directory, "activations_*.pt"))
    files = sorted(files)
    return files

def load_activations(pt_file):
    """Loads a single .pt file with structure:
       {"attentions": {...}, "input_ids":..., ...}"""
    return torch.load(pt_file)

# -- “urgent” detection subword approach --
def find_urgent_tokens_subword(decoded_tokens, urgent_words):
    urgent_mask = [False] * len(decoded_tokens)

    word_buffer = []
    word_start_idx = 0

    def flush_word(buffer, start_idx, end_idx):
        merged = "".join(buffer).replace("▁", "").replace("Ġ", "").lower()
        merged_clean = re.sub(r"\W+", "", merged)
        for w in urgent_words:
            w_clean = re.sub(r"\W+", "", w.lower())
            if merged_clean == w_clean:
                # Mark all tokens that contributed
                for idx in range(start_idx, end_idx):
                    urgent_mask[idx] = True

    def is_wordish(tok):
        # e.g. remove punctuation
        return len(re.sub(r"\W+", "", tok)) > 0

    for i, tok in enumerate(decoded_tokens):
        if is_wordish(tok):
            word_buffer.append(tok)
        else:
            # boundary => flush
            if word_buffer:
                flush_word(word_buffer, word_start_idx, i)
                word_buffer = []
            word_start_idx = i+1

    # end flush
    if word_buffer:
        flush_word(word_buffer, word_start_idx, len(decoded_tokens))

    return urgent_mask

def identify_token_ranges_robust(token_ids):
    decoded_tokens = tokenizer.convert_ids_to_tokens(token_ids, skip_special_tokens=False)
    # find colon
    colon_positions = [i for i, tok in enumerate(decoded_tokens) if ":" in tok]
    first_colon = colon_positions[0] if colon_positions else None

    # robust “urgent”
    urgent_mask = find_urgent_tokens_subword(decoded_tokens, URGENT_WORDS)

    # after-colon = task portion
    math_mask = [False]*len(decoded_tokens)
    if first_colon is not None and first_colon < len(decoded_tokens)-1:
        for idx in range(first_colon+1, len(decoded_tokens)):
            math_mask[idx] = True

    return {
        "decoded_tokens": decoded_tokens,
        "urgent_mask": urgent_mask,
        "math_mask": math_mask,
        "colon_pos": first_colon
    }

def identify_token_ranges_naive(token_ids):
    """
    Original naive approach:
      - We label any token that *contains* one of the URGENT_WORDS as "urgent".
      - We label tokens after the first colon as "math".
    """
    decoded_tokens = tokenizer.convert_ids_to_tokens(token_ids, skip_special_tokens=False)
    decoded_tokens_lower = [t.lower() for t in decoded_tokens]

    # Where is the first colon?
    colon_positions = [i for i, tok in enumerate(decoded_tokens) if ":" in tok]
    first_colon_pos = colon_positions[0] if len(colon_positions) > 0 else None

    # Mark urgent tokens (naive substring check)
    urgent_mask = [False] * len(decoded_tokens)
    for i, tok_l in enumerate(decoded_tokens_lower):
        for w in URGENT_WORDS:
            if w in tok_l:  # substring
                urgent_mask[i] = True
                break

    # Mark math tokens
    math_mask = [False] * len(decoded_tokens)
    if first_colon_pos is not None:
        for i in range(first_colon_pos+1, len(decoded_tokens)):
            math_mask[i] = True

    return {
        "decoded_tokens": decoded_tokens,
        "urgent_mask": urgent_mask,
        "math_mask": math_mask,
        "colon_pos": first_colon_pos
    }

# =============== KEY: “prompt vs. task” ===============
def make_prompt_task_masks(colon_pos, seq_len):
    """Return bool arrays for prompt portion [0..colon_pos]
       and task portion [colon_pos+1..]"""
    prompt_mask = np.zeros(seq_len, dtype=bool)
    task_mask   = np.zeros(seq_len, dtype=bool)
    if colon_pos is not None and colon_pos < seq_len:
        prompt_mask[:colon_pos+1] = True
        if colon_pos+1 < seq_len:
            task_mask[colon_pos+1:] = True
    else:
        # If no colon, maybe treat everything as prompt or everything as task
        # We'll do everything=task in that scenario, or you could do the opposite.
        task_mask[:] = True
    return prompt_mask, task_mask


# ====================== ATTENTION EXTRACTION (AGGREGATE) ======================
def extract_attention_stats(attentions, input_ids):
    """
    For each layer L in `attentions`, shape [B,H,S,S], we sum across S dimension=2,
    then average across heads dimension=1 => shape [B,S].
    Then we average across layers => [B,S].
    """
    stats_per_layer = {}
    for layer_name, attn_tensor in attentions.items():
        # shape: [B,H,S,S]
        # sum over dimension=2 => [B,H,S]
        attn_sum = attn_tensor.sum(dim=2)
        # mean over heads => [B,S]
        attn_mean = attn_sum.mean(dim=1)
        stats_per_layer[layer_name] = attn_mean

    layer_list = list(stats_per_layer.values())
    all_layers = torch.stack(layer_list, dim=0)  # [num_layers, B, S]
    avg_across_layers = all_layers.mean(dim=0)   # [B, S]

    return {
        "per_layer": stats_per_layer,
        "avg_layers": avg_across_layers,
        "input_ids": input_ids
    }

# ====================== ATTENTION EXTRACTION (PER-LAYER, PER-HEAD) ======================
def extract_per_layer_head_stats(attentions, input_ids):
    """
    Returns list of dicts with fields:
      layer, head, batch_idx,
      attn_sum (list[S]), attn_mean (list[S])
    """
    results = []
    sorted_layers = sorted(attentions.keys(), key=lambda x: int(x.split('_')[1]) if '_' in x else 0)
    B, S = input_ids.shape

    for layer_name in sorted_layers:
        attn_tensor = attentions[layer_name]  # [B,H,S,S]
        # sum across j => shape [B,H,S]
        attn_sum = attn_tensor.sum(dim=3)
        attn_mean = attn_tensor.mean(dim=3)

        layer_idx = int(layer_name.split('_')[1]) if '_' in layer_name else layer_name
        b_size, h_size, s_len, s_len2 = attn_tensor.shape
        if b_size!=B or s_len!=S or s_len2!=S:
            raise ValueError("Mismatch shape...")

        for b_idx in range(B):
            for h_idx in range(h_size):
                row = {
                    "layer": layer_idx,
                    "head": h_idx,
                    "batch_idx": b_idx,
                    "attn_sum": attn_sum[b_idx,h_idx].cpu().tolist(),
                    "attn_mean":attn_mean[b_idx,h_idx].cpu().tolist()
                }
                results.append(row)
    return results


# ====================== MAIN SCRIPT ======================
def main():
    normal_files = get_pt_files(DATA_NORMAL_DIR)
    urgent_files  = get_pt_files(DATA_URGENT_DIR)

    print(f"Found {len(normal_files)} normal .pt files.")
    print(f"Found {len(urgent_files)} urgent .pt files.")

    # For aggregator approach
    attention_records_normal = []
    attention_records_urgent  = []
    # For per-layer-head
    plh_records_normal = []
    plh_records_urgent  = []

    # Decide which colon-detection + urgent detection approach:
    identify_fn = identify_token_ranges_robust if ROBUST_URGENT_DETECTION else identify_token_ranges_naive

    def process_files(pt_files, prompt_type):
        agg_list = []
        plh_list = []
        for pt_file in pt_files:
            data = load_activations(pt_file)
            # ~ AGGREGATE
            batch_stats = extract_attention_stats(data["attentions"], data["input_ids"])
            B, S = batch_stats["avg_layers"].shape

            for i in range(B):
                # Convert to float32 for numpy
                attn_vals = batch_stats["avg_layers"][i].float().cpu().numpy()  # shape [S]
                masks_info = identify_fn(data["input_ids"][i].tolist())
                colon_pos  = masks_info["colon_pos"]

                # Make prompt vs. task masks
                prompt_mask, task_mask = make_prompt_task_masks(colon_pos, S)

                # sum total
                total_sum = attn_vals.sum()
                # sum prompt portion
                prompt_sum = attn_vals[prompt_mask].sum() if prompt_mask.any() else 0.0
                # sum task portion
                task_sum   = attn_vals[task_mask].sum() if task_mask.any() else 0.0

                frac_prompt = (prompt_sum / total_sum) if total_sum>1e-10 else 0.0
                frac_task   = (task_sum   / total_sum) if total_sum>1e-10 else 0.0

                # The old "urgent" approach
                urgent_mask = np.array(masks_info["urgent_mask"], dtype=bool)
                math_mask = np.array(masks_info["math_mask"], dtype=bool)
                avg_attn_urgent = attn_vals[urgent_mask].mean() if urgent_mask.any() else 0.0
                avg_attn_math = attn_vals[math_mask].mean() if math_mask.any() else 0.0

                # We'll store both old + new in the same record
                agg_list.append({
                    "batch_file": pt_file,
                    "prompt_type": prompt_type,
                    "avg_attn_urgent": avg_attn_urgent,
                    "avg_attn_math": avg_attn_math,
                    "avg_attn_all": attn_vals.mean(),
                    # new:
                    "prompt_sum": float(prompt_sum),
                    "task_sum":   float(task_sum),
                    "frac_prompt": float(frac_prompt),
                    "frac_task":   float(frac_task),
                })

            # ~ PER-LAYER-HEAD
            plh_data = extract_per_layer_head_stats(data["attentions"], data["input_ids"])
            for row in plh_data:
                b_idx = row["batch_idx"]
                colon_pos = identify_fn(data["input_ids"][b_idx].tolist())["colon_pos"]
                prompt_mask, task_mask = make_prompt_task_masks(colon_pos, S)

                attn_sum_array  = np.array(row["attn_sum"], dtype=np.float32)  # shape [S]
                total_sum       = attn_sum_array.sum()
                prompt_sum      = attn_sum_array[prompt_mask].sum() if prompt_mask.any() else 0.0
                task_sum        = attn_sum_array[task_mask].sum() if task_mask.any() else 0.0
                frac_prompt     = (prompt_sum / total_sum) if total_sum>1e-10 else 0.0
                frac_task       = (task_sum   / total_sum) if total_sum>1e-10 else 0.0

                plh_list.append({
                    "batch_file": pt_file,
                    "prompt_type": prompt_type,
                    "layer": row["layer"],
                    "head":  row["head"],
                    "batch_idx": b_idx,
                    "sum_all": float(total_sum),
                    "sum_prompt": float(prompt_sum),
                    "sum_task": float(task_sum),
                    "frac_prompt": float(frac_prompt),
                    "frac_task": float(frac_task),
                })
        return agg_list, plh_list

    # Process normal and urgent
    agg_norm, plh_norm = process_files(normal_files, "normal")
    agg_urgent, plh_urgent = process_files(urgent_files,  "urgent")
    attention_records_normal.extend(agg_norm)
    attention_records_urgent.extend(agg_urgent)
    plh_records_normal.extend(plh_norm)
    plh_records_urgent.extend(plh_urgent)

    # ~~~~~~~~~~~~~~~~~~~~~~
    # Create DataFrames
    # ~~~~~~~~~~~~~~~~~~~~~~
    df_normal = pd.DataFrame(attention_records_normal)
    df_urgent  = pd.DataFrame(attention_records_urgent)
    df_all    = pd.concat([df_normal, df_urgent], ignore_index=True)

    # Summaries: fraction of attention on prompt portion vs. task portion
    # group by prompt_type
    agg_mean = df_all.groupby("prompt_type")[["frac_prompt","frac_task"]].mean()
    print("=== Average fraction of attention on prompt vs. task portion ===")
    print(agg_mean)

    out_csv_agg = os.path.join(OUTPUT_DIR, "prompt_task_fraction_aggregate.csv")
    agg_mean.to_csv(out_csv_agg)
    print(f"Saved fraction summary CSV to: {out_csv_agg}")

    # Boxplot: fraction of attention on prompt portion
    plt.figure()
    df_all.boxplot(column="frac_prompt", by="prompt_type", grid=False)
    plt.suptitle("")
    plt.title("Fraction of Attention on Prompt Portion")
    plt.ylabel("Fraction of total attention (0~1)")
    plt.xlabel("Prompt Type")
    plt.savefig(os.path.join(OUTPUT_DIR, "boxplot_frac_prompt.png"))
    plt.close()

    # Boxplot: fraction of attention on task portion
    plt.figure()
    df_all.boxplot(column="frac_task", by="prompt_type", grid=False)
    plt.suptitle("")
    plt.title("Fraction of Attention on Task Portion")
    plt.ylabel("Fraction of total attention (0~1)")
    plt.xlabel("Prompt Type")
    plt.savefig(os.path.join(OUTPUT_DIR, "boxplot_frac_task.png"))
    plt.close()

    # ~~~~~~~~~~~~~~~~~~~~~~
    # Per-Layer-Head
    # ~~~~~~~~~~~~~~~~~~~~~~
    df_plh_norm = pd.DataFrame(plh_records_normal)
    df_plh_urgent = pd.DataFrame(plh_records_urgent)
    df_plh_all  = pd.concat([df_plh_norm, df_plh_urgent], ignore_index=True)

    # Summaries by layer
    plh_mean = df_plh_all.groupby(["prompt_type","layer"])[["frac_prompt","frac_task"]].mean()
    print("=== Per-layer average fraction of attention on prompt vs. task ===")
    print(plh_mean)

    out_csv_plh = os.path.join(OUTPUT_DIR, "prompt_task_fraction_perlayer.csv")
    plh_mean.to_csv(out_csv_plh)
    print(f"Saved fraction per-layer to: {out_csv_plh}")

    # For each prompt_type, boxplot of frac_prompt by layer
    for pt in df_plh_all["prompt_type"].unique():
        sub = df_plh_all[df_plh_all["prompt_type"] == pt]
        if sub.empty:
            continue
        plt.figure()
        sub.boxplot(column="frac_prompt", by="layer", grid=False)
        plt.suptitle("")
        plt.title(f"Fraction of Attention on Prompt vs. layer - {pt}")
        plt.ylabel("Fraction of total attention (0~1)")
        plt.xlabel("Layer")
        plt.savefig(os.path.join(OUTPUT_DIR, f"boxplot_perlayer_frac_prompt_{pt}.png"))
        plt.close()

    print("All new plots saved to:", OUTPUT_DIR)
    print("Done!")

if __name__ == "__main__":
    main()
