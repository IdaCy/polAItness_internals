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

# START -- just for convenience, only change those and that's it
PREP_MODEL = (
    os.environ.get("PREP_MODEL")
    or globals().get("PREP_MODEL")
    or "gemma-2-9b-it"
)
PREP_DIR = (
    os.environ.get("PREP_DIR")
    or globals().get("PREP_DIR")
    or "output/extractions/" + PREP_MODEL
)
ALL_PROMPT_TYPES = (
    os.environ.get("ALL_PROMPT_DIRS")
    or globals().get("ALL_PROMPT_DIRS")
    or f"explained insulting nicer1k normal1k pure reduced1k shortest1k urgent1k"
)
# END --- convenience

# list of pure names
all_types_list = [d.strip() for d in re.split(r"[, \n]+", ALL_PROMPT_TYPES) if d.strip()]  # <--- fix 'typoes' -> 'types'

# list of dirs of names
ALL_PROMPT_DIRS = ""
for dir_name in all_types_list:
    # Add a slash + space so they can be split again
    # e.g. "output/extractions/gemma-2-9b-it/normal1k "
    # This ensures the next step sees them as separate.
    ALL_PROMPT_DIRS += os.path.join(PREP_DIR, dir_name) + " "

# Split ALL_PROMPT_DIRS into a list of directories
all_dirs_list = [d.strip() for d in re.split(r"[, \n]+", ALL_PROMPT_DIRS) if d.strip()]

# Create a single underscored string for OUTPUT_DIR
def concatenate_dirs(all_prompt_dirs: list) -> str:
    return "_".join(all_prompt_dirs)

all_dirs_joined = concatenate_dirs(all_types_list)

MODEL_NAME = (
    os.environ.get("MODEL_NAME")
    or globals().get("MODEL_NAME")
    or "google/" + PREP_MODEL
)
OUTPUT_DIR = (
    os.environ.get("OUTPUT_DIR")
    or globals().get("OUTPUT_DIR")
    or "output/attention/" + PREP_MODEL + "/" + all_dirs_joined
)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# If you want a CSV of “nice” words:
NICE_WORDS_CSV = "analyses/nice_words.csv"
with open(NICE_WORDS_CSV, "r", encoding="utf-8") as f:
    NICE_WORDS = [line.strip() for line in f if line.strip()]

# If you want robust subword merging:
ROBUST_NICE_DETECTION = True

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

# -- “nice” detection subword approach --
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
                # Mark all tokens that contributed
                for idx in range(start_idx, end_idx):
                    nice_mask[idx] = True

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

    return nice_mask

def identify_token_ranges_robust(token_ids):
    decoded_tokens = tokenizer.convert_ids_to_tokens(token_ids, skip_special_tokens=False)
    # find colon
    colon_positions = [i for i, tok in enumerate(decoded_tokens) if ":" in tok]
    first_colon = colon_positions[0] if colon_positions else None

    # robust “nice”
    nice_mask = find_nice_tokens_subword(decoded_tokens, NICE_WORDS)

    # after-colon = task portion
    math_mask = [False]*len(decoded_tokens)
    if first_colon is not None and first_colon < len(decoded_tokens)-1:
        for idx in range(first_colon+1, len(decoded_tokens)):
            math_mask[idx] = True

    return {
        "decoded_tokens": decoded_tokens,
        "nice_mask": nice_mask,
        "math_mask": math_mask,
        "colon_pos": first_colon
    }

def identify_token_ranges_naive(token_ids):
    """
    Original naive approach:
      - We label any token that *contains* one of the NICE_WORDS as "nice".
      - We label tokens after the first colon as "math".
    """
    decoded_tokens = tokenizer.convert_ids_to_tokens(token_ids, skip_special_tokens=False)
    decoded_tokens_lower = [t.lower() for t in decoded_tokens]

    # Where is the first colon?
    colon_positions = [i for i, tok in enumerate(decoded_tokens) if ":" in tok]
    first_colon_pos = colon_positions[0] if len(colon_positions) > 0 else None

    # Mark nice tokens (naive substring check)
    nice_mask = [False] * len(decoded_tokens)
    for i, tok_l in enumerate(decoded_tokens_lower):
        for w in NICE_WORDS:
            if w in tok_l:  # substring
                nice_mask[i] = True
                break

    # Mark math tokens
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
    # For aggregator approach
    attention_records_normal = []
    attention_records_nicer  = []
    # For per-layer-head
    plh_records_normal = []
    plh_records_nicer  = []

    # Decide which colon-detection + nice detection approach:
    identify_fn = identify_token_ranges_robust if ROBUST_NICE_DETECTION else identify_token_ranges_naive

    def process_files(pt_files, prompt_type):
        agg_list = []
        plh_list = []
        for pt_file in pt_files:
            data = load_activations(pt_file)
            # ~ AGGREGATE
            batch_stats = extract_attention_stats(data["attentions"], data["input_ids"])
            B, S = batch_stats["avg_layers"].shape

            for i in range(B):
                attn_vals = batch_stats["avg_layers"][i].float().cpu().numpy()  # shape [S]
                masks_info = identify_fn(data["input_ids"][i].tolist())
                colon_pos  = masks_info["colon_pos"]

                # Make prompt vs. task masks
                prompt_mask, task_mask = make_prompt_task_masks(colon_pos, S)

                total_sum = attn_vals.sum()
                prompt_sum = attn_vals[prompt_mask].sum() if prompt_mask.any() else 0.0
                task_sum   = attn_vals[task_mask].sum() if task_mask.any() else 0.0

                frac_prompt = (prompt_sum / total_sum) if total_sum>1e-10 else 0.0
                frac_task   = (task_sum   / total_sum) if total_sum>1e-10 else 0.0

                nice_mask = np.array(masks_info["nice_mask"], dtype=bool)
                math_mask = np.array(masks_info["math_mask"], dtype=bool)
                avg_attn_nice = attn_vals[nice_mask].mean() if nice_mask.any() else 0.0
                avg_attn_math = attn_vals[math_mask].mean() if math_mask.any() else 0.0

                agg_list.append({
                    "batch_file": pt_file,
                    "prompt_type": prompt_type,
                    "avg_attn_nice": avg_attn_nice,
                    "avg_attn_math": avg_attn_math,
                    "avg_attn_all": attn_vals.mean(),
                    "prompt_sum": float(prompt_sum),
                    "task_sum":   float(task_sum),
                    "frac_prompt": float(frac_prompt),
                    "frac_task": float(frac_task),
                })

            # ~ PER-LAYER-HEAD
            plh_data = extract_per_layer_head_stats(data["attentions"], data["input_ids"])
            for row in plh_data:
                b_idx = row["batch_idx"]
                colon_pos = identify_fn(data["input_ids"][b_idx].tolist())["colon_pos"]
                prompt_mask, task_mask = make_prompt_task_masks(colon_pos, S)

                attn_sum_array  = np.array(row["attn_sum"], dtype=np.float32)
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

    # ====================== NEW BLOCK: process an arbitrary list of directories ======================
    all_attention_records = []
    all_plh_records = []

    print("=== Will process the following directories (ALL_PROMPT_DIRS) ===")
    for ddir in all_dirs_list:
        print(" -", ddir)
    print("=================================================================")

    for ddir in all_dirs_list:
        files = get_pt_files(ddir)
        pt_name = os.path.basename(ddir.rstrip("/"))  # e.g. "nicer1k"
        print(f"Found {len(files)} .pt files in: {ddir} (prompt_type='{pt_name}')")
        agg, plh = process_files(files, pt_name)
        all_attention_records.extend(agg)
        all_plh_records.extend(plh)

    # Build data frames from the old approach:
    df_normal = pd.DataFrame(attention_records_normal)
    df_nicer  = pd.DataFrame(attention_records_nicer)
    df_all    = pd.concat([df_normal, df_nicer], ignore_index=True)

    # Build data frames from the new multi-dir approach:
    df_multi = pd.DataFrame(all_attention_records)

    # Merge them all (drop duplicates if the same files appear):
    df_all = pd.concat([df_all, df_multi], ignore_index=True).drop_duplicates()

    # Summaries: fraction of attention on prompt portion vs. task portion
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

    # ----------------- Per-layer-head -----------------
    df_plh_norm = pd.DataFrame(plh_records_normal)
    df_plh_nice = pd.DataFrame(plh_records_nicer)
    df_plh_all  = pd.concat([df_plh_norm, df_plh_nice], ignore_index=True)

    df_plh_multi = pd.DataFrame(all_plh_records)
    df_plh_all = pd.concat([df_plh_all, df_plh_multi], ignore_index=True).drop_duplicates()

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
