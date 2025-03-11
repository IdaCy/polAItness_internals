# File: collab_modules/attention_analysis.py

import os
import re
import glob
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from transformers import AutoTokenizer

def find_nice_tokens_subword(decoded_tokens, nice_words):
    """
    Attempt to handle subword splitting so that 
    'please' => ['pl','ea','se'] is recognized as 'please'.

    We'll accumulate subwords until we hit punctuation or boundary,
    then compare the merged chunk to each nice_word (cleaned of punctuation).
    """
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
        # e.g. remove punctuation to see if there's any alphanumeric left
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

def identify_token_ranges_robust(tokenizer, token_ids, nice_words):
    """
    Splits tokens, finds first colon, 
    robustly detects 'nice' tokens with subword merging.
    """
    decoded_tokens = tokenizer.convert_ids_to_tokens(token_ids, skip_special_tokens=False)
    colon_positions = [i for i, tok in enumerate(decoded_tokens) if ":" in tok]
    first_colon = colon_positions[0] if colon_positions else None

    nice_mask = find_nice_tokens_subword(decoded_tokens, nice_words)

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

def identify_token_ranges_naive(tokenizer, token_ids, nice_words):
    """
    Original naive approach:
      - We label any token that *contains* one of the NICE_WORDS.
      - We label tokens after the first colon as 'math'.
    """
    decoded_tokens = tokenizer.convert_ids_to_tokens(token_ids, skip_special_tokens=False)
    decoded_tokens_lower = [t.lower() for t in decoded_tokens]
    colon_positions = [i for i, tok in enumerate(decoded_tokens) if ":" in tok]
    first_colon_pos = colon_positions[0] if len(colon_positions) > 0 else None

    nice_mask = [False]*len(decoded_tokens)
    for i, tok_l in enumerate(decoded_tokens_lower):
        for w in nice_words:
            if w in tok_l:
                nice_mask[i] = True
                break

    math_mask = [False]*len(decoded_tokens)
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
    """
    Return bool arrays for the prompt portion [0..colon_pos]
    and the task portion [colon_pos+1..] 
    """
    prompt_mask = np.zeros(seq_len, dtype=bool)
    task_mask = np.zeros(seq_len, dtype=bool)
    if colon_pos is not None and colon_pos < seq_len:
        prompt_mask[:colon_pos+1] = True
        if colon_pos+1 < seq_len:
            task_mask[colon_pos+1:] = True
    else:
        # If no colon, treat entire sequence as 'task'
        task_mask[:] = True
    return prompt_mask, task_mask

def extract_attention_stats(attentions, input_ids):
    """
    Averages across heads for each token i, summing over j => shape [B, S].
    Then average across layers => [B, S].
    """
    stats_per_layer = {}
    for layer_name, attn_tensor in attentions.items():
        # attn_tensor shape: [B,H,S,S]
        attn_sum = attn_tensor.sum(dim=2)   # => [B,H,S], summation over j
        attn_mean = attn_sum.mean(dim=1)   # => [B,S], average across heads
        stats_per_layer[layer_name] = attn_mean

    layer_list = list(stats_per_layer.values())
    all_layers = torch.stack(layer_list, dim=0)  # => [num_layers,B,S]
    avg_across_layers = all_layers.mean(dim=0)   # => [B,S]

    return {
        "per_layer": stats_per_layer,
        "avg_layers": avg_across_layers,  # [B,S]
        "input_ids": input_ids
    }

def extract_per_layer_head_stats(attentions, input_ids):
    """
    Return list of dicts with keys:
     'layer', 'head', 'batch_idx', 'attn_sum', 'attn_mean'
    where attn_sum[i] = sum_{j} attn[i,j] for each token i.
    """
    results = []
    # sort layers by numeric index
    sorted_layers = sorted(attentions.keys(), key=lambda x: int(x.split('_')[1]) if '_' in x else 0)
    B, S = input_ids.shape

    for layer_name in sorted_layers:
        attn_tensor = attentions[layer_name]  # [B,H,S,S]
        attn_sum = attn_tensor.sum(dim=3)   # => [B,H,S]
        attn_mean = attn_tensor.mean(dim=3) # => [B,H,S]

        layer_idx = int(layer_name.split('_')[1]) if '_' in layer_name else layer_name
        b_size, h_size, s_len, s_len2 = attn_tensor.shape
        if b_size != B or s_len != S or s_len2 != S:
            raise ValueError(f"Shape mismatch in {layer_name}")

        for b_idx in range(B):
            for h_idx in range(h_size):
                row = {
                    "layer": layer_idx,
                    "head": h_idx,
                    "batch_idx": b_idx,
                    "attn_sum": attn_sum[b_idx,h_idx].cpu().tolist(),
                    "attn_mean": attn_mean[b_idx,h_idx].cpu().tolist()
                }
                results.append(row)
    return results

def run_attention_analysis(
    model_name="google/gemma-2-2b-it",
    data_normal_dir="output/extractions/gemma2bit/normal",
    data_nicer_dir="output/extractions/gemma2bit/nicer",
    output_dir="output/attention/extra_normal_nicer",
    nice_words_csv="analyses/nice_words.csv",
    robust_nice_detection=True
):
    """
    Main function to run attention analysis. 
    Usage in Colab:
      from collab_modules.attention_analysis import run_attention_analysis
      run_attention_analysis(...your args here...)
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load your NICE words
    with open(nice_words_csv, "r", encoding="utf-8") as f:
        nice_words = [line.strip() for line in f if line.strip()]

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # We define the "identify_fn" depending on robust_nice_detection
    if robust_nice_detection:
        def identify_fn(ids):
            return identify_token_ranges_robust(tokenizer, ids, nice_words)
    else:
        def identify_fn(ids):
            return identify_token_ranges_naive(tokenizer, ids, nice_words)

    # gather .pt files
    def get_pt_files(directory):
        """Return sorted list of .pt files in a directory."""
        files = glob.glob(os.path.join(directory, "activations_*.pt"))
        return sorted(files)

    normal_files = get_pt_files(data_normal_dir)
    nicer_files = get_pt_files(data_nicer_dir)

    print(f"Found {len(normal_files)} normal .pt files in {data_normal_dir}")
    print(f"Found {len(nicer_files)} nicer .pt files  in {data_nicer_dir}")

    # Just like your HPC script
    attention_records_normal = []
    attention_records_nicer  = []
    plh_records_normal = []
    plh_records_nicer  = []

    def load_activations(pt_file):
        return torch.load(pt_file)

    def process_files(pt_files, prompt_type):
        agg_list = []
        plh_list = []
        for pt_file in pt_files:
            data = load_activations(pt_file)
            batch_stats = extract_attention_stats(data["attentions"], data["input_ids"])
            B, S = batch_stats["avg_layers"].shape

            for i in range(B):
                attn_vals = batch_stats["avg_layers"][i].float().cpu().numpy()  # shape [S]
                masks_info = identify_fn(batch_stats["input_ids"][i].tolist())
                colon_pos  = masks_info["colon_pos"]

                prompt_mask, task_mask = make_prompt_task_masks(colon_pos, S)
                total_sum = attn_vals.sum()
                prompt_sum = attn_vals[prompt_mask].sum() if prompt_mask.any() else 0.0
                task_sum   = attn_vals[task_mask].sum()   if task_mask.any() else 0.0
                frac_prompt = prompt_sum / total_sum if total_sum>1e-10 else 0.0
                frac_task   = task_sum   / total_sum if total_sum>1e-10 else 0.0

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
                    "task_sum": float(task_sum),
                    "frac_prompt": float(frac_prompt),
                    "frac_task": float(frac_task),
                })

            # per-layer-head
            plh_data = extract_per_layer_head_stats(data["attentions"], data["input_ids"])
            for row in plh_data:
                b_idx = row["batch_idx"]
                attn_sum_array = np.array(row["attn_sum"], dtype=np.float32)
                total_sum = attn_sum_array.sum()

                masks_info = identify_fn(data["input_ids"][b_idx].tolist())
                colon_pos = masks_info["colon_pos"]
                prompt_mask, task_mask = make_prompt_task_masks(colon_pos, S)
                prompt_sum = attn_sum_array[prompt_mask].sum() if prompt_mask.any() else 0.0
                task_sum   = attn_sum_array[task_mask].sum()   if task_mask.any() else 0.0
                frac_prompt = prompt_sum / total_sum if total_sum>1e-10 else 0.0
                frac_task   = task_sum   / total_sum if total_sum>1e-10 else 0.0

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

    # Process normal files
    agg_norm, plh_norm = process_files(normal_files, "normal")
    attention_records_normal.extend(agg_norm)
    plh_records_normal.extend(plh_norm)

    # Process nicer files
    agg_nice, plh_nice = process_files(nicer_files, "nicer")
    attention_records_nicer.extend(agg_nice)
    plh_records_nicer.extend(plh_nice)

    # ~ Combine
    df_normal = pd.DataFrame(attention_records_normal)
    df_nicer  = pd.DataFrame(attention_records_nicer)
    df_all    = pd.concat([df_normal, df_nicer], ignore_index=True)

    # Summaries
    agg_mean = df_all.groupby("prompt_type")[["frac_prompt","frac_task"]].mean()
    print("=== Average fraction of attention on prompt vs. task portion ===")
    print(agg_mean)

    os.makedirs(output_dir, exist_ok=True)

    out_csv_agg = os.path.join(output_dir, "prompt_task_fraction_aggregate.csv")
    agg_mean.to_csv(out_csv_agg)
    print(f"Saved fraction summary CSV to: {out_csv_agg}")

    # Boxplot fraction of attention on prompt
    plt.figure()
    df_all.boxplot(column="frac_prompt", by="prompt_type", grid=False)
    plt.suptitle("")
    plt.title("Fraction of Attention on Prompt Portion")
    plt.ylabel("Fraction of total attention (0~1)")
    plt.xlabel("Prompt Type")
    plt.savefig(os.path.join(output_dir, "boxplot_frac_prompt.png"))
    plt.close()

    # Boxplot fraction of attention on task
    plt.figure()
    df_all.boxplot(column="frac_task", by="prompt_type", grid=False)
    plt.suptitle("")
    plt.title("Fraction of Attention on Task Portion")
    plt.ylabel("Fraction of total attention (0~1)")
    plt.xlabel("Prompt Type")
    plt.savefig(os.path.join(output_dir, "boxplot_frac_task.png"))
    plt.close()

    # Per-Layer-Head
    df_plh_norm = pd.DataFrame(plh_records_normal)
    df_plh_nice = pd.DataFrame(plh_records_nicer)
    df_plh_all  = pd.concat([df_plh_norm, df_plh_nice], ignore_index=True)
    plh_mean = df_plh_all.groupby(["prompt_type","layer"])[["frac_prompt","frac_task"]].mean()
    print("=== Per-layer average fraction of attention on prompt vs. task ===")
    print(plh_mean)

    out_csv_plh = os.path.join(output_dir, "prompt_task_fraction_perlayer.csv")
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
        plt.savefig(os.path.join(output_dir, f"boxplot_perlayer_frac_prompt_{pt}.png"))
        plt.close()

    print("All new plots saved to:", output_dir)
    print("Done!")
