import os
import torch
import glob
import re
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from transformers import AutoTokenizer

# --------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------
MODEL_NAME = "google/gemma-2-2b"
DATA_NORMAL_DIR = "output/extractions/gemma2b/normal"
DATA_NICER_DIR = "output/extractions/gemma2b/nicer"
OUTPUT_DIR = "output/attention/normal_nicer"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# “Nice” words to look for (lowercased)
NICE_WORDS = ["please", "kindly", "humbly", "would you", "could you"]

# Taking same tokenizer that was used originally to decode input_ids. !
# If you have a local directory with the model:
# tokenizer = AutoTokenizer.from_pretrained("path/to/model")
# otherwise from HF:
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# --------------------------------------------------------------------
# Helper Functions
# --------------------------------------------------------------------
def get_pt_files(directory):
    """Return sorted list of .pt files in a directory."""
    files = glob.glob(os.path.join(directory, "activations_*.pt"))
    files = sorted(files)  # sorts by name, e.g. activations_00000_00004.pt
    return files

def load_activations(pt_file):
    """
    Loads a single .pt file. 
    We expect structure like:
      {
         "attentions": { "layer_0": <tensor>, "layer_5": <tensor>, ...},
         "hidden_states": ...,
         "input_ids": <tensor>,
         "topk_logits": ...,
         "topk_indices": ...,
         "final_predictions": [...]
      }
    """
    return torch.load(pt_file)

def tokenize_texts(texts):
    """Helper to quickly get token IDs, attention masks, etc."""
    return tokenizer(
        texts,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )

def extract_attention_stats(attentions, input_ids):
    """
    Given attentions dict for L layers, each shape = [batch_size, n_heads, seq_len, seq_len]
    and the input_ids of shape [batch_size, seq_len],
    returns:
      - a dictionary of summary stats if you’d like
      - or a DataFrame with token-wise attention, etc.

    For simplicity, we’ll compute:
      1) average self-attention each token receives from all heads at each layer
         (i.e., “which tokens are attended to?”).
      2) sum or average across heads, or across layers, etc.

    We can refine if needed.
    """
    # layers in attentions: e.g. attentions["layer_0"], shape = [B, nH, seq_len, seq_len]
    # We'll produce shape: [B, seq_len] that is average attention each token receives from all heads.
    # For now, averaging across heads *and* “who is looking?” dimension, so we get
    # “attention on token i from all heads & tokens j.”
    # Another approach is “how much each token i attends to token j,” etc., but for now: simpler.
    
    stats_per_layer = {}

    for layer_name, attn_tensor in attentions.items():
        # attn_tensor shape: (batch_size, num_heads, seq_len, seq_len)
        # We want to see how much attention each token gets from all heads.
        # So we can average out dimension=1 (heads) and dimension=2 (the “source” tokens).
        # Or we can do sum over “source tokens,” to measure total “focus received.”
        
        # sum over dimension=2 => sum of attention from all tokens in the sequence
        # then average over dimension=1 => average across heads
        # => shape [batch_size, seq_len]
        
        attn_sum = attn_tensor.sum(dim=2)              # sum over "which token is attending" => shape [B, H, seq_len]
        attn_mean = attn_sum.mean(dim=1)              # average across heads => shape [B, seq_len]
        
        stats_per_layer[layer_name] = attn_mean  # shape [B, seq_len]
    
    # We could combine layers (e.g. average across layers) or keep them separate:
    # For now, producing a single average over all extracted layers:
    # We'll stack them up => [num_layers, B, seq_len], then average.
    all_layers = torch.stack(list(stats_per_layer.values()), dim=0)  # [num_layers, B, seq_len]
    avg_across_layers = all_layers.mean(dim=0)  # => [B, seq_len]

    # Return per-layer plus overall
    return {
        "per_layer": stats_per_layer,         # dict of layer_name -> [B, seq_len]
        "avg_layers": avg_across_layers,      # shape [B, seq_len]
        "input_ids": input_ids
    }

def identify_token_ranges(token_ids, text_prompt):
    """
    This helps find:
      - which tokens are “nice words”
      - which tokens are in the math portion (after the first colon)
    For a single prompt. We'll make it flexible enough for a batch if needed.

    We'll do a naive approach:
      1) Re-decode the entire prompt from token_ids
      2) Compare each token (strings) to our NICE_WORDS
      3) Find location of first colon “:”
      4) Mark tokens after that colon as 'math portion' tokens
    For more robust detection, you might do a string-split approach on the raw text.
    """
    decoded_tokens = tokenizer.convert_ids_to_tokens(token_ids, skip_special_tokens=False)
    # Lowercased version for matching
    decoded_tokens_lower = [t.lower() for t in decoded_tokens]

    # Find the position of the first colon in the tokenized text
    # Because tokenizers can split `":"` into multiple pieces, we handle carefully:
    colon_positions = [i for i, tok in enumerate(decoded_tokens) if tok == ":" or tok == "Ġ:" or ":" in tok]
    first_colon_pos = colon_positions[0] if len(colon_positions) > 0 else None

    # Identify “nice” tokens
    # We'll label a token as NICE if it contains any nice word
    # (e.g. 'please' might be splitted as 'ple' + 'ase', so watch out for partial)
    # For a rough approach, just check if the subtoken matches or if it’s in the piece.
    # You could unify them if needed.
    nice_token_mask = [False] * len(decoded_tokens)
    for i, tok_l in enumerate(decoded_tokens_lower):
        for w in NICE_WORDS:
            if w in tok_l:
                nice_token_mask[i] = True
                break

    # Identify math portion tokens if index > first_colon_pos (strictly after)
    math_token_mask = [False] * len(decoded_tokens)
    if first_colon_pos is not None:
        for i in range(first_colon_pos+1, len(decoded_tokens)):
            math_token_mask[i] = True

    return {
        "decoded_tokens": decoded_tokens,
        "nice_mask": nice_token_mask,
        "math_mask": math_token_mask,
        "colon_pos": first_colon_pos
    }

# --------------------------------------------------------------------
# Main Analysis
# --------------------------------------------------------------------
def main():
    # 1) Gather the file paths
    normal_files = get_pt_files(DATA_NORMAL_DIR)
    nicer_files = get_pt_files(DATA_NICER_DIR)

    print(f"Found {len(normal_files)} normal .pt files.")
    print(f"Found {len(nicer_files)} nicer .pt files.")

    # These will accumulate stats
    attention_records_normal = []
    attention_records_nicer = []

    # 2) Process normal .pt files
    for pt_file in normal_files:
        data = load_activations(pt_file)
        # data["attentions"] is a dict: layer_x -> [batch, heads, seq_len, seq_len]
        # data["input_ids"] is shape [batch, seq_len]
        # We’ll extract stats:
        batch_attn_stats = extract_attention_stats(data["attentions"], data["input_ids"])
        # batch_attn_stats["avg_layers"] is [B, seq_len]

        # For each item in batch, figure out how many tokens are “nice” or “math”:
        # But these normal prompts might or might not have the word “please,” etc.
        B, seq_len = batch_attn_stats["avg_layers"].shape
        for i in range(B):
            # shape [seq_len]
            attn_on_tokens = batch_attn_stats["avg_layers"][i].numpy()
            input_ids_row = batch_attn_stats["input_ids"][i].tolist()

            # Let’s decode or do a partial decode for “nice” or “math” detection
            masks_info = identify_token_ranges(input_ids_row, None)  # text_prompt optional if needed
            nice_mask = np.array(masks_info["nice_mask"], dtype=bool)
            math_mask = np.array(masks_info["math_mask"], dtype=bool)

            # Summaries
            avg_attn_nice = attn_on_tokens[nice_mask].mean() if nice_mask.any() else 0.0
            avg_attn_math = attn_on_tokens[math_mask].mean() if math_mask.any() else 0.0
            avg_attn_all = attn_on_tokens.mean()

            attention_records_normal.append({
                "batch_file": pt_file,
                "avg_attn_nice": avg_attn_nice,
                "avg_attn_math": avg_attn_math,
                "avg_attn_all": avg_attn_all,
                "num_nice_tokens": nice_mask.sum(),
                "num_math_tokens": math_mask.sum(),
                "prompt_type": "normal"
            })

    # 3) Process nicer .pt files
    for pt_file in nicer_files:
        data = load_activations(pt_file)
        batch_attn_stats = extract_attention_stats(data["attentions"], data["input_ids"])
        B, seq_len = batch_attn_stats["avg_layers"].shape

        for i in range(B):
            attn_on_tokens = batch_attn_stats["avg_layers"][i].numpy()
            input_ids_row = batch_attn_stats["input_ids"][i].tolist()

            masks_info = identify_token_ranges(input_ids_row, None)
            nice_mask = np.array(masks_info["nice_mask"], dtype=bool)
            math_mask = np.array(masks_info["math_mask"], dtype=bool)

            avg_attn_nice = attn_on_tokens[nice_mask].mean() if nice_mask.any() else 0.0
            avg_attn_math = attn_on_tokens[math_mask].mean() if math_mask.any() else 0.0
            avg_attn_all = attn_on_tokens.mean()

            attention_records_nicer.append({
                "batch_file": pt_file,
                "avg_attn_nice": avg_attn_nice,
                "avg_attn_math": avg_attn_math,
                "avg_attn_all": avg_attn_all,
                "num_nice_tokens": nice_mask.sum(),
                "num_math_tokens": math_mask.sum(),
                "prompt_type": "nicer"
            })

    # 4) Create DataFrames
    df_normal = pd.DataFrame(attention_records_normal)
    df_nicer = pd.DataFrame(attention_records_nicer)
    df_all = pd.concat([df_normal, df_nicer], ignore_index=True)

    # 5) Aggregate Stats
    # Let’s see overall means:
    mean_stats = df_all.groupby("prompt_type")[["avg_attn_nice","avg_attn_math","avg_attn_all"]].mean()
    print("=== Overall Mean Attention Stats ===")
    print(mean_stats)

    # Save stats to CSV
    stats_csv_path = os.path.join(OUTPUT_DIR, "attention_summary.csv")
    mean_stats.to_csv(stats_csv_path)
    print(f"Saved summary CSV to: {stats_csv_path}")

    # 6) Plots
    # a) Compare average attention on nice words
    plt.figure()
    df_all.boxplot(column="avg_attn_nice", by="prompt_type", grid=False)
    plt.suptitle("")
    plt.title("Average Attention on Nice Words")
    plt.ylabel("Attention (mean over sequence for nice tokens)")
    plt.xlabel("Prompt Type")
    plt.savefig(os.path.join(OUTPUT_DIR, "plot_attn_on_nice_words.png"))
    plt.close()

    # b) Compare average attention on math portion
    plt.figure()
    df_all.boxplot(column="avg_attn_math", by="prompt_type", grid=False)
    plt.suptitle("")
    plt.title("Average Attention on Math Portion")
    plt.ylabel("Attention (mean for tokens after colon)")
    plt.xlabel("Prompt Type")
    plt.savefig(os.path.join(OUTPUT_DIR, "plot_attn_on_math.png"))
    plt.close()

    # c) Compare overall average attention
    plt.figure()
    df_all.boxplot(column="avg_attn_all", by="prompt_type", grid=False)
    plt.suptitle("")
    plt.title("Overall Average Attention on Prompt")
    plt.ylabel("Attention (mean over all tokens)")
    plt.xlabel("Prompt Type")
    plt.savefig(os.path.join(OUTPUT_DIR, "plot_attn_on_all.png"))
    plt.close()

    # ! Will add more specialised plots (e.g., distribution histograms, layer-by-layer differences, etc.)

    print("All plots saved to:", OUTPUT_DIR)

if __name__ == "__main__":
    main()
