#!/usr/bin/env python3

import os
import logging
import torch
import glob
import pandas as pd
from transformers import AutoTokenizer

# ------------------------------------------------------------------------
# 1. Configuration and Setup
# ------------------------------------------------------------------------
# These can be set via environment variables, Python globals, or left at defaults.
RUN_DIRECTORIES = globals().get("RUN_DIRECTORIES", "output/extractions/nice,output/extractions/mean")
OUTPUT_FILE = globals().get("OUTPUT_FILE", "output/attention_analysis.csv")
MODEL_NAME_FOR_DECODING = globals().get("MODEL_NAME_FOR_DECODING", "google/gemma-2-2b")

TOKEN_OF_INTEREST = globals().get("TOKEN_OF_INTEREST", "please")  
LAYERS_TO_ANALYZE = globals().get("LAYERS_TO_ANALYZE", [0, 5, 10, 15, 20, 25])
HEADS_TO_ANALYZE = globals().get("HEADS_TO_ANALYZE", None)  # None = all heads

# Fallback to environment variables, if set
RUN_DIRECTORIES = os.environ.get("RUN_DIRECTORIES", RUN_DIRECTORIES)
OUTPUT_FILE = os.environ.get("OUTPUT_FILE", OUTPUT_FILE)
MODEL_NAME_FOR_DECODING = os.environ.get("MODEL_NAME_FOR_DECODING", MODEL_NAME_FOR_DECODING)
TOKEN_OF_INTEREST = os.environ.get("TOKEN_OF_INTEREST", TOKEN_OF_INTEREST)

# We'll parse RUN_DIRECTORIES as a comma-separated string
RUN_DIRECTORIES = [d.strip() for d in RUN_DIRECTORIES.split(",")]

os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

LOG_FILE = globals().get("LOG_FILE", "logs/attention_analysis.log")
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

# ------------------------------------------------------------------------
# 1a. Set up Logging
# ------------------------------------------------------------------------
logger = logging.getLogger("AnalysisLogger")
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

logger.info("=== Starting attention analysis ===")
logger.info(f"Run directories: {RUN_DIRECTORIES}")
logger.info(f"Output file: {OUTPUT_FILE}")
logger.info(f"Token of interest: {TOKEN_OF_INTEREST}")
logger.info(f"Layers to analyze: {LAYERS_TO_ANALYZE}")
if HEADS_TO_ANALYZE is not None:
    logger.info(f"Heads to analyze: {HEADS_TO_ANALYZE}")
else:
    logger.info("Heads to analyze: all heads")

# ------------------------------------------------------------------------
# 2. Load a Tokenizer (for decoding input_ids)
# ------------------------------------------------------------------------
logger.info(f"Loading tokenizer for decoding: {MODEL_NAME_FOR_DECODING}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME_FOR_DECODING)
if tokenizer.pad_token is None and tokenizer.eos_token is not None:
    tokenizer.pad_token = tokenizer.eos_token

# ------------------------------------------------------------------------
# 3. Helper functions
# ------------------------------------------------------------------------
def find_token_positions(input_ids, token_of_interest):
    """
    Given a list of token IDs for a single sample and a target string (like "please"),
    return all positions where that token occurs. This is approximate, since
    "please" may be split into sub-tokens for some tokenizers.
    """
    decoded_tokens = [tokenizer.decode([tid], skip_special_tokens=True) for tid in input_ids]
    positions = []
    for i, t in enumerate(decoded_tokens):
        # Simple approach: we check if our token_of_interest is exactly the decoded single token
        # or is contained within it. Adjust as needed for your sub-token logic.
        if token_of_interest.lower() in t.lower():
            positions.append(i)
    return positions

def load_activations_from_dir(directory):
    """
    Finds all .pt files in a directory, loads them, and yields
    each sample's data structure.
    """
    pt_files = sorted(glob.glob(os.path.join(directory, "*.pt")))
    logger.info(f"Loading {len(pt_files)} .pt files from '{directory}'")

    for pt_file in pt_files:
        try:
            data = torch.load(pt_file)
            # data is typically the dictionary returned by `capture_activations()`
            # but if BATCH_SIZE > 1, it means data for that entire batch.
            # The dictionary has shape (or keys) for the entire batch, so
            # each key has shape [batch_size, ...].
            # We'll iterate over batch dimension if needed.
            
            # The next lines handle the possibility of multi-sample data:
            # - "attentions" is a dict like: {"layer_0": [batch, heads, seq_len, seq_len], ...}
            # - "input_ids" is [batch, seq_len]
            # - "final_predictions" is a list of length [batch]
            
            if not isinstance(data["input_ids"], torch.Tensor):
                logger.warning(f"Unexpected data format in {pt_file}")
                continue
            
            batch_size = data["input_ids"].size(0)
            for i in range(batch_size):
                yield {
                    "input_ids": data["input_ids"][i],
                    "attentions": {k: v[i] for k, v in data["attentions"].items()}, 
                    # v[i] shape: [heads, seq_len, seq_len]
                    "final_prediction": data["final_predictions"][i]
                }
        except Exception as e:
            logger.error(f"Failed to load '{pt_file}': {e}")

# ------------------------------------------------------------------------
# 4. Main Analysis
# ------------------------------------------------------------------------
results = []
for directory in RUN_DIRECTORIES:
    run_name = os.path.basename(directory.rstrip("/"))  # e.g. "nice" or "mean", etc.
    logger.info(f"--- Analyzing run directory: {run_name} ({directory}) ---")
    
    sample_counter = 0
    for sample_dict in load_activations_from_dir(directory):
        sample_counter += 1
        input_ids = sample_dict["input_ids"].tolist()
        final_pred = sample_dict["final_prediction"]
        attentions_dict = sample_dict["attentions"]

        # Find where the token_of_interest occurs
        positions_of_interest = find_token_positions(input_ids, TOKEN_OF_INTEREST)

        # If none found, skip or record zero attention
        if not positions_of_interest:
            # For example, we record 0.0 as average attention to that token
            # Or you can skip. We'll record zero for demonstration.
            avg_attn = 0.0
        else:
            # Let's just take an average across *all heads* in the specified layers,
            # to the token positions of interest. If HEADS_TO_ANALYZE is given, we
            # can restrict to those heads only.
            # attentions_dict["layer_X"] shape: [num_heads, seq_len, seq_len]
            layer_attentions = []
            for layer_idx_str, attn_tensor in attentions_dict.items():
                # layer_idx_str is something like "layer_0", "layer_5"...
                l_id = int(layer_idx_str.split("_")[1])
                if l_id not in LAYERS_TO_ANALYZE:
                    continue

                # attn_tensor shape: [num_heads, seq_len, seq_len]
                num_heads = attn_tensor.size(0)
                seq_len = attn_tensor.size(1)

                if HEADS_TO_ANALYZE is not None:
                    heads = HEADS_TO_ANALYZE
                else:
                    heads = range(num_heads)

                # For each head, each position i in [positions_of_interest], 
                # we look at how much attention the rest of the tokens place on i.
                # Typically, to measure "attention to token i", we look at attn[:, :, i]
                # But you might interpret it differently. We'll do "attention[: , i]"
                # meaning row i is "attention paid TO token i" if dimension order is
                # [head, dest, src], or [head, src, dest] depending on your model. 
                # Adjust if needed! We'll assume shape is [head, src_token, dest_token]
                # i.e. attn[h, q, k] = how much token q attends to token k. 
                # We'll measure how much all queries attend to the token_of_interest (the key).
                # So we'll sum or average over q dimension.
                
                # If you want "how much the token_of_interest attends to others," invert the indexing.
                
                # We'll do "how much all tokens attend to the token_of_interest," i.e. attn[:, :, i].
                # Then average over 'src' dimension (which is the q dimension).
                
                # For each head h in heads
                # Sum or average over src dimension (the second dimension).
                
                # We'll average attention across all heads, across all tokens, 
                # for the positions of interest, for this layer.
                # Then we can average across layers, or keep them separate. We'll keep them separate
                # for now, just to show how you'd store it.
                
                # We'll do a simple approach: average across heads, average across all queries, 
                # average across each position_of_interest. 
                
                layer_vals = []
                for h in heads:
                    # attn_tensor[h]. shape: [seq_len, seq_len]
                    # We average across ALL query positions for each position_of_interest
                    # Then average across the positions_of_interest.
                    # attn_for_interest = average( attn_tensor[h][:, i] for i in positions_of_interest )
                    sums = 0.0
                    count = 0
                    for pos_i in positions_of_interest:
                        # shape: [seq_len], each index is attention for query j -> key pos_i
                        sums += attn_tensor[h][:, pos_i].sum().item()
                        count += seq_len  # we summed across seq_len queries
                    if count > 0:
                        layer_vals.append(sums / count)

                if layer_vals:
                    # Average across the heads we used
                    avg_layer_val = sum(layer_vals) / len(layer_vals)
                    layer_attentions.append(avg_layer_val)

            if layer_attentions:
                avg_attn = sum(layer_attentions) / len(layer_attentions)
            else:
                avg_attn = 0.0

        # Store result for each sample
        results.append({
            "run_name": run_name,
            "sample_index": sample_counter,
            "has_token_of_interest": bool(positions_of_interest),
            "avg_attention_to_token_of_interest": avg_attn,
            "final_prediction": final_pred
        })

    logger.info(f"Processed {sample_counter} samples from {run_name}")

# ------------------------------------------------------------------------
# 5. Save and Summarize
# ------------------------------------------------------------------------
df_results = pd.DataFrame(results)
df_results.to_csv(OUTPUT_FILE, index=False)
logger.info(f"Saved analysis results to {OUTPUT_FILE}")

# Optional: print a quick groupby summary in console/log
summary = df_results.groupby("run_name")["avg_attention_to_token_of_interest"].mean()
logger.info("=== Average attention to token of interest by run_name ===")
logger.info(f"\n{summary}\n")

logger.info("Analysis complete.")

