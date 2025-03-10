#!/usr/bin/env python3

import os
import glob
import torch
import logging
import pandas as pd
from transformers import AutoTokenizer

# -----------------------------------------------------------------------------
# 1. Configuration and Setup
# -----------------------------------------------------------------------------
OUTPUT_DIR = globals().get("OUTPUT_DIR", "output/extractions/jb")  # Directory containing the .pt files
MODEL_NAME = globals().get("MODEL_NAME", "google/gemma-2-2b")
OUTPUT_CSV = globals().get("OUTPUT_CSV", "analysis/predictions_view.csv")

# Fallback to environment variables
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", OUTPUT_DIR)
MODEL_NAME = os.environ.get("MODEL_NAME", MODEL_NAME)
OUTPUT_CSV = os.environ.get("OUTPUT_CSV", OUTPUT_CSV)

os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

LOG_FILE = globals().get("LOG_FILE", "logs/view_predictions.log")
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

# -----------------------------------------------------------------------------
# 1a. Set up Logging
# -----------------------------------------------------------------------------
logger = logging.getLogger("ViewPredictionsLogger")
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

logger.info("=== Starting script to view predictions from extracted .pt files ===")
logger.info(f"Loading from directory: {OUTPUT_DIR}")
logger.info(f"Using model: {MODEL_NAME}")
logger.info(f"Will save CSV to: {OUTPUT_CSV}")

# -----------------------------------------------------------------------------
# 2. Load Tokenizer (for decoding input_ids if desired)
# -----------------------------------------------------------------------------
logger.info(f"Loading tokenizer from {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
# If pad_token is missing, set it to eos_token
if tokenizer.pad_token is None and tokenizer.eos_token is not None:
    tokenizer.pad_token = tokenizer.eos_token

# -----------------------------------------------------------------------------
# 3. Gather all .pt files and parse them
# -----------------------------------------------------------------------------
pt_files = sorted(glob.glob(os.path.join(OUTPUT_DIR, "*.pt")))
logger.info(f"Found {len(pt_files)} .pt files in {OUTPUT_DIR}")

all_rows = []

for pt_file in pt_files:
    logger.debug(f"Loading file: {pt_file}")
    try:
        data = torch.load(pt_file)
        # data is a dictionary with keys like:
        #   "hidden_states", "attentions", "topk_logits", "topk_indices",
        #   "input_ids", "final_predictions", etc.
        # Because we saved them in batches, each key is typically shaped [batch_size, ...].
        input_ids_batch = data["input_ids"]  # shape [batch, seq_len]
        predictions_batch = data["final_predictions"]  # list of strings, length = batch_size

        # We'll iterate over each item in the batch
        batch_size = input_ids_batch.size(0)
        for i in range(batch_size):
            input_ids = input_ids_batch[i]
            prompt_text = tokenizer.decode(input_ids, skip_special_tokens=True)
            prediction_text = predictions_batch[i]

            # Optional: shorten prompt if it's very long, for display
            short_prompt = (prompt_text[:200] + "...") if len(prompt_text) > 200 else prompt_text

            all_rows.append({
                "file": os.path.basename(pt_file),
                "batch_index": i,
                "prompt_text": short_prompt,
                "full_prompt_text": prompt_text,   # include full if you want
                "prediction": prediction_text
            })

    except Exception as e:
        logger.error(f"Could not load or parse file {pt_file}: {e}")

# -----------------------------------------------------------------------------
# 4. Convert to DataFrame, Inspect, and Save
# -----------------------------------------------------------------------------
df_results = pd.DataFrame(all_rows)
logger.info(f"Collected {len(df_results)} predictions total.")

df_results.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
logger.info(f"Saved all predictions to {OUTPUT_CSV}.")

# If you just want to show a preview in the console:
logger.info("=== Sample of predictions ===")
logger.info(f"\n{df_results.head(10)}")

logger.info("Done viewing predictions.")
