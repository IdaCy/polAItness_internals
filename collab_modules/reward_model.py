import os
import glob
import json
import csv
import torch
import logging
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline


def load_pt_files(pt_dir, logger=None):
    """
    Scans a directory for *.pt files produced by your inference script.
    Each .pt file should contain:
      - 'final_predictions': list of strings
      - 'original_indices': list of ints
      - possibly other fields: 'hidden_states', 'attentions', ...
    Returns a list of dicts, each dict has:
      {
        'pt_file': str,
        'original_index': int,
        'prediction': str
      }
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    pt_paths = sorted(glob.glob(os.path.join(pt_dir, "*.pt")))
    logger.info(f"Found {len(pt_paths)} .pt files in '{pt_dir}'.")
    all_items = []

    for pt_path in pt_paths:
        logger.debug(f"Loading: {pt_path}")
        data = torch.load(pt_path, map_location="cpu")
        # data should contain 'final_predictions' (list) and 'original_indices' (list).
        final_preds = data.get("final_predictions", [])
        orig_inds = data.get("original_indices", [])

        if len(final_preds) != len(orig_inds):
            logger.warning(f"Mismatch in lengths for {pt_path}: "
                           f"{len(final_preds)} preds vs. {len(orig_inds)} idxs")

        for i, pred_text in enumerate(final_preds):
            item = {
                "pt_file": os.path.basename(pt_path),
                "original_index": orig_inds[i] if i < len(orig_inds) else -1,
                "prediction": pred_text
            }
            all_items.append(item)

    logger.info(f"Loaded total {len(all_items)} predictions from directory.")
    return all_items


def load_reward_model(
    model_name="OpenAssistant/reward-model-deberta-v3-large-v2",
    device=0,
    logger=None
):
    """
    Loads a reward-model pipeline for text-classification from Hugging Face.
    Example: "OpenAssistant/reward-model-deberta-v3-large-v2"
    Returns a HuggingFace pipeline object that can score text.
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    logger.info(f"Loading reward model pipeline: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    rm_pipeline = pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        device=device,
        top_k=None,
        truncation=True
    )
    logger.info("Reward model pipeline loaded.")
    return rm_pipeline


def score_predictions_with_rm(items, rm_pipeline, logger=None):
    """
    Takes a list of dicts (as returned by load_pt_files), each with:
      {
        'pt_file': ...,
        'original_index': ...,
        'prediction': ...
      }
    Uses `rm_pipeline` to score each 'prediction'.
    Returns a new list of dicts:
      {
        'pt_file': ...,
        'original_index': ...,
        'prediction': ...,
        'reward_label': ...,
        'reward_score': ...
      }
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    predictions = [x["prediction"] for x in items]
    logger.info(f"Scoring {len(predictions)} predictions with reward model...")

    # Because large batches might cause OOM, do in mini-batches
    scored_items = []
    batch_size = 8
    for i in range(0, len(items), batch_size):
        batch = items[i : i + batch_size]
        texts = [it["prediction"] for it in batch]
        outputs = rm_pipeline(texts)

        # Each `outputs[j]` is typically a list of length=1 if top_k=None, or length>1 if top_k>1
        # But for "text-classification", it's often just a single dict: {'label': '...', 'score': X}
        for j, out in enumerate(outputs):
            if isinstance(out, list) and len(out) > 0:
                out = out[0]  # if pipeline returns a list for top_k
            result = {
                "pt_file": batch[j]["pt_file"],
                "original_index": batch[j]["original_index"],
                "prediction": batch[j]["prediction"],
                "reward_label": out["label"],
                "reward_score": float(out["score"])
            }
            scored_items.append(result)

    logger.info("Scoring complete.")
    return scored_items


def save_raw_scores_to_csv(scored_items, csv_path, logger=None):
    """
    Saves the raw (pt_file, original_index, prediction, reward_label, reward_score)
    to a CSV file.
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    fieldnames = ["pt_file", "original_index", "prediction", "reward_label", "reward_score"]
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for item in scored_items:
            writer.writerow(item)

    logger.info(f"Saved detailed scores to: {csv_path}")


def compute_and_save_aggregates(scored_items, agg_path, logger=None):
    """
    Computes min, max, avg of 'reward_score' and
    also creates a one-line comma-separated list of all scores plus an overall average.
    Writes the summary to two separate files:
      - {agg_path}.json with min, max, avg
      - {agg_path}.txt with a one-line CSV of scores, e.g. "0.72,0.66,0.99,..., average=0.81"
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    scores = [x["reward_score"] for x in scored_items]
    if len(scores) == 0:
        logger.warning("No scores found, skipping aggregates.")
        return

    mn = min(scores)
    mx = max(scores)
    avg = sum(scores) / len(scores)

    # Save to JSON
    agg_json = {
        "min_score": mn,
        "max_score": mx,
        "avg_score": avg,
        "count": len(scores)
    }
    agg_json_path = agg_path + ".json"
    with open(agg_json_path, "w", encoding="utf-8") as f:
        json.dump(agg_json, f, indent=2)
    logger.info(f"Saved aggregate stats (JSON) => {agg_json_path}")

    # Build one-line CSV of individual scores plus overall average
    # e.g. "0.76,0.43,0.88, average=0.69"
    line_scores = ",".join([f"{s:.3f}" for s in scores])
    line_output = f"{line_scores}, average={avg:.3f}\n"

    # Save to .txt
    agg_txt_path = agg_path + ".txt"
    with open(agg_txt_path, "w", encoding="utf-8") as f:
        f.write(line_output)
    logger.info(f"Saved one-line CSV with scores => {agg_txt_path}")


def score_predictions_and_save(
    pt_dir,
    rm_pipeline,
    raw_csv_path,
    agg_base_path,
    logger=None
):
    """
    High-level convenience function that:
      1) Loads .pt files from `pt_dir`
      2) Scores them with rm_pipeline
      3) Saves raw CSV to `raw_csv_path`
      4) Saves aggregates to `agg_base_path.{json,txt}`
      5) Returns the final list of scored items
    """
    if logger is None:
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)

    all_items = load_pt_files(pt_dir, logger=logger)
    scored = score_predictions_with_rm(all_items, rm_pipeline, logger=logger)

    save_raw_scores_to_csv(scored, raw_csv_path, logger=logger)
    compute_and_save_aggregates(scored, agg_base_path, logger=logger)

    # Return the final list
    return scored
