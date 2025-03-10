import os
import glob
import torch
import logging
import datetime
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report

# ------------------------------------------------------------------------
# 1. Configuration and Setup
# ------------------------------------------------------------------------

# Directories for "normal" vs "nicer" activation files
FIRST_DIR = (
    os.environ.get("FIRST_DIR")
    or globals().get("FIRST_DIR")
    or "output/extractions/gemma2b/normal"
)
SECOND_DIR = (
    os.environ.get("SECOND_DIR")
    or globals().get("SECOND_DIR")
    or "output/extractions/gemma2b/nicer"
)
LOG_NAME = (
    os.environ.get("LOG_NAME")
    or globals().get("LOG_NAME")
    or "normal_nicer"
)

EXTRACT_HIDDEN_LAYERS = (
    os.environ.get("EXTRACT_HIDDEN_LAYERS")
    or globals().get("EXTRACT_HIDDEN_LAYERS", [0, 5, 10, 15, 20, 25])
)
# If it might come in as a comma-separated string from env, parse it:
if isinstance(EXTRACT_HIDDEN_LAYERS, str):
    EXTRACT_HIDDEN_LAYERS = [int(x.strip()) for x in EXTRACT_HIDDEN_LAYERS.split(",")]

TRAIN_TEST_SPLIT_RATIO = float(
    os.environ.get("TRAIN_TEST_SPLIT_RATIO")
    or globals().get("TRAIN_TEST_SPLIT_RATIO", 0.8)
)

os.makedirs("logs", exist_ok=True)
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_FILE = f"logs/lin_class_" + LOG_NAME + "_{timestamp}.log"

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
console_handler.setFormatter(console_formatter)
logging.getLogger().addHandler(console_handler)

logging.info("Starting linear classification for 'normal' vs 'nicer' script...")
logging.info(f"FIRST_DIR: {FIRST_DIR}")
logging.info(f"SECOND_DIR: {SECOND_DIR}")
logging.info(f"EXTRACT_HIDDEN_LAYERS set to: {EXTRACT_HIDDEN_LAYERS}")
logging.info(f"TRAIN_TEST_SPLIT_RATIO set to: {TRAIN_TEST_SPLIT_RATIO}")
logging.info(f"Logging to: {LOG_FILE}")

# ------------------------------------------------------------------------
# 2. Load Activation Files
# ------------------------------------------------------------------------

def load_activation_files(directory):
    """
    Loads all .pt files from the specified directory.
    Each file is expected to be a dict, e.g.:
        {
          "hidden_states": {"layer_0": tensor_of_shape([batch_size, seq_len, hidden_dim]), ...},
          "attentions": {...},
          ...
        }
    """
    all_files = sorted(glob.glob(os.path.join(directory, "activations_*.pt")))
    if not all_files:
        logging.warning(f"No .pt files found in {directory}")
        return []
    logging.info(f"Found {len(all_files)} activation files in {directory}")
    
    data = []
    for pt_file in all_files:
        try:
            entry = torch.load(pt_file)
            data.append(entry)
        except Exception as e:
            logging.error(f"Could not load {pt_file}: {str(e)}")
    return data

# ------------------------------------------------------------------------
# 3. Build Dataset
# ------------------------------------------------------------------------

def build_dataset(normal_data, nicer_data, extract_layers):
    """
    We want to combine the "normal" data (label=0) and "nicer" data (label=1).
    For each loaded file from each set, we retrieve the specified layers from 
    'hidden_states' -> shape = [batch_size, seq_len, hidden_dim].
    
    We'll flatten across the batch dimension (since each file might contain multiple items),
    then average over tokens in seq_len. We'll skip anything that’s empty or NaN.
    
    Returns a dict: { layer_id: (X, y) } for each layer.
    """
    layer_datasets = {l: [] for l in extract_layers}
    
    # Helper to process data from a single directory (with a given label)
    def process_data(data_list, label):
        """
        data_list: list of dicts loaded from .pt files
        label: 0=normal, 1=nicer
        """
        for entry in data_list:
            hidden_states = entry.get("hidden_states", {})
            # For each layer in extract_layers, retrieve the corresponding activation
            for l in extract_layers:
                layer_key = f"layer_{l}"
                
                act = hidden_states.get(layer_key, None)
                if act is None:
                    # This layer wasn't extracted or doesn't exist in this file
                    continue
                # shape: [batch_size, seq_len, hidden_dim]
                # Convert to float32 for sklearn
                act_np = act.to(torch.float32).cpu().numpy()
                
                # Loop over each item in the batch dimension
                # and produce a single [hidden_dim] vector by averaging across seq_len
                for i in range(act_np.shape[0]):
                    sample_act = act_np[i]  # shape: [seq_len, hidden_dim]
                    if sample_act.shape[0] == 0:
                        logging.warning(
                            f"Skipping layer {layer_key}, batch idx {i} because seq_len=0."
                        )
                        continue
                    
                    vec = sample_act.nicer(axis=0)  # shape=[hidden_dim]
                    
                    if np.isnan(vec).any():
                        logging.warning(
                            f"Skipping layer {layer_key}, batch idx {i} due to NaN in nicer activation."
                        )
                        continue
                    
                    layer_datasets[l].append((vec, label))
    
    # Process normal (label=0) then nicer (label=1)
    process_data(normal_data, 0)
    process_data(nicer_data, 1)
    
    results = {}
    for l in extract_layers:
        data_list = layer_datasets[l]
        if len(data_list) == 0:
            logging.warning(f"No valid data for layer {l} (possibly all empty or missing).")
            continue
        
        X = np.array([item[0] for item in data_list], dtype=np.float32)
        y = np.array([item[1] for item in data_list], dtype=np.int64)
        
        results[l] = (X, y)
    return results

# ------------------------------------------------------------------------
# 4. Train and Evaluate Classifiers
# ------------------------------------------------------------------------

def train_and_evaluate_layer(X, y, layer_id, train_test_ratio=0.8):
    """
    Trains a LogisticRegression classifier on data from a single layer,
    returns classification metrics, logs them, and prints them.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=train_test_ratio, random_state=42, shuffle=True
    )

    clf = LogisticRegression(
        random_state=42,
        max_iter=1000,
        # Possibly adjust regularization, e.g.: C=1.0, penalty='l2'
    )

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")
    report = classification_report(
        y_test, y_pred, labels=[0, 1], target_names=["normal", "nicer"], zero_division=0
    )

    msg = (
        f"Layer {layer_id} -> Accuracy: {acc:.4f}, F1: {f1:.4f}\n"
        f"Classification report:\n{report}"
    )
    logging.info(msg)
    print(msg)

    return {
        "layer": layer_id,
        "accuracy": acc,
        "f1_score": f1,
        "report": report
    }

# ------------------------------------------------------------------------
# 5. Main Script Logic
# ------------------------------------------------------------------------

def main():
    # 1) Load data from both directories
    normal_data = load_activation_files(FIRST_DIR)
    nicer_data = load_activation_files(SECOND_DIR)
    
    if not normal_data and not nicer_data:
        logging.error("No activation data loaded from either directory. Exiting.")
        return
    elif not normal_data:
        logging.error("No activation data loaded from FIRST_DIR; cannot compare. Exiting.")
        return
    elif not nicer_data:
        logging.error("No activation data loaded from SECOND_DIR; cannot compare. Exiting.")
        return
    
    # 2) Build dataset per layer
    layer_data = build_dataset(normal_data, nicer_data, EXTRACT_HIDDEN_LAYERS)
    
    # 3) Train/evaluate a linear model for each layer
    all_results = []
    for layer_id, (X, y) in layer_data.items():
        res = train_and_evaluate_layer(X, y, layer_id, TRAIN_TEST_SPLIT_RATIO)
        all_results.append(res)
    
    # 4) Summarize
    summary = []
    for r in all_results:
        summary.append({
            "Layer": r["layer"],
            "Accuracy": r["accuracy"],
            "F1_score": r["f1_score"]
        })
    if summary:
        df = pd.DataFrame(summary)
        logging.info("Summary of results:")
        logging.info("\n" + df.to_string(index=False))
        print("Summary of results:")
        print(df.to_string(index=False))

    logging.info("Done. All results have been logged and printed.")

if __name__ == "__main__":
    main()
