import os
import glob
import torch
from tqdm import tqdm

# Directories containing the separate .pt files from your new inference script
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

DIFF_DIR = (
    os.environ.get("DIFF_DIR")
    or globals().get("DIFF_DIR")
    or "output/differences/normal_nicer"
)
os.makedirs(DIFF_DIR, exist_ok=True)

# List all SECOND .pt files
second_files = sorted(glob.glob(os.path.join(SECOND_DIR, "*.pt")))
first_files      = sorted(glob.glob(os.path.join(FIRST_DIR,      "*.pt")))

if len(second_files) != len(first_files):
    raise ValueError("Mismatch: # of second files != # of first files. Ensure same # of .pt files.")

print(f"Found {len(second_files)} second files in {SECOND_DIR}")
print(f"Found {len(first_files)} first files in {FIRST_DIR}")

for second_file, first_file in tqdm(zip(second_files, first_files),
                                  desc="Computing differences",
                                  total=len(second_files)):

    # Load the .pt data from both sets
    second_data = torch.load(second_file, map_location="cpu")
    first_data      = torch.load(first_file,      map_location="cpu")

    diff_data = {}

    # ----------------------------------------------------------------
    # 1) Hidden states difference
    #    In your new inference script, "hidden_states" is a dict:
    #       { "layer_5": [batch, seq_len, hidden_dim], ... }
    # ----------------------------------------------------------------
    if "hidden_states" in second_data and "hidden_states" in first_data:
        diff_data["hidden_states"] = {}
        for layer_key, second_tensor in second_data["hidden_states"].items():
            if layer_key in first_data["hidden_states"]:
                first_tensor = first_data["hidden_states"][layer_key]
                # Both are shape [batch, seq_len, hidden_dim]
                # Crop the seq dimension if needed
                min_batch  = min(second_tensor.size(0), first_tensor.size(0))
                min_seq    = min(second_tensor.size(1), first_tensor.size(1))
                diff_data["hidden_states"][layer_key] = \
                    second_tensor[:min_batch, :min_seq] - first_tensor[:min_batch, :min_seq]

    # ----------------------------------------------------------------
    # 2) Attentions difference (optional)
    #    In your new script, this is "attentions" not "attention_scores".
    #    The shape: [batch, num_heads, seq_len, seq_len]
    # ----------------------------------------------------------------
    if "attentions" in second_data and "attentions" in first_data:
        diff_data["attentions"] = {}
        for layer_key, second_tensor in second_data["attentions"].items():
            if layer_key in first_data["attentions"]:
                first_tensor = first_data["attentions"][layer_key]
                min_batch = min(second_tensor.size(0), first_tensor.size(0))
                min_heads = min(second_tensor.size(1), first_tensor.size(1))
                min_seq   = min(second_tensor.size(2), first_tensor.size(2))
                diff_data["attentions"][layer_key] = (
                    second_tensor[:min_batch, :min_heads, :min_seq, :min_seq]
                    - first_tensor[:min_batch, :min_heads, :min_seq, :min_seq]
                )

    # ----------------------------------------------------------------
    # 3) Top-K logits difference (optional)
    #    The new script calls it "topk_logits" (not "top_k_logits").
    #    Shape: [batch, seq_len, top_k]
    # ----------------------------------------------------------------
    if "topk_logits" in second_data and "topk_logits" in first_data:
        # We'll store one key: "topk_logits"
        second_topk = second_data["topk_logits"]
        first_topk      = first_data["topk_logits"]
        min_batch    = min(second_topk.size(0), first_topk.size(0))
        min_seq      = min(second_topk.size(1), first_topk.size(1))
        diff_data["topk_logits"] = (
            second_topk[:min_batch, :min_seq] - first_topk[:min_batch, :min_seq]
        )

    # ----------------------------------------------------------------
    # 4) Save the computed difference
    #    We'll base the filename on the second file’s name
    # ----------------------------------------------------------------
    base_filename = os.path.basename(second_file)
    diff_path = os.path.join(DIFF_DIR, base_filename)
    torch.save(diff_data, diff_path)
