import os
import glob
import torch
import logging
from tqdm import tqdm

################################################################################
# 1) LOADING HIDDEN STATES
################################################################################

def load_hidden_states_from_dir(dir_path, logger=None):
    """
    Loads all .pt files in the given directory and aggregates hidden states
    by original_index for each layer. Returns a dictionary:

      {
        original_index_1: {
           "layer_0": Tensor(seq_len, hidden_dim),
           "layer_5": Tensor(seq_len, hidden_dim),
           ...
        },
        original_index_2: { ... },
        ...
      }

    We rely on your run_inf .pt structure:
      - "original_indices": list of ints
      - "hidden_states": dict of layer_name -> FloatTensor(batch, seq_len, hidden_dim)
    """
    if logger is None:
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)

    data_dict = {}
    pt_files = sorted(glob.glob(os.path.join(dir_path, "*.pt")))
    logger.info(f"Found {len(pt_files)} '.pt' files in '{dir_path}'")

    for pt_file in tqdm(pt_files, desc=f"Loading from {dir_path}", leave=False):
        batch_data = torch.load(pt_file, map_location="cpu")

        original_indices = batch_data["original_indices"]
        hidden_map = batch_data["hidden_states"]  # e.g. {"layer_0": Tensor(batch, seq_len, hidden_dim), ...}

        for b_idx, orig_idx in enumerate(original_indices):
            if orig_idx not in data_dict:
                data_dict[orig_idx] = {}
            # For each layer in hidden_map, pick out the b-th slice
            for layer_key, layer_tensor in hidden_map.items():
                single_example = layer_tensor[b_idx]  # shape [seq_len, hidden_dim]
                data_dict[orig_idx][layer_key] = single_example

    logger.info(f"Loaded hidden states for {len(data_dict)} unique original_indices from '{dir_path}'.")
    return data_dict


def load_all_prompt_types(base_dir, prompt_types, logger=None):
    """
    Calls `load_hidden_states_from_dir` for each prompt_type in `prompt_types`.
    Assumes you have directories like: base_dir/<prompt_type>/*.pt

    Returns a dict: { 'normal': {...}, 'insulting': {...}, ... }
    """
    if logger is None:
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)

    prompt2data = {}
    for ptype in prompt_types:
        dir_path = os.path.join(base_dir, ptype)
        logger.info(f"Loading hidden states for prompt_type='{ptype}' from {dir_path}")
        prompt2data[ptype] = load_hidden_states_from_dir(dir_path, logger=logger)

    return prompt2data

################################################################################
# 2) COMPARING 'normal' TO OTHER PROMPT TYPES
################################################################################

def average_over_tokens(tensor_2d):
    """
    CHOICE: average across the seq_len dimension to get a single vector [hidden_dim].
    If you'd rather do 'take the final token', you can do: return tensor_2d[-1]
    """
    return tensor_2d.mean(dim=0)


def compute_difference_vectors_normal_vs_other(
    normal_data_dict,
    compare_data_dict,
    layers=None,
    logger=None
):
    """
    Compares hidden states from 'normal_data_dict' to 'compare_data_dict'.

    For each original_index in the intersection, we:
      1) Find the hidden states in 'normal' at each layer
      2) Find the hidden states in 'compare' at each layer
      3) Average across tokens -> single vector
      4) diff_vec = compare_avg - normal_avg

    Then we average these diff_vecs across *all shared original_indices* to get
    one final difference vector per layer.

    Returns a dict: { layer_key: mean_diff_vector }

    This effectively yields the "average difference" from normal to compare for that layer.
    """
    if logger is None:
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)

    # Figure out which layers are in the data if not explicitly provided
    if (layers is None) or (len(layers) == 0):
        if len(normal_data_dict) == 0:
            logger.warning("No data in normal_data_dict. Returning empty dict.")
            return {}
        example_idx = next(iter(normal_data_dict))
        layers = sorted(list(normal_data_dict[example_idx].keys()))
        logger.info(f"Inferred layers from normal_data_dict: {layers}")

    normal_indices = set(normal_data_dict.keys())
    compare_indices = set(compare_data_dict.keys())
    shared_indices = normal_indices.intersection(compare_indices)
    logger.info(f"Shared original_indices between normal & compare: {len(shared_indices)}")

    # We'll collect sum of differences for each layer, then divide by # of samples
    sum_diffs = {}
    count_diffs = {}
    for layer in layers:
        sum_diffs[layer] = None
        count_diffs[layer] = 0

    for idx in shared_indices:
        # For each layer, if it's present in both
        for layer in layers:
            if (layer not in normal_data_dict[idx]) or (layer not in compare_data_dict[idx]):
                continue
            normal_tensor = normal_data_dict[idx][layer]   # shape [seq_len1, hidden_dim]
            compare_tensor = compare_data_dict[idx][layer] # shape [seq_len2, hidden_dim]

            # CHOICE: average over tokens
            normal_avg = average_over_tokens(normal_tensor)
            compare_avg = average_over_tokens(compare_tensor)

            diff_vec = compare_avg - normal_avg  # shape [hidden_dim]

            if sum_diffs[layer] is None:
                sum_diffs[layer] = torch.zeros_like(diff_vec)
            sum_diffs[layer] += diff_vec
            count_diffs[layer] += 1

    # Now we produce the final average difference per layer
    layer_to_diff = {}
    for layer in layers:
        if count_diffs[layer] > 0:
            avg_diff = sum_diffs[layer] / count_diffs[layer]
            layer_to_diff[layer] = avg_diff
        else:
            logger.warning(f"Layer={layer} had 0 matching samples. Skipping from output.")
    return layer_to_diff


def compute_l2_and_abs_stats(diff_vec):
    """
    For a single difference vector (shape [hidden_dim]), compute:
      - L2 norm
      - mean absolute value
      - max absolute value
    Returns a dict of stats.
    """
    l2 = diff_vec.norm(p=2).item()  # L2 norm
    mean_abs = diff_vec.abs().mean().item()
    max_abs = diff_vec.abs().max().item()
    return {
        "l2_norm": l2,
        "mean_abs": mean_abs,
        "max_abs": max_abs
    }


def compare_normal_to_all(
    prompt2data,
    layers=None,
    normal_key="normal",
    save_dir=None,
    logger=None
):
    """
    Main function:
    - Takes a dict prompt2data: {ptype -> {orig_idx->{layer->Tensor(seq_len,hidden_dim)}}}
    - For each 'ptype' != normal_key, compute difference vectors (normal->ptype).
    - Print stats per layer (L2 norm, etc).
    - Optionally save difference vectors to disk (if save_dir is provided).

    NOTE: "difference" is effectively (compare - normal).
    """
    if logger is None:
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)

    # We assume prompt2data[normal_key] is the base
    if normal_key not in prompt2data:
        logger.error(f"Normal key '{normal_key}' not found in prompt2data. Aborting.")
        return

    normal_data_dict = prompt2data[normal_key]
    other_keys = [k for k in prompt2data.keys() if k != normal_key]

    # If layers not specified, infer from normal_data_dict
    if (layers is None) or (len(layers) == 0):
        if len(normal_data_dict) == 0:
            logger.warning("normal_data_dict is empty. Returning no results.")
            return
        example_idx = next(iter(normal_data_dict))
        layers = sorted(list(normal_data_dict[example_idx].keys()))
        logger.info(f"Inferred layers from normal_data_dict: {layers}")

    results = {}

    for ptype in other_keys:
        logger.info(f"=== Comparing normal -> {ptype} ===")
        compare_dict = prompt2data[ptype]

        diff_dict = compute_difference_vectors_normal_vs_other(
            normal_data_dict,
            compare_dict,
            layers=layers,
            logger=logger
        )
        # Print stats
        for layer, vec in diff_dict.items():
            stats = compute_l2_and_abs_stats(vec)
            logger.info(
                f"[{ptype}] {layer}: L2={stats['l2_norm']:.4f}, "
                f"meanAbs={stats['mean_abs']:.6f}, maxAbs={stats['max_abs']:.6f}"
            )
        results[ptype] = diff_dict

        # Optional: save difference vector
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"diff__{ptype}-normal.pt")
            torch.save(diff_dict, save_path)
            logger.info(f"Saved difference vectors {ptype} - normal => {save_path}")

    return results


################################################################################
# EXAMPLE USAGE (to paste in your notebook cell)
################################################################################

def example_usage():
    import logging
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # from collab_modules.compare_to_normal import (
    #     load_all_prompt_types, compare_normal_to_all
    # )

    base_dir = "/content/drive/MyDrive/polAItness_internals/output/extractions/gemma-2-9b-it"
    prompt_types = ["normal", "nicer", "urgent", "insulting", "reduced", "shortest", "explained", "pure"]

    # 1) Load all prompt types
    prompt2data = load_all_prompt_types(base_dir, prompt_types, logger=logger)

    # 2) Compare normal -> each other
    #    We'll check layers 0,5,10,15,20,25 for example
    layers = ["layer_0", "layer_5", "layer_10", "layer_15", "layer_20", "layer_25"]
    save_dir = "/content/drive/MyDrive/polAItness_internals/analyses/activations_diff/gemma-2-9b-it"

    compare_normal_to_all(
        prompt2data=prompt2data,
        layers=layers,
        normal_key="normal",
        save_dir=save_dir,
        logger=logger
    )

    logger.info("Done comparing normal to all other prompt types.")
