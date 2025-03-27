import os
import glob
import torch
import logging
from tqdm import tqdm
import numpy as np

################################################################################
# 1) LOADING HIDDEN STATES FROM .PT FILES
################################################################################

def load_hidden_states_from_dir(dir_path, logger=None):
    """
    Loads all .pt files in the given directory and aggregates hidden states
    by original_index for each layer. Returns a dictionary of the form:

        {
          original_index_1: {
             "layer_0": <FloatTensor shape [seq_len, hidden_dim]>,
             "layer_5": <FloatTensor shape [seq_len, hidden_dim]>,
             ...
          },
          original_index_2: { ... },
          ...
        }

    Assumes each .pt file is the output from run_inf, containing keys like:
      - "original_indices", "hidden_states", ...
      where hidden_states is a dict mapping layer name to
      Tensor(batch, seq_len, hidden_dim).

    If the same original_index appears in multiple .pt files, this function
    will store (or overwrite) based on the last file. Typically each index is
    unique across files, so that won't be an issue.
    """
    if logger is None:
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)

    data_dict = {}  # { original_index: { layer_key: Tensor(seq_len, hidden_dim) } }

    pt_files = sorted(glob.glob(os.path.join(dir_path, "*.pt")))
    logger.info(f"Found {len(pt_files)} '.pt' files in '{dir_path}'")

    for pt_file in tqdm(pt_files, desc=f"Loading from {dir_path}", leave=False):
        batch_data = torch.load(pt_file, map_location="cpu")

        original_indices = batch_data["original_indices"]
        hidden_map = batch_data["hidden_states"]  # e.g. { "layer_0": Tensor(batch, seq_len, hidden_dim), ... }

        # For each item in the batch, store the hidden states
        batch_size = len(original_indices)
        for b in range(batch_size):
            idx = original_indices[b]
            if idx not in data_dict:
                data_dict[idx] = {}

            # For each layer in hidden_map, pick out the b-th slice
            for layer_key, layer_tensor in hidden_map.items():
                # shape of layer_tensor: [batch, seq_len, hidden_dim]
                single_example = layer_tensor[b]  # shape [seq_len, hidden_dim]
                data_dict[idx][layer_key] = single_example

    logger.info(f"Loaded hidden states for {len(data_dict)} unique original_indices from '{dir_path}'.")
    return data_dict


################################################################################
# 2) AGGREGATING LOADED DATA: PROMPT TYPE -> DATA DICT
################################################################################

def load_all_prompt_types(base_dir, prompt_types, logger=None):
    """
    Convenience function that calls `load_hidden_states_from_dir` for each
    prompt_type in `prompt_types`, assuming your run_inf output directories
    are nested like:

        base_dir/<prompt_type>/*.pt

    Returns a dict:
        {
          'normal': { original_idx: {layer_key: Tensor(...)}, ... },
          'insulting': {...},
          'nicer': {...},
          ...
        }
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
# 3) MEAN DIFFERENCE VECTORS
################################################################################

def compute_mean_difference_vectors(
    base_data_dict, 
    compare_data_dict, 
    layers=None,
    logger=None
):
    """
    Takes two dictionaries (e.g. one from 'normal' prompt type, one from 'insulting'),
    finds their intersection of original_indices, and for each layer:
       1) Collect the difference [compare - base] for every token across all sequences,
       2) Compute the mean difference vector (size [hidden_dim]),
       3) Return them as { layer_name: mean_diff_vector }.

    By default, 'layers' should match the layer keys you extracted 
    (e.g. ["layer_0", "layer_5", ...]).
    If not provided, we automatically infer the layers from the first item in base_data_dict.

    NOTE: This function aggregates all tokens across all matching sequences to form 
    one single mean difference vector per layer.
    """
    if logger is None:
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)

    # If layers not given, try inferring from the first item in base_data_dict
    if layers is None or len(layers) == 0:
        if len(base_data_dict) == 0:
            logger.warning("base_data_dict is empty. Returning empty dict.")
            return {}
        example_idx = next(iter(base_data_dict))
        if len(base_data_dict[example_idx]) == 0:
            logger.warning("No layers found in base_data_dict's first example. Returning {}.")
            return {}
        layers = sorted(list(base_data_dict[example_idx].keys()))
        logger.info(f"Inferred layers from base_data_dict: {layers}")

    base_indices = set(base_data_dict.keys())
    compare_indices = set(compare_data_dict.keys())
    shared_indices = base_indices.intersection(compare_indices)
    logger.info(f"Shared original_indices between base & compare: {len(shared_indices)}")

    layer_to_diff = {}
    for layer in layers:
        diffs = []
        for idx in shared_indices:
            if layer not in base_data_dict[idx] or layer not in compare_data_dict[idx]:
                continue  # skip if missing in either

            base_tensor = base_data_dict[idx][layer]     # shape [seq_len, hidden_dim]
            comp_tensor = compare_data_dict[idx][layer]  # shape [seq_len, hidden_dim]

            # We'll do (comp - base) for each token, shape [seq_len, hidden_dim]
            diff_tensor = comp_tensor - base_tensor
            diffs.append(diff_tensor)

        if len(diffs) == 0:
            logger.warning(f"No data found for layer={layer} in intersection. Skipping.")
            continue

        # shape [ total_tokens, hidden_dim ]
        cat_diffs = torch.cat(diffs, dim=0)

        # Mean difference vector
        mean_diff = cat_diffs.mean(dim=0)  # shape [hidden_dim]
        layer_to_diff[layer] = mean_diff

        logger.info(f"Layer={layer}: shape of cat_diffs={cat_diffs.shape}, mean_diff shape={mean_diff.shape}")

    return layer_to_diff


################################################################################
# 4) ALL-PAIRS DIFFERENCES
################################################################################

def compute_all_pairs_mean_differences(
    prompt2data_dict,
    layers=None,
    logger=None
):
    """
    If you have multiple prompt types loaded, e.g.:
       {
         "normal": dict_of_hidden_states_for_each_index,
         "insulting": dict_of_hidden_states_for_each_index,
         "nicer": dict_of_hidden_states_for_each_index,
         ...
       }
    Then this function computes the mean difference vectors for *every* pair
    (compare - base) among them.

    Returns a nested dict, e.g.:
      result[(base_prompt, compare_prompt)] = {
          "layer_0": Tensor(...), 
          "layer_5": Tensor(...), 
          ...
      }

    'layers' can be manually specified or inferred from the first prompt data set.
    """
    if logger is None:
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)

    prompt_types = sorted(prompt2data_dict.keys())
    if len(prompt_types) == 0:
        logger.warning("No prompt types found in prompt2data_dict.")
        return {}

    # If 'layers' not provided, let's infer from the first prompt type
    if layers is None or len(layers) == 0:
        example_ptype = prompt_types[0]
        if len(prompt2data_dict[example_ptype]) == 0:
            logger.warning(f"No data found for first prompt type={example_ptype}, returning empty.")
            return {}
        example_idx = next(iter(prompt2data_dict[example_ptype]))
        layers = sorted(list(prompt2data_dict[example_ptype][example_idx].keys()))
        logger.info(f"Inferred layers from prompt type='{example_ptype}': {layers}")

    results = {}
    for i, base_type in enumerate(prompt_types):
        for j, compare_type in enumerate(prompt_types):
            if i == j:
                continue
            base_data = prompt2data_dict[base_type]
            compare_data = prompt2data_dict[compare_type]
            diff = compute_mean_difference_vectors(
                base_data, compare_data, layers=layers, logger=logger
            )
            results[(base_type, compare_type)] = diff
            logger.info(f"Computed difference: '{compare_type}' - '{base_type}' for layers={layers}")
    return results


################################################################################
# 5) OPTIONAL: PCA OR FURTHER ANALYSES ON DIFFERENCES
################################################################################

def gather_all_token_differences(
    base_data_dict,
    compare_data_dict,
    layer,
    logger=None
):
    """
    Gathers all token-level differences [compare - base] for a single layer
    across the intersection of original indices.

    Returns a (N x hidden_dim) tensor, where N = sum of seq_len across all
    matching samples. If no data, returns None.
    """
    if logger is None:
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)

    base_indices = set(base_data_dict.keys())
    compare_indices = set(compare_data_dict.keys())
    shared_indices = base_indices.intersection(compare_indices)

    all_diffs = []
    for idx in shared_indices:
        if layer not in base_data_dict[idx] or layer not in compare_data_dict[idx]:
            continue

        base_tensor = base_data_dict[idx][layer]    # shape [seq_len, hidden_dim]
        comp_tensor = compare_data_dict[idx][layer] # shape [seq_len, hidden_dim]
        diff_tensor = comp_tensor - base_tensor      # shape [seq_len, hidden_dim]
        all_diffs.append(diff_tensor)

    if not all_diffs:
        logger.warning(f"No overlapping data found for layer={layer}. Returning None.")
        return None

    cat_diffs = torch.cat(all_diffs, dim=0)  # shape [total_tokens, hidden_dim]
    return cat_diffs


def compute_pca_of_differences(
    base_data_dict, 
    compare_data_dict, 
    layer="layer_0", 
    n_components=5,
    logger=None
):
    """
    Collects all difference vectors [compare - base] for the specified layer,
    flattens them into a large matrix, and runs PCA on them.

    Returns (explained_variance_ratio_, components_) from sklearn PCA.
    """
    if logger is None:
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)

    try:
        from sklearn.decomposition import PCA
    except ImportError:
        logger.error("scikit-learn not found, please install it to use PCA.")
        return None, None

    cat_diffs = gather_all_token_differences(base_data_dict, compare_data_dict, layer, logger=logger)
    if cat_diffs is None:
        logger.warning("No diffs found, returning None.")
        return None, None

    logger.info(f"Running PCA on diff tensor: shape={cat_diffs.shape}")
    X = cat_diffs.numpy()
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)

    logger.info(f"Variance ratio for first {n_components} components: {pca.explained_variance_ratio_}")
    return pca.explained_variance_ratio_, pca.components_


################################################################################
# 6) OPTIONAL: SAVING / EXPORTING DIFFERENCE VECTORS
################################################################################

def save_difference_vectors_to_disk(diff_dict, save_path, logger=None):
    """
    Saves a dictionary of layer_name -> mean_diff_vector (Tensor) to a file using torch.save.

    Example 'diff_dict' structure:
      {
        "layer_0": Tensor([hidden_dim]),
        "layer_5": Tensor([hidden_dim]),
        ...
      }
    """
    if logger is None:
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)

    torch.save(diff_dict, save_path)
    logger.info(f"Saved difference vectors to {save_path}")


def save_all_pairwise_differences_to_disk(all_diffs, save_dir, logger=None):
    """
    Given a nested dict of structure:
      all_diffs[(base_prompt, compare_prompt)] = { "layer_0": Tensor(...), ... }

    This function saves each pair's dictionary to a separate file named:
      <save_dir>/diff__compare_prompt--base_prompt.pt
    """
    if logger is None:
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)

    os.makedirs(save_dir, exist_ok=True)
    for (base, compare), layer2vec in all_diffs.items():
        file_name = f"diff__{compare}--{base}.pt"
        file_path = os.path.join(save_dir, file_name)
        torch.save(layer2vec, file_path)
        logger.info(f"Saved difference for {compare} - {base} to {file_path}")
