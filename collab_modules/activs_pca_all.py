import os
import glob
import torch
import logging
from tqdm import tqdm
import matplotlib.pyplot as plt

try:
    from sklearn.decomposition import PCA
except ImportError:
    raise ImportError("scikit-learn is required for PCA. Please 'pip install scikit-learn' first.")


###############################################################################
# 1) LOADING HIDDEN STATES
###############################################################################

def load_hidden_states_from_dir(dir_path, logger=None):
    """
    Loads .pt files from a directory (like 'normal/' or 'insulting/'),
    each containing 'original_indices' and 'hidden_states'.

    Returns a dict:
      {
        original_index_1: { "layer_0": Tensor(seq_len, hidden_dim), "layer_5": ... },
        original_index_2: { ... },
        ...
      }
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
            for layer_key, layer_tensor in hidden_map.items():
                single_example = layer_tensor[b_idx]  # shape [seq_len, hidden_dim]
                data_dict[orig_idx][layer_key] = single_example

    logger.info(f"Loaded hidden states for {len(data_dict)} unique original_indices from '{dir_path}'.")
    return data_dict


def load_all_prompt_types(base_dir, prompt_types, logger=None):
    """
    For each prompt_type in prompt_types, calls load_hidden_states_from_dir.
    Returns a dict:
      {
         "normal": {orig_idx -> {layer -> Tensor(seq_len, hidden_dim)}},
         "insulting": {...},
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


###############################################################################
# 2) DIFFERENCE VECTORS + PCA
###############################################################################

def average_over_tokens(tensor_2d):
    """
    Aggregates across the seq_len dimension to get [hidden_dim].
    (You can change this to return tensor_2d[-1] if you prefer last-token only.)
    """
    return tensor_2d.mean(dim=0)


def gather_differences_for_layer(
    normal_dict,
    compare_dict,
    layer,
    logger=None
):
    """
    For a given layer string (e.g. "layer_10"), gather the difference vectors
    [compare_avg - normal_avg] across all shared original_indices.

    Returns a list of Tensors, each shape [hidden_dim], one per sample.
    """
    if logger is None:
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)

    shared_indices = set(normal_dict.keys()).intersection(set(compare_dict.keys()))
    diffs = []

    for idx in shared_indices:
        # If the layer is missing in either dict, skip
        if layer not in normal_dict[idx] or layer not in compare_dict[idx]:
            continue

        normal_tensor = normal_dict[idx][layer]   # [seq_len, hidden_dim]
        compare_tensor = compare_dict[idx][layer] # [seq_len, hidden_dim]

        normal_avg = average_over_tokens(normal_tensor)
        compare_avg = average_over_tokens(compare_tensor)

        diffs.append(compare_avg - normal_avg)  # shape [hidden_dim]

    return diffs


def gather_all_differences_across_prompt_types(
    prompt2data,
    normal_key="normal",
    layers=None,
    logger=None
):
    """
    Gathers difference vectors for each (layer, prompt_type != normal),
    returning a structure like:
      { layer -> [ big list of difference vectors (one per sample for all ptypes) ] }

    i.e. we combine "nicer - normal", "urgent - normal", "pure - normal", etc.
    into a single list for each layer. This helps us do a single PCA per layer
    across *all* differences from normal.
    """
    if logger is None:
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)

    if normal_key not in prompt2data:
        logger.error(f"normal_key '{normal_key}' not found in prompt2data.")
        return {}

    normal_data = prompt2data[normal_key]
    other_keys = [k for k in prompt2data.keys() if k != normal_key]

    # If layers is None, infer from normal_data
    if layers is None or len(layers) == 0:
        if len(normal_data) == 0:
            logger.warning("No data in normal_data, returning empty.")
            return {}
        example_idx = next(iter(normal_data))
        layers = sorted(list(normal_data[example_idx].keys()))
        logger.info(f"Inferred layers from normal_data: {layers}")

    # We'll store for each layer a big list of difference vectors
    layer_to_diffs = {layer: [] for layer in layers}

    for ptype in other_keys:
        compare_dict = prompt2data[ptype]
        logger.info(f"Gathering differences: {ptype} - {normal_key}")
        for layer in layers:
            diffs_this_layer = gather_differences_for_layer(
                normal_data, compare_dict, layer, logger=logger
            )
            # Extend the big list for that layer
            layer_to_diffs[layer].extend(diffs_this_layer)

    return layer_to_diffs


def run_pca_on_differences(
    diffs,  # list of Tensors shape [hidden_dim], or single big Tensor
    n_components=5,
    logger=None
):
    """
    Takes a list of difference vectors (each shape [hidden_dim]),
    stacks them into NxD, casts to float32, runs PCA (sklearn).

    Returns pca, X_pca, explained_variance_ratio
    """
    if logger is None:
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)

    if not diffs:
        logger.warning("Empty list of diffs, returning None.")
        return None, None, None

    # Stack them into a single NxD Tensor
    big_tensor = torch.stack(diffs, dim=0)  # shape [N, D]
    logger.info(f"Running PCA on difference matrix: shape={big_tensor.shape}")

    # *** CAST to float32 so np conversion won't fail with bfloat16 ***
    big_tensor = big_tensor.to(torch.float32)
    X = big_tensor.numpy()

    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    evr = pca.explained_variance_ratio_

    logger.info(f"Explained variance ratio (first {n_components} comps): {evr}")
    return pca, X_pca, evr


def analyze_dominant_pc_across_layers(
    prompt2data,
    normal_key="normal",
    layers=None,
    n_components=5,
    save_dir=None,
    logger=None
):
    """
    Main pipeline:

    1) Gather difference vectors for each layer, for all prompt types vs. normal.
    2) For each layer, run PCA on the combined difference vectors (ex: "nicer-normal", "pure-normal", etc.).
    3) Print & optionally plot the explained variance ratio.
    4) (Optional) Save principal components to disk.

    Returns a dict:
      layer_to_pca_results = {
         layer_0: { "pca": ..., "X_pca": ..., "explained_variance": ... },
         layer_5: { ... },
         ...
      }
    """
    if logger is None:
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)

    # Gather all difference vectors across all non-normal prompt types
    layer_to_diffs = gather_all_differences_across_prompt_types(
        prompt2data, normal_key=normal_key, layers=layers, logger=logger
    )

    layer_to_pca = {}
    for layer, diffs in layer_to_diffs.items():
        if not diffs:
            logger.warning(f"No diffs found for layer={layer}. Skipping PCA.")
            continue

        pca, X_pca, evr = run_pca_on_differences(diffs, n_components=n_components, logger=logger)
        if pca is None:
            continue

        layer_to_pca[layer] = {
            "pca": pca,
            "X_pca": X_pca,
            "explained_variance": evr
        }

        # Log the top 1-2 components if we have at least 2 comps
        logger.info(
            f"Layer {layer} => PC1 ratio={evr[0]:.4f}" +
            (f", PC2 ratio={evr[1]:.4f}" if len(evr) > 1 else "")
        )

        # Optionally save PCA results
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"pca_{layer}.pt")
            torch.save({
                "pca_components": pca.components_,
                "explained_variance": evr,
                "mean_": pca.mean_,
                "X_pca": X_pca
            }, save_path)
            logger.info(f"Saved PCA result for {layer} => {save_path}")

    # OPTIONAL: create a quick bar chart of PC1 explained variance for each layer
    if layer_to_pca:
        fig, ax = plt.subplots()
        sorted_layers = sorted(layer_to_pca.keys(), key=lambda x: int(x.split('_')[-1]))
        x_positions = range(len(sorted_layers))

        pc1_values = [layer_to_pca[ly]["explained_variance"][0] for ly in sorted_layers]

        ax.bar(x_positions, pc1_values, tick_label=sorted_layers)
        ax.set_ylabel("Explained Variance Ratio of PC1")
        ax.set_xlabel("Layer")
        ax.set_title("Dominant PC1 across Layers (All Prompt Differences vs. Normal)")

        if save_dir is not None:
            plot_path = os.path.join(save_dir, "dominant_PC1_across_layers.png")
            plt.savefig(plot_path)
            logger.info(f"Saved plot => {plot_path}")

        plt.show()

    return layer_to_pca
