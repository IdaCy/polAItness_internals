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
# 2) HELPER FUNCTIONS FOR DIFFERENCES & PCA
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


def run_pca_on_diffs(diffs, n_components=5, logger=None):
    """
    Takes a list of difference vectors (each shape [hidden_dim]),
    stacks them into NxD, casts to float32, runs PCA (sklearn).

    Returns (pca, X_pca, evr), where
      - pca: the fitted PCA object
      - X_pca: the PCA-transformed data (Nx(n_components))
      - evr: explained_variance_ratio_ array
    """
    if logger is None:
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)

    if not diffs:
        logger.warning("No difference vectors found, returning None.")
        return None, None, None

    big_tensor = torch.stack(diffs, dim=0)  # shape [N, D]
    big_tensor = big_tensor.to(torch.float32)  # Convert from bfloat16 if needed
    X = big_tensor.numpy()

    logger.info(f"Running PCA on shape={X.shape}")
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    evr = pca.explained_variance_ratio_

    logger.info(f"Explained variance ratio (PC1..): {evr}")
    return pca, X_pca, evr


###############################################################################
# 3) MAIN FUNCTION: PCA FOR EACH PROMPT TYPE vs NORMAL
###############################################################################

def analyze_each_prompt_type_separately(
    prompt2data,
    normal_key="normal",
    layers=None,
    n_components=5,
    save_dir=None,
    logger=None
):
    """
    For each prompt type p != normal_key:
      1. Gather difference vectors (p - normal) for each layer
      2. Run PCA on each layer separately (i.e. 1 PCA per layer)
      3. Summarize PC1 explained variance across layers (build a small chart)
      4. Save results

    We produce a single bar chart or line chart for each prompt type showing
    how PC1's explained variance ratio changes across layers.

    We'll store the PCA components in separate .pt files:
       pca_layer_<L>__<prompt_type>.pt

    We'll also store a chart named:
       pc1_across_layers__<prompt_type>.png
    """
    if logger is None:
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)

    if normal_key not in prompt2data:
        logger.error(f"'{normal_key}' not found in prompt2data, aborting.")
        return

    normal_data = prompt2data[normal_key]
    other_prompt_types = [ptype for ptype in prompt2data.keys() if ptype != normal_key]

    # If layers is None, infer them from normal_data's first sample
    if (layers is None) or len(layers) == 0:
        if len(normal_data) == 0:
            logger.warning("No data in normal_data. Nothing to do.")
            return
        example_idx = next(iter(normal_data))
        layers = sorted(list(normal_data[example_idx].keys()))
        logger.info(f"Inferred layers: {layers}")

    for ptype in other_prompt_types:
        logger.info(f"== PCA comparison for {ptype} vs. {normal_key} ==")
        comp_data = prompt2data[ptype]

        # We'll store PC1 ratio for each layer in a list for plotting
        pc1_ratios = []
        # We'll also store all PCA objects if you want them
        layer_to_pca = {}

        for layer in layers:
            # 1) Gather difference vectors for that layer
            diffs = gather_differences_for_layer(normal_data, comp_data, layer, logger=logger)
            if not diffs:
                logger.info(f"No diffs found for layer={layer} in {ptype}, skipping.")
                pc1_ratios.append(0.0)
                continue

            # 2) Run PCA
            pca, X_pca, evr = run_pca_on_diffs(diffs, n_components=n_components, logger=logger)
            if pca is None:
                logger.info(f"PCA returned None for {layer}, skipping.")
                pc1_ratios.append(0.0)
                continue

            # 3) We'll take the first PC's explained variance ratio
            pc1_ratio = evr[0] if len(evr) > 0 else 0.0
            pc1_ratios.append(pc1_ratio)

            layer_to_pca[layer] = {
                "pca": pca,
                "X_pca": X_pca,
                "explained_variance": evr
            }

            # 4) Save if requested
            if save_dir is not None:
                p_save_dir = os.path.join(save_dir, f"{ptype}_vs_{normal_key}")
                os.makedirs(p_save_dir, exist_ok=True)
                save_path = os.path.join(p_save_dir, f"pca_layer_{layer}__{ptype}.pt")
                torch.save({
                    "pca_components": pca.components_,
                    "explained_variance": evr,
                    "mean_": pca.mean_,
                    "X_pca": X_pca
                }, save_path)
                logger.info(f"Saved PCA for {ptype}, layer={layer} => {save_path}")

        # Now let's produce a chart for this prompt type's PC1 across layers
        fig, ax = plt.subplots()
        x_positions = range(len(layers))

        ax.bar(x_positions, pc1_ratios, tick_label=layers)
        ax.set_xlabel("Layer")
        ax.set_ylabel("Explained Variance Ratio (PC1)")
        ax.set_title(f"PC1 across layers: {ptype} vs. {normal_key}")

        if save_dir is not None:
            # We store it in e.g. <save_dir>/nicer_vs_normal/pc1_across_layers__nicer.png
            p_save_dir = os.path.join(save_dir, f"{ptype}_vs_{normal_key}")
            os.makedirs(p_save_dir, exist_ok=True)
            plot_path = os.path.join(p_save_dir, f"pc1_across_layers__{ptype}.png")
            plt.savefig(plot_path)
            logger.info(f"Saved PC1 chart for {ptype} => {plot_path}")

        plt.show()
