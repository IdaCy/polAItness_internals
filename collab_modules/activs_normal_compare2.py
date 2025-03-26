import os
import glob
import torch
import logging
from tqdm import tqdm
import matplotlib.pyplot as plt

def load_hidden_states_from_dir(dir_path, logger=None):
    if logger is None:
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)

    data_dict = {}
    pt_files = sorted(glob.glob(os.path.join(dir_path, "*.pt")))
    logger.info(f"Found {len(pt_files)} '.pt' files in '{dir_path}'")

    for pt_file in tqdm(pt_files, desc=f"Loading from {dir_path}", leave=False):
        batch_data = torch.load(pt_file, map_location="cpu")
        original_indices = batch_data["original_indices"]
        hidden_map = batch_data["hidden_states"]  # {"layer_0": Tensor(batch, seq_len, hidden_dim), ...}

        for b_idx, orig_idx in enumerate(original_indices):
            if orig_idx not in data_dict:
                data_dict[orig_idx] = {}
            for layer_key, layer_tensor in hidden_map.items():
                single_example = layer_tensor[b_idx]  # shape [seq_len, hidden_dim]
                data_dict[orig_idx][layer_key] = single_example

    logger.info(f"Loaded hidden states for {len(data_dict)} unique original_indices from '{dir_path}'.")
    return data_dict


def load_all_prompt_types(base_dir, prompt_types, logger=None):
    if logger is None:
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)

    prompt2data = {}
    for ptype in prompt_types:
        dir_path = os.path.join(base_dir, ptype)
        logger.info(f"Loading hidden states for prompt_type='{ptype}' from {dir_path}")
        prompt2data[ptype] = load_hidden_states_from_dir(dir_path, logger=logger)

    return prompt2data


def average_over_tokens(tensor_2d):
    """Aggregate hidden states across the seq_len dimension -> single vector [hidden_dim]."""
    return tensor_2d.mean(dim=0)


def compute_difference_vectors_normal_vs_other(
    normal_data_dict,
    compare_data_dict,
    layers=None,
    logger=None
):
    if logger is None:
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)

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

    sum_diffs = {}
    count_diffs = {}
    for layer in layers:
        sum_diffs[layer] = None
        count_diffs[layer] = 0

    for idx in shared_indices:
        for layer in layers:
            if (layer not in normal_data_dict[idx]) or (layer not in compare_data_dict[idx]):
                continue

            normal_tensor = normal_data_dict[idx][layer]   # shape [seq_len1, hidden_dim]
            compare_tensor = compare_data_dict[idx][layer] # shape [seq_len2, hidden_dim]

            normal_avg = average_over_tokens(normal_tensor)
            compare_avg = average_over_tokens(compare_tensor)

            diff_vec = compare_avg - normal_avg  # shape [hidden_dim]

            if sum_diffs[layer] is None:
                sum_diffs[layer] = torch.zeros_like(diff_vec)
            sum_diffs[layer] += diff_vec
            count_diffs[layer] += 1

    layer_to_diff = {}
    for layer in layers:
        if count_diffs[layer] > 0:
            avg_diff = sum_diffs[layer] / count_diffs[layer]
            layer_to_diff[layer] = avg_diff
        else:
            logger.warning(f"Layer={layer} had 0 matching samples. Skipping from output.")

    return layer_to_diff


def compute_l2_and_abs_stats(diff_vec):
    l2 = diff_vec.norm(p=2).item()
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
    Compares each prompt type to "normal", logs difference stats, and also
    generates a single line chart of L2 norms across the specified layers.

    If save_dir is given, difference vectors are saved as .pt files AND the plot
    is saved as "compare_normal_L2_plot.png".
    """

    if logger is None:
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)

    if normal_key not in prompt2data:
        logger.error(f"Normal key '{normal_key}' not found in prompt2data. Aborting.")
        return

    normal_data_dict = prompt2data[normal_key]
    other_keys = [k for k in prompt2data.keys() if k != normal_key]

    # Infer layers if not provided
    if (layers is None) or (len(layers) == 0):
        if len(normal_data_dict) == 0:
            logger.warning("normal_data_dict is empty. Returning no results.")
            return
        example_idx = next(iter(normal_data_dict))
        layers = sorted(list(normal_data_dict[example_idx].keys()))
        logger.info(f"Inferred layers from normal_data_dict: {layers}")

    # Prepare to store L2 norms for plotting
    # We'll store a list of L2 norms in the order of layers: {ptype -> [l2_layer0, l2_layer5, ...]}
    ptype_l2_values = {}

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

        # We'll gather L2 norms in the same layer order for plotting
        l2_list = []

        # Print stats per layer
        for layer in layers:
            if layer not in diff_dict:
                # Possibly a missing layer
                l2_list.append(0.0)
                continue

            vec = diff_dict[layer]
            stats = compute_l2_and_abs_stats(vec)
            logger.info(
                f"[{ptype}] {layer}: L2={stats['l2_norm']:.4f}, "
                f"meanAbs={stats['mean_abs']:.6f}, maxAbs={stats['max_abs']:.6f}"
            )
            l2_list.append(stats['l2_norm'])

        ptype_l2_values[ptype] = l2_list
        results[ptype] = diff_dict

        # Save difference vectors to disk if requested
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"diff__{ptype}-normal.pt")
            torch.save(diff_dict, save_path)
            logger.info(f"Saved difference vectors {ptype} - normal => {save_path}")

    # --- Make a single plot of L2 norms across layers for all prompt types ---
    if len(ptype_l2_values) > 0:
        fig, ax = plt.subplots()

        # We'll convert the layer names (e.g. 'layer_0', 'layer_5', ...) to something nice on the x-axis
        # e.g. x = 0..len(layers)-1, and the ticks are the layer names
        x_positions = range(len(layers))
        ax.set_xticks(list(x_positions))
        ax.set_xticklabels(layers, rotation=0)  # Show layer names as ticks

        # Now plot one line per prompt type
        for ptype, l2_list in ptype_l2_values.items():
            ax.plot(x_positions, l2_list, label=ptype)

        ax.set_xlabel("Layer")
        ax.set_ylabel("L2 Norm of (prompt_type - normal)")
        ax.set_title("Comparison of Prompt Type vs. Normal (Average Hidden State Differences)")
        ax.legend()

        # Save the figure if we have a directory
        if save_dir is not None:
            plot_path = os.path.join(save_dir, "compare_normal_L2_plot.png")
            plt.savefig(plot_path)
            logger.info(f"Saved L2 norm plot to {plot_path}")

        # Show in notebook
        plt.show()

    return results
