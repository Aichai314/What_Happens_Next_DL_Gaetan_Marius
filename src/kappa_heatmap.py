"""
Cohen Kappa Score Heatmap for MoE Diversity Analysis

Computes pairwise Cohen Kappa scores between models from their prediction
probabilities and displays the result as a seaborn heatmap.

Usage:
    python kappa_heatmap.py

Expected input: a dict mapping model names to numpy arrays of shape (N, C)
containing class probabilities (or logits) for N samples and C classes.
"""
import hydra
import torch
import numpy as np
from pathlib import Path
from omegaconf import DictConfig
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import cohen_kappa_score
from scipy.stats import entropy
from itertools import combinations
from evaluate_ensemble import _get_cache_path


def compute_kappa_matrix(
    model_probs: dict[str, np.ndarray],
) -> tuple[np.ndarray, list[str]]:
    """
    Computes the pairwise Cohen Kappa matrix from probability arrays.

    Args:
        model_probs: dict mapping model name -> (N, C) probability array

    Returns:
        kappa_matrix : (M, M) symmetric matrix of kappa scores
        model_names  : ordered list of model names
    """
    model_names = list(model_probs.keys())
    M = len(model_names)

    # Convert probabilities to hard predictions
    preds = {
        name: np.argmax(probs, axis=1)
        for name, probs in model_probs.items()
    }

    kappa_matrix = np.ones((M, M))  # diagonal = 1 by definition

    for i, j in combinations(range(M), 2):
        name_i, name_j = model_names[i], model_names[j]
        kappa = cohen_kappa_score(preds[name_i], preds[name_j])
        kappa_matrix[i, j] = kappa
        kappa_matrix[j, i] = kappa  # symmetric

    return kappa_matrix, model_names

def compute_kl_matrix(
    model_probs: dict[str, np.ndarray],
    epsilon: float = 1e-10
) -> tuple[np.ndarray, list[str]]:
    """
    Moyenne symétrique de la divergence KL entre distributions.
    Capture la diversité au niveau des probabilités, pas juste des prédictions.
    """
    model_names = list(model_probs.keys())
    M = len(model_names)
    kl_matrix = np.zeros((M, M))

    for i, j in combinations(range(M), 2):
        p = model_probs[model_names[i]] + epsilon
        q = model_probs[model_names[j]] + epsilon
        # KL symétrique : moyenne de KL(p||q) et KL(q||p)
        kl_pq = entropy(p.T, q.T).mean()
        kl_qp = entropy(q.T, p.T).mean()
        sym_kl = (kl_pq + kl_qp) / 2
        kl_matrix[i, j] = sym_kl
        kl_matrix[j, i] = sym_kl

    return kl_matrix, model_names


def plot_kappa_heatmap(
    kappa_matrix: np.ndarray,
    model_names: list[str],
    output_path: str | None = "kappa_heatmap.png",
) -> None:
    """
    Plots the Cohen Kappa matrix as a seaborn heatmap.

    Args:
        kappa_matrix : (M, M) symmetric matrix
        model_names  : list of model names for axis labels
        output_path  : if provided, saves the figure to this path
    """
    # Shorten names for readability (keep filename stem only)
    short_names = [n.split("/")[-1].replace(".pt", "") for n in model_names]

    fig, ax = plt.subplots(
        figsize=(max(8, len(model_names) * 0.9), max(6, len(model_names) * 0.8))
    )

    mask = np.zeros_like(kappa_matrix, dtype=bool)
    mask[np.triu_indices_from(mask, k=1)] = True  # mask upper triangle (redundant)

    sns.heatmap(
        kappa_matrix,
        mask=mask,
        annot=True,
        fmt=".3f",
        cmap="RdYlGn_r",   # red = high agreement (redundant), green = low (diverse)
        vmin=0.0,
        vmax=1.0,
        linewidths=0.5,
        linecolor="white",
        xticklabels=short_names,
        yticklabels=short_names,
        ax=ax,
        annot_kws={"size": 9},
    )

    ax.set_title("Pairwise Cohen Kappa — lower is more diverse", fontsize=13, pad=14)
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Heatmap saved to {output_path}")

    plt.show()


def plot_kappa_kl_heatmap(
    kappa_matrix: np.ndarray,
    kl_matrix: np.ndarray,
    model_names: list[str],
    output_path: str | None = "kappa_kl_heatmap.png",
) -> None:
    """
    Plots both Cohen Kappa and KL divergence matrices side by side.

    Args:
        kappa_matrix : (M, M) symmetric Cohen Kappa matrix
        kl_matrix    : (M, M) symmetric KL divergence matrix
        model_names  : list of model names for axis labels
        output_path  : if provided, saves the figure to this path
    """
    # Shorten names for readability (keep filename stem only)
    short_names = [n.split("/")[-1].replace(".pt", "") for n in model_names]

    fig, axes = plt.subplots(
        2, 1,
        figsize=(max(16, len(model_names)), max(6, len(model_names) * 1.8))
    )

    mask = np.zeros_like(kappa_matrix, dtype=bool)
    mask[np.triu_indices_from(mask, k=1)] = True  # mask upper triangle (redundant)

    # Plot Kappa matrix
    sns.heatmap(
        kappa_matrix,
        mask=mask,
        annot=True,
        fmt=".3f",
        cmap="RdYlGn_r",   # red = high agreement (redundant), green = low (diverse)
        vmin=0.0,
        vmax=1.0,
        linewidths=0.5,
        linecolor="white",
        xticklabels=short_names,
        yticklabels=short_names,
        ax=axes[0],
        annot_kws={"size": 8},
        cbar_kws={"label": "Cohen Kappa"},
    )
    axes[0].set_title("Cohen Kappa — lower is more diverse", fontsize=12, pad=14)
    axes[0].set_xticklabels(short_names, rotation=45, ha="right", fontsize=8)
    axes[0].set_yticklabels(short_names, rotation=0, fontsize=8)

    # Plot KL divergence matrix
    sns.heatmap(
        kl_matrix,
        mask=mask,
        annot=True,
        fmt=".3f",
        cmap="YlOrRd",     # higher KL = more diverse
        linewidths=0.5,
        linecolor="white",
        xticklabels=short_names,
        yticklabels=short_names,
        ax=axes[1],
        annot_kws={"size": 8},
        cbar_kws={"label": "Symmetric KL Divergence"},
    )
    axes[1].set_title("Symmetric KL Divergence — higher is more diverse", fontsize=12, pad=14)
    axes[1].set_xticklabels(short_names, rotation=45, ha="right", fontsize=8)
    axes[1].set_yticklabels(short_names, rotation=0, fontsize=8)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Heatmap saved to {output_path}")

    plt.show()

def load_model_probabilities(model_path: str, val_dir: str, tta: bool) -> np.ndarray:
    """
    Placeholder function to load model probabilities from checkpoints.

    In practice, replace this with your actual inference code that loads each
    model, runs it on the validation set, and collects the predicted probabilities.

    Args:
        model_path: path to a single checkpoint
        val_dir: path to the validation directory
        tta: whether to use test-time augmentation

    Returns:
        (N, C) probability array
    """
    cache_path = _get_cache_path(model_path, val_dir, TTA=tta)
    if cache_path.exists():
        print(f"Loading cached logits from {cache_path.name}...")
        expert_probs = np.load(cache_path)
        return expert_probs
    else:
        print(f"Cache not found for {model_path}. Please run inference to generate {cache_path.name}.")
        raise FileNotFoundError(f"Missing cache file: {cache_path}")


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # =========================================================
    # CONFIGURATION: Define your Kaggle Roster here
    # =========================================================
    # my_models = [
    #     "checkpoints/best_model_tsm_36-03.pt",
    #     "checkpoints/tsm_full_tdm_34-50.pt",
    #     "checkpoints/low_overfit_tdm_6channels_34-94.pt",
    #     "checkpoints/tsm_6channels_stem_37-95.pt",
    #     "checkpoints/attn_stage2_best_38-99.pt",
    #     "checkpoints/best_model_cnn_lstm_30-75.pt",
    #     "checkpoints/best_model_trn_29-53.pt",
    #     "checkpoints/best_model_x3d_xs_29-44.pt",
    #     "checkpoints/best_model_r2plus1d_30-97.pt",
    #     #"checkpoints/best_model_cnn_gru_30-54.pt",
    #     #"checkpoints/cnn_lstm_6channels_35-20.pt",
    #     "checkpoints/convnext_best_27-04.pt",
    #     "checkpoints/timesformer_best_24-33.pt",
    #     #"checkpoints/mobilenet_spatial_expert_38-09.pt",
    #     "checkpoints/mobilenet_motion_expert_33-92.pt",
    #     #"checkpoints/tsm_tdm_6channels_36_28.pt",
    #     #"checkpoints/mobilenet_6channels_37-58.pt",
    #     "checkpoints/efficientnet_6channels_39-78.pt",
    #     "checkpoints/efficientnet_attn_40-79.pt",
    #     "checkpoints/efficientnet_spatial_40-96.pt",
    #     "checkpoints/best_model_cnn_lstm_31-71.pt",
    #     "checkpoints/best_model_trn_32-90.pt",
    #     "checkpoints/efficientnet_tdm_39-87.pt",
    # ]
    my_models = [
        "checkpoints/timesformer_best_24-98.pt",
        "checkpoints/attn_stage2_best_38-99.pt",
        "checkpoints/best_model_cnn_lstm_30-75.pt",
        "checkpoints/best_model_trn_32-90.pt",
        "checkpoints/best_model_x3d_xs_29-64.pt",
        "checkpoints/R2Plus1D_high_ov_34-29.pt",
        "checkpoints/convnextv2_nano_30-36.pt",
        "checkpoints/efficientnetb0_motion_37-33.pt",
        "checkpoints/efficientnetb0_spatial_41-59.pt",
        "checkpoints/efficientnetb0_spatial_assym_41-11.pt",
        "checkpoints/efficientnetb0_tdn_40-13.pt",
        "checkpoints/efficientformer_tsm_attn_35-67.pt",
        "checkpoints/coatnet_tsm_37-21.pt",
        "checkpoints/mae_small_phase2_22-22.pt",
        "checkpoints/resnext_tsm_39-76.pt",
    ]

    val_dir = str(Path(cfg.dataset.val_dir).resolve())
    
    model_probs = {
        name: load_model_probabilities(name, val_dir, tta=True) for name in my_models
    }
    
    kappa_matrix, model_names = compute_kappa_matrix(model_probs)
    kl_matrix, _ = compute_kl_matrix(model_probs)
    plot_kappa_kl_heatmap(kappa_matrix, kl_matrix, model_names, output_path="kappa_kl_heatmap.png")

# ── Example usage ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    main()
