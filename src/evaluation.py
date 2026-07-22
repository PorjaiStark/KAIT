import os
import sys
import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
from scipy.stats import pearsonr
from sklearn.metrics import r2_score
from sklearn.decomposition import PCA

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.utils.run_logging import Tee, append_run_log


# ======================
# Config
# ======================

PRED_PATHS = [
    "outputs/predictions/test1_prediction.pt",
    "outputs/predictions/test2_prediction.pt",
]

# Base folder for all evaluation output. Each run gets its own timestamped
# subfolder under RUNS_DIR (so repeated runs stop overwriting each other's
# plots/metrics); RUNS_LOG accumulates one summary row per run for later
# comparison across runs.
BASE_OUTPUT_DIR = "outputs/evaluation"
RUNS_DIR = os.path.join(BASE_OUTPUT_DIR, "runs")
RUNS_LOG = os.path.join(BASE_OUTPUT_DIR, "runs_log.csv")

# Reassigned per-run in main() to RUNS_DIR/<run_id>; plot/print helpers
# below read this global so they don't need OUTPUT_DIR threaded through
# every call.
OUTPUT_DIR = BASE_OUTPUT_DIR

NUM_PLOTS = 20

# sample plot x-axis tick spacing, in days (7 = weekly, 14 = biweekly)
SAMPLE_PLOT_TICK_DAYS = 14

# geometry analysis: cap sample size for O(N^2)/heavier operations
MAX_COSINE_SAMPLE = 2000
MAX_PCA_SAMPLE = 20000



# ======================
# Load
# ======================

def load_prediction(path):

    data = torch.load(
        path,
        map_location="cpu"
    )

    predictions = data["prediction"]
    targets = data["target"]
    masks = data["mask"]

    # geometry-analysis captures (added alongside RTX 5090 inference changes;
    # absent for prediction files saved before that change)
    pre_transformer = data.get("pre_transformer")
    transformer_output = data.get("transformer_output")
    future_embedding = data.get("future_embedding")
    sequence_mask = data.get("sequence_mask")
    future_query_mask = data.get("future_query_mask")

    # real calendar dates per sample (list[str] "YYYY-MM-DD"), absent for
    # prediction files saved before this capture was added to inference.py
    future_dates = data.get("future_dates")

    # observe (input history) window, same absence caveat
    observe_ndvi = data.get("observe_ndvi")
    observe_dates = data.get("observe_dates")

    # which training run's checkpoint produced this file, for traceability
    # back to outputs/checkpoints/runs/<checkpoint_run_id>/
    checkpoint_run_id = data.get("checkpoint_run_id")

    return (
        predictions, targets, masks,
        pre_transformer, transformer_output, future_embedding,
        sequence_mask, future_query_mask, future_dates,
        observe_ndvi, observe_dates, checkpoint_run_id,
    )



# ======================
# Metrics
# ======================

def compute_metrics(
    predictions,
    targets,
    masks
):

    errors = []

    sq_errors = []


    pred_all = []
    target_all = []

    # NEW: track which sample (index into predictions/targets/masks list)
    # each flattened point came from, so we can color scatter plots by
    # sample/AOI identity later.
    sample_id_all = []


    for sample_idx, (pred, target, mask) in enumerate(zip(
        predictions,
        targets,
        masks
    )):

        pred = pred.squeeze(-1)
        target = target.squeeze(-1)

        mask = mask.bool()


        pred = pred[mask]
        target = target[mask]


        pred_all.extend(
            pred.tolist()
        )

        target_all.extend(
            target.tolist()
        )

        sample_id_all.extend(
            [sample_idx] * pred.shape[0]
        )


        error = pred - target


        errors.extend(
            error.tolist()
        )


        sq_errors.extend(
            (error ** 2).tolist()
        )


    errors = np.array(errors)


    mae = np.mean(
        np.abs(errors)
    )


    rmse = np.sqrt(
        np.mean(
            np.array(sq_errors)
        )
    )


    pred_all = np.array(pred_all)
    target_all = np.array(target_all)
    sample_id_all = np.array(sample_id_all)


    return (
        mae,
        rmse,
        pred_all,
        target_all,
        sample_id_all
    )



# ======================
# Correlation / R^2
# (NEW: catches "flat prediction" collapse that mean/std alone can miss)
# ======================

def compute_correlation_metrics(pred_all, target_all):

    corr, _ = pearsonr(pred_all, target_all)

    r2 = r2_score(target_all, pred_all)

    return corr, r2



# ======================
# Per-timestep variance
# (NEW: checks whether the model tracks dynamics over the future horizon,
#  or just predicts a near-constant value at every step, per Sample 1 plot)
# ======================

def compute_per_timestep_stats(predictions, targets, masks):
    """
    Assumes all samples share the same future horizon length. Stacks
    per-sample sequences into a (N, T) array using the mask to align
    valid timesteps; samples with a different valid length are skipped
    to keep the stack rectangular.
    """

    pred_rows = []
    target_rows = []

    horizon_lengths = []

    for pred, target, mask in zip(predictions, targets, masks):

        pred = pred.squeeze(-1)
        target = target.squeeze(-1)
        mask = mask.bool()

        pred = pred[mask]
        target = target[mask]

        horizon_lengths.append(pred.shape[0])

    if len(horizon_lengths) == 0:
        return None, None

    common_len = int(np.median(horizon_lengths))

    for pred, target, mask in zip(predictions, targets, masks):

        pred = pred.squeeze(-1)
        target = target.squeeze(-1)
        mask = mask.bool()

        pred = pred[mask]
        target = target[mask]

        if pred.shape[0] != common_len:
            continue

        pred_rows.append(pred.numpy())
        target_rows.append(target.numpy())

    if len(pred_rows) == 0:
        return None, None

    pred_stack = np.stack(pred_rows, axis=0)      # (N, T)
    target_stack = np.stack(target_rows, axis=0)  # (N, T)

    pred_std_per_step = pred_stack.std(axis=0)
    target_std_per_step = target_stack.std(axis=0)

    return pred_std_per_step, target_std_per_step



# ======================
# Within-sample correlation
# (NEW: global correlation mixes cross-sample baseline differences with
#  within-sample temporal dynamics. This isolates the latter -- whether
#  the model tracks the ups/downs *within* each sample's future window,
#  independent of how different one AOI's baseline NDVI is from another's.)
# ======================

def compute_within_sample_correlation(predictions, targets, masks):

    per_sample_corr = []

    for pred, target, mask in zip(predictions, targets, masks):

        pred = pred.squeeze(-1)
        target = target.squeeze(-1)
        mask = mask.bool()

        pred = pred[mask].numpy()
        target = target[mask].numpy()

        if len(pred) < 2:
            continue

        # skip near-constant targets: correlation is undefined/unstable
        if target.std() < 1e-6:
            continue

        corr = np.corrcoef(pred, target)[0, 1]

        if not np.isnan(corr):
            per_sample_corr.append(corr)

    per_sample_corr = np.array(per_sample_corr)

    if len(per_sample_corr) == 0:
        return None, None, per_sample_corr

    return per_sample_corr.mean(), per_sample_corr.std(), per_sample_corr



# ======================
# NEW: Per-sample correlation histogram
# ======================

def plot_within_sample_corr_hist(per_sample_corr):

    if per_sample_corr is None or len(per_sample_corr) == 0:
        print("Skipping within-sample correlation histogram: no valid samples")
        return

    plt.figure(figsize=(6, 4))

    plt.hist(per_sample_corr, bins=30, alpha=0.7, color="teal")

    plt.axvline(0, color="black", linestyle="--", linewidth=1)

    plt.xlabel("Per-sample correlation (prediction vs target)")
    plt.ylabel("Count")
    plt.title("Distribution of within-sample temporal correlation")
    plt.grid()
    plt.tight_layout()

    plt.savefig(
        os.path.join(OUTPUT_DIR, "within_sample_correlation_hist.png"),
        dpi=150
    )

    plt.close()



# ======================
# Plot samples
# ======================

def plot_samples(
    predictions,
    targets,
    masks,
    future_dates=None,
    observe_ndvi=None,
    observe_dates=None
):

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )


    n = min(
        NUM_PLOTS,
        len(predictions)
    )


    if future_dates is None:
        print(
            "Note: no future_dates in prediction file (re-run src/inference.py "
            "to regenerate it) -- sample plots will fall back to a plain "
            "timestep index instead of a true date axis."
        )

    show_observe = (
        future_dates is not None
        and observe_ndvi is not None
        and observe_dates is not None
    )

    if not show_observe and observe_ndvi is None:
        print(
            "Note: no observe_ndvi in prediction file (re-run src/inference.py "
            "to regenerate it) -- sample plots will not show the observed "
            "history window."
        )


    for i in range(n):

        pred = predictions[i].squeeze(-1)
        target = targets[i].squeeze(-1)

        mask = masks[i].bool()


        pred = pred[mask]
        target = target[mask]


        plt.figure(
            figsize=(6,4)
        )


        ax = plt.gca()

        if future_dates is not None:

            x = pd.to_datetime(future_dates[i])

        else:

            x = np.arange(len(target))


        x_true, y_true = x, target.numpy()
        x_pred, y_pred = x, pred.numpy()

        if show_observe:

            x_observe = pd.to_datetime(observe_dates[i])
            y_observe = observe_ndvi[i].numpy()

            ax.plot(
                x_observe,
                y_observe,
                marker=".",
                color="gray",
                label="Observed"
            )

            if len(x_observe) > 0:

                # bridge history -> forecast from the last REAL observed
                # point (not the anchor date itself, which usually has no
                # satellite pass exactly on it) so True/Prediction visually
                # continue from the last known value instead of starting
                # cold at the first future date
                last_observe_date = x_observe[-1]
                last_observe_value = y_observe[-1]

                x_true = pd.DatetimeIndex([last_observe_date]).append(x)
                y_true = np.concatenate([[last_observe_value], target.numpy()])

                x_pred = pd.DatetimeIndex([last_observe_date]).append(x)
                y_pred = np.concatenate([[last_observe_value], pred.numpy()])


        ax.plot(
            x_true,
            y_true,
            marker="o",
            label="True"
        )


        ax.plot(
            x_pred,
            y_pred,
            marker="x",
            label="Prediction"
        )


        if future_dates is not None:

            # true-scale date axis: points sit at their real calendar
            # position (irregular satellite-pass gaps, not evenly spaced),
            # with ticks fixed every SAMPLE_PLOT_TICK_DAYS days
            ax.xaxis.set_major_locator(
                mdates.DayLocator(interval=SAMPLE_PLOT_TICK_DAYS)
            )
            ax.xaxis.set_major_formatter(
                mdates.DateFormatter("%Y-%m-%d")
            )
            plt.xticks(rotation=45, ha="right")

            plt.xlabel("Date")

        else:

            plt.xlabel("Future timestep")

        plt.ylabel(
            "NDVI"
        )


        plt.title(
            f"Sample {i}"
        )


        plt.legend()

        plt.grid()


        plt.tight_layout()


        plt.savefig(
            os.path.join(
                OUTPUT_DIR,
                f"sample_{i}.png"
            ),
            dpi=150
        )


        plt.close()



# ======================
# Histogram
# ======================

def plot_histogram(
    pred_all,
    target_all
):

    plt.figure(
        figsize=(6,4)
    )


    plt.hist(
        target_all,
        bins=50,
        alpha=0.5,
        label="Target"
    )


    plt.hist(
        pred_all,
        bins=50,
        alpha=0.5,
        label="Prediction"
    )


    plt.xlabel(
        "NDVI"
    )

    plt.ylabel(
        "Count"
    )


    plt.legend()

    plt.grid()


    plt.tight_layout()


    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "histogram.png"
        ),
        dpi=150
    )


    plt.close()



# ======================
# Scatter plot (pred vs target) -- ORIGINAL, unchanged
# Best single plot for visually spotting flat-line / collapsed predictions.
# A well-behaved model hugs the y=x diagonal; a collapsed model looks like
# a near-vertical or near-horizontal band.
# ======================

def plot_scatter(pred_all, target_all):

    plt.figure(figsize=(6, 6))

    plt.scatter(
        target_all,
        pred_all,
        s=4,
        alpha=0.15
    )

    lims = [
        min(target_all.min(), pred_all.min()),
        max(target_all.max(), pred_all.max())
    ]

    plt.plot(lims, lims, color="red", linestyle="--", label="y = x")

    plt.xlabel("Target NDVI")
    plt.ylabel("Predicted NDVI")
    plt.title("Prediction vs Target")
    plt.legend()
    plt.grid()
    plt.tight_layout()

    plt.savefig(
        os.path.join(OUTPUT_DIR, "scatter_pred_vs_target.png"),
        dpi=150
    )

    plt.close()



# ======================
# NEW: Scatter plot colored by sample/AOI identity
#
# Purpose: the plain scatter plot mixes points from every sample together,
# so a flat "floor" band could either be (a) many different samples all
# collapsing to the same value, or (b) the model correctly memorizing a
# per-sample/per-AOI baseline level and just failing to track the temporal
# dynamics on top of it. Coloring by sample index lets us tell these apart:
# if each flat/off-diagonal band is dominated by a small number of colors
# (i.e. a few samples), that supports the "model learned per-sample/AOI
# baseline, not real temporal signal" hypothesis discussed earlier.
# ======================

def plot_scatter_by_sample(pred_all, target_all, sample_id_all):

    plt.figure(figsize=(6, 6))

    # Use a cyclic-ish colormap so many distinct samples are still
    # visually distinguishable rather than blurring into one hue.
    scatter = plt.scatter(
        target_all,
        pred_all,
        c=sample_id_all,
        cmap="tab20",
        s=6,
        alpha=0.5
    )

    lims = [
        min(target_all.min(), pred_all.min()),
        max(target_all.max(), pred_all.max())
    ]

    plt.plot(lims, lims, color="red", linestyle="--", label="y = x")

    cbar = plt.colorbar(scatter)
    cbar.set_label("Sample index (proxy for AOI/sequence identity)")

    plt.xlabel("Target NDVI")
    plt.ylabel("Predicted NDVI")
    plt.title("Prediction vs Target (colored by sample)")
    plt.legend()
    plt.grid()
    plt.tight_layout()

    plt.savefig(
        os.path.join(OUTPUT_DIR, "scatter_pred_vs_target_by_sample.png"),
        dpi=150
    )

    plt.close()



# ======================
# NEW: Per-timestep std plot
# ======================

def plot_per_timestep_std(pred_std_per_step, target_std_per_step):

    if pred_std_per_step is None:
        print("Skipping per-timestep std plot: no rectangular horizon found")
        return

    steps = np.arange(len(pred_std_per_step))

    plt.figure(figsize=(6, 4))

    plt.plot(steps, target_std_per_step, marker="o", label="Target std")
    plt.plot(steps, pred_std_per_step, marker="x", label="Prediction std")

    plt.xlabel("Future timestep")
    plt.ylabel("Std across samples")
    plt.title("Per-timestep variance: prediction vs target")
    plt.legend()
    plt.grid()
    plt.tight_layout()

    plt.savefig(
        os.path.join(OUTPUT_DIR, "per_timestep_std.png"),
        dpi=150
    )

    plt.close()



# ======================
# Embedding geometry analysis
#
# Captures what each stage of the model actually does to the
# representation geometry, using the four checkpoints model.py exposes:
#   pre_transformer     -- fusion + positional encoding output (input to
#                           the transformer encoder)
#   transformer_output  -- full observe+future sequence, after the
#                           transformer encoder
#   future_embedding    -- future-only tokens, i.e. the regression head's
#                           input (a subset of transformer_output's values)
#   prediction           -- the regression head's output (already analyzed
#                           by the plots above; not a high-dim embedding)
# ======================

def flatten_valid_tokens(embedding_list, mask_list):
    """
    embedding_list[i]: [T, D] per-sample embedding (possibly padded)
    mask_list[i]:      [T] bool mask, True at real (non-padded) positions

    Returns (N_tokens, D) array of every real token's embedding, pooled
    across all samples.
    """
    vectors = [
        emb.numpy()[mask.bool().numpy()]
        for emb, mask in zip(embedding_list, mask_list)
    ]
    return np.concatenate(vectors, axis=0)


def flatten_matching_values(value_list, mask_list):
    """
    Same masking/flattening as flatten_valid_tokens, but for a parallel
    per-token scalar (e.g. target NDVI) instead of an embedding -- used to
    color PCA scatter plots by a meaningful quantity.
    """
    values = [
        v.squeeze(-1).numpy()[mask.bool().numpy()]
        for v, mask in zip(value_list, mask_list)
    ]
    return np.concatenate(values, axis=0)


def subsample(vectors, max_n, seed=0, *aligned):
    """Randomly subsample vectors (and any aligned arrays) to at most max_n rows."""
    if len(vectors) <= max_n:
        return (vectors, *aligned) if aligned else vectors

    rng = np.random.RandomState(seed)
    idx = rng.choice(len(vectors), max_n, replace=False)

    result = vectors[idx]
    aligned_result = tuple(a[idx] for a in aligned)

    return (result, *aligned_result) if aligned else result


def compute_embedding_geometry(vectors):
    """
    vectors: (N, D) token embeddings from one stage.

    Returns norm stats, mean pairwise cosine similarity (collapse
    indicator -- near 1 means all tokens point the same direction), and
    effective rank (number of PCA components needed for 90% variance --
    low means the representation is concentrated in very few directions).
    """
    norms = np.linalg.norm(vectors, axis=1)

    cos_sample = subsample(vectors, MAX_COSINE_SAMPLE)
    normed = cos_sample / (np.linalg.norm(cos_sample, axis=1, keepdims=True) + 1e-8)
    sim_matrix = normed @ normed.T
    iu = np.triu_indices_from(sim_matrix, k=1)
    mean_cosine_sim = float(sim_matrix[iu].mean())

    pca_sample = subsample(vectors, MAX_PCA_SAMPLE)
    # full-rank (not capped) -- capping components would silently cap
    # effective_rank_90 too, making it useless for confirming healthy
    # (high) diversity and only able to detect collapse
    n_components = min(pca_sample.shape[1], pca_sample.shape[0] - 1)
    pca = PCA(n_components=n_components)
    pca.fit(pca_sample)

    cum_var = np.cumsum(pca.explained_variance_ratio_)
    effective_rank_90 = int(np.searchsorted(cum_var, 0.9) + 1)

    return {
        "norm_mean": float(norms.mean()),
        "norm_std": float(norms.std()),
        "mean_cosine_sim": mean_cosine_sim,
        "effective_rank_90": effective_rank_90,
        "pca": pca,
    }


def plot_geometry_pca_scatter(vectors, title, filename, color_values=None, color_label=None):

    plot_vectors, plot_colors = (
        subsample(vectors, MAX_PCA_SAMPLE, 0, color_values)
        if color_values is not None
        else (subsample(vectors, MAX_PCA_SAMPLE), None)
    )

    pca = PCA(n_components=2)
    proj = pca.fit_transform(plot_vectors)

    plt.figure(figsize=(6, 6))

    if plot_colors is not None:
        sc = plt.scatter(proj[:, 0], proj[:, 1], c=plot_colors, cmap="viridis", s=4, alpha=0.4)
        cbar = plt.colorbar(sc)
        if color_label:
            cbar.set_label(color_label)
    else:
        plt.scatter(proj[:, 0], proj[:, 1], s=4, alpha=0.3)

    var = pca.explained_variance_ratio_
    plt.xlabel(f"PC1 ({var[0]*100:.1f}% var)")
    plt.ylabel(f"PC2 ({var[1]*100:.1f}% var)")
    plt.title(title)
    plt.grid()
    plt.tight_layout()

    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=150)
    plt.close()


def plot_explained_variance_by_stage(stage_pcas, filename="geometry_explained_variance.png"):

    plt.figure(figsize=(7, 5))

    for name, pca in stage_pcas.items():
        cum_var = np.cumsum(pca.explained_variance_ratio_)
        plt.plot(np.arange(1, len(cum_var) + 1), cum_var, marker="o", markersize=3, label=name)

    plt.axhline(0.9, color="black", linestyle="--", linewidth=1, label="90% variance")
    plt.xlabel("Number of principal components")
    plt.ylabel("Cumulative explained variance")
    plt.title("Effective dimensionality by stage")
    plt.legend()
    plt.grid()
    plt.tight_layout()

    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=150)
    plt.close()


def run_geometry_analysis(
    pre_transformer, transformer_output, future_embedding,
    sequence_mask, future_query_mask, targets, masks,
):
    print("\n====================")
    print("Embedding geometry analysis")
    print("====================")

    pre_tf_tokens = flatten_valid_tokens(pre_transformer, sequence_mask)
    post_tf_tokens = flatten_valid_tokens(transformer_output, sequence_mask)
    future_emb_tokens = flatten_valid_tokens(future_embedding, masks)

    # observe vs future token type, for coloring pre/post-transformer scatter
    token_type = flatten_matching_values(
        [m.float().unsqueeze(-1) for m in future_query_mask], sequence_mask
    )

    # target NDVI value, for coloring the future_embedding scatter
    future_target_values = flatten_matching_values(
        [t.clone() for t in targets], masks
    )

    stages = {
        "pre_transformer (before encoder)": pre_tf_tokens,
        "transformer_output (after encoder)": post_tf_tokens,
        "future_embedding (before regression)": future_emb_tokens,
    }

    stage_pcas = {}
    stage_stats = {}

    for name, vectors in stages.items():
        geo = compute_embedding_geometry(vectors)
        stage_pcas[name] = geo["pca"]
        stage_stats[name] = {
            "n": len(vectors),
            "d": int(vectors.shape[1]),
            "norm_mean": geo["norm_mean"],
            "norm_std": geo["norm_std"],
            "mean_cosine_sim": geo["mean_cosine_sim"],
            "effective_rank_90": geo["effective_rank_90"],
        }

        print(f"\n{name}: N={len(vectors)}, D={vectors.shape[1]}")
        print(f"  norm mean/std        : {geo['norm_mean']:.4f} / {geo['norm_std']:.4f}")
        print(f"  mean pairwise cosine : {geo['mean_cosine_sim']:.4f}"
              " (near 1.0 = representation collapse)")
        print(f"  effective rank (90% var): {geo['effective_rank_90']} / {vectors.shape[1]} dims")

    plot_geometry_pca_scatter(
        pre_tf_tokens,
        "Pre-transformer embeddings (colored by observe/future)",
        "geometry_pca_pre_transformer.png",
        color_values=token_type,
        color_label="0=observe, 1=future",
    )

    plot_geometry_pca_scatter(
        post_tf_tokens,
        "Post-transformer embeddings (colored by observe/future)",
        "geometry_pca_transformer_output.png",
        color_values=token_type,
        color_label="0=observe, 1=future",
    )

    plot_geometry_pca_scatter(
        future_emb_tokens,
        "Pre-regression future embeddings (colored by target NDVI)",
        "geometry_pca_future_embedding.png",
        color_values=future_target_values,
        color_label="Target NDVI",
    )

    plot_explained_variance_by_stage(stage_pcas)

    print(f"\nSaved geometry plots to {OUTPUT_DIR}")

    return stage_stats



# ======================
# Main
# ======================

def run_one(pred_path):

    global OUTPUT_DIR

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + os.path.splitext(os.path.basename(pred_path))[0]
    OUTPUT_DIR = os.path.join(RUNS_DIR, run_id)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    real_stdout = sys.stdout
    console_log = open(os.path.join(OUTPUT_DIR, "console_log.txt"), "w")
    sys.stdout = Tee(real_stdout, console_log)

    metrics = {"run_id": run_id, "pred_path": pred_path}

    print(f"Run ID: {run_id}")
    print("Loading inference result...")


    (
        predictions, targets, masks,
        pre_transformer, transformer_output, future_embedding,
        sequence_mask, future_query_mask, future_dates,
        observe_ndvi, observe_dates, checkpoint_run_id,
    ) = load_prediction(
        pred_path
    )

    metrics["checkpoint_run_id"] = checkpoint_run_id
    print(f"Checkpoint run: {checkpoint_run_id}")


    print(
        "Number of samples:",
        len(predictions)
    )


    print(
        "Example prediction shape:",
        predictions[0].shape
    )


    mae, rmse, pred_all, target_all, sample_id_all = compute_metrics(
        predictions,
        targets,
        masks
    )


    print("\n====================")
    print("Evaluation")
    print("====================")


    print(
        f"MAE  : {mae:.6f}"
    )

    print(
        f"RMSE : {rmse:.6f}"
    )

    metrics["mae"] = float(mae)
    metrics["rmse"] = float(rmse)


    print("\n====================")
    print("Distribution check")
    print("====================")


    print(
        f"Prediction mean : {pred_all.mean():.6f}"
    )

    print(
        f"Prediction std  : {pred_all.std():.6f}"
    )


    print(
        f"Target mean     : {target_all.mean():.6f}"
    )

    print(
        f"Target std      : {target_all.std():.6f}"
    )


    ratio = (
        pred_all.std()
        /
        target_all.std()
    )


    print(
        f"\nStd ratio(pred/target): {ratio:.3f}"
    )

    metrics["pred_mean"] = float(pred_all.mean())
    metrics["pred_std"] = float(pred_all.std())
    metrics["target_mean"] = float(target_all.mean())
    metrics["target_std"] = float(target_all.std())
    metrics["std_ratio"] = float(ratio)


    if ratio < 0.5:
        print(
            "WARNING: Prediction variance is much smaller -> possible collapse"
        )


    # ---- correlation / R^2 ----
    print("\n====================")
    print("Correlation check")
    print("====================")

    corr, r2 = compute_correlation_metrics(pred_all, target_all)

    print(f"Pearson correlation : {corr:.4f}")
    print(f"R^2                 : {r2:.4f}")

    metrics["pearson_corr"] = float(corr)
    metrics["r2"] = float(r2)

    if corr < 0.3:
        print(
            "WARNING: Low correlation despite std ratio looking OK -> "
            "model likely not tracking per-sample signal (flat/collapsed prediction)"
        )


    # ---- within-sample (temporal) correlation ----
    print("\n====================")
    print("Within-sample correlation check")
    print("====================")

    within_corr_mean, within_corr_std, per_sample_corr = compute_within_sample_correlation(
        predictions, targets, masks
    )

    if within_corr_mean is not None:
        print(f"Mean per-sample correlation : {within_corr_mean:.4f}")
        print(f"Std of per-sample correlation: {within_corr_std:.4f}")
        print(f"Number of samples used       : {len(per_sample_corr)}")

        print(f"(Global correlation for reference: {corr:.4f})")

        metrics["within_sample_corr_mean"] = float(within_corr_mean)
        metrics["within_sample_corr_std"] = float(within_corr_std)
        metrics["within_sample_corr_n"] = int(len(per_sample_corr))

        if within_corr_mean < 0.3 and corr >= 0.3:
            print(
                "WARNING: Global correlation looks fine but mean per-sample "
                "correlation is low -> the model is likely predicting each "
                "sample's baseline level correctly, but NOT tracking the "
                "temporal ups/downs within each sample's future window."
            )
    else:
        print("Could not compute (no samples with valid variance)")


    # ---- per-timestep variance ----
    print("\n====================")
    print("Per-timestep variance check")
    print("====================")

    pred_std_per_step, target_std_per_step = compute_per_timestep_stats(
        predictions, targets, masks
    )

    if pred_std_per_step is not None:
        print("Pred std per step  :", np.round(pred_std_per_step, 4))
        print("Target std per step:", np.round(target_std_per_step, 4))

        metrics["pred_std_per_step"] = np.round(pred_std_per_step, 6).tolist()
        metrics["target_std_per_step"] = np.round(target_std_per_step, 6).tolist()

        step_ratio = pred_std_per_step / (target_std_per_step + 1e-8)
        low_variance_steps = np.where(step_ratio < 0.3)[0]

        if len(low_variance_steps) > 0:
            print(
                f"WARNING: timesteps {low_variance_steps.tolist()} have prediction "
                "std < 30% of target std -> model is flat at these steps"
            )
    else:
        print("Could not compute (samples have inconsistent horizon lengths)")


    plot_samples(
        predictions,
        targets,
        masks,
        future_dates,
        observe_ndvi,
        observe_dates
    )


    plot_histogram(
        pred_all,
        target_all
    )


    plot_scatter(pred_all, target_all)

    plot_scatter_by_sample(pred_all, target_all, sample_id_all)

    plot_per_timestep_std(pred_std_per_step, target_std_per_step)

    plot_within_sample_corr_hist(per_sample_corr)


    if pre_transformer is not None:
        metrics["embedding_geometry"] = run_geometry_analysis(
            pre_transformer, transformer_output, future_embedding,
            sequence_mask, future_query_mask, targets, masks,
        )
    else:
        print(
            "\nSkipping geometry analysis: prediction file has no "
            "embedding captures (re-run src/inference.py to regenerate it)"
        )


    print(
        "\nSaved to:",
        OUTPUT_DIR
    )

    with open(os.path.join(OUTPUT_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # Flat summary row for the master log -- nested fields (per-timestep
    # arrays, per-stage geometry) are dropped here since they don't fit a
    # CSV column; the full detail lives in this run's metrics.json.
    log_row = {
        k: v for k, v in metrics.items()
        if not isinstance(v, (list, dict))
    }
    log_row["timestamp"] = datetime.now().isoformat(timespec="seconds")
    if "embedding_geometry" in metrics:
        for stage_name, stats in metrics["embedding_geometry"].items():
            short = stage_name.split(" ")[0]  # e.g. "pre_transformer"
            log_row[f"{short}_effective_rank_90"] = stats["effective_rank_90"]
            log_row[f"{short}_mean_cosine_sim"] = stats["mean_cosine_sim"]

    append_run_log(log_row, RUNS_LOG)

    print(f"Appended run summary to: {RUNS_LOG}")

    sys.stdout = real_stdout
    console_log.close()


def main():

    for pred_path in PRED_PATHS:

        if not os.path.isfile(pred_path):
            print(f"Skipping {pred_path}: file not found")
            continue

        run_one(pred_path)



if __name__ == "__main__":
    main()