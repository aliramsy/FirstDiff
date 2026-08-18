from datasets import *
from torch.utils.data import DataLoader
from solver import *
from model.uncdit import UnconditionalTimeSeriesDiT

import torch
import torch.nn.functional as F

from sklearn.preprocessing import StandardScaler
from sklearn.covariance import LedoitWolf

import numpy as np
import pandas as pd
import csv
import os
import time


# ============================================================
# CONFIGURATION
# ============================================================

BATCH_SIZE = 128

WINDOW_SIZE = 96
WINDOW_SIZE_TRAIN = 128
STRIDE_TRAIN = 64
STRIDE = 96

GPU_ID = 0

# IMPORTANT:
# We train for exactly ONE epoch.
EPOCHS = 1

NOISE_STEPS = 100

HISTORY_SIZE = 1024
HEAD_NUMBER = 8
DiT_num_layers = 6
RUN = 1

INFO = f"GAT_SA_EXPTMP_TMPTMP_DEL_RUN_{RUN}"

D_MODEL = 256
HISTORY = True

# ------------------------------------------------------------
# Timing configuration
# ------------------------------------------------------------

# Number of timing repetitions after GPU warm-up.
NUM_TIMING_RUNS = 100

# Number of warm-up runs.
NUM_WARMUP_RUNS = 10

# Random seed for selecting the test window.
TIMING_SEED = 12345


# ============================================================
# DATA PATHS
# ============================================================

TRAIN_PATH = {
    'MSL': "datasets/MSL/MSL_train.npy",
    'SMAP': "datasets/SMAP/SMAP_train.npy",
    'SWaT': "datasets/SWaT/swat_train2.csv",
    'PSM': "datasets/PSM/train.csv",
    'SMD': "datasets/SMD/train.npy"
}

TEST_PATH = {
    'MSL': "datasets/MSL/MSL_test.npy",
    'SMAP': "datasets/SMAP/SMAP_test.npy",
    'SWaT': "datasets/SWaT/swat2.csv",
    'PSM': "datasets/PSM/test.csv",
    'SMD': "datasets/SMD/test.npy"
}

TEST_LABEL_PATH = {
    'MSL': "datasets/MSL/MSL_test_label.npy",
    'SMAP': "datasets/SMAP/SMAP_test_label.npy",
    'SWaT': None,
    'PSM': "datasets/PSM/test_label.csv",
    'SMD': "datasets/SMD/test_label.npy"
}

CHANNELS = {
    'MSL': 55,
    'SMAP': 25,
    'SWaT': 51,
    'PSM': 25,
    'SMD': 38
}

datasets = ["SMD", "SMAP", "SWaT", "MSL", "PSM"]


# ============================================================
# DEVICE
# ============================================================

if torch.cuda.is_available():
    device = f'cuda:{GPU_ID}'
else:
    device = 'cpu'

print("=" * 70)
print("DEVICE:", device)
print("=" * 70)


# ============================================================
# RANDOM SEEDS
# ============================================================

torch.manual_seed(42)
np.random.seed(42)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)


# ============================================================
# HELPER: CUDA SYNCHRONIZATION
# ============================================================

def synchronize():
    """
    Synchronize CUDA before/after timing.

    CUDA operations are asynchronous, so measuring without
    synchronization can significantly underestimate inference time.
    """
    if torch.cuda.is_available():
        torch.cuda.synchronize()


# ============================================================
# ONE-STEP MAHALANOBIS
# ============================================================

@torch.no_grad()
def one_step_mahalanobis(
    model,
    diffusion,
    x_clean,
    mu_eps,
    inv_cov_eps,
    device
):
    """
    One-step Mahalanobis inference.

    Pipeline:

        clean x
          |
          v
        x_t at t = noise_steps - 1
          |
          v
        ONE model forward pass
          |
          v
        predicted noise
          |
          v
        Mahalanobis distance

    This follows the one-step prediction used in the existing
    Solver implementation.
    """

    batch_size = x_clean.shape[0]

    # Start from the maximum diffusion timestep,
    # exactly as in the current inference code.
    noise_steps = torch.full(
        size=(batch_size,),
        fill_value=diffusion.noise_steps - 1,
        device=device,
        dtype=torch.long
    )

    # Add noise at t = 99.
    x_t, _ = diffusion.noise_time_series(
        x_clean,
        noise_steps
    )

    # ONE model forward pass.
    predicted_noise = model(
        x_curr=x_t,
        t=noise_steps
    )

    # Flatten exactly as in Solver.test().
    B, T, D = predicted_noise.shape

    pred_eps = predicted_noise.reshape(B * T, D)

    # Mahalanobis distance.
    diff_eps = pred_eps - mu_eps

    maha = torch.sqrt(
        torch.einsum(
            'bi,ij,bj->b',
            diff_eps,
            inv_cov_eps,
            diff_eps
        )
    )

    # Return a score for each time point.
    maha = maha.reshape(B, T)

    return maha


# ============================================================
# FULL-STEP RECONSTRUCTION
# ============================================================

@torch.no_grad()
def full_step_reconstruction(
    model,
    diffusion,
    x_clean,
    device
):
    """
    Full reverse-diffusion reconstruction.

    Starts at t = 99 and performs:

        99 -> 98 -> ... -> 0

    Therefore, with NOISE_STEPS=100, this performs
    100 model forward passes.
    """

    batch_size = x_clean.shape[0]

    # Start from maximum diffusion timestep.
    noise_steps = torch.full(
        size=(batch_size,),
        fill_value=diffusion.noise_steps - 1,
        device=device,
        dtype=torch.long
    )

    # Generate noisy input.
    x, _ = diffusion.noise_time_series(
        x_clean,
        noise_steps
    )

    # Full reverse diffusion.
    for j in range(
        diffusion.noise_steps - 1,
        -1,
        -1
    ):

        t = (
            j *
            torch.ones(
                x.shape[0],
                device=device
            )
        ).long()

        # One model forward pass.
        predicted_noise = model(
            x_curr=x,
            t=t
        )

        alpha = diffusion.alpha[t][:, None, None]
        alpha_hat = diffusion.alpha_hat[t][:, None, None]
        beta = diffusion.beta[t][:, None, None]

        if j > 0:
            noise = torch.randn_like(x)
        else:
            noise = torch.zeros_like(x)

        x = (
            1.0 / torch.sqrt(alpha)
            * (
                x
                - (
                    (1.0 - alpha)
                    / torch.sqrt(1.0 - alpha_hat)
                )
                * predicted_noise
            )
            + torch.sqrt(beta) * noise
        )

    return x


# ============================================================
# COMPUTE MAHALANOBIS STATISTICS
# ============================================================

@torch.no_grad()
def compute_mahalanobis_statistics(
    model,
    diffusion,
    val_loader,
    device
):
    """
    Calculate mu_eps and inverse covariance using the validation set.

    This follows the logic in Solver.val():

        predicted epsilon -> LedoitWolf -> mu_eps
        covariance -> inverse covariance

    This computation is NOT included in inference timing.
    """

    print("\nComputing Mahalanobis statistics...")

    model.eval()

    all_pred_eps = []

    for batch_idx, vdata in enumerate(val_loader):

        vinputs = vdata.to(device)

        batch_size = vinputs.shape[0]

        noise_steps = torch.full(
            size=(batch_size,),
            fill_value=diffusion.noise_steps - 1,
            device=device,
            dtype=torch.long
        )

        # Generate x_t.
        x, _ = diffusion.noise_time_series(
            vinputs,
            noise_steps
        )

        # One-step epsilon prediction.
        predicted_noise = model(
            x_curr=x,
            t=noise_steps
        )

        B, T, D = predicted_noise.shape

        pred_eps = predicted_noise.reshape(
            B * T,
            D
        )

        all_pred_eps.append(
            pred_eps.detach().cpu()
        )

    # Combine all validation predictions.
    all_pred_eps = torch.cat(
        all_pred_eps,
        dim=0
    )

    print(
        "Validation epsilon matrix:",
        tuple(all_pred_eps.shape)
    )

    # Ledoit-Wolf covariance estimation.
    lw_eps = LedoitWolf().fit(
        all_pred_eps.numpy()
    )

    mu_eps = torch.tensor(
        lw_eps.location_,
        device=device
    ).float()

    cov_eps = torch.tensor(
        lw_eps.covariance_,
        device=device
    ).float()

    # Numerical stabilizer.
    eps_reg = 1e-6

    cov_eps = (
        cov_eps
        + eps_reg
        * torch.eye(
            cov_eps.size(0),
            device=device
        )
    )

    inv_cov_eps = torch.inverse(
        cov_eps
    )

    return mu_eps, inv_cov_eps


# ============================================================
# SELECT RANDOM TEST WINDOW
# ============================================================

def select_random_test_window(
    test_dataset,
    seed=12345
):
    """
    Select one random test window from the dataset.
    """

    rng = np.random.default_rng(seed)

    index = rng.integers(
        low=0,
        high=len(test_dataset)
    )

    sample = test_dataset[index]

    # TestDataset returns:
    #
    #     data, labels
    #
    x = sample[0]
    label = sample[1]

    return (
        index,
        x,
        label
    )


# ============================================================
# TIMING FUNCTION
# ============================================================

def benchmark_inference(
    model,
    diffusion,
    x,
    mu_eps,
    inv_cov_eps,
    device,
    num_warmup=10,
    num_runs=100
):

    model.eval()

    # --------------------------------------------------------
    # Make sure batch dimension exists.
    # --------------------------------------------------------

    if x.dim() == 2:
        x = x.unsqueeze(0)

    x = x.to(
        device,
        non_blocking=True
    )

    # --------------------------------------------------------
    # Warm-up
    # --------------------------------------------------------

    print(
        f"\nGPU warm-up: {num_warmup} runs..."
    )

    with torch.no_grad():

        for _ in range(num_warmup):

            # One-step
            _ = one_step_mahalanobis(
                model,
                diffusion,
                x,
                mu_eps,
                inv_cov_eps,
                device
            )

            # Full-step
            _ = full_step_reconstruction(
                model,
                diffusion,
                x,
                device
            )

    synchronize()

    # ========================================================
    # ONE-STEP MAHALANOBIS TIMING
    # ========================================================

    one_step_times = []

    print(
        f"\nTiming one-step Mahalanobis "
        f"({num_runs} runs)..."
    )

    with torch.no_grad():

        for _ in range(num_runs):

            synchronize()

            start = time.perf_counter()

            _ = one_step_mahalanobis(
                model,
                diffusion,
                x,
                mu_eps,
                inv_cov_eps,
                device
            )

            synchronize()

            end = time.perf_counter()

            one_step_times.append(
                (end - start) * 1000.0
            )

    # ========================================================
    # FULL-STEP RECONSTRUCTION TIMING
    # ========================================================

    full_step_times = []

    print(
        f"Timing full-step reconstruction "
        f"({num_runs} runs)..."
    )

    with torch.no_grad():

        for _ in range(num_runs):

            synchronize()

            start = time.perf_counter()

            reconstructed = full_step_reconstruction(
                model,
                diffusion,
                x,
                device
            )

            synchronize()

            end = time.perf_counter()

            full_step_times.append(
                (end - start) * 1000.0
            )

    # ========================================================
    # STATISTICS
    # ========================================================

    one_step_times = np.asarray(
        one_step_times
    )

    full_step_times = np.asarray(
        full_step_times
    )

    results = {

        "one_step_mahalanobis": {
            "mean_ms": float(
                np.mean(one_step_times)
            ),
            "std_ms": float(
                np.std(
                    one_step_times,
                    ddof=1
                )
            ),
            "min_ms": float(
                np.min(one_step_times)
            ),
            "max_ms": float(
                np.max(one_step_times)
            ),
            "median_ms": float(
                np.median(one_step_times)
            )
        },

        "full_step_reconstruction": {
            "mean_ms": float(
                np.mean(full_step_times)
            ),
            "std_ms": float(
                np.std(
                    full_step_times,
                    ddof=1
                )
            ),
            "min_ms": float(
                np.min(full_step_times)
            ),
            "max_ms": float(
                np.max(full_step_times)
            ),
            "median_ms": float(
                np.median(full_step_times)
            )
        }
    }

    return results


# ============================================================
# MAIN EXPERIMENT
# ============================================================

all_results = []


for dataset_idx, dataset in enumerate(datasets):

    print("\n")
    print("=" * 80)
    print(
        f"DATASET: {dataset}"
    )
    print("=" * 80)

    # --------------------------------------------------------
    # Reproducibility
    # --------------------------------------------------------

    torch.manual_seed(42)
    np.random.seed(42)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    # --------------------------------------------------------
    # Load raw training data
    # --------------------------------------------------------

    if dataset in ['SWaT', 'PSM']:

        df = pd.read_csv(
            TRAIN_PATH[dataset]
        )

        if dataset == 'PSM':

            train_raw = df.values[:, 1:]
            train_raw = np.nan_to_num(
                train_raw
            )

        else:

            train_raw = df.values[:, :-1]

    else:

        train_raw = np.load(
            TRAIN_PATH[dataset]
        )

    # --------------------------------------------------------
    # Standardization
    # --------------------------------------------------------

    scaler = StandardScaler()

    scaler.fit(
        train_raw
    )

    # --------------------------------------------------------
    # Create datasets
    # --------------------------------------------------------

    if dataset in ['SWaT', 'PSM']:

        train_encoder = TrainDatasetCSV(
            TRAIN_PATH[dataset],
            scaler,
            None,
            WINDOW_SIZE_TRAIN,
            STRIDE_TRAIN,
            0.85
        )

        train = TrainDatasetCSV(
            TRAIN_PATH[dataset],
            scaler,
            None,
            WINDOW_SIZE,
            STRIDE_TRAIN,
            0.85
        )

        val = ValDatasetCSV(
            TRAIN_PATH[dataset],
            scaler,
            None,
            WINDOW_SIZE,
            STRIDE,
            0.85
        )

        test = TestDatasetCSV(
            TEST_PATH[dataset],
            TEST_LABEL_PATH[dataset],
            scaler,
            None,
            WINDOW_SIZE,
            STRIDE,
            1.0
        )

    else:

        train_encoder = TrainDataset(
            TRAIN_PATH[dataset],
            scaler,
            None,
            WINDOW_SIZE_TRAIN,
            STRIDE_TRAIN,
            0.85
        )

        train = TrainDataset(
            TRAIN_PATH[dataset],
            scaler,
            None,
            WINDOW_SIZE,
            STRIDE_TRAIN,
            0.85
        )

        val = ValDataset(
            TRAIN_PATH[dataset],
            scaler,
            None,
            WINDOW_SIZE,
            STRIDE,
            0.85
        )

        test = TestDataset(
            TEST_PATH[dataset],
            TEST_LABEL_PATH[dataset],
            scaler,
            None,
            WINDOW_SIZE,
            STRIDE,
            1.0
        )

    # --------------------------------------------------------
    # DataLoaders
    # --------------------------------------------------------

    train_dataloader = DataLoader(
        train,
        batch_size=BATCH_SIZE,
        shuffle=True,
        pin_memory=True
    )

    val_dataloader = DataLoader(
        val,
        batch_size=BATCH_SIZE,
        shuffle=False,
        pin_memory=True
    )

    test_dataloader = DataLoader(
        test,
        batch_size=BATCH_SIZE,
        shuffle=False,
        pin_memory=True
    )

    train_dataloader_encoder = DataLoader(
        train_encoder,
        batch_size=BATCH_SIZE,
        shuffle=True,
        pin_memory=True,
        drop_last=True
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    channels = CHANNELS[dataset]

    diffusion = Diffusion(
        noise_steps=NOISE_STEPS,
        device=device
    )

    uncdit = UnconditionalTimeSeriesDiT(
        target_seq_len=WINDOW_SIZE,
        num_sensors=channels,
        hidden_dim=D_MODEL,
        num_heads=HEAD_NUMBER,
        num_layers=DiT_num_layers
    ).to(device)

    # --------------------------------------------------------
    # Solver
    # --------------------------------------------------------

    experiment = {
        "dataset": dataset,
        "noise_steps": NOISE_STEPS,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "window_size": WINDOW_SIZE,
        "info": INFO
    }

    solver = Solver(
        uncdit,
        train_dataloader,
        val_dataloader,
        test_dataloader,
        diffusion=diffusion,
        experiment=experiment,
        device=device,
        gpu_id=GPU_ID,
        dataset=dataset,
        enumer=dataset_idx
    )

    # ========================================================
    # TRAIN EXACTLY ONE EPOCH
    # ========================================================

    print("\n")
    print("-" * 80)
    print(
        f"Training {dataset} for EXACTLY ONE epoch"
    )
    print("-" * 80)

    solver.diff_model.to(device)

    solver.optimizer = torch.optim.Adam(
        solver.diff_model.parameters(),
        lr=1e-3
    )

    solver.scheduler = torch.optim.lr_scheduler.ExponentialLR(
        solver.optimizer,
        gamma=0.95
    )

    solver.diff_model.train()

    train_loss = solver.train_one_epoch(
        epoch_index=0
    )

    solver.scheduler.step()

    print(
        f"\n{dataset} - One epoch training loss: "
        f"{train_loss:.8f}"
    )

    # ========================================================
    # COMPUTE MAHALANOBIS STATISTICS
    # ========================================================

    mu_eps, inv_cov_eps = (
        compute_mahalanobis_statistics(
            solver.diff_model,
            diffusion,
            val_dataloader,
            device
        )
    )

    # ========================================================
    # RANDOM TEST WINDOW
    # ========================================================

    random_index, x_test, test_label = (
        select_random_test_window(
            test,
            seed=TIMING_SEED + dataset_idx
        )
    )

    print("\n")
    print("-" * 80)
    print("RANDOM TEST WINDOW")
    print("-" * 80)

    print(
        f"Dataset       : {dataset}"
    )

    print(
        f"Window index  : {random_index}"
    )

    print(
        f"Window shape  : {tuple(x_test.shape)}"
    )

    print(
        f"Label shape   : {np.asarray(test_label).shape}"
    )

    # ========================================================
    # BENCHMARK
    # ========================================================

    timing_results = benchmark_inference(
        model=solver.diff_model,
        diffusion=diffusion,
        x=x_test,
        mu_eps=mu_eps,
        inv_cov_eps=inv_cov_eps,
        device=device,
        num_warmup=NUM_WARMUP_RUNS,
        num_runs=NUM_TIMING_RUNS
    )

    # ========================================================
    # PRINT RESULTS
    # ========================================================

    one = timing_results[
        "one_step_mahalanobis"
    ]

    full = timing_results[
        "full_step_reconstruction"
    ]

    speedup = (
        full["mean_ms"]
        /
        one["mean_ms"]
    )

    print("\n")
    print("=" * 80)
    print(
        f"RESULTS: {dataset}"
    )
    print("=" * 80)

    print("\nOne-step Mahalanobis:")
    print(
        f"  Mean   : {one['mean_ms']:.4f} ms"
    )
    print(
        f"  Std    : {one['std_ms']:.4f} ms"
    )
    print(
        f"  Median : {one['median_ms']:.4f} ms"
    )
    print(
        f"  Min    : {one['min_ms']:.4f} ms"
    )
    print(
        f"  Max    : {one['max_ms']:.4f} ms"
    )

    print("\nFull-step reconstruction:")
    print(
        f"  Mean   : {full['mean_ms']:.4f} ms"
    )
    print(
        f"  Std    : {full['std_ms']:.4f} ms"
    )
    print(
        f"  Median : {full['median_ms']:.4f} ms"
    )
    print(
        f"  Min    : {full['min_ms']:.4f} ms"
    )
    print(
        f"  Max    : {full['max_ms']:.4f} ms"
    )

    print(
        f"\nFull-step / One-step ratio: "
        f"{speedup:.2f}x"
    )

    # --------------------------------------------------------
    # Store result
    # --------------------------------------------------------

    all_results.append({
        "dataset": dataset,
        "random_window_index": random_index,
        "window_size": WINDOW_SIZE,
        "channels": channels,
        "noise_steps": NOISE_STEPS,
        "training_epochs": 1,

        "one_step_mean_ms":
            one["mean_ms"],

        "one_step_std_ms":
            one["std_ms"],

        "one_step_median_ms":
            one["median_ms"],

        "one_step_min_ms":
            one["min_ms"],

        "one_step_max_ms":
            one["max_ms"],

        "full_step_mean_ms":
            full["mean_ms"],

        "full_step_std_ms":
            full["std_ms"],

        "full_step_median_ms":
            full["median_ms"],

        "full_step_min_ms":
            full["min_ms"],

        "full_step_max_ms":
            full["max_ms"],

        "full_to_one_step_ratio":
            speedup
    })

    # --------------------------------------------------------
    # Free memory
    # --------------------------------------------------------

    del solver
    del uncdit
    del diffusion

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ============================================================
# SAVE RESULTS
# ============================================================

results_df = pd.DataFrame(
    all_results
)

output_file = (
    "inference_timing_results.csv"
)

results_df.to_csv(
    output_file,
    index=False
)

# ============================================================
# FINAL TABLE
# ============================================================

print("\n")
print("=" * 100)
print("FINAL INFERENCE-TIME RESULTS")
print("=" * 100)

print(
    results_df[
        [
            "dataset",
            "one_step_mean_ms",
            "one_step_std_ms",
            "full_step_mean_ms",
            "full_step_std_ms",
            "full_to_one_step_ratio"
        ]
    ].to_string(
        index=False
    )
)

print("\nResults saved to:")
print(output_file)


print("\nDone.")