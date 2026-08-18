import numpy as np
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import umap


# ============================================================
# Configuration
# ============================================================

FILE_PATH = "high_dim_data_SWaT_epoch_14.npz"

N_NORMAL = 100
N_ANOM = 100

RANDOM_SEED = 42


# ============================================================
# Load data
# ============================================================

data = np.load(FILE_PATH)

x_data = data["x_data"]
eps_data = data["eps_data"]

labels = data["labels"].astype(int)
dataset_name = str(data["dataset_name"])

print(f"Dataset : {dataset_name}")
print(f"X shape : {x_data.shape}")
print(f"Eps shape : {eps_data.shape}")
print(f"Labels : {labels.shape}")


# ============================================================
# Balanced sampling
# ============================================================

np.random.seed(RANDOM_SEED)

normal_idx = np.where(labels == 0)[0]
anom_idx = np.where(labels == 1)[0]

n_normal = min(N_NORMAL, len(normal_idx))
n_anom = min(N_ANOM, len(anom_idx))

sampled_normal = np.random.choice(
    normal_idx,
    n_normal,
    replace=False
)

sampled_anom = np.random.choice(
    anom_idx,
    n_anom,
    replace=False
)

# Same samples are used for X and epsilon
indices = np.concatenate([
    sampled_normal,
    sampled_anom
])

X = x_data[indices]
EPS = eps_data[indices]
Y = labels[indices]

print(f"\nSamples used for visualization:")
print(f"Normal  : {(Y == 0).sum()}")
print(f"Anomaly : {(Y == 1).sum()}")
print(f"Total   : {len(Y)}")


# ============================================================
# Helper function for scatter plots
# ============================================================

def scatter_plot(ax, coords, labels, title):

    normal = labels == 0
    anomaly = labels == 1

    # Normal samples
    ax.scatter(
        coords[normal, 0],
        coords[normal, 1],
        s=55,
        alpha=0.75,
        edgecolors="k",
        label=f"Normal ({normal.sum()})",
    )

    # Anomalous samples
    ax.scatter(
        coords[anomaly, 0],
        coords[anomaly, 1],
        s=80,
        alpha=0.9,
        marker="X",
        edgecolors="k",
        label=f"Anomaly ({anomaly.sum()})",
    )

    ax.set_title(
        title,
        fontsize=12,
        pad=10
    )

    ax.grid(
        alpha=0.3,
        linestyle="--"
    )

    ax.legend(
        loc="best"
    )


# ============================================================
# Visualization function
# ============================================================

def visualize_space(data_matrix, ax_tsne, ax_umap, title):

    print(f"\nProcessing {title}...")

    # --------------------------------------------------------
    # Standardization
    # --------------------------------------------------------

    scaler = StandardScaler()

    data_scaled = scaler.fit_transform(
        data_matrix
    )

    print(f"Standardized shape: {data_scaled.shape}")


    # ========================================================
    # t-SNE
    # ========================================================

    print(f"Running t-SNE on {title}...")

    tsne = TSNE(
        n_components=2,
        perplexity=30,
        learning_rate="auto",
        max_iter=1000,
        init="pca",
        random_state=RANDOM_SEED,
    )

    tsne_coords = tsne.fit_transform(
        data_scaled
    )

    scatter_plot(
        ax_tsne,
        tsne_coords,
        Y,
        f"t-SNE - {title}"
    )

    ax_tsne.set_xlabel("t-SNE 1")
    ax_tsne.set_ylabel("t-SNE 2")


    # ========================================================
    # UMAP
    # ========================================================

    print(f"Running UMAP on {title}...")

    reducer = umap.UMAP(
        n_neighbors=30,
        min_dist=0.05,
        metric="euclidean",
        random_state=RANDOM_SEED,
    )

    umap_coords = reducer.fit_transform(
        data_scaled
    )

    scatter_plot(
        ax_umap,
        umap_coords,
        Y,
        f"UMAP - {title}"
    )

    ax_umap.set_xlabel("UMAP 1")
    ax_umap.set_ylabel("UMAP 2")


# ============================================================
# Create figure
# ============================================================

fig, axes = plt.subplots(
    2,
    2,
    figsize=(14, 12)
)

#fig.suptitle(
#    f"Representation Comparison ({dataset_name})",
#    fontsize=18,
#    y=0.98
#)


# ============================================================
# Row 1: Input X
# ============================================================

print("\n========================================")
print("Input X")
print("========================================")

visualize_space(
    X,
    axes[0, 0],
    axes[0, 1],
    "Input X"
)


# ============================================================
# Row 2: Predicted Noise
# ============================================================

print("\n========================================")
print("Predicted Noise")
print("========================================")

visualize_space(
    EPS,
    axes[1, 0],
    axes[1, 1],
    r"Predicted Noise $\hat{\epsilon}$"
)


# ============================================================
# Final layout
# ============================================================

plt.tight_layout(
    rect=[0, 0, 1, 0.96]
)


# ============================================================
# Save figure
# ============================================================

save_name = (
    f"representation_tsne_umap_{dataset_name}.png"
)

plt.savefig(
    save_name,
    dpi=300,
    bbox_inches="tight"
)

print(
    f"\nSaved figure to {save_name}"
)


# ============================================================
# Display
# ============================================================

plt.show()