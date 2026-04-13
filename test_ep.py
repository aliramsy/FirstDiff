import re
import numpy as np
from collections import defaultdict

file_path = "14.txt"

metrics = [
    "total_cond_scores",
    "total_uncond_scores",
    "total_cond_uncond_scores",
    "total_half_scores",
    "epsilon"
]

stats = defaultdict(lambda: {m: [] for m in metrics})

threshold = None

pattern = re.compile(
    r"(True Negative|True Positive|False Negative|False Positive)\s*\|\s*index:\s*\d+\s*\|\s*history_anomalous:\s*(\d+)\s*\|\s*"
    r"total_cond_scores:\s*([eE0-9\.\-]+)\s*\|\s*"
    r"total_uncond_scores:\s*([eE0-9\.\-]+)\s*\|\s*"
    r"total_cond_uncond_scores:\s*([eE0-9\.\-]+)\s*\|\s*"
    r"total_half_scores:\s*([eE0-9\.\-]+)\s*\|\s*"
    r"epsilon:\s*([eE0-9\.\-]+)"
)

label_map = {
    "True Negative": "TN",
    "True Positive": "TP",
    "False Negative": "FN",
    "False Positive": "FP"
}

with open(file_path, "r") as f:
    for line in f:

        if line.startswith("threshold:"):
            threshold = float(line.split(":")[1].strip())

        match = pattern.search(line)

        if match:
            label = label_map[match.group(1)]
            history = int(match.group(2))

            values = {
                "total_cond_scores": float(match.group(3)),
                "total_uncond_scores": float(match.group(4)),
                "total_cond_uncond_scores": float(match.group(5)),
                "total_half_scores": float(match.group(6)),
                "epsilon": float(match.group(7)),
            }

            history_group = "history=0" if history == 0 else "history>0"

            for m in metrics:
                stats[(label, history_group)][m].append(values[m])

# aggregate label only
label_only = defaultdict(lambda: {m: [] for m in metrics})

for (label, hist), values in stats.items():
    for m in metrics:
        label_only[label][m].extend(values[m])


def print_stats(name, data):
    if len(data["epsilon"]) == 0:
        print(f"{name}: no samples")
        return

    print(name)

    for m in metrics:
        arr = np.array(data[m])
        print(f"  {m} mean: {np.mean(arr):.6e}")
        print(f"  {m} std : {np.std(arr):.6e}")

    print()


print("===== Threshold =====")
print(threshold)
print()

print("===== Overall TN / TP / FN / FP =====")

for label in ["TN", "TP", "FN", "FP"]:
    print_stats(label, label_only[label])


print("===== Split by history_anomalous =====")

for label in ["TN", "TP", "FN", "FP"]:
    for hist in ["history=0", "history>0"]:
        key = (label, hist)
        print_stats(f"{label} ({hist})", stats[key])


print("\n===== Epsilon Threshold Analysis for TP & FP =====")

# thresholds for epsilon
eps_thresholds = [25, 26, 27, 28]

tp_eps = label_only["TP"]["epsilon"]
fp_eps = label_only["FP"]["epsilon"]

print("\n--- TP epsilon analysis (epsilon < threshold) ---")
print(f"Total TP samples: {len(tp_eps)}\n")

for t in eps_thresholds:
    count = sum(e < t for e in tp_eps)
    print(f"threshold = {t:2d}  |  TP epsilon below threshold = {count}")

print("\n--- FP epsilon analysis (epsilon < threshold) ---")
print(f"Total FP samples: {len(fp_eps)}\n")

for t in eps_thresholds:
    count = sum(e < t for e in fp_eps)
    print(f"threshold = {t:2d}  |  FP epsilon below threshold = {count}")

print("\n===== Epsilon Threshold Analysis (history_anomalous > 0) =====")

eps_thresholds = [25, 26, 27, 28]

tp_eps = stats[("TP", "history>0")]["epsilon"]
fp_eps = stats[("FP", "history>0")]["epsilon"]

print("\n--- TP epsilon analysis (history>0, epsilon < threshold) ---")
print(f"Total TP samples (history>0): {len(tp_eps)}\n")

for t in eps_thresholds:
    count = sum(e < t for e in tp_eps)
    print(f"threshold = {t:2d} | TP epsilon below threshold = {count}")

print("\n--- FP epsilon analysis (history>0, epsilon < threshold) ---")
print(f"Total FP samples (history>0): {len(fp_eps)}\n")

for t in eps_thresholds:
    count = sum(e < t for e in fp_eps)
    print(f"threshold = {t:2d} | FP epsilon below threshold = {count}")

print("===== TP Above Threshold =====")

tp_scores = label_only["TP"]["total_half_scores"]

count_above = sum(s > threshold for s in tp_scores)
total_tp = len(tp_scores)

print(f"Threshold: {threshold}")
print(f"TP above threshold: {count_above}/{total_tp}")
print(f"Ratio: {count_above/total_tp:.4f}")


print("\n===== TN Above Threshold (All Metrics) =====")

for m in metrics:
    scores = label_only["TN"][m]
    count = sum(s > threshold for s in scores)
    total = len(scores)

    print(f"{m}: {count}/{total} ({count/total:.3f})")

