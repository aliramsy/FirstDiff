import re
import numpy as np
from collections import defaultdict

file_path = "14.txt"

# 1. Added total_first_scores to metrics list
metrics = [
    "total_cond_scores",
    "total_uncond_scores",
    "total_cond_uncond_scores",
    "total_half_scores",
    "total_first_scores", # <--- NEW
    "epsilon"
]

stats = defaultdict(lambda: {m: [] for m in metrics})
threshold = None

# 2. Updated Regex pattern to capture total_first_scores
# Note: I added a group for total_first_scores between half_scores and epsilon
pattern = re.compile(
    r"(True Negative|True Positive|False Negative|False Positive)\s*\|\s*index:\s*\d+\s*\|\s*history_anomalous:\s*(\d+)\s*\|\s*"
    r"total_cond_scores:\s*([eE0-9\.\-]+)\s*\|\s*"
    r"total_uncond_scores:\s*([eE0-9\.\-]+)\s*\|\s*"
    r"total_cond_uncond_scores:\s*([eE0-9\.\-]+)\s*\|\s*"
    r"total_half_scores:\s*([eE0-9\.\-]+)\s*\|\s*"
    r"total_first_scores:\s*([eE0-9\.\-]+)\s*\|\s*" # <--- NEW regex group
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

            # 3. Map captured groups to values (groups shifted by 1 due to the new insertion)
            values = {
                "total_cond_scores": float(match.group(3)),
                "total_uncond_scores": float(match.group(4)),
                "total_cond_uncond_scores": float(match.group(5)),
                "total_half_scores": float(match.group(6)),
                "total_first_scores": float(match.group(7)), # <--- NEW
                "epsilon": float(match.group(8)),           # <--- Shifted from 7 to 8
            }

            history_group = "history=0" if history == 0 else "history>0"

            for m in metrics:
                stats[(label, history_group)][m].append(values[m])

# Aggregate label only
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

# --- Output Sections ---

print("===== Threshold =====")
print(threshold)
print()

print("===== Overall TN / TP / FN / FP =====")
for label in ["TN", "TP", "FN", "FP"]:
    print_stats(label, label_only[label])

# ... [Epsilon Analysis sections remain same, as they use label_only["TP"]["epsilon"]] ...

print("===== TP Above Threshold (Check total_half and total_first) =====")
tp_half = label_only["TP"]["total_half_scores"]
tp_first = label_only["TP"]["total_first_scores"] # <--- NEW

for score_name, scores in [("Half Scores", tp_half), ("First Scores", tp_first)]:
    count_above = sum(s > threshold for s in scores)
    total_tp = len(scores)
    ratio = count_above / total_tp if total_tp > 0 else 0
    print(f"{score_name} - TP above threshold: {count_above}/{total_tp} (Ratio: {ratio:.4f})")

print("\n===== TN Above Threshold (All Metrics including First) =====")
for m in metrics:
    scores = label_only["TN"][m]
    total = len(scores)
    if total > 0:
        count = sum(s > threshold for s in scores)
        print(f"{m}: {count}/{total} ({count/total:.3f})")
    else:
        print(f"{m}: No samples found.")