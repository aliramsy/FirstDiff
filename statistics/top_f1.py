import re
from heapq import nlargest

fname = "log.txt"

# Regex to get f1 value
f1_pattern = re.compile(r"f1:\s*([0-9]*\.?[0-9]+)")

lines_with_f1 = []

with open(fname, "r", encoding="utf-8", errors="ignore") as f:
    for idx, line in enumerate(f, start=1):
        m = f1_pattern.search(line)
        if not m:
            continue
        f1 = float(m.group(1))
        lines_with_f1.append((f1, idx, line.rstrip("\n")))

# Get top 10 by f1
top10 = nlargest(10, lines_with_f1, key=lambda x: x[0])

print("Top 10 lines by f1:\n")
for rank, (f1, lineno, text) in enumerate(top10, start=1):
    # Try to extract epoch and loss if present on the same line
    epoch_match = re.search(r"epoch[:\s]+(\d+)", text, re.IGNORECASE)
    loss_match = re.search(r"loss[:\s]+([0-9]*\.?[0-9]+)", text, re.IGNORECASE)

    epoch_str = f" | epoch={epoch_match.group(1)}" if epoch_match else ""
    loss_str = f" | loss={loss_match.group(1)}" if loss_match else ""

    print(f"#{rank} (line {lineno}) f1={f1}{epoch_str}{loss_str}")
    print(text)
    print("-" * 80)
