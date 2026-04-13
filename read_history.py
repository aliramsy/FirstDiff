true_count = 0
false_count = 0

with open("instances.txt", "r") as f:
    for line in f:
        if "history_anomalous: True" in line:
            true_count += 1
        elif "history_anomalous: False" in line:
            false_count += 1

total = true_count + false_count

print(f"Anomalous history count: {true_count}")
print(f"Normal history count: {false_count}")
print(f"Total instances: {total}")

if false_count > 0:
    ratio = true_count / false_count
    print(f"Ratio (anomalous / normal): {ratio:.4f}")
else:
    print("Ratio undefined (no normal history found)")

if total > 0:
    percent = (true_count / total) * 100
    print(f"Percent anomalous history: {percent:.2f}%")
