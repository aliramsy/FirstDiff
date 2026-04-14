def find_best_f1(file_path):
    best_raw = {"f1": -1.0, "line": ""}
    best_adj = {"f1": -1.0, "line": ""}

    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if "f1:" not in line:
                continue
            
            # Extract F1 value
            try:
                # Splitting by 'f1:' and then taking the value before the next comma
                f1_val = float(line.split("f1:")[1].split(",")[0].strip())
            except (IndexError, ValueError):
                continue

            # Check if it's RAW or ADJ and compare
            if line.startswith("[RAW]"):
                if f1_val > best_raw["f1"]:
                    best_raw["f1"] = f1_val
                    best_raw["line"] = line
            elif line.startswith("[ADJ]"):
                if f1_val > best_adj["f1"]:
                    best_adj["f1"] = f1_val
                    best_adj["line"] = line

    print("--- Best RAW Metrics ---")
    print(best_raw["line"] if best_raw["line"] else "No RAW data found.")
    print("\n--- Best ADJ Metrics ---")
    print(best_adj["line"] if best_adj["line"] else "No ADJ data found.")

# Usage
find_best_f1("log.txt")
