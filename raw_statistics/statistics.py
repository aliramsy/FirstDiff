import os
import re
from collections import defaultdict, Counter
import matplotlib.pyplot as plt

# Configuration
NUM_FILES = 15
WINDOW_SIZE = 96

# Data structures to store parsed information
# file_id -> list of tuples (index, history)
fp_data = defaultdict(list)
fn_data = defaultdict(list)

# Regex to parse the lines
# Example line: False Positive | index: 1985 | history_anomalous: 774
line_pattern = re.compile(r'(False Positive|False Negative)\s*\|\s*index:\s*(\d+)\s*\|\s*history_anomalous:\s*(\d+)')

def parse_files():
    for i in range(NUM_FILES):
        filename = f"0{i}.txt"
        if not os.path.exists(filename):
            print(f"Warning: File {filename} not found.")
            continue
            
        with open(filename, 'r') as f:
            for line in f:
                match = line_pattern.search(line)
                if match:
                    error_type = match.group(1)
                    idx = int(match.group(2))
                    history = int(match.group(3))
                    
                    if error_type == 'False Positive':
                        fp_data[i].append((idx, history))
                    else:
                        fn_data[i].append((idx, history))

def count_repetitions(data_dict, item_extractor):
    """
    Counts how many files each item (index or window) appears in.
    Returns a dictionary mapping: occurrence_count -> list of items.
    """
    item_occurrences = defaultdict(set)
    for file_id, records in data_dict.items():
        items_in_file = set(item_extractor(record) for record in records)
        for item in items_in_file:
            item_occurrences[item].add(file_id)
            
    # Group by number of files they appeared in
    occurrence_groups = defaultdict(list)
    for item, files in item_occurrences.items():
        occurrence_groups[len(files)].append(item)
        
    return occurrence_groups

def print_repetitions(occurrence_groups, name):
    print(f"\n--- {name} Repetitions Across Files ---")
    for count in range(NUM_FILES, 0, -1):
        items = occurrence_groups.get(count, [])
        print(f"Present in exactly {count} files: {len(items)} items")
        if len(items) > 0:
            # Print first 20 to avoid cluttering the console if there are too many
            print(f"   Sample items: {items[:20]}{'...' if len(items) > 20 else ''}")

def analyze_history_ratio(data_dict, item_extractor, name):
    print(f"\n--- Ratio of {name} with 0 anomalous history per file ---")
    for file_id in range(NUM_FILES):
        if file_id not in data_dict or len(data_dict[file_id]) == 0:
            print(f"File {file_id}: No {name} data.")
            continue
            
        # Group histories by the item (index or window)
        item_histories = defaultdict(list)
        for record in data_dict[file_id]:
            item = item_extractor(record)
            hist = record[1]
            item_histories[item].append(hist)
            
        zero_history_count = 0
        total_items = len(item_histories)
        
        for item, histories in item_histories.items():
            # For windows, we average the histories. For indices, it's just the exact history.
            avg_hist = sum(histories) / len(histories)
            if avg_hist == 0:
                zero_history_count += 1
                
        ratio = zero_history_count / total_items if total_items > 0 else 0
        print(f"File {file_id}: {zero_history_count}/{total_items} = {ratio:.4f}")

def plot_history_bar_charts(data_dict, item_extractor, name_prefix):
    for file_id in range(NUM_FILES):
        if file_id not in data_dict or len(data_dict[file_id]) == 0:
            continue
            
        item_histories = defaultdict(list)
        for record in data_dict[file_id]:
            item = item_extractor(record)
            hist = record[1]
            item_histories[item].append(hist)
            
        # Calculate average history for the item (applicable for windows)
        hist_counts = Counter()
        for item, histories in item_histories.items():
            avg_hist = round(sum(histories) / len(histories))
            hist_counts[avg_hist] += 1
            
        histories, counts = zip(*sorted(hist_counts.items()))
        
        plt.figure(figsize=(10, 5))
        plt.bar(histories, counts, color='skyblue', edgecolor='black')
        plt.title(f"{name_prefix} - File {file_id}: Incidences per Anomalous History")
        plt.xlabel("Anomalous History")
        plt.ylabel("Number of Incidences")
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        # Save the plot
        safe_name = name_prefix.replace(' ', '_')
        plt.savefig(f"{safe_name}_File_{file_id}.png")
        plt.close()
    print(f"\nSaved bar charts for {name_prefix} to current directory.")

def main():
    parse_files()
    
    # Extractors
    index_extractor = lambda x: x[0]
    window_extractor = lambda x: x[0] // WINDOW_SIZE

    print("================ INDEX LEVEL ANALYSIS ================")
    # 1. FP index repetitions
    fp_index_reps = count_repetitions(fp_data, index_extractor)
    print_repetitions(fp_index_reps, "FP Indices")
    
    # 2. FN index repetitions
    fn_index_reps = count_repetitions(fn_data, index_extractor)
    print_repetitions(fn_index_reps, "FN Indices")
    
    # 3. FP indices zero history ratio
    analyze_history_ratio(fp_data, index_extractor, "FP Indices")
    
    # 4. FN indices zero history ratio
    analyze_history_ratio(fn_data, index_extractor, "FN Indices")
    
    # 5. Plot FN indices history charts
    plot_history_bar_charts(fn_data, index_extractor, "Index Level FN")
    
    # 6. Plot FP indices history charts
    plot_history_bar_charts(fp_data, index_extractor, "Index Level FP")
    

    print("\n\n================ WINDOW LEVEL ANALYSIS (SIZE = 96) ================")
    # 1 (Window). FP window repetitions
    fp_window_reps = count_repetitions(fp_data, window_extractor)
    print_repetitions(fp_window_reps, "FP Windows")
    
    # 2 (Window). FN window repetitions
    fn_window_reps = count_repetitions(fn_data, window_extractor)
    print_repetitions(fn_window_reps, "FN Windows")
    
    # 3 (Window). FP windows zero history ratio
    analyze_history_ratio(fp_data, window_extractor, "FP Windows")
    
    # 4 (Window). FN windows zero history ratio
    analyze_history_ratio(fn_data, window_extractor, "FN Windows")
    
    # 5 (Window). Plot FN windows history charts
    plot_history_bar_charts(fn_data, window_extractor, "Window Level FN")
    
    # 6 (Window). Plot FP windows history charts
    plot_history_bar_charts(fp_data, window_extractor, "Window Level FP")

if __name__ == "__main__":
    main()
