import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
import itertools

def point_adjustment(gt, preds):
    """Applies the standard Point Adjustment (PA) protocol."""
    adjusted_preds = preds.copy()
    in_anomaly = False
    start = -1
    
    for i in range(len(gt)):
        if gt[i] == 1 and not in_anomaly:
            in_anomaly = True
            start = i
        elif gt[i] == 0 and in_anomaly:
            in_anomaly = False
            end = i
            if np.sum(preds[start:end]) > 0:
                adjusted_preds[start:end] = 1
        elif i == len(gt) - 1 and in_anomaly:
            end = i + 1
            if np.sum(preds[start:end]) > 0:
                adjusted_preds[start:end] = 1
                
    return adjusted_preds

def grid_search_weighted_vote(filepath):
    print(f"Parsing {filepath}...\n")
    
    gt_list = []
    cond_list = []
    uncond_list = []
    halfway_list = []
    epsilon_list = []
    
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            # Skip empty lines or header/epoch lines
            if not line or line.startswith("epoch:") or line.startswith("threshold:"):
                continue
                
            # Determine Ground Truth
            if line.startswith("True Positive") or line.startswith("False Negative"):
                gt = 1
            elif line.startswith("True Negative") or line.startswith("False Positive"):
                gt = 0
            else:
                continue

            parts = [p.strip() for p in line.split('|')]
            
            try:
                cond_score = float([p for p in parts if "total_cond_scores:" in p][0].split(':')[1].strip())
                uncond_score = float([p for p in parts if "total_uncond_scores:" in p][0].split(':')[1].strip())
                half_score = float([p for p in parts if "total_half_scores:" in p][0].split(':')[1].strip())
                eps_score = float([p for p in parts if "epsilon:" in p][0].split(':')[1].strip())
            except IndexError:
                continue
                
            gt_list.append(gt)
            cond_list.append(cond_score)
            uncond_list.append(uncond_score)
            halfway_list.append(half_score)
            epsilon_list.append(eps_score)

    gt = np.array(gt_list)
    cond_scores = np.array(cond_list)
    uncond_scores = np.array(uncond_list)
    half_scores = np.array(halfway_list)
    eps_scores = np.array(epsilon_list)
    
    # ==========================================
    # 4 Threshold Arrays (4x4x4x4 = 256 combinations)
    # Adjust these based on your specific data ranges
    # ==========================================
    cond_thresholds   = [ .81]
    uncond_thresholds = [0.60]
    half_thresholds   = [.89]
    eps_thresholds    = [25+i/10 for i in range(90)] 
    
    combinations = list(itertools.product(cond_thresholds, uncond_thresholds, half_thresholds, eps_thresholds))
    
    results = []
    
    print(f"Testing {len(combinations)} combinations.")
    print("Voting rule: Cond=2 votes, Uncond=1, Half=1, Eps=1. Threshold >= 3 votes.\n")
    
    for t_cond, t_uncond, t_half, t_eps in combinations:
        # Generate individual binary votes
        vote_cond   = (cond_scores > t_cond).astype(int)
        vote_uncond = (uncond_scores > t_uncond).astype(int)
        vote_half   = (half_scores > t_half).astype(int)
        vote_eps    = (eps_scores > t_eps).astype(int)
        
        # Apply weights: cond gets * 2, others get * 1
        total_votes = vote_cond + 0*vote_uncond + vote_half + vote_eps
        
        # Anomaly if total votes >= 3
        ensemble_preds = (total_votes >= 1).astype(int)
        
        # Apply Point Adjustment
        adj_preds = point_adjustment(gt, ensemble_preds)
        
        # Calculate metrics
        p = precision_score(gt, adj_preds, zero_division=0)
        r = recall_score(gt, adj_preds, zero_division=0)
        f1 = f1_score(gt, adj_preds, zero_division=0)
        
        results.append({
            'cond': t_cond, 'uncond': t_uncond, 'half': t_half, 'eps': t_eps,
            'p': p, 'r': r, 'f1': f1
        })
        
    # Sort by Adjusted F1 score
    results.sort(key=lambda x: x['f1'], reverse=True)
    
    print("=== TOP 10 WEIGHTED COMBINATIONS ===")
    for r in results[:10]:
        print(f"Cond: {r['cond']:.2f} | Unc: {r['uncond']:.2f} | Half: {r['half']:.2f} | Eps: {r['eps']:.1f}  =>  Adj P: {r['p']:.4f} | Adj R: {r['r']:.4f} | Adj F1: {r['f1']:.4f}")

if __name__ == "__main__":
    file_path = "14.txt"  # Replace with your actual file name
    grid_search_weighted_vote(file_path)
