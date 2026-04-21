import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score
import itertools

def point_adjustment(gt, preds):
    """
    Applies the Point Adjustment (PA) protocol: 
    If a single point in an anomaly segment is correctly predicted, 
    the entire segment is labeled as a prediction (1).
    """
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

def calculate_metrics_with_delay(gt, preds):
    """
    Calculates standard metrics and Anomaly Detection Delay (ADD).
    ADD: Steps from anomaly start to first detection.
    """
    f1 = f1_score(gt, preds, zero_division=0)
    p = precision_score(gt, preds, zero_division=0)
    r = recall_score(gt, preds, zero_division=0)
    
    delays = []
    in_anomaly = False
    detected = False
    start_idx = 0
    
    for i in range(len(gt)):
        if gt[i] == 1 and not in_anomaly:
            in_anomaly = True
            start_idx = i
            detected = False
        elif gt[i] == 1 and in_anomaly and not detected:
            if preds[i] == 1:
                delays.append(i - start_idx)
                detected = True
        elif gt[i] == 0 and in_anomaly:
            in_anomaly = False
            if not detected:
                delays.append(i - start_idx) 
                
    avg_delay = np.mean(delays) if delays else 0
    return p, r, f1, avg_delay

def grid_search_weighted_vote(filepath):
    print(f"Parsing {filepath} and extracting features...\n")
    
    data = {
        'gt': [], 'cond': [], 'uncond': [], 'half': [], 'first': [],
        'eps': [], 'res': [], 'm_eps': [], 'm_res': [], 
        'cos_e': [], 'cos_r': [], 'p1_e': [], 'p2_e': [],
        'p1_r': [], 'p2_r': []
    }
    
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or any(line.startswith(s) for s in ["epoch:", "threshold:"]):
                continue
            gt = 1 if any(s in line for s in ["True Positive", "False Negative"]) else 0
            parts = [p.strip() for p in line.split('|')]
            l_map = {p.split(':')[0].strip(): float(p.split(':')[1].strip()) for p in parts if ':' in p}
            
            try:
                data['gt'].append(gt)
                data['cond'].append(l_map['total_cond_scores'])
                data['uncond'].append(l_map['total_uncond_scores'])
                data['half'].append(l_map['total_half_scores'])
                data['first'].append(l_map['total_first_scores'])
                data['eps'].append(l_map['epsilon'])
                data['res'].append(l_map['residual'])
                data['m_eps'].append(l_map['maha_eps'])
                data['m_res'].append(l_map['maha_res'])
                data['cos_e'].append(l_map['cos_eps'])
                data['cos_r'].append(l_map['cos_res'])
                data['p1_e'].append(l_map['pca1_eps'])
                data['p2_e'].append(l_map['pca2_eps'])
                data['p1_r'].append(l_map['pca1_res'])
                data['p2_r'].append(l_map['pca2_res'])
            except KeyError: continue

    # 1. Pre-convert to NumPy arrays
    gt = np.array(data['gt'])
    s = {k: np.array(v) for k, v in data.items() if k != 'gt'}

    # 2. Grid configuration
    t_ranges = {
        't_cond':  [0],
        't_uncon': [0],
        't_half':  [0.7 + i/100 for i in range(20)],
        't_first': [0],
        't_eps':   [0],
        't_res':   [0],
        't_m_eps': [0],
        't_m_res': [0],
        't_cos_e': [0],
        't_cos_r': [0],
        't_p1_e':  [0],
        't_p2_e':  [0],
        't_p1_r':  [0],
        't_p2_r':  [0]
    }
    
    combinations = list(itertools.product(*t_ranges.values()))
    results = []
    
    # Calculate Range-AUC-PR (Average Precision) on unconditional as baseline
    try:
        rauc_pr = average_precision_score(gt, s['half'])
    except:
        rauc_pr = 0.0

    print(f"Testing {len(combinations)} combinations...\n")

    for tc, tun, th, tf, te, tr, tme, tmr, tce, tcr, tp1e, tp2e, tp1r, tp2r in combinations:
        
        # --- Generate Binary Votes for all 14 features ---
        #v = (
        #    (s['cond'] > tc).astype(int) + (s['uncond'] > tun).astype(int) + 
        #    (s['half'] > th).astype(int) + (s['first'] > tf).astype(int) + 
        #    (s['eps'] > te).astype(int) + (s['res'] > tr).astype(int) + 
        #    (s['m_eps'] > tme).astype(int) + (s['m_res'] > tmr).astype(int) + 
        #    (s['cos_e'] > tce).astype(int) + (s['cos_r'] > tcr).astype(int) + 
        #    (s['p1_e'] > tp1e).astype(int) + (s['p2_e'] > tp2e).astype(int) + 
        #    (s['p1_r'] > tp1r).astype(int) + (s['p2_r'] > tp2r).astype(int)
        #)
        v = (s['half'] > th).astype(int)
        # Ensemble Prediction (Anomaly if at least 1 vote)
        ensemble_preds = (v >= 1).astype(int)
        
        # --- APPLY POINT ADJUSTMENT ---
        adj_preds = point_adjustment(gt, ensemble_preds)
        #adj_preds = ensemble_preds
        # Calculate Metrics and Delay
        p, r, f1, add = calculate_metrics_with_delay(gt, adj_preds)
        
        results.append({
            'params': (tc, tun, th, tf, te, tr, tme, tmr, tce, tcr, tp1e, tp2e, tp1r, tp2r),
            'p': p, 'r': r, 'f1': f1, 'rauc': rauc_pr, 'add': add
        })

    results.sort(key=lambda x: x['f1'], reverse=True)

    # 3. Print Results
    header = f"{'Cnd':<5}|{'Unc':<5}|{'Hlf':<5}|{'Fst':<5}|{'Eps':<5}|{'Res':<5}|{'MEp':<5}|{'MRe':<5}|{'CoE':<5}|{'CoR':<5}|{'P1E':<5}|{'P2E':<5}|{'P1R':<5}|{'P2R':<5} || {'PA-P':<7} | {'PA-R':<7} | {'PA-F1':<7} | {'R-PR':<7} | {'ADD':<7}"
    print(header)
    print("-" * len(header))

    for res in results[:10]:
        tc, tun, th, tf, te, tr, tme, tmr, tce, tcr, tp1e, tp2e, tp1r, tp2r = res['params']
        vals = f"{tc:<5.2f}|{tun:<5.2f}|{th:<5.2f}|{tf:<5.2f}|{te:<5.2f}|{tr:<5.2f}|{tme:<5.2f}|{tmr:<5.2f}|{tce:<5.2f}|{tcr:<5.2f}|{tp1e:<5.2f}|{tp2e:<5.2f}|{tp1r:<5.2f}|{tp2r:<5.2f}"
        print(f"{vals} || {res['p']:<7.4f} | {res['r']:<7.4f} | {res['f1']:<7.4f} | {res['rauc']:<7.4f} | {res['add']:<7.2f}")

if __name__ == "__main__":
    grid_search_weighted_vote("14.txt")
