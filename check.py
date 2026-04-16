import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
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

def grid_search_weighted_vote(filepath):
    print(f"Parsing {filepath} and extracting features...\n")
    
    # 1. Data Collection
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
            l_map = {}
            for p in parts:
                if ':' in p:
                    key, val = p.split(':', 1)
                    l_map[key.strip()] = float(val.strip())
            
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
            except KeyError:
                continue

    gt = np.array(data['gt'])
    
    # 2. Threshold Grid Configuration
    # Set features you don't want to search to [0]
    t_ranges = {
        't_cond':  [.7+i/100 for i in range(250)],
        't_uncon': [0],
        't_half':  [0],
        't_first': [0],
        't_eps':   [0],
        't_res':   [0],
        't_m_eps': [0], # Search Mahalanobis Epsilon
        't_m_res': [0], # Search Mahalanobis Residual
        't_cos_e': [0],
        't_cos_r': [0],
        't_p1_e':  [0],
        't_p2_e':  [0],
        't_p1_r':  [0],
        't_p2_r':  [0]
    }
    
    combinations = list(itertools.product(*t_ranges.values()))
    results = []
    
    print(f"Testing {len(combinations)} combinations...\n")

    for tc, tun, th, tf, te, tr, tme, tmr, tce, tcr, tp1e, tp2e, tp1r, tp2r in combinations:
        
        # --- Generate Binary Votes ---
        v_cond  = (np.array(data['cond']) > tc).astype(int)
        v_uncon = (np.array(data['uncond']) > tun).astype(int)
        v_half  = (np.array(data['half']) > th).astype(int)
        v_first = (np.array(data['first']) > tf).astype(int)
        v_eps   = (np.array(data['eps']) > te).astype(int)
        v_res   = (np.array(data['res']) > tr).astype(int)
        v_m_eps = (np.array(data['m_eps']) > tme).astype(int)
        v_m_res = (np.array(data['m_res']) > tmr).astype(int)
        v_cos_e = (np.array(data['cos_e']) > tce).astype(int)
        v_cos_r = (np.array(data['cos_r']) > tcr).astype(int)
        v_p1_e  = (np.array(data['p1_e']) > tp1e).astype(int)
        v_p2_e  = (np.array(data['p2_e']) > tp2e).astype(int)
        v_p1_r  = (np.array(data['p1_r']) > tp1r).astype(int)
        v_p2_r  = (np.array(data['p2_r']) > tp2r).astype(int)

        # --- Weighted Ensemble Logic ---
        # Modify weights here (e.g., v_cond * 2)
        #total_votes = (v_cond + v_uncon + v_half + v_first + v_eps + v_res + v_m_eps + v_m_res + v_cos_e + v_cos_r + v_p1_e + v_p2_e + v_p1_r + v_p2_r)
        total_votes = v_cond
        # Sensitivity threshold (Anomaly if total votes >= X)
        ensemble_preds = (total_votes >= 1).astype(int)
        
        # Point Adjustment
        adj_preds = point_adjustment(gt, ensemble_preds)
        
        # Metrics
        f1 = f1_score(gt, adj_preds, zero_division=0)
        p = precision_score(gt, adj_preds, zero_division=0)
        r = recall_score(gt, adj_preds, zero_division=0)
        
        results.append({
            'params': (tc, tun, th, tf, te, tr, tme, tmr, tce, tcr, tp1e, tp2e, tp1r, tp2r),
            'p': p, 'r': r, 'f1': f1
        })

    # Sort and Display results
    results.sort(key=lambda x: x['f1'], reverse=True)

    header = f"{'Cnd':<5}|{'Unc':<5}|{'Hlf':<5}|{'Fst':<5}|{'Eps':<5}|{'Res':<5}|{'MEp':<5}|{'MRe':<5}|{'CoE':<5}|{'CoR':<5}|{'P1E':<5}|{'P2E':<5}|{'P1R':<5}|{'P2R':<5} || {'P':<7} | {'R':<7} | {'F1':<7}"
    print(header)
    print("-" * len(header))

    for res in results[:10]:
        tc, tun, th, tf, te, tr, tme, tmr, tce, tcr, tp1e, tp2e, tp1r, tp2r = res['params']
        vals = f"{tc:<5.2f}|{tun:<5.2f}|{th:<5.2f}|{tf:<5.2f}|{te:<5.2f}|{tr:<5.2f}|{tme:<5.2f}|{tmr:<5.2f}|{tce:<5.2f}|{tcr:<5.2f}|{tp1e:<5.2f}|{tp2e:<5.2f}|{tp1r:<5.2f}|{tp2r:<5.2f}"
        print(f"{vals} || {res['p']:<7.4f} | {res['r']:<7.4f} | {res['f1']:<7.4f}")

if __name__ == "__main__":
    grid_search_weighted_vote("14.txt")
