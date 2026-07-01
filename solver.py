import torch.nn.functional as F
import torch
from sklearn.decomposition import PCA
from sklearn.covariance import LedoitWolf
from torch.utils.tensorboard import SummaryWriter
from torcheval.metrics.functional import binary_f1_score, binary_precision, binary_recall
from model.Diffusion import *
from sklearn.metrics import auc
import numpy as np
from vus.metrics import get_metrics

class Solver():
    # add autoencoder to the init function it is after self
    def __init__(self, diff_model, train_loader, val_loader, test_loader, diffusion=None, mask_data=True, anomaly_ratio=0.05, experiment=None, device='cuda', gpu_id=0, decomposer = None, dataset = None):
        #self.autoencoder = autoencoder
        self.decomposer = decomposer
        self.dataset = dataset
        self.mask_data = mask_data
        self.diff_model = diff_model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.loss_fn = F.mse_loss
        self.anomaly_ratio = anomaly_ratio
        self.alpha = 10
        self.diffusion = diffusion
        self.experiment_name = f"runs/AE_diff_{experiment['dataset']}_noise({experiment['noise_steps']})_epochs({experiment['epochs']})_batch({experiment['batch_size']})_window({experiment['window_size']})_{experiment['info']}"
        self.model_name = f"AE_diff_{experiment['dataset']}_noise({experiment['noise_steps']})_epochs({experiment['epochs']})_batch({experiment['batch_size']})_window({experiment['window_size']})_{experiment['info']}"
        
        self.optimizer = None
        if device != 'cpu':
            self.device = f"cuda:{gpu_id}"
        else:
            self.device = device
        
        self.tb_writer = SummaryWriter(self.experiment_name)

    def calculate_mask(self, input, output, window_size, batch_size, dim):
        """
        Greater score corresponds to possible anomalies
        """
        scores = torch.square((output - input))
        scores_reshaped = scores.reshape(-1, window_size)
        scores_sorted, _ =  torch.sort(scores_reshaped, dim=1)
        thres_ind = int(0.95 * window_size)
        threshold_values = scores_sorted[:, thres_ind].view(batch_size, dim, 1)
        return (scores < threshold_values).type(torch.float32)

    def train_one_epoch(self, epoch_index):
        running_loss = 0.
        last_loss = 0.
        

        for i, data in enumerate(self.train_loader):
            inputs= data
            inputs = inputs.to(self.device)
            
            self.optimizer.zero_grad()
            
            if self.mask_data:
                #outputs = self.autoencoder(inputs)
                pass

            ### Data masking
            if self.mask_data:
                mask = self.calculate_mask(inputs, outputs, inputs.shape[2], inputs.shape[0], inputs.shape[1])
                ae_x = mask * inputs + (1. - mask) * torch.rand_like(inputs)
            else:
                ae_x = inputs

            if self.decomposer:
                x_low, x_mid, x_high = self.decomposer(ae_x)
                x_low = x_low.to(self.device)
                x_mid = x_mid.to(self.device)
                x_high = x_high.to(self.device)

            ##### DIFFUSION
            t = self.diffusion.sample_timesteps(ae_x.shape[0]).to(self.device)
            x_t, noise = self.diffusion.noise_time_series(ae_x, t)
            #predicted_noise = self.diff_model(x_t, t, ae_x)
            if self.decomposer:
                predicted_noise = self.diff_model(x_curr=x_t, t=t, c_mid=x_mid, coarse=x_low)
            else:
                predicted_noise = self.diff_model(x_t, t)


            diff_loss = self.loss_fn(predicted_noise, noise)
            ####### DIFFUSION
            if self.mask_data:
                loss = self.alpha * diff_loss
            else:
                loss = diff_loss

            if self.mask_data:
                loss += self.loss_fn(outputs, inputs)

            loss.backward()
            
            self.optimizer.step()
            
            running_loss += loss.item()
            last_loss = running_loss / (i + 1) 
                
        return last_loss
    
    def train(self, epochs):
        #self.autoencoder.to(self.device)
        self.diff_model.to(self.device)
        if self.mask_data:
            self.optimizer = torch.optim.Adam(list(self.autoencoder.parameters()) + list(self.diff_model.parameters()), lr=1e-3)
        else:
            self.optimizer = torch.optim.Adam(self.diff_model.parameters(), lr=1e-3)
            self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.95)
        for epoch in range(epochs):
            #self.autoencoder.train()
            self.diff_model.train()

            avg_loss = self.train_one_epoch(epoch)
            self.scheduler.step()
 
            avg_vloss = self.val(epoch)
            self.tb_writer.add_scalars('Loss', {"Train" : avg_loss, "Val" : avg_vloss}, epoch)
            print(f"EPOCH {epoch} LOSS train {avg_loss} valid {avg_vloss}")
            if epoch == epochs - 1:
                self.test(epoch)

            #if epoch % 5 == 0 and epoch !=0:
            #    self.save_model([f'AE_{epoch}', f'Diffusion_{epoch}'])

        self.tb_writer.flush()
        #self.save_model(f'Diffusion_{epoch}')

    
    def denoise_process(self, i, x, batch_num=128, epoch=None):

        for j in range(self.diffusion.noise_steps - 1, -1, -1):
            t = (j * torch.ones(x.shape[0])).long().to(self.device)       
            self.diff_model.eval()

            predicted_noise = self.diff_model(x_curr=x, t=t)
            if j == self.diffusion.noise_steps - 10:
                ret_predicted_noise_cond = predicted_noise.clone()

            if j % 100 == 0:
                print(f"Epoch: {epoch}, Batch No. {i}, Denoise step: {j}, Total Batches: {batch_num}")
            
            alpha = self.diffusion.alpha[t][:, None, None]
            alpha_hat = self.diffusion.alpha_hat[t][:, None, None]
            beta = self.diffusion.beta[t][:, None, None]
            
            if j > 0:
                noise = torch.randn_like(x)
            else:
                noise = torch.zeros_like(x)
            
            x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
            
            if j == int(self.diffusion.noise_steps/2):
                ret_half_noise = x.clone()
                #ret_half_noise = x_low + x_mid + ret_half_noise
            if j == self.diffusion.noise_steps - 1:
                ret_first_noise = x.clone()
                #ret_first_noise = x_low + x_mid + ret_first_noise
        #x = x + x_low + x_mid

        self.diff_model.train()
        return x, ret_predicted_noise_cond, ret_half_noise, ret_first_noise

    
    def val(self, epoch=0):
        #self.autoencoder.eval()
        self.diff_model.eval()

        running_vloss = 0.0
        all_pred_eps = []
        all_residuals = []
        
        with torch.no_grad():
            for i, vdata in enumerate(self.val_loader):
                vinputs = vdata
                vinputs = vinputs.to(self.device)

                if self.mask_data:
                    #voutputs = self.autoencoder(vinputs)
                    pass

                ### Data masking
                if self.mask_data:
                    mask = self.calculate_mask(vinputs, voutputs, vinputs.shape[2], vinputs.shape[0], vinputs.shape[1])
                    ae_x = mask * vinputs + (1. - mask) * torch.rand_like(vinputs)
                else:
                    ae_x = vinputs
            
                ##### DIFFUSION
                if self.decomposer:
                    x_low, x_mid, x_high = self.decomposer(ae_x)
                    x_low = x_low.to(self.device)
                    x_mid = x_mid.to(self.device)
                    x_high = x_high.to(self.device)

                noise_steps = torch.full(size=(ae_x.shape[0],), fill_value=self.diffusion.noise_steps - 1).to(self.device)
                x, eps = self.diffusion.noise_time_series(ae_x, noise_steps)

                diff_voutputs, predicted_noise, diff_voutputs_cond_half, diff_voutputs_cond_first = self.denoise_process(
                    i, x, len(self.val_loader), epoch=epoch
                )
                pred_eps = predicted_noise.detach()
                B, T, D = pred_eps.shape
                pred_eps = pred_eps.reshape(B * T, D)
                eps_flat = eps.reshape(B * T, D)
                residual = eps_flat - pred_eps
                all_pred_eps.append(pred_eps)
                all_residuals.append(residual)

                ####### DIFFUSION
                vloss = F.mse_loss(diff_voutputs, ae_x)
                if self.mask_data:
                    vloss += self.loss_fn(voutputs, vinputs)
                running_vloss += vloss.item()

        all_pred_eps = torch.cat(all_pred_eps, dim=0)
        all_residuals = torch.cat(all_residuals, dim=0)
        # Fit LedoitWolf ONCE
        lw_eps = LedoitWolf().fit(all_pred_eps.cpu().numpy())
        
        # Single consistent mean
        self.mu_eps = torch.tensor(lw_eps.location_, device=self.device).float()
        # ===== Mean direction for cosine similarity =====
        self.mu_eps_dir = self.mu_eps / (torch.norm(self.mu_eps) + 1e-12)
        
        # Stable inverse covariance
        cov_eps = torch.tensor(lw_eps.covariance_, device=self.device).float()
        
        # small numerical stabilizer
        eps_reg = 1e-6
        cov_eps = cov_eps + eps_reg * torch.eye(cov_eps.size(0), device=self.device)
        
        self.inv_cov_eps = torch.inverse(cov_eps)

        # ===============================
        # 4️⃣ Mahalanobis, Cosine, & PCA on residual
        # ===============================
        lw_res = LedoitWolf().fit(all_residuals.cpu().numpy())
        
        self.mu_res = torch.tensor(lw_res.location_, device=self.device).float()
        
        # ===== NEW: Mean direction for residual cosine similarity =====
        self.mu_res_dir = self.mu_res / (torch.norm(self.mu_res) + 1e-12)
        
        cov_res = torch.tensor(lw_res.covariance_, device=self.device).float()
        cov_res = cov_res + 1e-6 * torch.eye(cov_res.size(0), device=self.device)
        
        self.inv_cov_res = torch.inverse(cov_res)

        avg_vloss = running_vloss / len(self.val_loader)
        return avg_vloss
    
    def adjust_preds(self, preds, labels):
        anomaly_flag = False
        preds = preds.copy()
        for i in range(len(labels)):
                if labels[i] == 1 and preds[i] == 1 and not anomaly_flag:
                    anomaly_flag = True
                    for j in range(i, -1, -1):
                        if labels[j] == 0:
                            break
                        else:
                            if preds[j] == 0:
                                preds[j] = 1
                    for j in range(i, len(labels)):
                        if labels[j] == 0:
                            break
                        else:
                            if preds[j] == 0:
                                preds[j] = 1
                elif labels[i] == 0:
                    anomaly_flag = False
                if anomaly_flag:
                    preds[i] = 1
        return torch.from_numpy(preds).type(torch.float32)
    
    
    def calculate_add(self, raw_predict, actual):
        """
        Calculates the Anomaly Detection Delay (ADD).
        """
        # If they are already numpy arrays, skip the .is_cuda check
        if isinstance(raw_predict, torch.Tensor):
            if raw_predict.is_cuda:
                raw_predict = raw_predict.cpu()
            raw_predict = raw_predict.numpy()
            
        if isinstance(actual, torch.Tensor):
            if actual.is_cuda:
                actual = actual.cpu()
            actual = actual.numpy()
    
        # Now use numpy logic for the diff
        import numpy as np
        diff = np.diff(actual, prepend=0.0)
        
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]
        
        if len(ends) < len(starts):
            ends = np.append(ends, len(actual))
            
        if len(starts) == 0:
            return 0.0
    
        delays = []
        for s, e in zip(starts, ends):
            segment_preds = raw_predict[s:e]
            first_hit_idx = np.argmax(segment_preds)
            
            if segment_preds[first_hit_idx] == 1:
                delays.append(first_hit_idx)
                
        return sum(delays) / len(delays) if len(delays) > 0 else 0.0
    
    def _get_anomaly_ranges(self, labels):
        """Helper to extract start and end indices of continuous anomaly segments."""
        diff = np.diff(np.insert(labels, 0, 0))
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]
        if len(ends) < len(starts):
            ends = np.append(ends, len(labels))
        return list(zip(starts, ends))

    from vus.metrics import get_metrics


    def compute_range_metrics(self, labels, scores):
        """Computes standardized VLDB 2022 Range-AUC and VUS metrics."""
        try:
            y_true = labels.detach().cpu().numpy().astype(int).flatten()
            y_scores = scores.detach().cpu().numpy().astype(float).flatten()            

            # Dynamically find the median anomaly length to set the sliding window buffer
            diff = np.diff(np.concatenate([[0], y_true, [0]]))
            starts = np.where(diff == 1)[0]
            ends = np.where(diff == -1)[0]
            lengths = ends - starts
            sliding_window = (
                max(int(np.median(lengths)), 16)
                if len(lengths) > 0
                else 100
            )

            # Compute all official benchmark metrics simultaneously
            metrics = get_metrics(
                score=y_scores,
                labels=y_true,
                metric="all",
                slidingWindow=sliding_window,
            )

            return {
                "rauc_roc": float(metrics["R_AUC_ROC"]),
                "rauc_pr": float(metrics["R_AUC_PR"]),
                "vus_roc": float(metrics["VUS_ROC"]),
                "vus_pr": float(metrics["VUS_PR"]),
            }
        except Exception as e:
            print(f"Error computing VUS metrics: {e}")
            return {"rauc_roc": 0.0, "rauc_pr": 0.0, "vus_roc": 0.0, "vus_pr": 0.0}

    
    def test(self, epoch):

        running_tloss = 0.0
        total_scores = torch.empty(0).to(self.device)
        total_labels = torch.empty(0).to(self.device)
        total_pred_scores = torch.empty(0).to(self.device)
        total_half_cur_scores = torch.empty(0).to(self.device)
        total_first_cur_scores = torch.empty(0).to(self.device)
        total_maha_eps = torch.empty(0).to(self.device)
        total_cos_eps = torch.empty(0).to(self.device)
        total_maha_res = torch.empty(0).to(self.device)
        total_cos_res = torch.empty(0).to(self.device)
        total_res_scores = torch.empty(0).to(self.device) 


        all_inputs = []
        all_outputs = []

        with torch.no_grad():
            for i, tdata in enumerate(self.test_loader):

                tinputs, labels = tdata[0], tdata[1]
                tinputs = tinputs.to(self.device)
                labels = labels.to(self.device)

                if self.mask_data:
                    #toutputs = self.autoencoder(tinputs)
                    pass

                if self.mask_data:
                    mask = self.calculate_mask(
                        tinputs,
                        toutputs,
                        tinputs.shape[2],
                        tinputs.shape[0],
                        tinputs.shape[1],
                    )
                    ae_x = mask * tinputs + (1. - mask) * torch.rand_like(tinputs)
                else:
                    ae_x = tinputs

                if self.decomposer:
                    x_low, x_mid, x_high = self.decomposer(ae_x)
                    x_low = x_low.to(self.device)
                    x_mid = x_mid.to(self.device)
                    x_high = x_high.to(self.device)

                noise_steps = torch.full(
                    size=(ae_x.shape[0],),
                    fill_value=self.diffusion.noise_steps - 1,
                ).to(self.device)

                x, eps = self.diffusion.noise_time_series(ae_x, noise_steps)

                diff_toutputs_cond, predicted_noise, diff_toutputs_cond_half, diff_toutputs_cond_first = self.denoise_process(
                    i, x, len(self.test_loader), epoch=epoch
                )
                diff_toutputs = diff_toutputs_cond 

                all_inputs.append(ae_x.detach().cpu())
                all_outputs.append(diff_toutputs.detach().cpu())

                # ======== PCA, Mahalanobis distance ========
                B, T, D = predicted_noise.shape
                pred_eps = predicted_noise.reshape(B * T, D)
                eps_flat = eps.reshape(B * T, D)   
                residual = eps_flat - pred_eps

                # Epsilon Mahalanobis
                diff_eps = pred_eps - self.mu_eps
                maha_eps = torch.sqrt(
                    torch.einsum('bi,ij,bj->b', diff_eps, self.inv_cov_eps, diff_eps)
                ) 
                
                # ===== Cosine similarity to mean epsilon & residual direction =====
                pred_eps_norm = pred_eps / (torch.norm(pred_eps, dim=1, keepdim=True) + 1e-12)
                cos_eps = torch.matmul(pred_eps_norm, self.mu_eps_dir)      
                
                res_norm = residual / (torch.norm(residual, dim=1, keepdim=True) + 1e-12)
                cos_res = torch.matmul(res_norm, self.mu_res_dir) # NEW

                # Residual Mahalanobis
                diff_res = residual - self.mu_res
                maha_res = torch.sqrt(
                    torch.einsum('bi,ij,bj->b', diff_res, self.inv_cov_res, diff_res)
                )
                
                total_maha_eps = torch.cat([total_maha_eps, maha_eps.reshape(-1)])
                total_maha_res = torch.cat([total_maha_res, maha_res.reshape(-1)])
                
                total_cos_eps = torch.cat([total_cos_eps, cos_eps.reshape(-1)])
                total_cos_res = torch.cat([total_cos_res, cos_res.reshape(-1)]) # NEW

                # ... (Keep your existing CFG divergence code here) ...
                
                # Create a score based on magnitude for residual just like you did for pred_noise
                preds_scores = torch.sum(torch.abs(predicted_noise), dim=2)
                res_scores = torch.sum(torch.abs(residual.reshape(B, T, D)), dim=2) # NEW
                
                total_pred_scores = torch.cat(
                    [total_pred_scores, preds_scores.reshape(-1)]
                )
                total_res_scores = torch.cat(
                    [total_res_scores, res_scores.reshape(-1)] # NEW
                )

                preds = torch.square((diff_toutputs - ae_x))
                half_cur = torch.square((diff_toutputs_cond_half - ae_x))
                first_cur = torch.square((diff_toutputs_cond_first - ae_x))
                    
                anomaly_scores = torch.mean(preds, dim=2)
                preds_scores = torch.sum(torch.abs(predicted_noise), dim=2)
                half_cur_scores = torch.mean(half_cur, dim=2)
                first_cur_scores = torch.mean(first_cur, dim=2)


                total_scores = torch.cat(
                    [total_scores, anomaly_scores.reshape(-1)]
                )
                
                total_half_cur_scores = torch.cat(
                    [total_half_cur_scores, half_cur_scores.reshape(-1)]
                )

                total_first_cur_scores = torch.cat(
                    [total_first_cur_scores, first_cur_scores.reshape(-1)]
                )


                total_labels = torch.cat(
                    [total_labels, labels.reshape(-1)]
                )

                tloss = F.mse_loss(diff_toutputs, ae_x)

                if self.mask_data:
                    tloss += self.loss_fn(toutputs, tinputs)

                running_tloss += tloss.item()

        # =========================
        # THRESHOLD GRID (FIXED)
        # =========================
        threshold_grid = torch.linspace(0.0, 0.300, 300, device=self.device)


        def evaluate_score(score_tensor, labels, adjusted=True):
        
            score_tensor = score_tensor.to(self.device)
            labels = labels.to(self.device)

            vus_dict = self.compute_range_metrics(labels, score_tensor)

            best = {
                "f1": 0.0,
                "p": 0.0,
                "r": 0.0,
                "add": 0.0,
                "rauc": vus_dict["rauc_roc"],
                "rauc_pr": vus_dict["rauc_pr"],
                "vus_roc": vus_dict["vus_roc"],
                "vus_pr": vus_dict["vus_pr"],
                "thresh_ratio": 0.0,
                "thresh_value": 0.0,
            }

            all_results = []

            for ratio in threshold_grid:
            
                thresh_value = torch.quantile(score_tensor, 1. - ratio)

                preds = (score_tensor >= thresh_value).float()
                lbls = labels

                # =========================
                # ADJUSTMENT FIX
                # =========================
                if adjusted:
                    preds_np = preds.detach().cpu().numpy()
                    lbls_np = lbls.detach().cpu().numpy()

                    preds_adj = self.adjust_preds(preds_np, lbls_np)
                    # Faster and removes the warning
                    preds_final = preds_adj.to(self.device).detach()
                else:
                    preds_final = preds

                # =========================
                # METRICS
                # =========================
                f1 = binary_f1_score(preds_final, lbls.int())
                p = binary_precision(preds_final, lbls.int())
                r = binary_recall(preds_final, lbls.int())

                # ADD must use RAW preds
                add = self.calculate_add(preds.detach().cpu().numpy(), lbls.detach().cpu().numpy())

                all_results.append((ratio.item(), thresh_value.item(), float(f1)))

                if f1 > best["f1"]:
                    best = {
                        "f1": float(f1),
                        "p": float(p),
                        "r": float(r),
                        "add": float(add),

                        "rauc": vus_dict["rauc_roc"],
                        "rauc_pr": vus_dict["rauc_pr"],
                        "vus_roc": vus_dict["vus_roc"],
                        "vus_pr": vus_dict["vus_pr"],

                        "thresh_ratio": float(ratio.item()),
                        "thresh_value": float(thresh_value.item())
                    }

            return best, all_results


        # =========================
        # SCORE DICTIONARY
        # =========================
        score_dict = {
            "total": total_scores,
            "maha_eps": total_maha_eps,
            "maha_res": total_maha_res,
            "epsilon": total_pred_scores,
            "residual": total_res_scores,
            "cos_eps": total_cos_eps,
            "cos_res": total_cos_res,
        }

        results = {}

        for name, score_tensor in score_dict.items():
            best_adj, all_adj = evaluate_score(score_tensor, total_labels, adjusted=True)
            best_raw, all_raw = evaluate_score(score_tensor, total_labels, adjusted=False)

            results[name] = {
                "adj": best_adj,
                "raw": best_raw,
                "all_adj": all_adj,
                "all_raw": all_raw
            }


        # =========================
        # NEIGHBOR FUNCTION (FIXED SAFE INDEXING)
        # =========================
        def get_neighbors(all_results, best_ratio):
            # Extract all ratios from the 250-point results
            ratios = [r for r, _, _ in all_results]
        
            if best_ratio not in ratios:
                idx = min(range(len(ratios)), key=lambda i: abs(ratios[i] - best_ratio))
            else:
                idx = ratios.index(best_ratio)
        
            # To get 41 neighbors (20 left, the best one, 20 right):
            lower_bound = max(0, idx - 20)
            upper_bound = min(len(ratios), idx + 21)
            
            # Slice the ratios list to get the neighbors
            neighbor_ratios = ratios[lower_bound:upper_bound]
        
            return neighbor_ratios

        # =========================
        # COMBINATION SEARCH
        # =========================
        def combination_search(base_name, other_name, adjusted=True):
            base_results = results[base_name]["all_adj" if adjusted else "all_raw"]
            other_results = results[other_name]["all_adj" if adjusted else "all_raw"]

            base_best = results[base_name]["adj" if adjusted else "raw"]["thresh_ratio"]
            other_best = results[other_name]["adj" if adjusted else "raw"]["thresh_ratio"]

            base_neighbors = get_neighbors(base_results, base_best)
            other_neighbors = get_neighbors(other_results, other_best)

            # === FIX: Moved normalization block to the top of the function scope ===
            s1 = score_dict[base_name]
            s2 = score_dict[other_name]

            # Min-Max Normalization to bring both to the same scale [0, 1]
            s1_norm = (s1 - s1.min()) / (s1.max() - s1.min() + 1e-12)
            s2_norm = (s2 - s2.min()) / (s2.max() - s2.min() + 1e-12)

            combined_continuous_scores = s1_norm + s2_norm

            # Calculate range metrics safely using the fully resolved score matrix
            vus_metrics = self.compute_range_metrics(
                total_labels,
                combined_continuous_scores
            )

            continuous_rauc = vus_metrics["rauc_roc"]
            continuous_rauc_pr = vus_metrics["rauc_pr"]
            continuous_vus_roc = vus_metrics["vus_roc"]
            continuous_vus_pr = vus_metrics["vus_pr"]

            # Initialize best_combo dictionary
            best_combo = {
                "f1": 0.0,
                "p": 0.0,
                "r": 0.0,
                "add": 0.0,
                "rauc": float(continuous_rauc),
                "rauc_pr": float(continuous_rauc_pr),
                "vus_roc": float(continuous_vus_roc),
                "vus_pr": float(continuous_vus_pr)
            }

            lbls = total_labels.to(self.device)

            for r1 in base_neighbors:
                thresh1 = torch.quantile(s1, 1. - r1)
                preds1 = s1 >= thresh1

                for r2 in other_neighbors:
                    thresh2 = torch.quantile(s2, 1. - r2)
                    preds2 = s2 >= thresh2

                    preds_raw = (preds1 | preds2).float()

                    if adjusted:
                        preds_np = preds_raw.detach().cpu().numpy()
                        lbls_np = lbls.detach().cpu().numpy()
                        preds_adj = self.adjust_preds(preds_np, lbls_np)
                        preds_final = preds_adj.to(self.device)
                    else:
                        preds_final = preds_raw

                    f1 = binary_f1_score(preds_final, lbls.int())

                    if f1 > best_combo["f1"]:
                        p = binary_precision(preds_final, lbls.int())
                        r = binary_recall(preds_final, lbls.int())
                        add = self.calculate_add(preds_raw.detach().cpu().numpy(), lbls.detach().cpu().numpy())

                        best_combo = {
                            "f1": float(f1),
                            "p": float(p),
                            "r": float(r),
                            "add": float(add),
                            "rauc": float(continuous_rauc),
                            "rauc_pr": float(continuous_rauc_pr),
                            "vus_roc": float(continuous_vus_roc),
                            "vus_pr": float(continuous_vus_pr),
                            "ratio1": r1,
                            "ratio2": r2
                        }

            return best_combo


        # =========================
        # COMBINATIONS
        # =========================
        combo_results = {}

        pairs = ["epsilon", "residual", "cos_res", "cos_eps", "maha_res"]

        for name in pairs:
            combo_results[f"total+{name}_adj"] = combination_search("total", name, adjusted=True)
            combo_results[f"total+{name}_raw"] = combination_search("total", name, adjusted=False)

        if epoch == 14:
            with open("log_roc.txt", "a") as f:
            
                f.write(f"\n===== Epoch {epoch} BEST RESULTS =====\n")
                f.write(f"\n===== dataset {self.dataset} BEST RESULTS =====\n")
        
                # =========================
                # SINGLE DETECTOR RESULTS
                # =========================
                for name in results:
                
                    adj = results[name]["adj"]
                    raw = results[name]["raw"]
        
                    # ---------- ADJUSTED ----------
                    f.write(f"\n[{name.upper()} - ADJ]\n")
                    f.write(f"Best Threshold Ratio: {adj['thresh_ratio']}\n")
                    f.write(f"Precision: {adj['p']}\n")
                    f.write(f"Recall: {adj['r']}\n")
                    f.write(f"F1: {adj['f1']}\n")
                    f.write(f"ADD: {adj['add']}\n")
                    f.write(f"R-AUC-ROC: {adj['rauc']}\n")
                    f.write(f"R-AUC-PR: {adj['rauc_pr']}\n")
                    f.write(f"VUS-ROC: {adj['vus_roc']}\n")
                    f.write(f"VUS-PR: {adj['vus_pr']}\n")
                    
        
                    # ---------- RAW ----------
                    f.write(f"\n[{name.upper()} - RAW]\n")
                    f.write(f"Best Threshold Ratio: {raw['thresh_ratio']}\n")
                    f.write(f"Precision: {raw['p']}\n")
                    f.write(f"Recall: {raw['r']}\n")
                    f.write(f"F1: {raw['f1']}\n")
                    f.write(f"ADD: {raw['add']}\n")
                    f.write(f"R-AUC-ROC: {adj['rauc']}\n")
                    f.write(f"R-AUC-PR: {adj['rauc_pr']}\n")
                    f.write(f"VUS-ROC: {adj['vus_roc']}\n")
                    f.write(f"VUS-PR: {adj['vus_pr']}\n")
        
        
                # =========================
                # COMBINATION RESULTS
                # =========================
                f.write("\n--- COMBINATIONS (Best by F1) ---\n")

                for k, v in combo_results.items():
                    f.write(f"\n{k}:\n")
                    f.write(f"  F1: {v['f1']:.4f}\n")
                    f.write(f"  Precision: {v['p']:.4f}\n")
                    f.write(f"  Recall: {v['r']:.4f}\n")
                    f.write(f"  ADD: {v['add']:.2f}\n")
                    f.write(f"  R-AUC-ROC: {v['rauc']:.4f}\n")
                    f.write(f"  Ratio1: {v['ratio1']}\n")
                    f.write(f"  Ratio2: {v['ratio2']}\n")
                    f.write(f"  R-AUC-ROC: {v['rauc']:.4f}\n")
                    f.write(f"  R-AUC-PR: {v['rauc_pr']:.4f}\n")


        # =========================
        # FIX: TENSORBOARD & FILE LOGGING
        # =========================
        # We will use the 'total' score's best adjusted results as the main anchor for plotting and logs.
        best_total_adj = results["total"]["adj"]

        self.tb_writer.add_scalars(
            "Scores_Total_Adj",
            {
                "P": best_total_adj["p"],
                "R": best_total_adj["r"],
                "F1": best_total_adj["f1"],
                "Threshold": best_total_adj["thresh_value"],
            },
            epoch # Added epoch here so it plots correctly on the x-axis
        )

        self.tb_writer.flush()

        # Reconstruct the best predictions array to dump into the text file
        best_threshold = best_total_adj["thresh_value"]
        best_raw_predictions = (total_scores >= best_threshold).float().cpu().numpy()
        
        # adjust_preds returns a Tensor, so we call .numpy() on it
        preds = self.adjust_preds(best_raw_predictions, total_labels.cpu().numpy()).numpy() 
        labels = total_labels.cpu().int().numpy()

        # Ensure all score arrays are moved to CPU and converted to numpy for writing
        total_pred_scores_np = total_pred_scores.cpu().numpy() 
        total_half_cur_scores_np = total_half_cur_scores.cpu().numpy()
        total_first_cur_scores_np = total_first_cur_scores.cpu().numpy()
        total_scores_np = total_scores.cpu().numpy()
        total_maha_eps_np = total_maha_eps.cpu().numpy()
        total_cos_eps_np = total_cos_eps.cpu().numpy()
        total_maha_res_np = total_maha_res.cpu().numpy()
        total_cos_res_np = total_cos_res.cpu().numpy()
        total_res_scores_np = total_res_scores.cpu().numpy()
        
        limit = len(preds)
        
        with open(f"logs/{self.dataset}_num.txt", "w") as f:
            f.write(f"epoch: {epoch}\n")
            f.write(f"threshold: {best_threshold}\n")
            anomaly_counter = -1
            for i in range(limit):
                if i % 96 == 0:
                    anomaly_counter += 1
                if preds[i] == 1 and labels[i] == 0:
                    f.write(
                        f"False Positive | index: {i} | "
                        f"total_cond_scores: {total_scores_np[i]} | "
                        f"total_half_scores: {total_half_cur_scores_np[i]} | "
                        f"total_first_scores: {total_first_cur_scores_np[i]} | "
                        f"epsilon: {total_pred_scores_np[i]} | "
                        f"residual: {total_res_scores_np[i]} | "
                        f"maha_eps: {total_maha_eps_np[i]} | "
                        f"maha_res: {total_maha_res_np[i]} | "
                        f"cos_eps: {total_cos_eps_np[i]} | "
                        f"cos_res: {total_cos_res_np[i]}\n"
                    )
                if preds[i] == 0 and labels[i] == 1:
                    f.write(
                        f"False Negative | index: {i} | "
                        f"total_cond_scores: {total_scores_np[i]} | "
                        f"total_half_scores: {total_half_cur_scores_np[i]} | "
                        f"total_first_scores: {total_first_cur_scores_np[i]} | "
                        f"epsilon: {total_pred_scores_np[i]} | "
                        f"residual: {total_res_scores_np[i]} | "
                        f"maha_eps: {total_maha_eps_np[i]} | "
                        f"maha_res: {total_maha_res_np[i]} | "
                        f"cos_eps: {total_cos_eps_np[i]} | "
                        f"cos_res: {total_cos_res_np[i]}\n"
                    )
                if preds[i] == 1 and labels[i] == 1:
                    f.write(
                        f"True Positive | index: {i} | "
                        f"total_cond_scores: {total_scores_np[i]} | "
                        f"total_half_scores: {total_half_cur_scores_np[i]} | "
                        f"total_first_scores: {total_first_cur_scores_np[i]} | "
                        f"epsilon: {total_pred_scores_np[i]} | "
                        f"residual: {total_res_scores_np[i]} | "
                        f"maha_eps: {total_maha_eps_np[i]} | "
                        f"maha_res: {total_maha_res_np[i]} | "
                        f"cos_eps: {total_cos_eps_np[i]} | "
                        f"cos_res: {total_cos_res_np[i]}\n"
                    )
                if preds[i] == 0 and labels[i] == 0:
                    f.write(
                        f"True Negative | index: {i} | "
                        f"total_cond_scores: {total_scores_np[i]} | "
                        f"total_half_scores: {total_half_cur_scores_np[i]} | "
                        f"total_first_scores: {total_first_cur_scores_np[i]} | "
                        f"epsilon: {total_pred_scores_np[i]} | "
                        f"residual: {total_res_scores_np[i]} | "
                        f"maha_eps: {total_maha_eps_np[i]} | "
                        f"maha_res: {total_maha_res_np[i]} | "
                        f"cos_eps: {total_cos_eps_np[i]} | "
                        f"cos_res: {total_cos_res_np[i]}\n"
                    )

        #preds = best_raw_predictions.int().numpy()
        #labels = best_raw_labels_final.int().numpy()
        #limit = len(preds)
        #with open(f"rawlogs/{epoch}.txt", "w") as f:
        #    f.write(f"epoch: {epoch}\n")
        #    f.write(f"threshold: {best_threshold}\n")
        #    anomaly_counter = -1
        #    for i in range(limit):
        #        if i % 96 == 0:
        #            anomaly_counter += 1
        #        if preds[i] == 1 and labels[i] == 0:
        #            f.write(...)

