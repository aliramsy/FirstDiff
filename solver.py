import torch.nn.functional as F
import torch
from sklearn.decomposition import PCA
from sklearn.covariance import LedoitWolf
from torch.utils.tensorboard import SummaryWriter
from torcheval.metrics.functional import binary_f1_score, binary_precision, binary_recall, binary_auroc
from model.Diffusion import *

class Solver():
    # add autoencoder to the init function it is after self
    def __init__(self, diff_model, train_loader, val_loader, test_loader, diffusion=None, mask_data=True, anomaly_ratio=0.05, experiment=None, device='cuda', gpu_id=0, decomposer = None):
        #self.autoencoder = autoencoder
        self.decomposer = decomposer
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

        for epoch in range(epochs):
            #self.autoencoder.train()
            self.diff_model.train()

            avg_loss = self.train_one_epoch(epoch)
 
            avg_vloss = self.val(epoch)
            self.tb_writer.add_scalars('Loss', {"Train" : avg_loss, "Val" : avg_vloss}, epoch)
            print(f"EPOCH {epoch} LOSS train {avg_loss} valid {avg_vloss}")
            self.test(epoch)

            #if epoch % 5 == 0 and epoch !=0:
            #    self.save_model([f'AE_{epoch}', f'Diffusion_{epoch}'])

        self.tb_writer.flush()
        self.save_model(f'Diffusion_{epoch}')

    
    def denoise_process(self, i, x, batch_num=128, epoch=None):

        for j in range(self.diffusion.noise_steps - 1, -1, -1):
            t = (j * torch.ones(x.shape[0])).long().to(self.device)       
            self.diff_model.eval()

            predicted_noise = self.diff_model(x_curr=x, t=t)
            if j == self.diffusion.noise_steps - 1:
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
        Calculates the Anomaly Detection Delay (ADD) quickly using vectorization.
        """
        # Ensure they are on CPU to avoid sync overhead if not already
        if actual.is_cuda:
            actual = actual.cpu()
        if raw_predict.is_cuda:
            raw_predict = raw_predict.cpu()

        # Find where actual transitions from 0 to 1 (starts) and 1 to 0 (ends)
        diff = torch.diff(actual, prepend=torch.tensor([0.0]))
        
        starts = torch.where(diff == 1)[0]
        ends = torch.where(diff == -1)[0]
        
        # Handle edge case: if the sequence ends while still in an anomaly
        if len(ends) < len(starts):
            ends = torch.cat([ends, torch.tensor([len(actual)])])
            
        if len(starts) == 0:
            return 0.0

        delays = []
        
        # Iterate over the segments (fast, because there are usually few segments)
        for s, e in zip(starts, ends):
            segment_preds = raw_predict[s:e]
            
            # argmax is extremely fast and returns the index of the first '1' 
            # (or 0 if no 1s exist, so we double-check if it actually hit)
            first_hit_idx = torch.argmax(segment_preds).item()
            
            if segment_preds[first_hit_idx] == 1:
                delays.append(first_hit_idx)
                
        if len(delays) == 0:
            return 0.0
            
        return sum(delays) / len(delays)

    

    def get_best_score(self, f1, p, r, rocauc, threshold, scores: dict, key='f1'):
        if f1 > scores[key]:
            scores[key] = f1
            scores['p'] = p
            scores['r'] = r
            scores['rocauc'] = rocauc
            scores['thres'] = threshold
    
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

                total_pred_scores = torch.cat(
                    [total_pred_scores, preds_scores.reshape(-1)]
                )

                total_labels = torch.cat(
                    [total_labels, labels.reshape(-1)]
                )

                tloss = F.mse_loss(diff_toutputs, ae_x)

                if self.mask_data:
                    tloss += self.loss_fn(toutputs, tinputs)

                running_tloss += tloss.item()

        avg_tloss = running_tloss / len(self.test_loader)

        best_scores = {"f1": 0, "p": 0, "r": 0, "rocauc": 0}
        best_raw_scores = {"f1": 0, "p": 0, "r": 0, "rocauc": 0}

        f = open("results.txt", "w")
        f.write(f"epoch: {epoch} \n")

        best_predictions = None
        best_labels_final = None
        best_threshold = None
        total_labels = total_labels.detach().cpu()

        for anomaly_threshold in torch.arange(0., 0.15 ,0.01):

            thresh = torch.quantile(
                total_scores, 1. - anomaly_threshold.to(self.device)
            )

            raw_predictions = torch.where(
                total_scores >= thresh, 1., 0.
            )
            raw_predictions = raw_predictions.detach().cpu()

            raw_tp = torch.sum(
                (raw_predictions == 1) & (total_labels == 1)
            ).item()

            raw_fp = torch.sum(
                (raw_predictions == 1) & (total_labels == 0)
            ).item()

            raw_tn = torch.sum(
                (raw_predictions == 0) & (total_labels == 0)
            ).item()

            raw_fn = torch.sum(
                (raw_predictions == 0) & (total_labels == 1)
            ).item()

            raw_f1 = binary_f1_score(
                raw_predictions, total_labels, threshold=anomaly_threshold
            )
            if raw_f1 > best_raw_scores["f1"]:
                best_raw_predictions = raw_predictions.clone()
                best_raw_labels_final = total_labels.clone()
                best_threshold_raw = anomaly_threshold

            raw_p = binary_precision(
                raw_predictions,
                total_labels.type(torch.int32),
                threshold=anomaly_threshold,
            )

            raw_r = binary_recall(
                raw_predictions,
                total_labels.type(torch.int32),
                threshold=anomaly_threshold,
            )

            raw_roc = binary_auroc(raw_predictions, total_labels)

            # --- NEW ADD CALCULATION HERE ---
            avg_delay = self.calculate_add(raw_predictions, total_labels)

            adj_predictions = self.adjust_preds(
                raw_predictions.detach().cpu().numpy(),
                total_labels.numpy(),
            )

            adj_tp = torch.sum(
                (adj_predictions == 1) & (total_labels == 1)
            ).item()

            adj_fp = torch.sum(
                (adj_predictions == 1) & (total_labels == 0)
            ).item()

            adj_tn = torch.sum(
                (adj_predictions == 0) & (total_labels == 0)
            ).item()

            adj_fn = torch.sum(
                (adj_predictions == 0) & (total_labels == 1)
            ).item()

            adj_f1 = binary_f1_score(
                adj_predictions, total_labels, threshold=anomaly_threshold
            )
            # select best
            if adj_f1 > best_scores["f1"]:
                best_predictions = adj_predictions.clone()
                best_labels_final = total_labels.clone()
                best_threshold = anomaly_threshold


            adj_p = binary_precision(
                adj_predictions,
                total_labels.type(torch.int32),
                threshold=anomaly_threshold,
            )

            adj_r = binary_recall(
                adj_predictions,
                total_labels.type(torch.int32),
                threshold=anomaly_threshold,
            )

            adj_roc = binary_auroc(adj_predictions, total_labels)

            self.get_best_score(
                adj_f1,
                adj_p,
                adj_r,
                adj_roc,
                anomaly_threshold,
                best_scores,
            )

            msg_raw = (
                f"[RAW] Thresh {anomaly_threshold:.3f}, "
                f"f1: {raw_f1:.4f}, p: {raw_p:.4f}, r: {raw_r:.4f}, "
                f"ROC: {raw_roc:.4f} | ADD: {avg_delay:.2f} | "
                f"TP:{raw_tp} FP:{raw_fp} TN:{raw_tn} FN:{raw_fn}"
            )

            print(msg_raw)
            #f.write(msg_raw + "\n")

            msg_adj = (
                f"[ADJ] Thresh {anomaly_threshold:.3f}, "
                f"f1: {adj_f1:.4f}, p: {adj_p:.4f}, r: {adj_r:.4f}, "
                f"ROC: {adj_roc:.4f} | ADD: {avg_delay:.2f} | "
                f"TP:{adj_tp} FP:{adj_fp} TN:{adj_tn} FN:{adj_fn}"
            )


            print(msg_adj)
            #f.write(msg_adj + "\n")
            #f.write("-" * 50 + "\n")
            with open("log.txt", "a") as f:
                f.write(msg_raw + "\n")
                f.write(msg_adj + "\n")
                f.write("-" * 50 + "\n")

            total_predictions = adj_predictions

        f.close()

        self.tb_writer.add_scalars(
            "Scores",
            {
                "P": best_scores["p"],
                "R": best_scores["r"],
                "F1": best_scores["f1"],
                "ROCAUC": best_scores["rocauc"],
                "Threshold": best_scores["thres"],
            },
        )

        self.tb_writer.flush()

        preds = best_predictions.int().numpy()
        best_threshold = best_threshold.numpy()
        total_pred_scores = total_pred_scores.cpu().numpy() 
        total_half_cur_scores = total_half_cur_scores.cpu().numpy()
        total_first_cur_scores = total_first_cur_scores.cpu().numpy()
        total_scores = total_scores.cpu().numpy()
        total_maha_eps = total_maha_eps.cpu().numpy()
        total_cos_eps = total_cos_eps.cpu().numpy()
        total_maha_res = total_maha_res.cpu().numpy()
        total_cos_res = total_cos_res.cpu().numpy()
        total_res_scores = total_res_scores.cpu().numpy()
        labels = best_labels_final.int().numpy()
        limit = len(preds)
        with open(f"logs/{epoch}.txt", "w") as f:
            f.write(f"epoch: {epoch}\n")
            f.write(f"threshold: {best_threshold}\n")
            anomaly_counter = -1
            for i in range(limit):
                if i % 96 == 0:
                    anomaly_counter += 1
                if preds[i] == 1 and labels[i] == 0:
                    f.write(
                        f"False Positive | index: {i} | "
                        f"total_cond_scores: {total_scores[i]} | "
                        f"total_half_scores: {total_half_cur_scores[i]} | "
                        f"total_first_scores: {total_first_cur_scores[i]} | "
                        f"epsilon: {total_pred_scores[i]} | "
                        f"residual: {total_res_scores[i]} | "
                        f"maha_eps: {total_maha_eps[i]} | "
                        f"maha_res: {total_maha_res[i]} | "
                        f"cos_eps: {total_cos_eps[i]} | "
                        f"cos_res: {total_cos_res[i]}\n"
                    )
                if preds[i] == 0 and labels[i] == 1:
                    f.write(
                        f"False Negative | index: {i} | "
                        f"total_cond_scores: {total_scores[i]} | "
                        f"total_half_scores: {total_half_cur_scores[i]} | "
                        f"total_first_scores: {total_first_cur_scores[i]} | "
                        f"epsilon: {total_pred_scores[i]} | "
                        f"residual: {total_res_scores[i]} | "
                        f"maha_eps: {total_maha_eps[i]} | "
                        f"maha_res: {total_maha_res[i]} | "
                        f"cos_eps: {total_cos_eps[i]} | "
                        f"cos_res: {total_cos_res[i]}\n"
                    )
                if preds[i] == 1 and labels[i] == 1:
                    f.write(
                        f"True Positive | index: {i} | "
                        f"total_cond_scores: {total_scores[i]} | "
                        f"total_half_scores: {total_half_cur_scores[i]} | "
                        f"total_first_scores: {total_first_cur_scores[i]} | "
                        f"epsilon: {total_pred_scores[i]} | "
                        f"residual: {total_res_scores[i]} | "
                        f"maha_eps: {total_maha_eps[i]} | "
                        f"maha_res: {total_maha_res[i]} | "
                        f"cos_eps: {total_cos_eps[i]} | "
                        f"cos_res: {total_cos_res[i]}\n"
                    )
                if preds[i] == 0 and labels[i] == 0:
                    f.write(
                        f"True Negative | index: {i} | "
                        f"total_cond_scores: {total_scores[i]} | "
                        f"total_half_scores: {total_half_cur_scores[i]} | "
                        f"total_first_scores: {total_first_cur_scores[i]} | "
                        f"epsilon: {total_pred_scores[i]} | "
                        f"residual: {total_res_scores[i]} | "
                        f"maha_eps: {total_maha_eps[i]} | "
                        f"maha_res: {total_maha_res[i]} | "
                        f"cos_eps: {total_cos_eps[i]} | "
                        f"cos_res: {total_cos_res[i]}\n"
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
        #            f.write(
        #                f"False Positive | index: {i} | "
        #                f"total_cond_scores: {total_scores[i]} | "
        #                f"total_half_scores: {total_half_cur_scores[i]} | "
        #                f"total_first_scores: {total_first_cur_scores[i]} | "
        #                f"epsilon: {total_pred_scores[i]} | "
        #                f"residual: {total_res_scores[i]} | "
        #                f"pca1_eps: {total_pca1[i]} | "
        #                f"pca2_eps: {total_pca2[i]} | "
        #                f"pca1_res: {total_pca1_res[i]} | "
        #                f"pca2_res: {total_pca2_res[i]} | "
        #                f"maha_eps: {total_maha_eps[i]} | "
        #                f"maha_res: {total_maha_res[i]} | "
        #                f"cos_eps: {total_cos_eps[i]} | "
        #                f"cos_res: {total_cos_res[i]}\n"
        #            )
        #        if preds[i] == 0 and labels[i] == 1:
        #            f.write(
        #                f"False Negative | index: {i} | "
        #                f"total_cond_scores: {total_scores[i]} | "
        #                f"total_half_scores: {total_half_cur_scores[i]} | "
        #                f"total_first_scores: {total_first_cur_scores[i]} | "
        #                f"epsilon: {total_pred_scores[i]} | "
        #                f"residual: {total_res_scores[i]} | "
        #                f"pca1_eps: {total_pca1[i]} | "
        #                f"pca2_eps: {total_pca2[i]} | "
        #                f"pca1_res: {total_pca1_res[i]} | "
        #                f"pca2_res: {total_pca2_res[i]} | "
        #                f"maha_eps: {total_maha_eps[i]} | "
        #                f"maha_res: {total_maha_res[i]} | "
        #                f"cos_eps: {total_cos_eps[i]} | "
        #                f"cos_res: {total_cos_res[i]}\n"
        #            )
        #        if preds[i] == 1 and labels[i] == 1:
        #            f.write(
        #                f"True Positive | index: {i} | "
        #                f"total_cond_scores: {total_scores[i]} | "
        #                f"total_half_scores: {total_half_cur_scores[i]} | "
        #                f"total_first_scores: {total_first_cur_scores[i]} | "
        #                f"epsilon: {total_pred_scores[i]} | "
        #                f"residual: {total_res_scores[i]} | "
        #                f"pca1_eps: {total_pca1[i]} | "
        #                f"pca2_eps: {total_pca2[i]} | "
        #                f"pca1_res: {total_pca1_res[i]} | "
        #                f"pca2_res: {total_pca2_res[i]} | "
        #                f"maha_eps: {total_maha_eps[i]} | "
        #                f"maha_res: {total_maha_res[i]} | "
        #                f"cos_eps: {total_cos_eps[i]} | "
        #                f"cos_res: {total_cos_res[i]}\n"
        #            )
        #        if preds[i] == 0 and labels[i] == 0:
        #            f.write(
        #                f"True Negative | index: {i} | "
        #                f"total_cond_scores: {total_scores[i]} | "
        #                f"total_half_scores: {total_half_cur_scores[i]} | "
        #                f"total_first_scores: {total_first_cur_scores[i]} | "
        #                f"epsilon: {total_pred_scores[i]} | "
        #                f"residual: {total_res_scores[i]} | "
        #                f"pca1_eps: {total_pca1[i]} | "
        #                f"pca2_eps: {total_pca2[i]} | "
        #                f"pca1_res: {total_pca1_res[i]} | "
        #                f"pca2_res: {total_pca2_res[i]} | "
        #                f"maha_eps: {total_maha_eps[i]} | "
        #                f"maha_res: {total_maha_res[i]} | "
        #                f"cos_eps: {total_cos_eps[i]} | "
        #                f"cos_res: {total_cos_res[i]}\n"
        #            )


    def save_model(self, names : 'Diffusion'):
        folder = "trained_models"
        #torch.save(self.autoencoder.state_dict(), f"{folder}/{names[0]}.pth")
        print(f"Saved PyTorch Model State to {names}.pth")
        torch.save(self.diff_model.state_dict(), f"{folder}/{names}.pth")

    def load_model(self, names: list[str] = [f'AE', f'Diffusion']):
        names = [f"{name}_{self.model_name}" for name in names]
        folder = "trained_models"
        #self.autoencoder.to(self.device)
        #self.autoencoder.load_state_dict(torch.load(f"{folder}/{names[0]}.pth"))
        self.diff_model.to(self.device)
        self.diff_model.load_state_dict(torch.load(f"{folder}/{names[1]}.pth"))
