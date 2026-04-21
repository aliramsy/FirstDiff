from datasets import *
from torch.utils.data import DataLoader
from solver import *
from model.AE import AutoEncoder
from model.DiffModel import DiffModel
import torch
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
#from model.cnn_encoder import HistoryEncoderCNN
#from model.cross_attention import HistoryEncoderPerceiver
#from model.dit import TimeSeriesDiT
from model.uncdit import UnconditionalTimeSeriesDiT
#from model.vit_encoder import HistoryEncoderViT
from model.timeseriesdit import TimeSeriesDiT
from model.mrdd import MRDDFrequencyDecomposer


BATCH_SIZE = 64
WINDOW_SIZE = 128
STRIDE = 96
GPU_ID = 0
EPOCHS = 20
NOISE_STEPS = 100
HISTORY_SIZE = 1024
DiT_num_layers = 6
RUN = 1
INFO = f"GAT_SA_EXPTMP_TMPTMP_DEL_RUN_{RUN}"
D_MODEL = 256
HISTORY = True
TRAIN_PATH = {'MSL': "datasets/MSL/MSL_train.npy", 'SMAP': "datasets/SMAP/SMAP_train.npy", 'SWaT': "datasets/SWaT/swat_train2.csv",
            'PSM': "datasets/PSM/train.csv"}
TEST_PATH = {'MSL': "datasets/MSL/MSL_test.npy", 'SMAP': "datasets/SMAP/SMAP_test.npy",  'SWaT': "datasets/SWaT/swat2.csv",
             'PSM': "datasets/PSM/test.csv"}
TEST_LABEL_PATH = {'MSL': "datasets/MSL/MSL_test_label.npy", 'SMAP': "datasets/SMAP/SMAP_test_label.npy",  'SWaT': None,
                   'PSM': "datasets/PSM/test_label.csv"}
CHANNELS = {'MSL': 55, 'SMAP': 25,  'SWaT': 51, 'PSM': 25}


CHANNELS = {'SWaT': 51}

dataset = "SWaT"

if dataset in ['SWaT', 'PSM']:
    df = pd.read_csv(TRAIN_PATH[dataset])
    if dataset == 'PSM':
        train_raw = df.values[:, 1:]
        train_raw = np.nan_to_num(train_raw)
    else:
        train_raw = df.values[:, :-1]
else:
    train_raw = np.load(TRAIN_PATH[dataset])

scaler = StandardScaler()
scaler.fit(train_raw)

if dataset in ['SWaT', 'PSM']:
    train = TrainDatasetCSV(TRAIN_PATH[dataset], scaler, None, WINDOW_SIZE, STRIDE, 0.85)
    val = ValDatasetCSV(TRAIN_PATH[dataset], scaler, None, WINDOW_SIZE, STRIDE, 0.85)
    test = TestDatasetCSV(TEST_PATH[dataset], TEST_LABEL_PATH[dataset], scaler, None, WINDOW_SIZE, STRIDE, 1.0)
else:
    train = TrainDataset(TRAIN_PATH[dataset], scaler, None, WINDOW_SIZE, STRIDE, 0.85)
    val = ValDataset(TRAIN_PATH[dataset], scaler, None, WINDOW_SIZE, STRIDE, 0.85)
    test = TestDataset(TEST_PATH[dataset], TEST_LABEL_PATH[dataset], scaler, None, WINDOW_SIZE, STRIDE, 1.0)

train_dataloader = DataLoader(train, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
val_dataloader = DataLoader(val, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)
test_dataloader = DataLoader(test, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)

if torch.cuda.is_available():
    device = f'cuda:{GPU_ID}'
else:
    device = 'cpu'

channels = CHANNELS[dataset]

autoencoder = AutoEncoder(channels=channels, h_dim=channels, h_dim_2=256, device=device)
diff_model = DiffModel(c_in=channels, c_out=channels, device=device)
#dit_model = TimeSeriesDiT(num_sensors=CHANNELS[dataset], hidden_dim=D_MODEL, num_layers=DiT_num_layers).to(device)
dit_model = TimeSeriesDiT(target_seq_len=WINDOW_SIZE,
        history_seq_len=32,
        num_sensors=CHANNELS[dataset],
        hidden_dim=D_MODEL,
        num_heads=8,
        num_layers=DiT_num_layers).to(device)

decomposer = MRDDFrequencyDecomposer().to(device)

diffusion = Diffusion(noise_steps=NOISE_STEPS, device=device)
#history_model = HistoryEncoderCNN(in_channels=CHANNELS[dataset], d_model=D_MODEL).to(device)
#history_model = HistoryEncoderPerceiver(seq_len= HISTORY_SIZE, in_channels=CHANNELS[dataset], num_latents=32, d_model=D_MODEL, nhead=8)
#history_model = HistoryEncoderViT(seq_len= HISTORY_SIZE, in_channels=CHANNELS[dataset], num_patches=32, d_model=D_MODEL)

#uncdit = UnconditionalTimeSeriesDiT(target_seq_len=WINDOW_SIZE, num_sensors=channels, hidden_dim=D_MODEL, num_heads=8, num_layers=DiT_num_layers)

experiment = {"dataset": dataset, "noise_steps": NOISE_STEPS, "epochs": EPOCHS, "batch_size": BATCH_SIZE, "window_size": WINDOW_SIZE, 'info': INFO}
#solver = Solver(autoencoder, diff_model, train_dataloader, val_dataloader, test_dataloader, diffusion=diffusion, mask_data=True, anomaly_ratio=0.05, experiment=experiment, device=device, gpu_id=GPU_ID)
solver = Solver(decomposer, dit_model, train_dataloader, val_dataloader, test_dataloader, diffusion=diffusion, mask_data=False, anomaly_ratio=0.05, experiment=experiment, device=device, gpu_id=GPU_ID)

solver.train(EPOCHS)
solver.load_model()
#uncomment
#solver.test()