from datasets import *
from torch.utils.data import DataLoader
from solver import *
from model.AE import AutoEncoder
#from model.DiffModel import DiffModel
import torch
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
#from model.cnn_encoder import HistoryEncoderCNN
#from model.cross_attention import HistoryEncoderPerceiver
#from model.dit import TimeSeriesDiT
from model.uncdit import UnconditionalTimeSeriesDiT
#from model.vit_encoder import HistoryEncoderViT
#from model.timeseriesdit import TimeSeriesDiT
#from model.mrdd import MRDDFrequencyDecomposer
#from model.clt import SpectralSignatureEncoder, pretrain_spectral_encoder

BATCH_SIZE = 128
WINDOW_SIZE = 96
WINDOW_SIZE_TRAIN = 128
STRIDE_TRAIN = 64
STRIDE = 96
GPU_ID = 0
EPOCHS = 15
NOISE_STEPS = 100
HISTORY_SIZE = 1024
HEAD_NUMBER = 8
DiT_num_layers = 6
RUN = 1
INFO = f"GAT_SA_EXPTMP_TMPTMP_DEL_RUN_{RUN}"
D_MODEL = 256
HISTORY = True
TRAIN_PATH = {'MSL': "datasets/MSL/MSL_train.npy", 'SMAP': "datasets/SMAP/SMAP_train.npy", 'SWaT': "datasets/SWaT/swat_train2.csv",
            'PSM': "datasets/PSM/train.csv", 'SMD': "datasets/SMD/train.npy"}
TEST_PATH = {'MSL': "datasets/MSL/MSL_test.npy", 'SMAP': "datasets/SMAP/SMAP_test.npy",  'SWaT': "datasets/SWaT/swat2.csv",
             'PSM': "datasets/PSM/test.csv", 'SMD': "datasets/SMD/test.npy"}
TEST_LABEL_PATH = {'MSL': "datasets/MSL/MSL_test_label.npy", 'SMAP': "datasets/SMAP/SMAP_test_label.npy",  'SWaT': None,
        'PSM': "datasets/PSM/test_label.csv", 'SMD':"datasets/SMD/test_label.npy"}
CHANNELS = {'MSL': 55, 'SMAP': 25,  'SWaT': 51, 'PSM': 25, 'SMD' : 38}

datasets = ["SWaT", "MSL", "PSM", "SMD", "SMAP"]

for i in range(2):
    for dataset in datasets:
        torch.manual_seed(42)
        np.random.seed(42)
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
            train_encoder = TrainDatasetCSV(TRAIN_PATH[dataset], scaler, None, WINDOW_SIZE_TRAIN, STRIDE_TRAIN, 0.85)
            train = TrainDatasetCSV(TRAIN_PATH[dataset], scaler, None, WINDOW_SIZE, STRIDE_TRAIN, 0.85)
            val = ValDatasetCSV(TRAIN_PATH[dataset], scaler, None, WINDOW_SIZE, STRIDE, 0.85)
            test = TestDatasetCSV(TEST_PATH[dataset], TEST_LABEL_PATH[dataset], scaler, None, WINDOW_SIZE, STRIDE, 1.0)
        else:
            train_encoder = TrainDataset(TRAIN_PATH[dataset], scaler, None, WINDOW_SIZE_TRAIN, STRIDE_TRAIN, 0.85)
            train = TrainDataset(TRAIN_PATH[dataset], scaler, None, WINDOW_SIZE, STRIDE_TRAIN, 0.85)
            val = ValDataset(TRAIN_PATH[dataset], scaler, None, WINDOW_SIZE, STRIDE, 0.85)
            test = TestDataset(TEST_PATH[dataset], TEST_LABEL_PATH[dataset], scaler, None, WINDOW_SIZE, STRIDE, 1.0)

        train_dataloader = DataLoader(train, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
        val_dataloader = DataLoader(val, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)
        test_dataloader = DataLoader(test, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)

        train_dataloader_encoder = DataLoader(train_encoder, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, drop_last=True)

        if torch.cuda.is_available():
            device = f'cuda:{GPU_ID}'
        else:
            device = 'cpu'

        channels = CHANNELS[dataset]
        #spectral_encoder = SpectralSignatureEncoder(num_sensors=CHANNELS[dataset], hidden_dim=D_MODEL, window_size=WINDOW_SIZE_TRAIN)
        #pretrain_spectral_encoder(encoder=spectral_encoder, train_loader=train_dataloader_encoder, device=device, epochs=1000)

        #dit_model = TimeSeriesDiT(target_seq_len=WINDOW_SIZE,num_sensors=CHANNELS[dataset],hidden_dim=D_MODEL,num_layers=DiT_num_layers).to(device)

        #dit_model.coarse_encoder = spectral_encoder
        #for param in dit_model.coarse_encoder.parameters():
        #    param.requires_grad = False

        diffusion = Diffusion(noise_steps=NOISE_STEPS, device=device)
        #history_model = HistoryEncoderCNN(in_channels=CHANNELS[dataset], d_model=D_MODEL).to(device)
        #history_model = HistoryEncoderPerceiver(seq_len= HISTORY_SIZE, in_channels=CHANNELS[dataset], num_latents=32, d_model=D_MODEL, nhead=8)
        #history_model = HistoryEncoderViT(seq_len= HISTORY_SIZE, in_channels=CHANNELS[dataset], num_patches=32, d_model=D_MODEL)

        uncdit = UnconditionalTimeSeriesDiT(target_seq_len=WINDOW_SIZE, num_sensors=channels, hidden_dim=D_MODEL, num_heads=HEAD_NUMBER, num_layers=DiT_num_layers).to(device)

        experiment = {"dataset": dataset, "noise_steps": NOISE_STEPS, "epochs": EPOCHS, "batch_size": BATCH_SIZE, "window_size": WINDOW_SIZE, 'info': INFO}

        solver = Solver(uncdit, train_dataloader, val_dataloader, test_dataloader, 
                    diffusion=diffusion, mask_data=False, experiment=experiment, 
                    device=device, gpu_id=GPU_ID, dataset = dataset)
        solver.train(EPOCHS)

        del solver
        del uncdit
        #del dit_model
        del diffusion
        #del spectral_encoder
        torch.cuda.empty_cache()
        #solver.load_model()
        #solver.test()
