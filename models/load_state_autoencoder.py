import torch
import numpy as np
import pandas as pd
from icecream import ic
import os
import joblib
from models.submodels.RNN_model import RNN
from models.submodels.AE_model import AE, train_ae
from models.submodels.train import train_rnn, HIDDEN_DIMS, N_LAYERS
from models.ai_clinician import get_policy_bins
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


ai_instance = None

def get_instance():
    global ai_instance
    if ai_instance is None:
        ai_instance = RecordPredictor()
    return ai_instance

opt = {'ae_path': './models/submodels/ae_models.pth', 'rnn_path': './models/submodels/rnn_models.pth', 'study_pkl': './models/submodels/ae_study.pkl'}

class RecordPredictor:

    colbin = ['gender','mechvent','max_dose_vaso','re_admission']
    colnorm=['age','Weight_kg','GCS','HR','SysBP','MeanBP','DiaBP','RR','Temp_C','FiO2_1',
        'Potassium','Sodium','Chloride','Glucose','Magnesium','Calcium',
        'Hb','WBC_count','Platelets_count','PTT','PT','Arterial_pH','paO2','paCO2',
        'Arterial_BE','HCO3','Arterial_lactate','SOFA','SIRS','Shock_Index','PaO2_FiO2','cumulated_balance']
    collog=['SpO2','BUN','Creatinine','SGOT','SGPT','Total_bili','INR','input_total','input_4hourly','output_total','output_4hourly']

    def __init__(self):

        if not os.path.isfile(opt['ae_path']) or not os.path.isfile(opt['study_pkl']):
            train_ae(opt)
        save_ae_pth = torch.load(opt['ae_path'])
        ae_study = joblib.load(opt['study_pkl'])
        self.ae_model = AE(ae_study.best_trial, ae_study.best_trial.params['latent_n_units'])
        self.ae_model.load_state_dict(save_ae_pth['model_state'])

        if not os.path.isfile(opt['rnn_path']):
            train_rnn(opt)
        save_rnn_pth = torch.load(opt['rnn_path'])
        self.rnn_model = RNN(self.ae_model.latent_features, self.ae_model.latent_features, HIDDEN_DIMS, N_LAYERS)
        self.rnn_model.load_state_dict(save_rnn_pth['model_state'])

        self.MIMICtable = pd.read_csv('models/data/final_table.csv')

        with open('models/data/MIMICraw.npy', 'rb') as f:
            MIMICraw = np.load(f, allow_pickle=True)
        
        self.colunm_mean = np.concatenate([np.mean(MIMICraw[:, 4:36], axis=0), np.mean(np.log(MIMICraw[:, 36:] + 0.1), axis=0)])
        self.colunm_mean[np.isnan(self.colunm_mean)] = 0
        self.colunm_std = np.concatenate([np.std(MIMICraw[:, 4:36], axis=0), np.std(np.log(MIMICraw[:, 36:] + 0.1), axis=0)])
        self.colunm_std[np.isnan(self.colunm_std)] = 0
    
    def predict(self, records, actions):

        if type(records) is not pd.core.frame.DataFrame:
            print('Record to be predict should be a pandas DataFrame.')
            return
        if np.isin(records.columns, RecordPredictor.colbin).sum() + \
                np.isin(records.columns, RecordPredictor.colnorm).sum() + \
                np.isin(records.columns, RecordPredictor.collog).sum() != \
                len(RecordPredictor.colbin) + \
                len(RecordPredictor.colnorm) + \
                len(RecordPredictor.collog):
            print('Wrong length of record.')
            return

        records = records.loc[:, RecordPredictor.colbin + RecordPredictor.colnorm + RecordPredictor.collog]

        # for i, col in enumerate(RecordPredictor.colnorm + RecordPredictor.collog):
        #     records[col] = pd.cut(records[col], self.norm_and_log_bins[i], labels=range(10)) if i < len(RecordPredictor.colnorm) else \
                # pd.cut(np.log(records[col] + 0.1), self.norm_and_log_bins[i], labels=range(10))
        
        _, _, vc_bins, io_bins = get_policy_bins(self.MIMICtable)

        for i in range(len(records)):
            io_action = actions[i] // 5
            vc_action = actions[i] % 5

            records.loc[i, 'max_dose_vaso'] = vc_bins[vc_action][1]
            records.loc[i, 'input_4hourly'] = io_bins[io_action][1]

        for col in RecordPredictor.colbin:
            records[col] = records[col] - 0.5

        # print(records, self.colunm_mean, self.colunm_std)

        for i, col in enumerate(RecordPredictor.colnorm + RecordPredictor.collog):
            records[col] = (records[col] - self.colunm_mean[i]) / self.colunm_std[i] if i < len(RecordPredictor.colnorm) else \
                (np.log(records[col] + 0.1) - self.colunm_mean[i]) / self.colunm_std[i]


        records = records.to_numpy().astype('float64')

        records[np.isnan(records)] = 0
        records[np.isinf(records)] = 0
        records[:, 2] = np.log(records[:, 2] + 6)   # MAX DOSE NORAD 
        records[:, 44] = 2 * records[:, 44]   # increase weight of input_4hourly
        records = torch.tensor(records).to(device, dtype=torch.float)

        with torch.no_grad():

            self.ae_model.eval()

            hidden = self.ae_model.encode(records)[0]

            self.rnn_model.eval()

            input_pack = pack_padded_sequence(hidden.reshape(1, -1, hidden.shape[1]), torch.tensor([hidden.shape[0]]), batch_first=True)

            output, _ = self.rnn_model(input_pack)

            preds = self.ae_model.decode(output).numpy()[:, -1, :]

        preds = pd.DataFrame(preds, columns = RecordPredictor.colbin + RecordPredictor.colnorm + RecordPredictor.collog)

        for col in RecordPredictor.colbin:
            if col != 'max_dose_vaso':
                preds.loc[preds[col] > 0, col] = 1
                preds.loc[preds[col] <= 0, col] = 0
            else:
                preds[col] += 0.5
        
        for i, col in enumerate(RecordPredictor.colnorm + RecordPredictor.collog):
            if i < len(RecordPredictor.colnorm):
                preds[col] = preds[col] * self.colunm_std[i] + self.colunm_mean[i]
            else:
                preds[col] = np.exp(preds[col] * self.colunm_std[i] + self.colunm_mean[i]) - 0.1
        return preds

        
        