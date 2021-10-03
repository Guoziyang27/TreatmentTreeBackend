import torch
from models.state_autoencoder.model import AE as AEmodel
import numpy as np
import pandas as pd
from icecream import ic

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


ai_instance = None

def get_instance(forward_num=4):
    global ai_instance
    if ai_instance is None:
        ai_instance = StateAutoencoder(forward_num)
    return ai_instance

class StateAutoencoder:

    colbin = ['gender','mechvent','max_dose_vaso','re_admission']
    colnorm=['age','Weight_kg','GCS','HR','SysBP','MeanBP','DiaBP','RR','Temp_C','FiO2_1',
        'Potassium','Sodium','Chloride','Glucose','Magnesium','Calcium',
        'Hb','WBC_count','Platelets_count','PTT','PT','Arterial_pH','paO2','paCO2',
        'Arterial_BE','HCO3','Arterial_lactate','SOFA','SIRS','Shock_Index','PaO2_FiO2','cumulated_balance']
    collog=['SpO2','BUN','Creatinine','SGOT','SGPT','Total_bili','INR','input_total','input_4hourly','output_total','output_4hourly']

    def __init__(self, forward_num=4):
        save_pth = torch.load('models/state_autoencoder/all_models.pth')
        self.model = AEmodel()
        self.model.load_state_dict(save_pth['model_state'])

        with open('models/state_autoencoder/norm_and_log_bins.npy', 'rb') as f:
            self.norm_and_log_bins = np.load(f, allow_pickle=True)
        with open('models/state_autoencoder/mean_std_list.npy', 'rb') as f:
            self.mean_std_list = np.load(f, allow_pickle=True)
        self.forward_num = forward_num
    
    def predict(self, records, actions):


        if type(records) is not pd.core.frame.DataFrame:
            print('Record to be predict should be a pandas DataFrame.')
            return
        if np.isin(records.columns, StateAutoencoder.colbin).sum() + \
                np.isin(records.columns, StateAutoencoder.colnorm).sum() + \
                np.isin(records.columns, StateAutoencoder.collog).sum() != \
                len(StateAutoencoder.colbin) + \
                len(StateAutoencoder.colnorm) + \
                len(StateAutoencoder.collog):
            print('Wrong length of record.')
            return

        records = records.loc[:, StateAutoencoder.colbin + StateAutoencoder.colnorm + StateAutoencoder.collog]

        # for i, col in enumerate(StateAutoencoder.colnorm + StateAutoencoder.collog):
        #     records[col] = pd.cut(records[col], self.norm_and_log_bins[i], labels=range(10)) if i < len(StateAutoencoder.colnorm) else \
                # pd.cut(np.log(records[col] + 0.1), self.norm_and_log_bins[i], labels=range(10))
        

        for col in StateAutoencoder.colbin:
            records[col] = records[col] - 0.5

        for i, col in enumerate(StateAutoencoder.colnorm + StateAutoencoder.collog):
            records[col] = (records[col] - self.mean_std_list[i, 0]) / self.mean_std_list[i, 1] if i < len(StateAutoencoder.colnorm) else \
                (np.log(records[col] + 0.1) - self.mean_std_list[i, 0]) / self.mean_std_list[i, 1]

        
        records.loc[:, 'actions'] = actions
        
        records = records.to_numpy().astype('float64')

        records[np.isnan(records)] = 0

        if len(records) < self.forward_num:
            records = np.concatenate([np.full([self.forward_num - len(records), records.shape[1]], -1), records])
        


        records = torch.tensor([records]).to(device, dtype=torch.float)

        with torch.no_grad():
            self.model.eval()

            output = self.model(records)

            preds = np.squeeze(output.detach().cpu().numpy())
        
        preds = pd.DataFrame([preds], columns = StateAutoencoder.colbin + StateAutoencoder.colnorm + StateAutoencoder.collog)

        for col in StateAutoencoder.colbin:
            if col != 'max_dose_vaso':
                preds.loc[preds[col] > 0, col] = 1
                preds.loc[preds[col] <= 0, col] = 0
            else:
                preds[col] += 0.5


        # for i, col in enumerate(StateAutoencoder.colnorm + StateAutoencoder.collog):
        #     if i < len(StateAutoencoder.colnorm):
        #         preds[col] = self.norm_and_log_bins[i][preds[col].astype('int') + 1]
        #     else:
        #         preds[col] = np.exp(self.norm_and_log_bins[i][preds[col].astype('int') + 1])
        
        for i, col in enumerate(StateAutoencoder.colnorm + StateAutoencoder.collog):
            if i < len(StateAutoencoder.colnorm):
                preds[col] = preds[col] * self.mean_std_list[i, 1] + self.mean_std_list[i, 0]
            else:
                preds[col] = np.exp(preds[col] * self.mean_std_list[i, 1] + self.mean_std_list[i, 0]) - 0.1
        return preds

        
        