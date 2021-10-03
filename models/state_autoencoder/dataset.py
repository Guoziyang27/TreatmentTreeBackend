import torch.utils.data as data
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from scipy import stats
from sklearn.cluster import KMeans

class Dataset(data.Dataset):
    def __init__(self, forward_num=4):
        MIMICtable = pd.read_csv('../data/final_table.csv')

        colbin = ['gender','mechvent','max_dose_vaso','re_admission']
        colnorm=['age','Weight_kg','GCS','HR','SysBP','MeanBP','DiaBP','RR','Temp_C','FiO2_1',
            'Potassium','Sodium','Chloride','Glucose','Magnesium','Calcium',
            'Hb','WBC_count','Platelets_count','PTT','PT','Arterial_pH','paO2','paCO2',
            'Arterial_BE','HCO3','Arterial_lactate','SOFA','SIRS','Shock_Index','PaO2_FiO2','cumulated_balance']
        collog=['SpO2','BUN','Creatinine','SGOT','SGPT','Total_bili','INR','input_total','input_4hourly','output_total','output_4hourly']

        colbin_id = [MIMICtable.columns.tolist().index(name) for name in colbin]
        colnorm_id = [MIMICtable.columns.tolist().index(name) for name in colnorm]
        collog_id = [MIMICtable.columns.tolist().index(name) for name in collog]
        nra = 5

        MIMICraw = MIMICtable.loc[:, colbin + colnorm + collog]

        self.mean_std_list = np.empty([len(colnorm) + len(collog), 2])
        bins_top = 0

        for col in colbin:
            MIMICraw[col] = MIMICraw[col] - 0.5

        for col in colnorm:
            self.mean_std_list[bins_top] = (MIMICraw[col].mean(), MIMICraw[col].std())

            MIMICraw[col] = (MIMICraw[col] - self.mean_std_list[bins_top, 0]) / self.mean_std_list[bins_top, 1]

            bins_top += 1
        
        for col in collog:
            self.mean_std_list[bins_top] = (np.log(0.1+MIMICraw[col]).mean(), np.log(0.1+MIMICraw[col]).std())
            MIMICraw[col] = (np.log(0.1 + MIMICraw[col]) - self.mean_std_list[bins_top, 0]) / self.mean_std_list[bins_top, 1]
            bins_top += 1
        MIMICraw = MIMICraw.to_numpy().astype('float64')

        with open('mean_std_list.npy', 'wb') as f:
            np.save(f, self.mean_std_list)

        MIMICraw[np.isnan(MIMICraw)] = 0
        # MIMICraw = MIMICraw.astype('int')
        reformat5 = MIMICtable.to_numpy()

        iol = MIMICtable.columns.tolist().index('input_4hourly')
        vcl = MIMICtable.columns.tolist().index('max_dose_vaso')

        nact = nra ** 2
        a = reformat5[:, iol]                   #IV fluid
        a = rankdata(a[a>0]) / len(a[a>0])   # excludes zero fluid (will be action 1)
        
        iof = np.floor((a+0.2499999999)*4)  #converts iv volume in 4 actions
        a = reformat5[:, iol]
        a = a>0  #location of non-zero fluid in big matrix
        io = np.ones(len(reformat5))  #array of ones, by default     
        io[a] = iof + 1   #where more than zero fluid given: save actual action
        vc = np.copy(reformat5[:, vcl])
        vcr = rankdata(vc[vc != 0]) / len(vc[vc != 0])
        vcr = np.floor((vcr+0.249999999999)*4)  #converts to 4 bins
        vcr[vcr == 0] = 1
        vc[vc != 0] = vcr + 1
        vc[vc == 0] = 1
        ma1 = np.array([ np.median(reformat5[io==1,iol]),  np.median(reformat5[io==2,iol]),  np.median(reformat5[io==3,iol]),  np.median(reformat5[io==4,iol]),  np.median(reformat5[io==5,iol])])
        #median dose of drug in all bins
        ma2 = np.array([ np.median(reformat5[vc==1,vcl]),  np.median(reformat5[vc==2,vcl]),  np.median(reformat5[vc==3,vcl]),  np.median(reformat5[vc==4,vcl]),  np.median(reformat5[vc==5,vcl])])
        
        med = np.array([io, vc])
        uniqueValues, _, actionbloc = np.unique(med, axis=1, return_index=True, return_inverse=True)


        outcome = actionbloc

        start_bloc = np.where(reformat5[:, 0] == 1)[0]

        items = []

        for i in range(len(start_bloc)):
            if i < len(start_bloc) - 1:
                # print(start_bloc[i], start_bloc[i + 1] - forward_num - 1)
                for j in range(int(start_bloc[i] + 1), int(start_bloc[i + 1])):
                    items.append((j, min(forward_num, j - start_bloc[i])))
            else:
                for j in range(int(start_bloc[i] + 1), int(len(reformat5))):
                    items.append((j, min(forward_num, j - start_bloc[i])))

        self.outcome = np.array(outcome)
        self.state_data = np.array(MIMICraw)
        self.items = items
        self.forward_num = forward_num

        pass

    def __getitem__(self, idx):
        item_id, forward_num = self.items[idx]
        prev_rec = np.array([np.concatenate([self.state_data[item_id - i - 1], [self.outcome[item_id - i - 1]]]) for i in range(forward_num)])
        if forward_num < self.forward_num:
            prev_rec = np.concatenate([np.full([self.forward_num - forward_num, prev_rec.shape[1]], -1), prev_rec])
        end_rec = np.array([self.state_data[item_id]])
        return prev_rec, end_rec

    def __len__(self):
        return len(self.items)
