import copy
import numpy

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from scipy import stats
from sklearn.cluster import KMeans
from icecream import ic

ai_instance = None


def get_instance(data_pth=None):
    global ai_instance
    if ai_instance is None:
        ai_instance = AI_Clinician(data_pth)
    return ai_instance


def get_policy_bins(MIMICtable):
    reformat5 = MIMICtable.to_numpy()
    iol = MIMICtable.columns.tolist().index('input_4hourly')
    vcl = MIMICtable.columns.tolist().index('max_dose_vaso')
    a = reformat5[:, iol]  # IV fluid
    a = rankdata(a[a > 0]) / len(a[a > 0])  # excludes zero fluid (will be action 1)

    iof = np.floor((a + 0.2499999999) * 4)  # converts iv volume in 4 actions
    a = reformat5[:, iol]
    a = a > 0  # location of non-zero fluid in big matrix
    io = np.ones(len(reformat5))  # array of ones, by default
    io[a] = iof + 1  # where more than zero fluid given: save actual action
    vc = np.copy(reformat5[:, vcl])
    vcr = rankdata(vc[vc != 0]) / len(vc[vc != 0])
    vcr = np.floor((vcr + 0.249999999999) * 4)  # converts to 4 bins
    vcr[vcr == 0] = 1
    vc[vc != 0] = vcr + 1
    vc[vc == 0] = 1
    ma1 = np.array(
        [np.median(reformat5[io == 1, iol]), np.median(reformat5[io == 2, iol]), np.median(reformat5[io == 3, iol]),
         np.median(reformat5[io == 4, iol]), np.median(reformat5[io == 5, iol])])
    # median dose of drug in all bins
    ma2 = np.array(
        [np.median(reformat5[vc == 1, vcl]), np.median(reformat5[vc == 2, vcl]), np.median(reformat5[vc == 3, vcl]),
         np.median(reformat5[vc == 4, vcl]), np.median(reformat5[vc == 5, vcl])])

    med = np.array([io, vc])
    uniqueValues, _, actionbloc = np.unique(med, axis=1, return_index=True, return_inverse=True)

    vc_bins = [[min(reformat5[vc == i, vcl]), np.median(reformat5[vc == i, vcl]), max(reformat5[vc == i, vcl])] for i in
               range(1, 6)]
    io_bins = [[min(reformat5[io == i, iol]), np.median(reformat5[io == i, iol]), max(reformat5[io == i, iol])] for i in
               range(1, 6)]

    return vc, io, vc_bins, io_bins


class AI_Clinician:
    # interested 47 columns
    colbin = ['gender', 'mechvent', 'max_dose_vaso', 're_admission']
    colnorm = ['age', 'Weight_kg', 'GCS', 'HR', 'SysBP', 'MeanBP', 'DiaBP', 'RR', 'Temp_C', 'FiO2_1',
               'Potassium', 'Sodium', 'Chloride', 'Glucose', 'Magnesium', 'Calcium',
               'Hb', 'WBC_count', 'Platelets_count', 'PTT', 'PT', 'Arterial_pH', 'paO2', 'paCO2',
               'Arterial_BE', 'HCO3', 'Arterial_lactate', 'SOFA', 'SIRS', 'Shock_Index', 'PaO2_FiO2',
               'cumulated_balance']
    collog = ['SpO2', 'BUN', 'Creatinine', 'SGOT', 'SGPT', 'Total_bili', 'INR', 'input_total', 'input_4hourly',
              'output_total', 'output_4hourly']

    nr_reps = 200  # nr of repetitions (total nr models)
    nclustering = 32  # how many times we do clustering (best solution will be chosen)
    prop = 0.25  # proportion of the data we sample for clustering
    gamma = 0.99  # gamma
    transthres = 5  # threshold for pruning the transition matrix
    polkeep = 0  # count of saved policies
    ncl = 750  # nr of states
    nra = 5  # nr of actions (2 to 10)
    ncv = 5  # nr of crossvalidation runs (each is 80# training / 20# test)

    def __init__(self, data_pth=None):
        if data_pth is None:
            data_pth = 'models/data/'
        print('Reading data ', end='')
        MIMICtable = pd.read_csv(data_pth + 'mimic_record.csv')

        # Data_mat = mat73.loadmat(data_pth + 'Data_160219.mat')

        with open(data_pth + 'MIMICzs.npy', 'rb') as f:
            MIMICzs = np.load(f, allow_pickle=True)
        with open(data_pth + 'recqvi.npy', 'rb') as f:
            recqvi = np.load(f, allow_pickle=True)
        with open(data_pth + 'allpols.npy', 'rb') as f:
            allpols = np.load(f, allow_pickle=True)

        print('[DONE]')

        # MIMICraw = Data_mat['MIMICraw']
        # MIMICzs = Data_mat['MIMICzs']
        # recqvi = Data_mat['recqvi']
        # idxs = Data_mat['idxs']
        # OA = Data_mat['OA']
        # allpols = np.array(Data_mat['allpols'], dtype=object)

        ## IDENTIFIES BEST MODEL HERE
        recqvi = np.delete(recqvi, np.s_[30:], 1)

        r = recqvi
        r = np.delete(r, np.s_[29:], 1)

        # SORT RECQVI BY COL 24 / DESC
        bestpol = r[max(r[:, 23]) == r[:, 23], 0]  # model maximising 95# LB of value of AI policy in MIMIC test set

        ## RECOVER BEST MODEL and TEST IT
        print('Recovering the best model ', end='')
        a = np.hstack(allpols[:, 0])
        outcome = 9  # HOSPITAL MORTALITY = 8 / 90d MORTA = 10
        ii = np.isin(a, bestpol)  # position of best model in the array allpols

        # RECOVER MODEL DATA
        # Qoff = allpols[ii, 1][0]
        Qon = allpols[ii, 2][0]
        physpol = allpols[ii, 3][0]
        softpol = allpols[ii, 4][0]
        transitionr = allpols[ii, 5][0]
        transitionr2 = allpols[ii, 6][0]
        R = allpols[ii, 7][0]
        C = allpols[ii, 8][0]
        train = allpols[ii, 9][0]
        test = ~train
        qldata3train = allpols[ii, 10][0]
        qldata3test = allpols[ii, 11][0]
        # qldata2 = allpols[ii, 12][0]

        kmeans = KMeans(n_clusters=AI_Clinician.ncl, random_state=0, n_init=AI_Clinician.nclustering, max_iter=30,
                        n_jobs=2).fit(C)
        idx = kmeans.predict(MIMICzs)  # N-D nearest point search: look for points closest to each centroid
        print('[DONE]')

        center_cluster_id = kmeans.predict(C)
        center_cluster_pos = np.zeros(AI_Clinician.ncl, dtype=np.int)
        for i in range(len(center_cluster_id)):
            center_cluster_pos[center_cluster_id[i]] = i

        self.vc, self.io, self.vc_bins, self.io_bins = get_policy_bins(MIMICtable)

        self.idx = idx
        self.Qon = Qon
        self.kmeans = kmeans
        self.kmeans_center = C[center_cluster_pos]
        self.train = train
        self.transitionr2 = transitionr2
        self.qldata3train = qldata3train
        self.qldata3test = qldata3test
        self.MIMICtable = MIMICtable
        self.MIMICzs = MIMICzs
        self.mean_std = (MIMICtable[AI_Clinician.colnorm].mean(), MIMICtable[AI_Clinician.colnorm].std(),
                         np.log(0.1 + MIMICtable[AI_Clinician.collog]).mean(),
                         np.log(0.1 + MIMICtable[AI_Clinician.collog]).std())
        self.state_statics_store = [None for _ in range(AI_Clinician.ncl)]

    def trans_zs(self, record):
        return [record[AI_Clinician.colbin] - 0.5, (record[AI_Clinician.colnorm] - self.mean_std[0]) / self.mean_std[1],
                (np.log(0.1 + record[AI_Clinician.collog]) - self.mean_std[2]) / self.mean_std[3]]

    def cluster_state(self, record):

        # if type(record) is not pd.core.frame.DataFrame:
        #     print('Record to be predict should be a pandas DataFrame.')
        #     return
        if np.isin(record.columns, AI_Clinician.colbin).sum() + \
                np.isin(record.columns, AI_Clinician.colnorm).sum() + \
                np.isin(record.columns, AI_Clinician.collog).sum() != \
                len(AI_Clinician.colbin) + \
                len(AI_Clinician.colnorm) + \
                len(AI_Clinician.collog):
            print('Wrong length of record.')
            return

        recordzs = np.hstack(self.trans_zs(record))

        recordzs[np.isnan(recordzs)] = 0
        recordzs[:, 2] = np.log(recordzs[:, 2] + 6)  # MAX DOSE NORAD
        recordzs[:, 44] = 2 * recordzs[:, 44]  # increase weight of this variable

        predict_idx = self.kmeans.predict(recordzs)

        return predict_idx

    def predict_action(self, state_idx, n_branch=3):
        Qs = np.argsort(self.Qon[state_idx, :])[::-1]
        Qs_possibility = np.sort(self.Qon[state_idx, :])[::-1]
        return Qs[:n_branch], Qs_possibility[:n_branch]

    def predict_state(self, action_idx, pre_state_idx, n_branch=2):
        States = np.argsort(self.transitionr2[pre_state_idx, :, action_idx])[::-1]
        States_possibility = np.sort(self.transitionr2[pre_state_idx, :, action_idx])[::-1]
        # ic(States)
        # ic(States_possibility)
        return States[:n_branch], States_possibility[:n_branch]

    def state_statics(self, state_idx):

        if self.state_statics_store[state_idx] is not None:
            return self.state_statics_store[state_idx]

        start_bloc = np.where(self.MIMICtable['bloc'] == 1)[0]

        same_state_record = np.where(self.idx == state_idx)[0]
        same_state_and_mortality_record = np.where((self.idx == state_idx) & (self.MIMICtable.to_numpy()[:, 9] == 1))[0]

        same_state_record_his = np.histogram(same_state_record, bins=start_bloc.tolist() + [np.inf])[0]
        same_state_and_mortality_record_his = \
        np.histogram(same_state_and_mortality_record, bins=start_bloc.tolist() + [np.inf])[0]

        statics = {'mortality': sum(same_state_and_mortality_record_his) / sum(same_state_record_his)}

        self.state_statics_store[state_idx] = statics

        # mortality = self.MIMICtable.to_numpy()[self.idx == state_idx, 9]
        return statics

    def predict_record(self, state_id):
        record_zs = self.kmeans_center[state_id]
        preds = pd.DataFrame([record_zs], columns=AI_Clinician.colbin + AI_Clinician.colnorm + AI_Clinician.collog)

        preds.loc[:, 'max_dose_vaso'] = np.exp(preds.loc[:, 'max_dose_vaso']) - 6  # MAX DOSE NORAD
        preds.loc[:, 'input_4hourly'] = preds.loc[:, 'input_4hourly'] / 2  # increase weight of input_4hourly

        # print(preds)

        for col in AI_Clinician.colbin:
            if col != 'max_dose_vaso':
                preds.loc[preds[col] > 0, col] = 1
                preds.loc[preds[col] <= 0, col] = 0
            else:
                preds[col] += 0.5

        for i, col in enumerate(AI_Clinician.colnorm + AI_Clinician.collog):
            if i < len(AI_Clinician.colnorm):
                preds[col] = preds[col] * self.mean_std[1][i] + self.mean_std[0][i]
            else:
                preds[col] = np.exp(preds[col] * self.mean_std[3][i - len(AI_Clinician.colnorm)] + self.mean_std[2][
                    i - len(AI_Clinician.colnorm)]) - 0.1
        return preds


if __name__ == '__main__':
    data_pth = 'data/'
    ai_clinician = AI_Clinician(data_pth)
    print(ai_clinician.kmeans_center)
    print(ai_clinician.kmeans.predict(ai_clinician.kmeans_center))
