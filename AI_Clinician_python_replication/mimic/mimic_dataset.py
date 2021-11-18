import pandas as pd
import numpy as np
from datetime import datetime
from scipy.spatial.distance import cdist
import scipy.io as sio
from sklearn.impute import KNNImputer
import sys
from tqdm import tqdm
import os

sys.path.append("..")
from mimic.sample_name import sample
from mimic.utils import deloutabove, deloutbelow, SAH, fixgaps


def mimicDataset():
    data_dir = 'extracted_data/'
    output_dir = '../models/data/'

    if os.path.isdir(output_dir):
        os.mkdir(output_dir)

    if os.path.isfile(output_dir + 'mimic_record.csv'):
        reformat4t = pd.read_csv(output_dir + 'mimic_record.csv')
        return reformat4t

    abx = pd.read_csv(data_dir + 'abx.csv', header=None)
    abx = abx.rename(columns=lambda c: c + 1)
    culture = pd.read_csv(data_dir + 'culture.csv', header=None)
    culture = culture.rename(columns=lambda c: c + 1)
    microbio = pd.read_csv(data_dir + 'microbio.csv', header=None)
    microbio = microbio.rename(columns=lambda c: c + 1)
    demog = pd.read_csv(data_dir + 'demog.csv', header=None)
    demog = demog.rename(columns=lambda c: c + 1)
    ce010 = pd.read_csv(data_dir + 'ce01000000.csv', header=None)
    ce010 = ce010.rename(columns=lambda c: c + 1)
    ce1020 = pd.read_csv(data_dir + 'ce10000002000000.csv', header=None)
    ce1020 = ce1020.rename(columns=lambda c: c + 1)
    ce2030 = pd.read_csv(data_dir + 'ce20000003000000.csv', header=None)
    ce2030 = ce2030.rename(columns=lambda c: c + 1)
    ce3040 = pd.read_csv(data_dir + 'ce30000004000000.csv', header=None)
    ce3040 = ce3040.rename(columns=lambda c: c + 1)
    ce4050 = pd.read_csv(data_dir + 'ce40000005000000.csv', header=None)
    ce4050 = ce4050.rename(columns=lambda c: c + 1)
    ce5060 = pd.read_csv(data_dir + 'ce50000006000000.csv', header=None)
    ce5060 = ce5060.rename(columns=lambda c: c + 1)
    ce6070 = pd.read_csv(data_dir + 'ce60000007000000.csv', header=None)
    ce6070 = ce6070.rename(columns=lambda c: c + 1)
    ce7080 = pd.read_csv(data_dir + 'ce70000008000000.csv', header=None)
    ce7080 = ce7080.rename(columns=lambda c: c + 1)
    ce8090 = pd.read_csv(data_dir + 'ce80000009000000.csv', header=None)
    ce8090 = ce8090.rename(columns=lambda c: c + 1)
    ce90100 = pd.read_csv(data_dir + 'ce900000010000000.csv', header=None)
    ce90100 = ce90100.rename(columns=lambda c: c + 1)
    labU = pd.concat(
        [pd.read_csv(data_dir + 'labs_ce.csv', header=None), pd.read_csv(data_dir + 'labs_le.csv', header=None)],
        ignore_index=True)
    labU = labU.rename(columns=lambda c: c + 1)
    MV = pd.read_csv(data_dir + 'mechvent.csv', header=None)
    MV = MV.rename(columns=lambda c: c + 1)
    inputpreadm = pd.read_csv(data_dir + 'preadm_fluid.csv', header=None)
    inputpreadm = inputpreadm.rename(columns=lambda c: c + 1)
    inputMV = pd.read_csv(data_dir + 'fluid_mv.csv', header=None)
    inputMV = inputMV.rename(columns=lambda c: c + 1)
    vasoMV = pd.read_csv(data_dir + 'vaso_mv.csv', header=None)
    vasoMV = vasoMV.rename(columns=lambda c: c + 1)
    UOpreadm = pd.read_csv(data_dir + 'preadm_uo.csv', header=None)
    UOpreadm = UOpreadm.rename(columns=lambda c: c + 1)
    UO = pd.read_csv(data_dir + 'uo.csv', header=None)
    UO = UO.rename(columns=lambda c: c + 1)

    def datetimeToEpoch(dateString):
        return dateString
        striptimeFormatString = '%Y-%m-%d %H:%M:%S'
        return datetime.strptime(dateString, striptimeFormatString).timestamp()

    def dateToEpoch(dateString):
        return dateString
        striptimeFormatString = '%Y-%m-%d'
        return datetime.strptime(dateString, striptimeFormatString).timestamp()

    # ii = microbio[3].isnull()
    # microbio.loc[ii, 3] = microbio[ii][4]
    # microbio.loc[:, 4] = microbio[:][3]
    # microbio.loc[:, 3] = 0
    # microbio.loc[:, 5] = 0

    bacterio = pd.concat([microbio, culture], ignore_index=True)

    demog.loc[np.isnan(demog[16]), 16] = 0
    demog.loc[np.isnan(demog[17]), 17] = 0
    demog.loc[np.isnan(demog[18]), 18] = 0

    inputMV.loc[:, 8] = inputMV[:][7] * inputMV[:][6] / inputMV[:][5]

    for i in tqdm(range(len(bacterio)), desc='bacterio'):
        if bacterio.loc[i, 3] == 0:
            o = (bacterio.loc[i, 4])
            subjectid = bacterio.loc[i, 1]
            hadmid = bacterio.loc[i, 2]
            ii = demog.index[demog[1] == subjectid].tolist()
            # jj = demog.index[(demog[1] == subjectid) & (demog[2] == hadmid)].tolist()
            for j in range(len(ii)):
                if (o >= (demog.loc[ii[j], 8]) - 48 * 3600 and o <= (demog.loc[ii[j], 9]) + 48 * 3600) or len(ii) == 1:
                    bacterio.loc[i, 3] = demog.loc[ii[j], 3]
            # if bacterio.loc[i, 3] == 0 and len(jj) == 1:
            #     bacterio.loc[i, 3] = demog.loc[jj[0], 3]

    dropi = []
    for i in tqdm(range(len(abx)), desc='abx'):
        if np.isnan(abx.loc[i, 2]) and not np.isnan(abx.loc[i, 3]):
            o = (abx.loc[i, 3])
            hadmid = abx.loc[i, 1]
            ii = demog.index[demog[2] == hadmid].tolist()
            for j in range(len(ii)):
                if (o >= (demog.loc[ii[j], 8]) - 48 * 3600 and o <= (demog.loc[ii[j], 9]) + 48 * 3600) or len(ii) == 1:
                    abx.loc[i, 2] = demog.loc[ii[j], 3]
        if np.isnan(abx.loc[i, 3]):
            dropi.append(i)

    abx = abx.drop(dropi)

    icustayids = np.unique(abx[2])  # [x for x in set(abx[:, 1].tolist()) if not np.isnan(x)] #[0:38000]
    icustayids = icustayids[~np.isnan(icustayids)]
    # icustayids = [x for x in set(abx[2].tolist()) if not np.isnan(x)][0:300]
    print(icustayids)

    onset = pd.DataFrame(np.zeros([len(icustayids), 3]), index=icustayids, columns=range(1, 4))

    for icustayid in tqdm(icustayids, desc='onset'):

        ab = abx.loc[abx[2] == icustayid, 3].reset_index(drop=True)
        bact = bacterio.loc[bacterio[3] == icustayid, 4].reset_index(drop=True)
        subj_bact = bacterio.loc[bacterio[3] == icustayid, 1].reset_index(drop=True)

        if not ab.empty and not bact.empty:

            D = cdist(np.array(ab).reshape(-1, 1), np.array(bact).reshape(-1, 1)) / 3600

            for i in range(len(D)):
                M, I = D[i].min(), D[i].argmin()
                ab1 = ab.loc[i]
                bact1 = bact.loc[I]

                # print(ab1, bact1)

                if M <= 24 and ab1 <= bact1:
                    onset.loc[icustayid, 1] = subj_bact[0]
                    onset.loc[icustayid, 2] = icustayid
                    onset.loc[icustayid, 3] = ab1
                    break
                elif M <= 72 and ab1 >= bact1:
                    onset.loc[icustayid, 1] = subj_bact[0]
                    onset.loc[icustayid, 2] = icustayid
                    onset.loc[icustayid, 3] = bact1
                    break

    print(sum(onset.loc[onset[3] > 0, 3]))
    print(onset)

    referenceMatrices = sio.loadmat('./reference_matrices_add_mimiciv.mat')

    # referenceMatrices = sio.loadmat('./reference_matrices.mat')

    def ismember(A, B):
        return np.array([int(B in a) for a in A])

    # for i in range(len(labU)):
    #     locb = ismember(referenceMatrices['Reflabs'], labU.loc[i, 3])
    #     labU.loc[i, 3] = locb.argmax() + 1

    # for i in range(len(ce010)):
    #     locb = ismember(referenceMatrices['Refvitals'], ce010.loc[i, 3])
    #     ce010.loc[i, 3] = locb.argmax() + 1
    # for i in range(len(ce1020)):
    #     locb = ismember(referenceMatrices['Refvitals'], ce1020.loc[i, 3])
    #     ce1020.loc[i, 3] = locb.argmax() + 1
    # for i in range(len(ce2030)):
    #     locb = ismember(referenceMatrices['Refvitals'], ce2030.loc[i, 3])
    #     ce2030.loc[i, 3] = locb.argmax() + 1
    # for i in range(len(ce3040)):
    #     locb = ismember(referenceMatrices['Refvitals'], ce3040.loc[i, 3])
    #     ce3040.loc[i, 3] = locb.argmax() + 1
    # for i in range(len(ce4050)):
    #     locb = ismember(referenceMatrices['Refvitals'], ce4050.loc[i, 3])
    #     ce4050.loc[i, 3] = locb.argmax() + 1
    # for i in range(len(ce5060)):
    #     locb = ismember(referenceMatrices['Refvitals'], ce5060.loc[i, 3])
    #     ce5060.loc[i, 3] = locb.argmax() + 1
    # for i in range(len(ce6070)):
    #     locb = ismember(referenceMatrices['Refvitals'], ce6070.loc[i, 3])
    #     ce6070.loc[i, 3] = locb.argmax() + 1
    # for i in range(len(ce7080)):
    #     locb = ismember(referenceMatrices['Refvitals'], ce7080.loc[i, 3])
    #     ce7080.loc[i, 3] = locb.argmax() + 1
    # for i in range(len(ce8090)):
    #     locb = ismember(referenceMatrices['Refvitals'], ce8090.loc[i, 3])
    #     ce8090.loc[i, 3] = locb.argmax() + 1
    # for i in range(len(ce90100)):
    #     locb = ismember(referenceMatrices['Refvitals'], ce90100.loc[i, 3])
    #     ce90100.loc[i, 3] = locb.argmax() + 1

    reformat = np.empty([4000000, 68])  # final table
    reformat[:] = np.NaN
    reformat = pd.DataFrame(reformat, columns=range(1, 69))  # final table

    qstime = pd.DataFrame(np.zeros([len(icustayids), 4]), index=icustayids, columns=range(1, 5))
    winb4 = 49  # lower limit for inclusion of data (48h before time flag)
    winaft = 25  # upper limit (24h after)
    irow = 0  # recording row for summary table

    for i, icustayid in tqdm(enumerate(icustayids), desc='initialization of reformat'):
        # print(onset.loc[icustayid, :])
        # print(i,  len(icustayids))
        qst = onset.loc[icustayid, 3]  # flag for presumed infection
        if qst > 0:  # if we have a flag
            d1 = demog.loc[demog[3] == icustayid, [11, 5]].values.tolist()  # age of patient + discharge time

            if d1[0][0] > 18:  # if older than 18 years old

                # CHARTEVENTS
                if icustayid < 31000000:
                    temp = ce010.loc[ce010[1] == icustayid, :]
                elif icustayid >= 31000000 and icustayid < 32000000:
                    temp = ce1020.loc[ce1020[1] == icustayid, :]
                elif icustayid >= 32000000 and icustayid < 33000000:
                    temp = ce2030.loc[ce2030[1] == icustayid, :]
                elif icustayid >= 33000000 and icustayid < 34000000:
                    temp = ce3040.loc[ce3040[1] == icustayid, :]
                elif icustayid >= 34000000 and icustayid < 35000000:
                    temp = ce4050.loc[ce4050[1] == icustayid, :]
                elif icustayid >= 35000000 and icustayid < 36000000:
                    temp = ce5060.loc[ce5060[1] == icustayid, :]
                elif icustayid >= 36000000 and icustayid < 37000000:
                    temp = ce6070.loc[ce6070[1] == icustayid, :]
                elif icustayid >= 37000000 and icustayid < 38000000:
                    temp = ce7080.loc[ce7080[1] == icustayid, :]
                elif icustayid >= 38000000 and icustayid < 39000000:
                    temp = ce8090.loc[ce8090[1] == icustayid, :]
                elif icustayid >= 39000000:
                    temp = ce90100.loc[ce90100[1] == icustayid, :]

                ii = temp.index[(temp[2] >= qst - (winb4 + 4) * 3600) & (
                            temp[2] <= qst + (winaft + 4) * 3600)]  # time period of interest -4h and +4h
                temp = temp.loc[ii, :]  # only time period of interest

                # LABEVENTS
                ii = labU.index[labU[1] == icustayid]
                temp2 = labU.loc[ii, :]
                ii = temp2.index[(temp2[2] >= qst - (winb4 + 4) * 3600) & (
                            temp2[2] <= qst + (winaft + 4) * 3600)]  # time period of interest -4h and +4h
                temp2 = temp2.loc[ii, :]  # only time period of interest

                # Mech Vent + ?extubated
                ii = MV[1] == icustayid
                temp3 = MV.loc[ii, :]
                ii = (temp3[2] >= qst - (winb4 + 4) * 3600) & (
                            temp3[2] <= qst + (winaft + 4) * 3600)  # time period of interest -4h and +4h
                temp3 = temp3.loc[ii, :]  # only time period of interest

                t = pd.unique(pd.concat([temp[2], temp2[2], temp3[2]],
                                        ignore_index=True))  # list of unique timestamps from all 3 sources / sorted in ascending order

                if len(t) > 0:
                    for i in range(len(t)):

                        reformat.loc[irow, 1] = i  # timestep
                        reformat.loc[irow, 2] = icustayid
                        reformat.loc[irow, 3] = (t[i])  # charttime
                        # CHARTEVENTS
                        ii = temp[2] == t[i]
                        if not temp.loc[ii, 3].empty:
                            col = temp.loc[ii, 3].values.tolist()[0]
                            value = temp.loc[ii, 4].values.tolist()[0]
                            reformat.loc[irow, 3 + col] = value  # (locb(:,1)) #store available values

                        # LAB VALUES
                        ii = temp2[2] == t[i]
                        if not temp2.loc[ii, 3].empty:
                            col = temp2.loc[ii, 3].values.tolist()[0]
                            value = temp2.loc[ii, 4].values.tolist()[0]
                            reformat.loc[irow, 31 + col] = value  # store available values

                        # MV
                        ii = temp3[2] == t[i]
                        if np.sum(ii) > 0:
                            value = temp3.loc[ii, 3:4].values.tolist()[0]
                            reformat.loc[irow, 67:68] = value  # store available values
                        else:
                            reformat.loc[irow, 67:68] = np.NaN

                        irow += 1

                    qstime.loc[
                        icustayid, 1] = qst  # flag for presumed infection / this is time of sepsis if SOFA >=2 for this patient
                    # HERE I SAVE FIRST and LAST TIMESTAMPS, in QSTIME, for each ICUSTAYID
                    qstime.loc[icustayid, 2] = (t[0])  # first timestamp
                    qstime.loc[icustayid, 3] = (t[len(t) - 1])  # last timestamp
                    qstime.loc[icustayid, 4] = (d1[0][1])  # dischargetime

    reformat = reformat.drop(range(irow, len(reformat))).reset_index(drop=True)

    reformat.to_csv('init_reformat.csv', index=False, sep=',')
    qstime.to_csv('init_qstime.csv', sep=',')

    # reformat = pd.read_csv('init_reformat.csv')
    # reformat.columns = range(1,69)
    # qstime = pd.read_csv('init_qstime.csv', index_col='Unnamed: 0')
    # qstime.columns = range(1,5)

    print('filtering age ', end='')
    reformat = deloutabove(reformat, 5, 300)
    print('[DONE]')

    # HR
    print('filtering HR ', end='')
    reformat = deloutabove(reformat, 8, 250)
    print('[DONE]')

    # BP
    print('filtering BP ', end='')
    reformat = deloutabove(reformat, 9, 300)
    reformat = deloutbelow(reformat, 10, 0)
    reformat = deloutabove(reformat, 10, 200)
    reformat = deloutbelow(reformat, 11, 0)
    reformat = deloutabove(reformat, 11, 200)
    print('[DONE]')

    # RR
    print('filtering RR ', end='')
    reformat = deloutabove(reformat, 12, 80)
    print('[DONE]')

    # SpO2
    print('filtering SpO2 ', end='')
    reformat = deloutabove(reformat, 13, 150)
    ii = reformat.index[reformat[13] > 100]
    reformat.loc[ii, 13] = 100
    print('[DONE]')

    # temp
    print('filtering temp ', end='')
    ii = (reformat[14] > 90) & np.isnan(reformat[15])
    reformat.loc[ii, 15] = reformat.loc[ii, 14]
    reformat = deloutabove(reformat, 14, 90)
    print('[DONE]')

    # interface / is in col 22

    # FiO2
    print('filtering FiO2 ', end='')
    reformat = deloutabove(reformat, 23, 100)
    ii = reformat[23] < 1
    reformat.loc[ii, 23] = reformat.loc[ii, 23] * 100
    reformat = deloutbelow(reformat, 23, 20)
    reformat = deloutabove(reformat, 24, 1.5)
    print('[DONE]')

    # O2 FLOW
    print('filtering O2 FLOW ', end='')
    reformat = deloutabove(reformat, 25, 70)
    print('[DONE]')

    # PEEP
    print('filtering PEEP ', end='')
    reformat = deloutbelow(reformat, 26, 0)
    reformat = deloutabove(reformat, 26, 40)
    print('[DONE]')

    # TV
    print('filtering TV ', end='')
    reformat = deloutabove(reformat, 27, 1800)
    print('[DONE]')

    # MV
    print('filtering MV ', end='')
    reformat = deloutabove(reformat, 28, 50)
    print('[DONE]')

    # K+
    print('filtering K+ ', end='')
    reformat = deloutbelow(reformat, 32, 1)
    reformat = deloutabove(reformat, 32, 15)
    print('[DONE]')

    # Na
    print('filtering Na ', end='')
    reformat = deloutbelow(reformat, 33, 95)
    reformat = deloutabove(reformat, 33, 178)
    print('[DONE]')

    # Cl
    print('filtering Cl ', end='')
    reformat = deloutbelow(reformat, 34, 70)
    reformat = deloutabove(reformat, 34, 150)
    print('[DONE]')

    # Glc
    print('filtering Glc ', end='')
    reformat = deloutbelow(reformat, 35, 1)
    reformat = deloutabove(reformat, 35, 1000)
    print('[DONE]')

    # Creat
    print('filtering Creat ', end='')
    reformat = deloutabove(reformat, 37, 150)
    print('[DONE]')

    # Mg
    print('filtering Mg ', end='')
    reformat = deloutabove(reformat, 38, 10)
    print('[DONE]')

    # Ca
    print('filtering Ca ', end='')
    reformat = deloutabove(reformat, 39, 20)
    print('[DONE]')

    # ionized Ca
    print('filtering ionized Ca ', end='')
    reformat = deloutabove(reformat, 40, 5)
    print('[DONE]')

    # CO2
    print('filtering CO2 ', end='')
    reformat = deloutabove(reformat, 41, 120)
    print('[DONE]')

    # SGPT/SGOT
    print('filtering SGPT/SGOT ', end='')
    reformat = deloutabove(reformat, 42, 10000)
    reformat = deloutabove(reformat, 43, 10000)
    print('[DONE]')

    # Hb/Ht
    print('filtering Hb/Ht ', end='')
    reformat = deloutabove(reformat, 50, 20)
    reformat = deloutabove(reformat, 51, 65)
    print('[DONE]')

    # WBC
    print('filtering WBC ', end='')
    reformat = deloutabove(reformat, 53, 500)
    print('[DONE]')

    # plt
    print('filtering plt ', end='')
    reformat = deloutabove(reformat, 54, 2000)
    print('[DONE]')

    # INR
    print('filtering INR ', end='')
    reformat = deloutabove(reformat, 58, 20)
    print('[DONE]')

    # pH
    print('filtering pH ', end='')
    reformat = deloutbelow(reformat, 59, 6.7)
    reformat = deloutabove(reformat, 59, 8)
    print('[DONE]')

    # po2
    print('filtering po2 ', end='')
    reformat = deloutabove(reformat, 60, 700)
    print('[DONE]')

    # pco2
    print('filtering pco2 ', end='')
    reformat = deloutabove(reformat, 61, 200)
    print('[DONE]')

    # BE
    print('filtering BE ', end='')
    reformat = deloutbelow(reformat, 62, -50)
    print('[DONE]')

    # lactate
    print('filtering lactate ', end='')
    reformat = deloutabove(reformat, 63, 30)
    print('[DONE]')

    # ####################################################################
    # some more data manip / imputation from existing values

    # estimate GCS from RASS - data from Wesley JAMA 2003
    print('estimate GCS from RASS - data from Wesley JAMA 2003 ', end='')
    ii = np.isnan(reformat[6]) & (reformat[7] >= 0)
    reformat.loc[ii, 6] = 15
    ii = np.isnan(reformat[6]) & (reformat[7] == -1)
    reformat.loc[ii, 6] = 14
    ii = np.isnan(reformat[6]) & (reformat[7] == -2)
    reformat.loc[ii, 6] = 12
    ii = np.isnan(reformat[6]) & (reformat[7] == -3)
    reformat.loc[ii, 6] = 11
    ii = np.isnan(reformat[6]) & (reformat[7] == -4)
    reformat.loc[ii, 6] = 6
    ii = np.isnan(reformat[6]) & (reformat[7] == -5)
    reformat.loc[ii, 6] = 3
    print('[DONE]')

    # FiO2
    print('estimate FiO2 ', end='')
    ii = ~np.isnan(reformat[23]) & np.isnan(reformat[24])
    reformat.loc[ii, 24] = reformat.loc[ii, 23] / 100
    ii = ~np.isnan(reformat[24]) & np.isnan(reformat[23])
    reformat.loc[ii, 23] = reformat.loc[ii, 24] * 100
    print('[DONE]')

    reformatsah = SAH(reformat, referenceMatrices["sample_and_hold"])  # do SAH first to handle this task

    # NO FiO2, YES O2 flow, no interface OR cannula
    print('NO FiO2, YES O2 flow, no interface OR cannula ', end='')
    ii = np.where(np.isnan(reformatsah.loc[:, 23]) & ~np.isnan(reformatsah.loc[:, 25]) & (
                (reformatsah.loc[:, 22] == 0) | (reformatsah.loc[:, 22] == 2)))[0]
    reformat.loc[ii[reformatsah.loc[ii, 25] <= 15], 23] = 70
    reformat.loc[ii[reformatsah.loc[ii, 25] <= 12], 23] = 62
    reformat.loc[ii[reformatsah.loc[ii, 25] <= 10], 23] = 55
    reformat.loc[ii[reformatsah.loc[ii, 25] <= 8], 23] = 50
    reformat.loc[ii[reformatsah.loc[ii, 25] <= 6], 23] = 44
    reformat.loc[ii[reformatsah.loc[ii, 25] <= 5], 23] = 40
    reformat.loc[ii[reformatsah.loc[ii, 25] <= 4], 23] = 36
    print('1')
    reformat.loc[ii[reformatsah.loc[ii, 25] <= 3], 23] = 32
    print('1')
    reformat.loc[ii[reformatsah.loc[ii, 25] <= 2], 23] = 28
    print('1')
    reformat.loc[ii[reformatsah.loc[ii, 25] <= 1], 23] = 24
    print('[DONE]')

    # NO FiO2, NO O2 flow, no interface OR cannula
    print('NO FiO2, NO O2 flow, no interface OR cannula ', end='')
    ii = np.isnan(reformatsah.loc[:, 23]) & np.isnan(reformatsah.loc[:, 25]) & ((reformatsah.loc[:, 22] == 0) | (
                reformatsah.loc[:, 22] == 2))  # no fio2 given and o2flow given, no interface OR cannula
    reformat.loc[ii, 23] = 21
    print('[DONE]')

    # NO FiO2, YES O2 flow, face mask OR.... OR ventilator (assume it's face mask)
    print('NO FiO2, YES O2 flow, face mask OR.... OR ventilator (assume it\'s face mask) ', end='')
    ii = np.isnan(reformatsah.loc[:, 23]) & ~np.isnan(reformatsah.loc[:, 25]) & (
                np.isnan(reformatsah.loc[:, 22]) | (reformatsah.loc[:, 22] == 1) | (reformatsah.loc[:, 22] == 3) | (
                    reformatsah.loc[:, 22] == 4) | (reformatsah.loc[:, 22] == 5) | (reformatsah.loc[:, 22] == 6) | (
                            reformatsah.loc[:, 22] == 9) | (reformatsah.loc[:, 22] == 10))
    reformat.loc[ii & (reformatsah.loc[ii, 25] <= 15), 23] = 75
    reformat.loc[ii & (reformatsah.loc[ii, 25] <= 12), 23] = 69
    reformat.loc[ii & (reformatsah.loc[ii, 25] <= 10), 23] = 66
    reformat.loc[ii & (reformatsah.loc[ii, 25] <= 8), 23] = 58
    reformat.loc[ii & (reformatsah.loc[ii, 25] <= 6), 23] = 40
    reformat.loc[ii & (reformatsah.loc[ii, 25] <= 4), 23] = 36
    print('[DONE]')

    # NO FiO2, NO O2 flow, face mask OR ....OR ventilator
    print('NO FiO2, NO O2 flow, face mask OR ....OR ventilator ', end='')
    ii = np.isnan(reformatsah.loc[:, 23]) & np.isnan(reformatsah.loc[:, 25]) & (
                np.isnan(reformatsah.loc[:, 22]) | (reformatsah.loc[:, 22] == 1) | (reformatsah.loc[:, 22] == 3) | (
                    reformatsah.loc[:, 22] == 4) | (reformatsah.loc[:, 22] == 5) | (reformatsah.loc[:, 22] == 6) | (
                            reformatsah.loc[:, 22] == 9) | (
                            reformatsah.loc[:, 22] == 10))  # no fio2 given and o2flow given, no interface OR cann
    reformat.loc[ii, 23] = np.NaN
    print('[DONE]')

    # NO FiO2, YES O2 flow, Non rebreather mask
    print('NO FiO2, YES O2 flow, Non rebreather mask ', end='')
    ii = np.isnan(reformatsah.loc[:, 23]) & ~np.isnan(reformatsah.loc[:, 25]) & (reformatsah.loc[:, 22] == 7)
    reformat.loc[ii & (reformatsah.loc[ii, 25] >= 10), 23] = 90
    reformat.loc[ii & (reformatsah.loc[ii, 25] >= 15), 23] = 100
    reformat.loc[ii & (reformatsah.loc[ii, 25] < 10), 23] = 80
    reformat.loc[ii & (reformatsah.loc[ii, 25] <= 8), 23] = 70
    reformat.loc[ii & (reformatsah.loc[ii, 25] <= 6), 23] = 60
    print('[DONE]')

    # NO FiO2, NO O2 flow, NRM
    print('NO FiO2, NO O2 flow, NRM ', end='')
    ii = np.isnan(reformatsah.loc[:, 23]) & np.isnan(reformatsah.loc[:, 25]) & (
                reformatsah.loc[:, 22] == 7)  # no fio2 given and o2flow given, no interface OR cann
    reformat.loc[ii, 23] = np.NaN
    print('[DONE]')

    # update again FiO2 columns
    print(' update again FiO2 columns ', end='')
    ii = ~np.isnan(reformat.loc[:, 23]) & np.isnan(reformat.loc[:, 24])
    reformat.loc[ii, 24] = reformat.loc[ii, 23] / 100
    ii = ~np.isnan(reformat.loc[:, 24]) & np.isnan(reformat.loc[:, 23])
    reformat.loc[ii, 23] = reformat.loc[ii, 24] * 100
    print('[DONE]')

    # BP
    print('BP ', end='')
    ii = ~np.isnan(reformat.loc[:, 9]) & ~np.isnan(reformat.loc[:, 10]) & np.isnan(reformat.loc[:, 11])
    reformat.loc[ii, 11] = (3 * reformat.loc[ii, 10] - reformat.loc[ii, 9]) / 2
    ii = ~np.isnan(reformat.loc[:, 9]) & ~np.isnan(reformat.loc[:, 11]) & np.isnan(reformat.loc[:, 10])
    reformat.loc[ii, 10] = (reformat.loc[ii, 9] + 2 * reformat.loc[ii, 11]) / 3
    ii = ~np.isnan(reformat.loc[:, 10]) & ~np.isnan(reformat.loc[:, 11]) & np.isnan(reformat.loc[:, 9])
    reformat.loc[ii, 9] = 3 * reformat.loc[ii, 10] - 2 * reformat.loc[ii, 11];
    print('[DONE]')

    # TEMP
    # some values recorded in the wrong column
    print('TEMP ', end='')
    ii = (reformat.loc[:, 15] > 25) & (reformat.loc[:, 15] < 45)  # tempF close to 37deg??!
    reformat.loc[ii, 14] = reformat.loc[ii, 15]
    reformat.loc[ii, 15] = np.NaN
    ii = (reformat.loc[:, 14] > 70)  # tempC > 70?!!! probably degF
    reformat.loc[ii, 15] = reformat.loc[ii, 14]
    reformat.loc[ii, 14] = np.NaN
    ii = ~np.isnan(reformat.loc[:, 14]) & np.isnan(reformat.loc[:, 15]);
    reformat.loc[ii, 15] = reformat.loc[ii, 14] * 1.8 + 32;
    ii = ~np.isnan(reformat.loc[:, 15]) & np.isnan(reformat.loc[:, 14]);
    reformat.loc[ii, 14] = (reformat.loc[ii, 15] - 32) / 1.8;
    print('[DONE]')

    # Hb/Ht
    print(' Hb/Ht ', end='')
    ii = ~np.isnan(reformat.loc[:, 50]) & np.isnan(reformat.loc[:, 51])
    reformat.loc[ii, 51] = (reformat.loc[ii, 50] * 2.862) + 1.216
    ii = ~np.isnan(reformat.loc[:, 51]) & np.isnan(reformat.loc[:, 50])
    reformat.loc[ii, 50] = (reformat.loc[ii, 51] - 1.216) / 2.862
    print('[DONE]')

    # BILI
    print('BILI ', end='')
    ii = ~np.isnan(reformat.loc[:, 44]) & np.isnan(reformat.loc[:, 45])
    reformat.loc[ii, 45] = (reformat.loc[ii, 44] * 0.6934) - 0.1752
    ii = ~np.isnan(reformat.loc[:, 45]) & np.isnan(reformat.loc[:, 44])
    reformat.loc[ii, 44] = (reformat.loc[ii, 45] + 0.1752) / 0.6934
    print('[DONE]')

    ## ########################################################################
    #                      SAMPLE AND HOLD on RAW DATA
    # ########################################################################

    reformat = SAH(reformat.loc[:, 1:68], referenceMatrices["sample_and_hold"])

    # reformat.to_csv('init_reformat_2.csv', index=False, sep=',')
    # qstime.to_csv('init_qstime_2.csv', index=False, sep=',')

    # reformat = pd.read_csv('init_reformat_2.csv')
    # reformat.columns = range(1,69)
    # qstime = pd.read_csv('init_qstime.csv', index_col='Unnamed: 0')
    # qstime.columns = range(1,5)

    timestep = 4  # resolution of timesteps, in hours
    irow = 0
    icustayidlist = np.unique(reformat[2])  # [x for x in set(reformat[2].values.tolist()) if not np.isnan(x)]
    icustayidlist = icustayidlist[~np.isnan(icustayidlist)]
    reformat2 = pd.DataFrame(np.empty([len(reformat), 84]), columns=range(1, 85))  # output array
    reformat2[:] = np.NaN
    npt = len(icustayidlist)  # number of patients
    # Adding 2 empty cols for future shock index=HR/SBP and P/F
    reformat.loc[:, 69] = np.NaN
    reformat.loc[:, 70] = np.NaN

    for i in tqdm(range(npt), desc='reformat2'):

        # print(i, npt)

        icustayid = icustayidlist[i]  # 1 to 100000, NOT 200 to 300K!

        # CHARTEVENTS AND LAB VALUES
        temp = reformat.loc[reformat[2] == icustayid, :].reset_index(drop=True)  # subtable of interest
        beg = temp.loc[0, 3]  # timestamp of first record

        # IV FLUID STUFF
        iv = inputMV[1] == icustayid  # rows of interest in inputMV
        input = inputMV.loc[iv, :]  # subset of interest

        startt = input[2]  # start of all infusions and boluses
        endt = input[3]  # end of all infusions and boluses
        rate = input[8]  # rate of infusion (is NaN for boluses) || corrected for tonicity

        pread = inputpreadm.loc[inputpreadm[1] == icustayid, 2]  # preadmission volume
        if not pread.empty:  # store the value, if available
            totvol = np.sum(pread)
        else:
            totvol = 0  # if not documented: it's zero

        # compute volume of fluid given before start of record!!!
        t0 = 0
        t1 = beg
        # input from MV (4 ways to compute)
        infu = np.sum(rate * (endt - startt) * ((endt <= t1) & (startt >= t0)) / 3600 + rate * (endt - t0) * (
                    (startt <= t0) & (endt <= t1) & (endt >= t0)) / 3600 + rate * (t1 - startt) * (
                                  (startt >= t0) & (endt >= t1) & (startt <= t1)) / 3600 + rate * (t1 - t0) * (
                                  (endt >= t1) & (startt <= t0)) / 3600)
        # all boluses received during this timestep, from inputMV (need to check rate is NaN) and inputCV (simpler):
        bolus = np.sum(input.loc[np.isnan(input[6]) & (input[2] >= t0) & (input[2] <= t1), 7])
        totvol = np.sum([totvol, infu, bolus])

        # VASOPRESSORS
        iv = vasoMV[1] == icustayid  # rows of interest in vasoMV
        vaso1 = vasoMV.loc[iv, :]  # subset of interest
        startv = vaso1[3]  # start of VP infusion
        endv = vaso1[4]  # end of VP infusions
        ratev = vaso1[5]  # rate of VP infusion

        # DEMOGRAPHICS / gender, age, elixhauser, re-admit, died in hosp?, died within
        # 48h of out_time (likely in ICU or soon after), died within 90d after admission?
        demogi = demog.index[demog[3] == icustayid][0]
        dem = [demog.loc[demogi, 15], demog.loc[demogi, 11], demog.loc[demogi, 18], demog.loc[demogi, 6] > 1,
               demog.loc[demogi, 16], abs((demog.loc[demogi, 13]) - (demog.loc[demogi, 9])) < (24 * 3600 * 2),
               demog.loc[demogi, 17], (qstime.loc[icustayid, 4] - qstime.loc[icustayid, 3]) / 3600]

        # URINE OUTPUT
        iu = UO[1] == icustayid  # rows of interest in inputMV
        output = UO.loc[iu, :]  # subset of interest
        pread = UOpreadm.loc[UOpreadm[1] == icustayid, 4]  # preadmission UO
        if not pread.empty:  # store the value, if available
            UOtot = np.sum(pread)
        else:
            UOtot = 0
        # adding the volume of urine produced before start of recording!    
        UOnow = np.sum(output.loc[(output[2] >= t0) & (output[2] <= t1), 4])  # t0 and t1 defined above
        UOtot = np.sum([UOtot, UOnow])

        for j in range(0, 79, timestep):  # -52 until +28 = 80 hours in total
            t0 = 3600 * j + beg  # left limit of time window
            t1 = 3600 * (j + timestep) + beg  # right limit of time window
            ii = (temp[3] >= t0) & (temp[3] <= t1)  # index of items in this time period
            if sum(ii) > 0:
                # ICUSTAY_ID, OUTCOMES, DEMOGRAPHICS
                reformat2.loc[irow, 1] = (j / timestep) + 1  # 'bloc' = timestep (1,2,3...)
                reformat2.loc[irow, 2] = icustayid  # icustay_ID
                reformat2.loc[irow, 3] = 3600 * j + beg  # t0 = lower limit of time window
                reformat2.loc[irow, 4:11] = dem  # demographics and outcomes

                # CHARTEVENTS and LAB VALUES (+ includes empty cols for shock index and P/F)
                value = temp.loc[ii, :]  # records all values in this timestep

                # #####################   DISCUSS ADDING STUFF HERE / RANGE, MIN, MAX ETC   ################

                # if sum(ii)==1:   #if only 1 row of values at this timestep
                #     reformat2.loc[irow,12:78] = value.loc[:,4:]
                # else:
                reformat2.loc[irow, 12:78] = np.nanmean(value.loc[:, 4:], axis=0)  # mean of all available values

                # VASOPRESSORS
                # for CV: dose at timestamps.
                # for MV: 4 possibles cases, each one needing a different way to compute the dose of VP actually administered:
                # ----t0---start----end-----t1----
                # ----start---t0----end----t1----
                # -----t0---start---t1---end
                # ----start---t0----t1---end----

                # MV
                v = ((endv >= t0) & (endv <= t1)) | ((startv >= t0) & (endv <= t1)) | (
                            (startv >= t0) & (startv <= t1)) | ((startv <= t0) & (endv >= t1))

                if not ratev.loc[v].empty:
                    v1 = np.nanmedian(ratev.loc[v])
                    v2 = np.nanmax(ratev.loc[v])
                    if not np.isnan(v1) and not np.isnan(v2):
                        reformat2.loc[irow, 79] = v1  # median of dose of VP
                        reformat2.loc[irow, 80] = v2  # max dose of VP

                # INPUT FLUID
                # input from MV (4 ways to compute)
                infu = np.sum(rate * (endt - startt) * ((endt <= t1) & (startt >= t0)) / 3600 + rate * (endt - t0) * (
                            (startt <= t0) & (endt <= t1) & (endt >= t0)) / 3600 + rate * (t1 - startt) * (
                                          (startt >= t0) & (endt >= t1) & (startt <= t1)) / 3600 + rate * (t1 - t0) * (
                                          (endt >= t1) & (startt <= t0)) / 3600)
                # all boluses received during this timestep, from inputMV (need to check rate is NaN) and inputCV (simpler):
                bolus = np.sum(input.loc[np.isnan(input[6]) & (input[2] >= t0) & (input[2] <= t1), 7])
                # sum fluid given
                totvol = np.sum([totvol, infu, bolus])
                reformat2.loc[irow, 81] = totvol  # total fluid given
                reformat2.loc[irow, 82] = np.sum([infu, bolus])  # fluid given at this step

                # UO
                UOnow = np.sum(output.loc[(output[2] >= t0) & (output[2] <= t1), 4])
                UOtot = np.sum([UOtot, UOnow])
                reformat2.loc[irow, 83] = UOtot  # total UO
                reformat2.loc[irow, 84] = np.sum(UOnow)  # UO at this step

                # CUMULATED BALANCE
                reformat2.loc[irow, 85] = totvol - UOtot  # cumulated balance

                irow += 1

    reformat2 = reformat2.drop(range(irow, len(reformat2))).reset_index(drop=True)

    ## ########################################################################
    #    CONVERT TO TABLE AND DELETE VARIABLES WITH EXCESSIVE MISSINGNESS
    # ########################################################################

    dataheaders = sample + ['Shock_Index', 'PaO2_FiO2']
    # dataheaders=regexprep(dataheaders,'['']','');
    dataheaders = ['bloc', 'icustayid', 'charttime', 'gender', 'age', 'elixhauser', 're_admission', 'died_in_hosp',
                   'died_within_48h_of_out_time', 'mortality_90d', 'delay_end_of_record_and_discharge_or_death'] + \
                  dataheaders + ['median_dose_vaso', 'max_dose_vaso', 'input_total', 'input_4hourly', 'output_total',
                                 'output_4hourly', 'cumulated_balance']

    # print(len(dataheaders), dataheaders)
    reformat2t = reformat2
    reformat2t.columns = dataheaders
    miss = np.isnan(reformat2).sum() / len(reformat2)

    # TODO: if values have less than 70# missing values (over 30# of values present): I keep them
    reformat3t = reformat2t.loc[:, :]
    # reformat3t=reformat2t.loc[:, [True] * 11 + (miss[11:74] < 0.90).tolist() + [True] * 11]

    ## ########################################################################
    #             HANDLING OF MISSING VALUES  &  CREATE REFORMAT4T
    # ########################################################################

    # Do linear interpol where missingness is low (kNN imputation doesnt work if all rows have missing values)
    reformat3 = reformat3t
    miss = np.isnan(reformat3).sum() / len(reformat3)
    ii = (miss > 0) & (miss < 0.05)  # less than 5# missingness
    mechventcol = reformat3t.columns.tolist().index('mechvent')

    for i in range(10, mechventcol):  # correct col by col, otherwise it does it wrongly
        if ii[i]:
            reformat3.loc[:, reformat3.columns.tolist()[i]] = fixgaps(reformat3.loc[:, reformat3.columns.tolist()[i]])

    # KNN IMPUTATION -  Done on chunks of 10K records.

    mechventcol = reformat3t.columns.tolist().index('mechvent')
    ref = reformat3.loc[:, reformat3.columns[10:mechventcol]]  # columns of interest

    knnimputer = KNNImputer(n_neighbors=1)
    for i in tqdm(range(0, len(reformat3) - 9999, 10000),
                  desc='impute NaN data'):  # dataset divided in 5K rows chunks (otherwise too large)
        missNotAll = np.isnan(ref.loc[i:i + 9999, :]).sum() != len(ref.loc[i:i + 9999, :])
        ref.loc[i:i + 9999, missNotAll] = knnimputer.fit_transform(ref.loc[i:i + 9999, missNotAll])

    missNotAll = np.isnan(ref.loc[len(reformat3) - 9999:len(reformat3), :]).sum() != len(
        ref.loc[len(reformat3) - 9999:len(reformat3), :])

    ref.loc[len(reformat3) - 9999:len(reformat3), missNotAll] = knnimputer.fit_transform(
        ref.loc[len(reformat3) - 9999:len(reformat3), missNotAll])

    # I paste the data interpolated, but not the demographics and the treatments
    reformat3t.loc[:, reformat3.columns[10:mechventcol]] = ref

    reformat4t = reformat3t
    reformat4 = reformat4t

    ## ########################################################################
    #        COMPUTE SOME DERIVED VARIABLES: P/F, Shock Index, SOFA, SIRS...
    # ########################################################################

    # CORRECT GENDER
    reformat4t['gender'] = reformat4t['gender'] - 1

    # CORRECT AGE > 200 yo
    ii = reformat4t['age'] > 150
    reformat4t.loc[ii, 'age'] = 91

    # FIX MECHVENT
    reformat4t.mechvent[np.isnan(reformat4t.mechvent)] = 0
    reformat4t.mechvent[reformat4t.mechvent > 0] = 1

    # FIX Elixhauser missing values
    reformat4t.elixhauser[np.isnan(reformat4t.elixhauser)] = np.nanmedian(
        reformat4t.elixhauser)  # use the median value / only a few missing data points

    # vasopressors / no NAN
    ii = np.isnan(reformat4t.median_dose_vaso)
    reformat4t.median_dose_vaso[ii] = 0
    ii = np.isnan(reformat4.max_dose_vaso)
    reformat4t.max_dose_vaso[ii] = 0

    # re-compute P/F with no missing values...
    if 'paO2' in reformat4.columns and 'FiO2_1' in reformat4.columns:
        print('re-compute P/F')
        reformat4t.PaO2_FiO2 = reformat4.paO2 / reformat4.FiO2_1

    # recompute SHOCK INDEX without NAN and INF
    # p=find(ismember(reformat4t.Properties.VariableNames,{'HR'}));
    # f=find(ismember(reformat4t.Properties.VariableNames,{'SysBP'}));
    # a=find(ismember(reformat4t.Properties.VariableNames,{'Shock_Index'}));

    if 'HR' in reformat4.columns and 'SysBP' in reformat4.columns:
        print('re-compute SHOCK')
        reformat4.Shock_Index = reformat4.HR / reformat4.SysBP
    reformat4.Shock_Index[np.isinf(reformat4.Shock_Index)] = np.NaN
    d = np.nanmean(reformat4.Shock_Index)
    reformat4.Shock_Index[np.isnan(reformat4.Shock_Index)] = d  # replace NaN with average value ~ 0.8
    reformat4t.Shock_Index = reformat4.Shock_Index

    # SOFA - at each timepoint
    # need (in this order):  P/F  MV  PLT  TOT_BILI  MAP  NORAD(max)  GCS  CR  UO
    s = pd.DataFrame(reformat4t.loc[:,
                     ['PaO2_FiO2', 'Platelets_count', 'Total_bili', 'MeanBP', 'max_dose_vaso', 'GCS', 'Creatinine',
                      'output_4hourly']])
    s.columns = range(1, 9)

    p = pd.DataFrame([[0, 1, 2, 3, 4]])

    s1 = pd.DataFrame(
        [s[1] > 400, (s[1] >= 300) & (s[1] < 400), (s[1] >= 200) & (s[1] < 300), (s[1] >= 100) & (s[1] < 200),
         s[1] < 100], index=range(5))
    s2 = pd.DataFrame(
        [s[2] > 150, (s[2] >= 100) & (s[2] < 150), (s[2] >= 50) & (s[2] < 100), (s[2] >= 20) & (s[2] < 50), s[2] < 20],
        index=range(5))
    s3 = pd.DataFrame(
        [s[3] < 1.2, (s[3] >= 1.2) & (s[3] < 2), (s[3] >= 2) & (s[3] < 6), (s[3] >= 6) & (s[3] < 12), s[3] > 12],
        index=range(5))
    s4 = pd.DataFrame([s[4] >= 70, (s[4] < 70) & (s[4] >= 65), (s[4] < 65), (s[5] > 0) & (s[5] <= 0.1), s[5] > 0.1],
                      index=range(5))
    s5 = pd.DataFrame(
        [s[6] > 14, (s[6] > 12) & (s[6] <= 14), (s[6] > 9) & (s[6] <= 12), (s[6] > 5) & (s[6] <= 9), s[6] <= 5],
        index=range(5))
    s6 = pd.DataFrame(
        [s[7] < 1.2, (s[7] >= 1.2) & (s[7] < 2), (s[7] >= 2) & (s[7] < 3.5), ((s[7] >= 3.5) & (s[7] < 5)) | (s[8] < 84),
         (s[7] > 5) | (s[8] < 34)], index=range(5))

    nrcol = len(reformat4.columns)  # nr of variables in data
    for new_col in range(nrcol + 1, nrcol + 8):
        reformat4[new_col] = 0
    for i in tqdm(range(len(reformat4)), desc='calculate SOFA and SIRS'):

        ms1 = max(p.loc[0, s1.loc[:, i]]) if not p.loc[0, s1.loc[:, i]].empty else 0
        ms2 = max(p.loc[0, s2.loc[:, i]]) if not p.loc[0, s2.loc[:, i]].empty else 0
        ms3 = max(p.loc[0, s3.loc[:, i]]) if not p.loc[0, s3.loc[:, i]].empty else 0
        ms4 = max(p.loc[0, s4.loc[:, i]]) if not p.loc[0, s4.loc[:, i]].empty else 0
        ms5 = max(p.loc[0, s5.loc[:, i]]) if not p.loc[0, s5.loc[:, i]].empty else 0
        ms6 = max(p.loc[0, s6.loc[:, i]]) if not p.loc[0, s6.loc[:, i]].empty else 0

        t = ms1 + ms2 + ms3 + ms4 + ms5 + ms6  # SUM OF ALL 6 CRITERIA
        if t is not None:
            reformat4.loc[i, range(nrcol + 1, nrcol + 8)] = [ms1, ms2, ms3, ms4, ms5, ms6, t]

    # SIRS - at each timepoint |  need: temp HR RR PaCO2 WBC 
    s = reformat4t.loc[:, ['Temp_C', 'HR', 'RR', 'paCO2', 'WBC_count']]
    s.columns = range(1, 6)

    s1 = (s[1] >= 38) | (s[1] <= 36)  # count of points for all criteria of SIRS
    s2 = (s[2] > 90)
    s3 = (s[3] >= 20) | (s[4] <= 32)
    s4 = (s[5] >= 12) | (s[5] < 4)
    reformat4.loc[:, nrcol + 8] = s1.astype(int) + s2.astype(int) + s3.astype(int) + s4.astype(int)

    # adds 2 cols for SOFA and SIRS, if necessary
    # if 'SIRS' not in reformat4t.columns:
    #     reformat4t.loc[:, 'SOFA'] = 0
    #     reformat4t.loc[:, 'SIRS'] = 0

    # records values
    # reformat4t.loc[:, 'SOFA'] = reformat4.loc[:, reformat4.columns[len(reformat4.columns) - 2]]
    # reformat4t.loc[:, 'SIRS'] = reformat4.loc[:, reformat4.columns[len(reformat4.columns) - 1]]
    reformat4t = reformat4t.rename(columns={(nrcol + 7): 'SOFA', (nrcol + 8): 'SIRS'})

    ## ########################################################################
    #                            EXCLUSION OF SOME PATIENTS 
    # ########################################################################

    print(len(set(reformat4t.icustayid)))  # count before

    # check for patients with extreme UO = outliers = to be deleted (>40 litres of UO per 4h!!)
    # a=reformat4t.output_4hourly>12000
    # i=list(set(reformat4t.icustayid(a).values.tolist()))
    # i=find(ismember(reformat4t.icustayid,i));
    # reformat4t(i,:)=[];
    reformat4t = reformat4t.drop(reformat4t[reformat4t.output_4hourly > 12000].index).reset_index(drop=True)

    # some have bili = 999999
    # a=find(reformat4t.Total_bili>10000); 
    # i=unique(reformat4t.icustayid(a));
    # i=find(ismember(reformat4t.icustayid,i));
    # reformat4t(i,:)=[];
    reformat4t = reformat4t.drop(reformat4t[reformat4t.Total_bili > 10000].index).reset_index(drop=True)

    # check for patients with extreme INTAKE = outliers = to be deleted (>10 litres of intake per 4h!!)
    # a=find(reformat4t.input_4hourly>10000);
    # i=unique(reformat4t.icustayid(a));  # 28 ids
    # i=find(ismember(reformat4t.icustayid,i));
    # reformat4t(i,:)=[];
    reformat4t = reformat4t.drop(reformat4t[reformat4t.input_4hourly > 10000].index).reset_index(drop=True)

    # #### exclude early deaths from possible withdrawals ####
    # stats per patient
    q = reformat4t.bloc == 1
    # fence_posts=find(q(:,1)==1);
    num_of_trials = len(set(reformat4t.icustayid))  # size(fence_posts,1);
    a = pd.DataFrame(
        [reformat4t.icustayid, reformat4t.mortality_90d, reformat4t.max_dose_vaso, reformat4t.SOFA]).transpose()

    a.columns = ['id', 'mortality_90d', 'vaso', 'sofa']
    # a.Properties.VariableNames={'id','mortality_90d','vaso','sofa'};
    d = a.groupby('id', as_index=False).agg('max')
    d['GroupCount'] = a.groupby('id').size().reset_index(drop=True)

    # finds patients who match our criteria
    e = pd.DataFrame([False] * num_of_trials)
    for i in range(num_of_trials):
        if d.mortality_90d[i] == 1:
            ii = (reformat4t.icustayid == d.id[i]) & (reformat4t.bloc == d.GroupCount[i])  # last row for this patient
            if sum(ii) > 0:
                e.loc[i, 0] = ((reformat4t.max_dose_vaso[ii].values.tolist()[0] == 0) & (d.vaso[i] > 0.3) & (
                            reformat4t.SOFA[ii].values.tolist()[0] >= d.sofa[i] / 2))
            else:
                e.loc[i, 0] = False
    r = d.id[e[0] & (d.GroupCount < 20)]  # ids to be removed
    # ii=ismember(reformat4t.icustayid,r)
    reformat4t = reformat4t.drop(reformat4t.loc[reformat4t.icustayid.isin(r), :].index).reset_index(drop=True)

    # exclude patients who died in ICU during data collection period
    ii = (reformat4t.bloc == 1) & (reformat4t.died_within_48h_of_out_time == 1) & (
                reformat4t.delay_end_of_record_and_discharge_or_death < 24)
    # ii=ismember(icustayidlist,reformat4t.icustayid(ii));
    # reformat4t(ii,:)=[];
    reformat4t = reformat4t.drop(reformat4t.loc[ii, :].index).reset_index(drop=True)

    print(len(set(reformat4t.icustayid)))  # count after

    ## #######################################################################
    #                       CREATE SEPSIS COHORT
    # ########################################################################

    # create array with 1 row per icu admission
    # keep only patients with flagged sepsis (max sofa during time period of interest >= 2)
    # we assume baseline SOFA of zero (like other publications)

    sepsis = pd.DataFrame(np.zeros([30000, 5]), columns=range(1, 6))
    irow = 0

    for icustayid in icustayidlist:
        ii = reformat4t.icustayid == icustayid
        # if mod(icustayid,10000)==0:
        #     disp([num2str(icustayid/1000), ' #'])
        if sum(ii) > 0:
            sofa = reformat4t.SOFA[ii]
            sirs = reformat4t.SIRS[ii]
            sepsis.loc[irow, 1] = icustayid
            sepsis.loc[irow, 2] = reformat4t.mortality_90d[ii].values.tolist()[0]  # 90-day mortality
            sepsis.loc[irow, 3] = max(sofa)
            sepsis.loc[irow, 4] = max(sirs)
            sepsis.loc[irow, 5] = qstime.loc[icustayid, 1]  # time of onset of sepsis
            irow += 1
    print(irow)
    # sepsis.to_csv('sepsis_mimic_test.csv', index=False, sep=',')
    sepsis = sepsis.drop(range(irow, len(sepsis)))
    # sepsis.to_csv('sepsis_mimic_testt.csv', index=False, sep=',')

    sepsis.columns = ['icustayid', 'morta_90d', 'max_sofa', 'max_sirs', 'sepsis_time']

    # delete all non-sepsis
    # sepsis(sepsis.max_sofa<2,:)=[];

    sepsis = sepsis.drop(sepsis.loc[sepsis.max_sofa < 2, :].index)

    # sepsisTable = pd.DataFrame() #TODO: IDENTIFIES THE COHORT OF PATIENTS WITH SEPSIS in MIMIC-IV
    sepsis.to_csv(output_dir + 'sepsis_mimic.csv', index=False, sep=',')
    reformat4t.to_csv(output_dir + 'mimic_record.csv', index=False, sep=',')
    return reformat4t
