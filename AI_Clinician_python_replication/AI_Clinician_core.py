import os.path

import pandas as pd
import numpy as np
from scipy import stats
from tqdm import tqdm
from sklearn.cluster import KMeans
from scipy.stats import rankdata
from MDPtoolbox import mdp_policy_iteration_with_Q
from scipy.io import savemat
from offpolicy import offpolicy_multiple_eval_010518, offpolicy_eval_tdlearning_with_morta
import matplotlib.pyplot as plt
import matlab.engine


output_dir = '../models/data/'
data_dir = '../sepsis_data/'


def AI_Clinician_core(MIMICtable):

    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    # ############################  MODEL PARAMETERS   #####################################

    print('####  INITIALISATION  ####') 

    nr_reps = 100               # nr of repetitions (total nr models)
    save_iters = 10
    # nr_reps = 1
    nclustering = 32            # how many times we do clustering (best solution will be chosen)
    prop = 0.25                 # proportion of the data we sample for clustering
    gamma = 0.99                # gamma
    transthres = 5              # threshold for pruning the transition matrix
    polkeep = 0                 # count of saved policies
    ncl = 750                   # nr of states
    nra = 5                     # nr of actions (2 to 10)
    ncv = 5                     # nr of crossvalidation runs (each is 80# training / 20# test)
    OA = np.empty([752, nr_reps])       # record of optimal actions
    OA[:] = np.NaN
    recqvi = np.empty([nr_reps*2,30])  # saves data about each model (1 row per model)
    recqvi[:] = np.NaN
    allpols = np.empty((nr_reps, 15), dtype=object)  # saving best candidate models
    allpols[:] = 0

    max_mimiciv_wis = -np.inf


    #################   Convert training data and compute conversion factors    ######################

    # # all 47 columns of interest
    colbin = ['gender','mechvent','max_dose_vaso','re_admission']
    colnorm=['age','Weight_kg','GCS','HR','SysBP','MeanBP','DiaBP','RR','Temp_C','FiO2_1',
        'Potassium','Sodium','Chloride','Glucose','Magnesium','Calcium',
        'Hb','WBC_count','Platelets_count','PTT','PT','Arterial_pH','paO2','paCO2',
        'Arterial_BE','HCO3','Arterial_lactate','SOFA','SIRS','Shock_Index','PaO2_FiO2','cumulated_balance']
    collog=['SpO2','BUN','Creatinine','SGOT','SGPT','Total_bili','INR','input_total','input_4hourly','output_total','output_4hourly']

    colbin_id = [MIMICtable.columns.tolist().index(name) for name in colbin]
    colnorm_id = [MIMICtable.columns.tolist().index(name) for name in colnorm]
    collog_id = [MIMICtable.columns.tolist().index(name) for name in collog]


    # MIMICtable = MIMICtable.loc[MIMICtable[colbin + colnorm + collog].isna().sum(axis=1) == 0, :]

    # find patients who died in ICU during data collection period
    reformat5 = MIMICtable.to_numpy()
    # reformat5=reformat5(ikeep,:);
    icustayidlist = MIMICtable.icustayid
    icuuniqueids = np.unique(icustayidlist) #list of unique icustayids from MIMIC
    idxs = np.empty([len(icustayidlist),nr_reps]) #record state membership test cohort
    idxs[:] = np.NaN

    MIMICraw = MIMICtable.loc[:, colbin + colnorm + collog]

    print(MIMICraw.isna().sum())
    print(len(MIMICraw))


    MIMICraw = MIMICraw.to_numpy()  # RAW values

    MIMICzs = np.hstack([reformat5[:, colbin_id]-0.5, stats.zscore(reformat5[:,colnorm_id], ddof=1, nan_policy='omit'), stats.zscore(np.log(0.1+reformat5[:, collog_id]), ddof=1, nan_policy='omit')])
    MIMICzs[np.isnan(MIMICzs)] = 0
    MIMICzs[:, 2] = np.log(MIMICzs[:, 2] + 6)   # MAX DOSE NORAD 
    MIMICzs[:, 44] = 2 * MIMICzs[:, 44]   # increase weight of this variable

    matlab_eng = matlab.engine.start_matlab()

    main_loop_tqdm = tqdm(range(nr_reps), desc='MAIN LOOP', leave=True)
    def disp(desc):
        main_loop_tqdm.set_description(desc)
        main_loop_tqdm.refresh()
    for modl in main_loop_tqdm:  # MAIN LOOP OVER ALL MODELS
    
        N = len(icuuniqueids) #total number of rows to choose from
        grp = np.floor(ncv*np.random.rand(N)+1)  #list of 1 to 5 (20# of the data in each grp) -- this means that train/test MIMIC split are DIFFERENT in all the 500 models
        crossval = 1
        trainidx = icuuniqueids[crossval != grp]
        testidx = icuuniqueids[crossval == grp]
        train = np.isin(icustayidlist, trainidx)
        test = np.isin(icustayidlist, testidx)
        X = MIMICzs[train, :]
        Xtestmimic = MIMICzs[~train, :]
        blocs = reformat5[train, 0]
        bloctestmimic = reformat5[~train, 0]
        ptid = reformat5[train, 1]
        ptidtestmimic = reformat5[~train, 1]
        outcome = 9 #   HOSP _ MORTALITY = 8 / 90d MORTA = 10
        Y90 = reformat5[train, outcome]
        

        #######   find best clustering solution (lowest intracluster variability)  ####################
        disp('MAIN LOOP (CLUSTERING)')
        N = len(X) #total number of rows to choose from
        sampl = X[np.floor(np.random.rand(N)+prop) == 1, :]
        sampl = matlab.double(sampl.tolist())
        # TODO: adjust parameters to run faster
        # kmeans = KMeans(n_clusters=ncl, random_state=0, n_init=nclustering, max_iter=30, n_jobs=2).fit(sampl)
        _, C = matlab_eng.kmeans(sampl, ncl, nargout=2)
        # idx = kmeans.predict(X)  #N-D nearest point search: look for points closest to each centroid
        idx = np.asarray(matlab_eng.knnsearch(C, matlab.double(X.tolist())), dtype='int').reshape(-1) - 1

        ############################# CREATE ACTIONS  ########################
        disp('MAIN LOOP (CREATE ACTIONS)')
        nact = nra ** 2
        
        iol = MIMICtable.columns.tolist().index('input_4hourly')
        vcl = MIMICtable.columns.tolist().index('max_dose_vaso')
        
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
        actionbloctrain = actionbloc[train]
        uniqueValuesdose = np.array([ ma2[uniqueValues[1].astype('int') - 1], ma1[uniqueValues[0].astype('int') - 1]])  # median dose of each bin for all 25 actions
        
        
        # ###################################################################################################################################
        disp('MAIN LOOP (CREATE QLDATA3)')
        r = np.array([100, -100])
        r2 = np.array([r[0] * (2 * (1 - Y90) - 1), r[1] * (2 * (1 - Y90) - 1)])
        qldata = np.array([blocs, idx, actionbloctrain, Y90, r2[0], r2[1]]).transpose()  # contains bloc / state / action / outcome&reward     #1 = died
        qldata3 = np.zeros([int(np.floor(len(qldata)*1.2)), 4])
        c = -1
        abss = [ncl+1, ncl] #absorbing states numbers
        
        for i in range(len(qldata) - 1):
            c += 1
            qldata3[c, :] = qldata[i, :4]
            if qldata[i+1, 0] == 1: #end of trace for this patient
                c += 1
                qldata3[c, :] = [qldata[i, 0]+1, abss[int(qldata[i, 3])], -1, qldata[i, 4]]
            
        qldata3 = np.delete(qldata3, np.s_[c+1:], 0)

        
        # ###################################################################################################################################
        disp('MAIN LOOP (CREATE TRANSITION MATRIX T(S\'\',S,A))')

        def transition_matrix(T):
            transitionr = np.zeros([ncl+2,ncl+2,nact])  #this is T(S',S,A)
            sums0a0 = np.zeros([ncl+2,nact])
            
            for i in range(len(qldata3) - 1):
        
                if qldata3[i+1, 0] != 1:  # if we are not in the last state for this patient = if there is a transition to make!
                    S0 = int(qldata3[i, 1])
                    S1 = int(qldata3[i + 1, 1])
                    acid = int(qldata3[i, 2])
                    if not T:
                        transitionr[S1,S0,acid] = transitionr[S1,S0,acid] + 1
                    else:
                        transitionr[S0,S1,acid] = transitionr[S0,S1,acid] + 1

                    sums0a0[S0,acid] = sums0a0[S0,acid] + 1
            

            sums0a0[sums0a0 <= transthres] = 0  #delete rare transitions [those seen less than 5 times = bottom 50#!!]

            for i in range(ncl + 2):
                for j in range(nact):
                    if sums0a0[i, j] == 0:
                        if not T:
                            transitionr[:, i, j] = 0
                        else:
                            transitionr[i, :, j] = 0

                    else:
                        if not T:
                            transitionr[:, i, j] = transitionr[:, i, j] / sums0a0[i, j]
                        else:
                            transitionr[i, :, j] = transitionr[i, :, j] / sums0a0[i, j]
                            
            
            
            transitionr[np.isnan(transitionr)] = 0  #replace NANs with zeros
            transitionr[np.isinf(transitionr)] = 0  #replace NANs with zeros

            physpol = sums0a0 / np.tile(sum(sums0a0.T).T, (sums0a0.shape[1], 1)).T

            return [transitionr, physpol]

        
        transitionr, physpol = transition_matrix(False)    #physicians policy: what action was chosen in each state
        
        disp('MAIN LOOP (CREATE TRANSITION MATRIX T[S,S\'\',A])')
        
        transitionr2, _ = transition_matrix(True)
        
        # #################################################################################################################################
        disp('MAIN LOOP (CREATE REWARD MATRIX  R[S,A])')
        # CF sutton& barto bottom 1998 page 106. i compute R[S,A] from R[S'SA] and T[S'SA]
        r3 = np.zeros([ncl+2,ncl+2,nact])
        r3[ncl, :, :] = -100
        r3[ncl + 1, :, :] = 100
        R = sum(transitionr * r3)
        # R = squeeze(R)   #remove 1 unused dimension

        # ###################################################################################################################################
        disp('MAIN LOOP (POLICY ITERATION)')

        _,_,_, Qon = mdp_policy_iteration_with_Q(transitionr2, R, gamma, np.ones(ncl+2))
        OptimalAction = Qon.argmax(axis=1)  #deterministic 
        OA[:, modl] = OptimalAction #save optimal actions
        
        disp('MAIN LOOP (OFF-POLICY EVALUATION - MIMIC TRAIN SET)')

        def evaluation_data(blocs, idx, actionbloctrain, Y90, ptid, reformatiol, reformatvcl):
            # create new version of QLDATA3
            r = np.array([100, -100])
            r2 = np.array([r[0] * (2 * (1 - Y90) - 1), r[1] * (2 * (1 - Y90) - 1)])
            models = OptimalAction[idx]                  #optimal action for each record
            modeldosevaso = uniqueValuesdose[0, models]      #dose reco in this model
            modeldosefluid = uniqueValuesdose[1, models]     #dose reco in this model
            qldata = np.array([blocs, idx, actionbloctrain, Y90, np.zeros(len(idx)), r2[0], ptid, reformatiol, reformatvcl, modeldosefluid, modeldosevaso, Y90]).transpose()  # contains bloc / state / action / outcome&reward     #1 = died
            qldata3 = np.zeros([int(np.floor(len(qldata)*1.2)), 13])
            c = -1
            abss = [ncl+1, ncl] #absorbing states numbers

            for i in range(len(qldata) - 1):
                c += 1
                qldata3[c, :] = qldata[i, [0, 1, 2, 4, 6, 6, 6, 6, 7, 8, 9, 10, 11]]
                if qldata[i+1, 0] == 1: #end of trace for this patient
                    c += 1
                    qldata3[c, :] = [qldata[i, 0]+1, abss[int(qldata[i, 3])], 0, qldata[i,5], 0, 0, 0, qldata[i,6], qldata[i,7], qldata[i,8], qldata[i,9], qldata[i,10], qldata[i,11]]

            qldata3 = np.delete(qldata3, np.s_[c+1:], 0)

            # add pi[s,a] and b[s,a]
            p = 0.01 #softening policies
            softpi = physpol # behavior policy = clinicians'

            for i in range(ncl):
                ii = softpi[i,:] == 0
                z = p / ii.sum()
                nz = p / (~ii).sum()
                softpi[i, ii] = z
                softpi[i, ~ii] = softpi[i,~ii] - nz

            softb = np.zeros([ncl + 2, 25]) + p / 24 #"optimal" policy = target policy = evaluation policy

            for i in range(ncl):
                softb[i, OptimalAction[i]] = 1 - p

            for i in range(len(qldata3)):  #adding the probas of policies to qldata3
                if qldata3[i, 1] <= ncl:
                    qldata3[i, 4] = softpi[int(qldata3[i,1]), int(qldata3[i,2])]
                    qldata3[i, 5] = softb[int(qldata3[i,1]), int(qldata3[i,2])]
                    qldata3[i, 6] = OptimalAction[int(qldata3[i,1])]   #optimal action
            return qldata3

        qldata3 = evaluation_data(blocs, idx, actionbloctrain, Y90, ptid, reformat5[train, iol], reformat5[train, vcl])


        qldata3train = qldata3

        bootql, bootwis = offpolicy_multiple_eval_010518(qldata3, physpol, 0.99, 1, 6, ncl)


        recqvi[modl, 0] = modl
        recqvi[modl, 3] = np.nanmean(bootql)
        recqvi[modl, 4] = np.quantile(bootql, 0.99)
        recqvi[modl, 5] = np.nanmean(bootwis)  #we want this as high as possible
        recqvi[modl, 6] = np.quantile(bootwis, 0.05)  #we want this as high as possible


        # testing on MIMIC-test
        disp('MAIN LOOP (OFF-POLICY EVALUATION - MIMIC TEST SET)')
            
        # create new version of QLDATA3 with MIMIC TEST samples
        # idxtest = kmeans.predict(Xtestmimic)
        idxtest = np.asarray(matlab_eng.knnsearch(C, matlab.double(Xtestmimic.tolist())), dtype='int').reshape(-1) - 1
        idxs[test, modl] = idxtest  #important: record state membership of test cohort

        actionbloctest = actionbloc[~train]
        Y90test = reformat5[~train, outcome]
        qldata3 = evaluation_data(bloctestmimic, idxtest, actionbloctest, Y90test, ptidtestmimic, reformat5[test, iol], reformat5[test, vcl])

        qldata3test = qldata3

        bootmimictestql, bootmimictestwis = offpolicy_multiple_eval_010518( qldata3,physpol, 0.99,1,6,2000)

        recqvi[modl, 18] = np.quantile(bootmimictestql, 0.95)   #PHYSICIANS' 95# UB
        recqvi[modl, 19] = np.nanmean(bootmimictestql)
        recqvi[modl, 20] = np.quantile(bootmimictestql, 0.99)
        recqvi[modl, 21] = np.nanmean(bootmimictestwis)    
        recqvi[modl, 22] = np.quantile(bootmimictestwis, 0.01)  
        recqvi[modl, 23] = np.quantile(bootmimictestwis, 0.05)  #AI 95# LB, we want this as high as possible


        if recqvi[modl, 23] > 0 and recqvi[modl, 23] > max_mimiciv_wis * 4 / 5:# & recqvi[modl,14]>0:   # if 95# LB is >0 : save the model [otherwise it's pointless]
            # TODO
            max_mimiciv_wis = max(max_mimiciv_wis, recqvi[modl, 23])
            disp('MAIN LOOP (  GOOD MODEL FOUND - SAVING IT)')
            allpols[polkeep, 0] = modl
            allpols[polkeep, 2] = Qon
            allpols[polkeep, 3] = physpol
            allpols[polkeep, 5] = transitionr
            allpols[polkeep, 6] = transitionr2
            allpols[polkeep, 7] = R
            allpols[polkeep, 8] = np.asarray(C, dtype='double') # kmeans.cluster_centers_
            allpols[polkeep, 9] = train
            allpols[polkeep, 10] = qldata3train
            allpols[polkeep, 11] = qldata3test
            # allpols[polkeep][12] = qldata2
            polkeep += 1
        if modl % save_iters == 0:
            with open(output_dir + 'MIMICzs.npy', 'wb') as f:
                np.save(f, MIMICzs)
            with open(output_dir + 'recqvi.npy', 'wb') as f:
                np.save(f, recqvi)
            with open(output_dir + 'allpols.npy', 'wb') as f:
                np.save(f, allpols)
            
    recqvi = np.delete(recqvi, np.s_[nr_reps:], 0)

    # tic
    #     save['D:\BACKUP MIT PC\Data_160219.mat', '-v7.3'];
    # toc
    with open(output_dir + 'MIMICzs.npy', 'wb') as f:
        np.save(f, MIMICzs)
    with open(output_dir + 'recqvi.npy', 'wb') as f:
        np.save(f, recqvi)
    with open(output_dir + 'allpols.npy', 'wb') as f:
        np.save(f, allpols)

    # savemat("Data_160219.mat", {
    #     'MIMICraw': MIMICraw,
    #     'MIMICzs': MIMICzs,
    #     'recqvi': recqvi,
    #     'idxs': idxs,
    #     'OA': OA,
    #     'allpols': allpols
    # })

def test_model():
    MIMICtable = pd.read_csv(data_dir + 'mimic_record.csv')
#region load model

    nr_reps = 200               # nr of repetitions (total nr models)
    # nr_reps = 1
    nclustering = 32            # how many times we do clustering (best solution will be chosen)
    prop = 0.25                 # proportion of the data we sample for clustering
    gamma = 0.99                # gamma
    transthres = 5              # threshold for pruning the transition matrix
    polkeep = 0                 # count of saved policies
    ncl = 750                   # nr of states
    nra = 5                     # nr of actions (2 to 10)
    ncv = 5                     # nr of crossvalidation runs (each is 80# training / 20# test)
    OA = np.empty([752, nr_reps])       # record of optimal actions
    OA[:] = np.NaN
    recqvi = np.empty([nr_reps*2,30])  # saves data about each model (1 row per model)
    recqvi[:] = np.NaN
    allpols = np.empty((nr_reps, 15), dtype=object)  # saving best candidate models
    allpols[:] = 0

    max_mimiciv_wis = 0


    #################   Convert training data and compute conversion factors    ######################

    # # all 47 columns of interest
    colbin = ['gender','mechvent','max_dose_vaso','re_admission']
    colnorm=['age','Weight_kg','GCS','HR','SysBP','MeanBP','DiaBP','RR','Temp_C','FiO2_1',
        'Potassium','Sodium','Chloride','Glucose','Magnesium','Calcium',
        'Hb','WBC_count','Platelets_count','PTT','PT','Arterial_pH','paO2','paCO2',
        'Arterial_BE','HCO3','Arterial_lactate','SOFA','SIRS','Shock_Index','PaO2_FiO2','cumulated_balance']
    collog=['SpO2','BUN','Creatinine','SGOT','SGPT','Total_bili','INR','input_total','input_4hourly','output_total','output_4hourly']

    colbin_id = [MIMICtable.columns.tolist().index(name) for name in colbin]
    colnorm_id = [MIMICtable.columns.tolist().index(name) for name in colnorm]
    collog_id = [MIMICtable.columns.tolist().index(name) for name in collog]


    # MIMICtable = MIMICtable.loc[MIMICtable[colbin + colnorm + collog].isna().sum(axis=1) == 0, :]

    # find patients who died in ICU during data collection period
    reformat5 = MIMICtable.to_numpy()
    iol = MIMICtable.columns.tolist().index('input_4hourly')
    vcl = MIMICtable.columns.tolist().index('max_dose_vaso')



    # reformat5=reformat5(ikeep,:);
    icustayidlist = MIMICtable.icustayid
    icuuniqueids = np.unique(icustayidlist) #list of unique icustayids from MIMIC
    idxs = np.empty([len(icustayidlist),nr_reps]) #record state membership test cohort
    idxs[:] = np.NaN


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
    print(actionbloc.max())

    # Data_mat = mat73.loadmat('Data_160219III.mat')

    with open(output_dir + 'MIMICzs.npy', 'rb') as f:
        MIMICzs = np.load(f, allow_pickle=True)
    with open(output_dir + 'recqvi.npy', 'rb') as f:
        recqvi = np.load(f, allow_pickle=True)
    with open(output_dir + 'allpols.npy', 'rb') as f:
        allpols = np.load(f, allow_pickle=True)

    # MIMICraw = Data_mat['MIMICraw']
    # MIMICzs = Data_mat['MIMICzs']
    # recqvi = Data_mat['recqvi']
    # idxs = Data_mat['idxs']
    # OA = Data_mat['OA']
    # allpols = np.array(Data_mat['allpols'], dtype=object)
    
    ## IDENTIFIES BEST MODEL HERE
    recqvi = np.delete(recqvi, np.s_[30:], 1)

    # recqvi[:,31:end]=[];

    r = recqvi
    r = np.delete(r, np.s_[29:], 1)

    # r[r[:,14]<0,:]=[];  #delete models with poor value in MIMIC test set
    # r = np.delete(r, r[:,13] < 0, 0) #TODO


    # SORT RECQVI BY COL 24 / DESC
    bestpol = r[max(r[:,23]) == r[:,23], 0]   # model maximising 95# LB of value of AI policy in MIMIC test set

    print(max(r[:,23]), max(r[:,18]))


    ## RECOVER BEST MODEL and TEST IT
    print('RECOVER BEST MODEL')
    a = np.hstack(allpols[:, 0])
    outcome = 9 #   HOSPITAL MORTALITY = 8 / 90d MORTA = 10
    ii = np.isin(a, bestpol) #position of best model in the array allpols

    # RECOVER MODEL DATA
    Qon = allpols[ii, 2][0]
    physpol = allpols[ii, 3][0]
    transitionr = allpols[ii, 5][0]
    transitionr2 = allpols[ii, 6][0]
    R = allpols[ii, 7][0]
    C = allpols[ii, 8][0]
    train = allpols[ii, 9][0]
    test = ~train
    qldata3train = allpols[ii, 10][0]
    qldata3test = allpols[ii, 11][0]
    # qldata2 = allpols[ii, 12][0]

    kmeans = KMeans(n_clusters=ncl, random_state=0, n_init=nclustering, max_iter=30, n_jobs=2).fit(C)

    idx = kmeans.predict(MIMICzs[train, :])  #N-D nearest point search: look for points closest to each centroid
    OptimalAction = Qon.argmax(axis=1)  #deterministic 

    idxtest = kmeans.predict(MIMICzs[test, :]) #np.hstack(idxs[test, int(a[ii])])          #state of records from training set
    actionbloctrain = actionbloc[train]
    actionbloctest = actionbloc[test]        #actionbloc is constant across clustering solutions
    Y90 = reformat5[train, outcome]
    Y90test =  reformat5[test, outcome]
    blocs = reformat5[train, 0]
    bloctestmimic = reformat5[test, 0]
    vcl = reformat5[test, vcl] # TODO: use vcl here.
    iol = reformat5[test, iol]
    ptid = reformat5[train, 1]
    ptidtestmimic = reformat5[test, 1]

    idxtestint = idxtest.astype('int64')

#endregion


##### SOME TEST WORK
#region figures
    fig = plt.figure(figsize=(20, 10))

    bootmimictestql, bootmimictestwis = offpolicy_multiple_eval_010518(qldata3test, physpol, 0.99,1,500,80)
    bootmimictestql = np.matlib.repmat(bootmimictestql, int(np.floor(len(bootmimictestwis)/ len(bootmimictestql))), 1).ravel()

    counts, _, _ = np.histogram2d(bootmimictestql, bootmimictestwis, bins=(np.arange(-105, 103, 2.5), np.arange(-105, 103, 2.5)))
    counts = counts.T
    counts = np.flipud(counts)

    offpolicy_eval_plot = fig.add_subplot(231)
    logcounts = np.log10(counts + 1)
    # logcounts[np.isinf(logcounts)] = 0
    im = offpolicy_eval_plot.imshow(logcounts, cmap=plt.get_cmap('jet'))
    fig.colorbar(im, ax=offpolicy_eval_plot)
    offpolicy_eval_plot.set_xticks(np.arange(1, 90, 10))
    offpolicy_eval_plot.set_xticklabels(['-100','-75','-50','-25','0','25','50','75','100'])
    offpolicy_eval_plot.set_yticks(np.arange(1, 90, 10))
    offpolicy_eval_plot.set_yticklabels(['100', '75','50','25','0','-25','-50','-75','-100'])
    xmin, xmax, ymin, ymax = offpolicy_eval_plot.axis()
    offpolicy_eval_plot.plot([xmin, ymin], [xmax, ymax], 'r', linewidth=2)
    offpolicy_eval_plot.set_xlabel('Clinicans\' policy value')
    offpolicy_eval_plot.set_ylabel('AI policy value')

    ####

    counts, _, _ = np.histogram2d(med[0], med[1], bins=(np.arange(1, 7), np.arange(1, 7)))
    counts /=  len(med[0])
    # counts = np.flipud(counts)

    actual_action_plot = fig.add_subplot(232, projection='3d')
    _x = np.arange(5)
    _y = np.arange(5)
    _xx, _yy = np.meshgrid(_x, _y)
    x, y = _xx.ravel(), _yy.ravel()

    top = counts.ravel()
    bottom = np.zeros_like(top)
    width = depth = 1
    actual_action_plot.bar3d(x, y, bottom, width, depth, top, shade=True)
    actual_action_plot.set_xticks(range(5))
    actual_action_plot.set_yticklabels(['>475', '173-475','44-173','0-44','0'])
    actual_action_plot.set_yticks(range(5))
    actual_action_plot.set_xticklabels(['0', '0.003-0.08','0.08-0.20','0.22-0.363','>0.363'])

    actual_action_plot.set_xlabel('Vasopressor dose')
    actual_action_plot.set_ylabel('     IV fluids dose')
    actual_action_plot.set_title('Clinicians\' policy')

    ####
    ai_action_plot = fig.add_subplot(233, projection='3d')


    OA1 = OptimalAction[idxtest.astype('int64')] + 1
    a = np.array([OA1, np.floor((OA1 - 0.0001) / 5) + 1, OA1 - np.floor(OA1 / 5) * 5])
    a[2, a[2, :] == 0] = 5
    med = a[[1, 2], :]

    counts, _, _ = np.histogram2d(med[0], med[1], bins=(np.arange(1, 7), np.arange(1, 7)))
    counts /=  len(med[0])
    # counts = np.flipud(counts)
    top = counts.ravel()
    ai_action_plot.bar3d(x, y, bottom, width, depth, top, shade=True)
    ai_action_plot.set_xticks(range(5))
    ai_action_plot.set_yticklabels(['>475', '173-475','44-173','0-44','0'])
    ai_action_plot.set_yticks(range(5))
    ai_action_plot.set_xticklabels(['0', '0.003-0.08','0.08-0.20','0.22-0.363','>0.363'])
    ai_action_plot.set_xlabel('Vasopressor dose')
    ai_action_plot.set_ylabel('     IV fluids dose')
    ai_action_plot.set_title('AI policy')

    ####

    t = np.arange(-1250, 1251, 100)
    t2 = np.arange(-1.05, 1.06, 0.1)

    nr_reps = 200
    p = np.unique(qldata3test[:, 7])
    prop = 10000 / len(p)
    prop = min(prop, 0.75)

    qldata = np.zeros([len(qldata3test[qldata3test[:, 2] != 0, :]), 18]) # TODO

    qldata[:, :13] = qldata3test[qldata3test[:, 2] != 0, :]
    qldata[:, 13] = qldata[:, 9] - qldata[:, 11]
    qldata[:, 14] = qldata[:, 8] - qldata[:, 10]

    
    r = pd.DataFrame(qldata[:, [7, 12, 13, 14]])
    r.columns = ['id','morta','vaso','ivf']
    d = r.groupby('id').agg(['mean','median','sum'])
    print(len(d), len(p), len(np.unique(r.id)))
    groupCount = r.groupby('id').count()['morta'].to_numpy()
    d3 = np.array([d.morta['mean'].to_numpy(), d.vaso['mean'].to_numpy(), d.ivf['mean'].to_numpy(), d.vaso['median'].to_numpy(), d.ivf['median'].to_numpy(), d.ivf['sum'].to_numpy(), groupCount]).T

    r1 = np.zeros([len(t) - 1, nr_reps, 2])
    r2 = np.zeros([len(t2) - 1, nr_reps, 2])

    for rep in range(nr_reps):
        ii = np.floor(np.random.rand(len(p)) + prop)
        d4 = d3[ii == 1, :]

        a = []
        b = []

        for i in range(len(t) - 1):
            ii = (d4[:, 4] >= t[i]) & (d4[:, 4] <= t[i + 1])
            a.append([t[i], t[i + 1], sum(ii), np.nanmean(d4[ii, 0]), np.nanstd(d4[ii, 0])])
        
        a = np.array(a)
        
        r1[:, rep, 0] = a[:, 3]
        r1[:, rep, 1] = a[:, 2]


        for i in range(len(t2) - 1):
            ii = (d4[:, 3] >= t2[i]) & (d4[:, 3] <= t2[i + 1])
            b.append([t2[i], t2[i + 1], sum(ii), np.nanmean(d4[ii, 0]), np.nanstd(d4[ii, 0])])

        b = np.array(b)
        
        r2[:, rep, 0] = b[:, 3]
        r2[:, rep, 1] = b[:, 2]
    
    a1 = np.nanmean(r1[:, :, 0], axis=1)
    a2 = np.nanmean(r2[:, :, 0], axis=1)

    def get_s(r, t):
        s1 = np.zeros(len(t) - 1)
        for i in range(len(t) - 1):

            dbg1 = np.ones(int(np.nansum(r[i, :, 0] * r[i, :, 1])))
            dbg2 = np.zeros(int(np.nansum((1 - r[i, :, 0]) * r[i, :, 1])))
            dbg3 = np.sqrt(np.nansum(r[i, :, 1]))
            dbg4 = np.nanstd(np.concatenate((dbg1, dbg2)))

            s1[i] = np.nanstd(np.concatenate((np.ones(int(np.nansum(r[i, :, 0] * r[i, :, 1]))), 
                    np.zeros(int(np.nansum((1 - r[i, :, 0]) * r[i, :, 1])))))) / \
                    np.sqrt(np.nansum(r[i, :, 1]))
        return s1
    s1 = get_s(r1, t)
    s2 = get_s(r2, t2)


    IV_dose_plot = fig.add_subplot(234)

    f = 10
    
    h = IV_dose_plot.plot(a1, 'b', linewidth=1)
    IV_dose_plot.plot(a1+f*s1,'b:', linewidth=1)
    IV_dose_plot.plot(a1-f*s1,'b:', linewidth=1)

    
    IV_dose_plot.plot([len(a1)/2, len(a1)/2], [0, 1], 'k:')

    

    IV_dose_plot.set_xlim([1, len(a1)])
    IV_dose_plot.set_ylim([0, 1])

    t = t.astype('float64')

    t -= (t[-1] - t[-2]) / 2
    t = np.round(t,2)
    t = t[1:-1:2]
    print(t)
    IV_dose_plot.set_xticks(np.arange(0, 2*len(t), 2))
    IV_dose_plot.set_xticklabels(t.tolist())
    for tick in IV_dose_plot.get_xticklabels():
        tick.set_rotation(90)
    IV_dose_plot.set_xlabel('Average dose excess per patient')
    IV_dose_plot.set_ylabel('Mortality')

    IV_dose_plot.set_title('Intravenous fluids')


    vaso_dose_plot = fig.add_subplot(235)

    f = 10
    
    h = vaso_dose_plot.plot(a2, 'b', linewidth=1)
    vaso_dose_plot.plot(a2+f*s2,'b:', linewidth=1)
    vaso_dose_plot.plot(a2-f*s2,'b:', linewidth=1)

    
    vaso_dose_plot.plot([len(a2)/2, len(a2)/2], [0, 1], 'k:')

    
    vaso_dose_plot.set_xlim([1, len(a2)])
    vaso_dose_plot.set_ylim([0, 1])

    t2 -= (t2[-1] - t2[-2]) / 2
    t2 = np.round(t2,2)
    t2 = t2[1:-1:2]
    print(t2)
    vaso_dose_plot.set_xticks(np.arange(0, 2*len(t2), 2))
    vaso_dose_plot.set_xticklabels(t2.tolist())

    for tick in vaso_dose_plot.get_xticklabels():
        tick.set_rotation(90)
    vaso_dose_plot.set_xlabel('Average dose excess per patient')
    vaso_dose_plot.set_ylabel('Mortality')

    vaso_dose_plot.set_title('Vasopressors')


    plt.show()
    


#endregion


#region numberical result

    # VASOPRESSORS
    j = abs((qldata[:, 9] - qldata[:, 11]) / qldata[:, 9]) * 100 # PCT difference btw given and reco  VASOPRESSORS
    qldata[:, 16] = (abs(qldata[:, 13]) <= 0.02) | (j <= 10)   #close dose
    ii = qldata[:, 16] == 1
    # sum[ii]/numel[ii]  # how many received close to optimal dose?
    qldata[ii, 16] = qldata[ii, 16] + 1 # category 2 = dose similar
    ii = (qldata[:, 16] == 0) & (qldata[:, 13] < 0) #less than reco
    qldata[ii, 16] = 1 # category 1
    ii = (qldata[:, 16] == 0) & (qldata[:, 13] > 0) #more than reco
    qldata[ii, 16] = 3 # category 3

    # stats for all 3 categories
    a = []
    for i in range(1, 4):
        ii = qldata[:, 16] == i   #rows in qldata who corresp to this category
        j = qldata[ii, 13]   #dose

        a.append([sum(ii) / len(ii), np.mean(qldata[ii, 12]), np.std(qldata[ii, 12], ddof=1) / np.sqrt(sum(ii)), np.quantile(j, 0.25), np.median(j), np.quantile(j, 0.75)])

    # FLUIDS
    j = abs((qldata[:, 8] - qldata[:, 10]) / qldata[:, 8]) * 100# PCT difference btw given and reco FLUIDS
    qldata[:, 17] = (j <= 10) | (abs(qldata[:, 14]) <= 40)   #close dose [40 ml/4h = 10 ml / h]
    ii = qldata[:, 17] == 1
    # sum[ii]/numel[ii] # how many received close to optimal dose?
    qldata[ii, 17] = qldata[ii, 17] + 1   # category 2 = dose similar
    ii = (qldata[:, 17] == 0) & (qldata[:, 14] < 0) #less than reco
    qldata[ii, 17] = 1  #cat 1
    ii = (qldata[:, 17] == 0) & (qldata[:, 14] > 0) #more than reco
    qldata[ii, 17] = 3   # cat 3


    # stats for all 3 categories
    # a=[];
    for i in range(1, 4):
        ii = qldata[:, 17] == i   #rows in qldata who corresp to this category
        j = qldata[ii, 14] / 4   #dose in ml/h

        a.append([sum(ii) / len(ii), np.mean(qldata[ii, 12]), np.std(qldata[ii, 12]) / np.sqrt(sum(ii)), np.quantile(j, 0.25), np.median(j), np.quantile(j, 0.75)])

    a = np.array(a)

    i = a[0, 0] / (a[0, 0] + a[2, 0])
    ii = a[5, 0] / (a[3, 0] + a[5, 0])
    a = pd.DataFrame(a)
    a.index = ['Vaso: Less than reco', 'Vaso: Similar dose', 'Vaso: More than reco', 'Fluids: Less than reco' ,'Fluids: Similar dose' ,'Fluids: More than reco']
    a.columns = ['fraction','avg_mortality','SEM','Q1_dose','Q2_dose','Q3_dose']

    print(a)
    print(' Among patients who did not receive the recommended dose of vasopressors, fraction of patients who received less than recommended : ', i)

    print(' Among patients who did not receive the recommended dose of IV fluids, fraction of patients who received more than recommended : ', ii)



    ## #########    dose of drugs in the 5 bins      ###############

    iol = MIMICtable.columns.tolist().index('input_4hourly')
    vcl = MIMICtable.columns.tolist().index('max_dose_vaso')

    print(reformat5[:,vcl])

    #boundaries + median doses for each action
    print('VASOPRESSORS')
    for i in range(1, 6):
        print([min(reformat5[vc==i,vcl]) if len(reformat5[vc==i,vcl]) > 0 else np.NaN, np.median(reformat5[vc==i,vcl]), max(reformat5[vc==i,vcl]) if len(reformat5[vc==i,vcl]) > 0 else np.NaN])


    print('IV FLUIDS')
    for i in range(1, 6):
        print([min(reformat5[io==i,iol]) if len(reformat5[io==i,iol]) > 0 else np.NaN, np.median(reformat5[io==i,iol]), max(reformat5[io==i,iol]) if len(reformat5[io==i,iol]) > 0 else np.NaN])
#endregion