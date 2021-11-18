import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from tqdm import tqdm

def deloutabove( reformat, var, thres ):

    ii = reformat.index[reformat[var] > thres]
    reformat.loc[ii, var] = np.NaN
    return reformat

def deloutbelow( reformat,var,thres ):

    ii = reformat.index[reformat[var] < thres]
    reformat.loc[ii, var] = np.NaN
    return reformat

def SAH(temp, vitalslab_hold):

    # Matthieu Komorowski - Imperial College London 2017 
    # will copy a value in the rows below if the missing values are within the
    # hold period for this variable (e.g. 48h for weight, 2h for HR...)
    # vitalslab_hold = 2x55 cell (with row1 = strings of names ; row 2 = hold time)

    hold = vitalslab_hold[1]
    temp_np = temp.to_numpy()
    nrow = temp_np.shape[0]
    ncol = temp_np.shape[1]

    lastcharttime = np.zeros(ncol)
    lastvalue = np.zeros(ncol)
    oldstayid = temp_np[0, 1]

    print('getting SAH')

    for i in range(3, ncol):

        for j in tqdm(range(nrow), desc=f'{i}/{ncol-1} it'):
            
    
            if oldstayid != temp_np[j, 1]:
                lastcharttime[:] = 0
                lastvalue[:] = 0
                oldstayid = temp_np[j, 1]
            
                    
            if not np.isnan(temp_np[j, i]):
                lastcharttime[i] = temp_np[j, 2]
                lastvalue[i] = temp_np[j, i]
            
            
            if j > 0:
                if np.isnan(temp_np[j, i]) and temp_np[j, 1] == oldstayid and temp_np[j, 2] - lastcharttime[i] <= hold[i-3][0][0] * 3600: #note : hold has 53 cols, temp_np has 55
                    temp_np[j, i] = lastvalue[i]
    temp.loc[:,:] = temp_np
    return temp

def fixgaps(x):
# FIXGAPS Linearly interpolates gaps in a time series
# YOUT=FIXGAPS(YIN) linearly interpolates over NaN
# in the input time series (may be complex), but ignores
# trailing and leading NaN.
#

# R. Pawlowicz 6/Nov/99

    y = x

    bd = np.isnan(x)
    gd = x.index[~bd]

    bd.loc[1:(min(gd)-1)] = 0
    bd.loc[(max(gd)+1):] = 0
    y[bd] = interp1d(gd,x(gd))(x.index(bd).tolist())
    return y

