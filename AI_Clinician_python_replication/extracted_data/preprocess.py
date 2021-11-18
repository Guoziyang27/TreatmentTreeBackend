import numpy as np
import pandas as pd
import scipy.io as sio
import fileinput
from tqdm import tqdm
from datetime import datetime
import json

item_id_file_list = ['labs_le.csv', 'labs_ce.csv', 
'ce01000000.csv',
# 'ce10000002000000.csv',
# 'ce20000003000000.csv',
# 'ce30000004000000.csv',
# 'ce40000005000000.csv',
# 'ce50000006000000.csv',
# 'ce60000007000000.csv',
# 'ce70000008000000.csv',
# 'ce80000009000000.csv',
# 'ce900000010000000.csv',
]

date_epoch_file_list = ['abx.csv',
'culture.csv',
'microbio.csv',
'demog.csv',
'ce01000000.csv',
'ce10000002000000.csv',
'ce20000003000000.csv',
'ce30000004000000.csv',
'ce40000005000000.csv',
'ce50000006000000.csv',
'ce60000007000000.csv',
'ce70000008000000.csv',
'ce80000009000000.csv',
'ce900000010000000.csv',
'labs_ce.csv',
'labs_le.csv',
'mechvent.csv',
'preadm_fluid.csv',
'fluid_mv.csv',
'vaso_mv.csv',
'preadm_uo.csv',
'uo.csv']

mat_name = {'labs_le.csv': 'Reflabs', 'labs_ce.csv': 'Reflabs', 'ce.csv': 'Refvitals'}

if __name__ == '__main__':

    referenceMatrices = sio.loadmat('../reference_matrices_add_mimiciv.mat')
    with open('itemid.json', 'r') as f:
        itemid_lists = json.loads(f.read())

    def ismember(A, B):
        return np.array([ int(B in a) for a in A ]).argmax() + 1
    
    # def datetimeToEpoch(dateString):
    #     if not isinstance(dateString, str):
    #         return dateString
    #     striptimeFormatString = '%Y-%m-%d %H:%M:%S'
    #     return datetime.strptime(dateString, striptimeFormatString).timestamp()
    

    # def dateToEpoch(dateString):
    #     striptimeFormatString = '%Y-%m-%d'
    #     return datetime.strptime(dateString, striptimeFormatString).timestamp()

    print('simplify the itemid')
    for file_name in item_id_file_list:
        with fileinput.FileInput(file_name, inplace=True, backup='.bak') as f:
            for line in tqdm(f, desc=file_name):

                file_name_index = file_name if file_name in itemid_lists else 'ce.csv'
                itemid_list = itemid_lists[file_name_index]
                for itemid in itemid_list:
                    if line.find(',' + str(itemid) + ',') != -1:
                        print(line.replace(',' + str(itemid) + ',', ',' + str(ismember(referenceMatrices[mat_name[file_name_index]], itemid)) + ','), end='')
                
    # print('transform date/time data to epoch')
    # for file_name in date_epoch_file_list:
    #     table = pd.read_csv(file_name)
    #     table.to_csv(file_name + '.bak')
    #     time_cols = [col for col in table.columns if col.endswith('time') or col.endswith('date')]
    #     date_cols = [col for col in table.columns if col == 'dod']
    #     for i in tqdm(range(len(table)), desc=file_name):
    #         for col in date_cols:
    #             table.loc[i, col] = dateToEpoch(table.loc[i, col])
    #         for col in time_cols:
    #             table.loc[i, col] = datetimeToEpoch(table.loc[i, col])
    #     table.to_csv(file_name)



        