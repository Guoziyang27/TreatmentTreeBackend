from google_auth_oauthlib import flow

from google.cloud import bigquery
from google.cloud import bigquery_storage

import pandas as pd
import numpy as np
from tqdm import tqdm

file_list = [
    'culture',
    'microbio',
    'abx',
    'demog',
    'ce',
    'labs_ce',
    'labs_le',
    'uo',
    'preadm_uo',
    'fluid_mv',
    'preadm_fluid',
    'vaso_mv',
    'mechvent'
]

bqclient = None
bqstorageclient = None

def query(file_name):
    sql_dir = 'data_extract_sql/'
    export_dir = 'extracted_data/'

    if file_name != 'ce':

        with open(sql_dir + file_name + '.sql', 'r') as f:
            query = f.read()

        query_result = bqclient.query(query)

        result = np.array([[str(result[i]) for i in range(len(result))] for result in tqdm(query_result, desc=file_name)])
        result[result == 'None'] = ''
        np.savetxt(export_dir + file_name + '.csv', result, delimiter=",", fmt="%s")

        # d = query_result.result().to_dataframe(bqstorage_client=bqstorageclient,
        #         progress_bar_type='tqdm',)
        # d.to_csv(export_dir + file_name + '.csv', index=False)
        # print(result[:10])

    else:

        with open(sql_dir + file_name + '.sql', 'r') as f:
            query = f.read()

        for i in range(0,10000000,1000000):

            query_ = query.format(30000000+i, 31000000+i)

            query_result = bqclient.query(query_)

            result = np.array([[str(result[i]) for i in range(len(result))] for result in tqdm(query_result, desc=file_name+str(i)+str(i+1000000))])
            result[result == 'None'] = ''
            np.savetxt(export_dir + file_name+str(i)+str(i+1000000) + '.csv', result, delimiter=",", fmt="%s")

            # d = query_result.result().to_dataframe(bqstorage_client=bqstorageclient,
            #     progress_bar_type='tqdm',)
            # d.to_csv(export_dir + file_name + str(i)+str(i+1000000) + '.csv', index=False)
            # print(result[:10])

    

def main():
    global bqclient
    launch_browser = True

    appflow = flow.InstalledAppFlow.from_client_secrets_file(
        "./client_secret.json", scopes=["https://www.googleapis.com/auth/bigquery"]
    )


    if launch_browser:
        appflow.run_local_server()
    else:
        appflow.run_console()
    project = 'rebuilt-mimic-iv'
    credentials = appflow.credentials
    bqclient = bigquery.Client(project=project, credentials=credentials)
    bqstorageclient = bigquery_storage.BigQueryReadClient(credentials=credentials)

    for i, file_name in enumerate(file_list):
        query(file_name)
        print(str(i + 1) + '/' + str(len(file_list)))

if __name__ == '__main__':
    main()