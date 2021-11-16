# Backend of TreatmentTree

## Install

   Install from `requirements.txt`
   ```
   python -m pip install -r requirements.txt
   ```

## Prepare MIMIC-IV Data

Get access to MIMIC-IV from https://physionet.org/content/mimiciv/1.0/. Then select "Request access using Google BigQuery".

<img src="./assets/access_to_mimiciv.png"/>

Setting up your client credentials if needed. Here is a guide for using client credential to authenticate the API:

https://cloud.google.com/bigquery/docs/authentication/end-user-installed

<img src="./assets/manually_creating_credentials.png"/>

Put `client_secrets.json` under `pre-process/`.

Extract data and preprocess it.

```shell
cd pre-process/
mkdir extracted_data/
python data_extract_MIMICIV.py
cd extracted_data/
python preprocess.py
```

## Train Models

### AI Clinician

```shell
cd pre-process/
python main.py
```

### Patient's State Predictor



## Start Backend

```
python app.py
```

This will run a backend on port 5000 of your localhost.