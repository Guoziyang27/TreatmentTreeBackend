import copy
import json
import sys
from numpy import number
import pandas as pd
import numpy as np
from functools import partial
import random

from flask import Blueprint, request

from icecream import ic
import os
from sklearn.manifold import MDS

from models.ai_clinician import get_instance as get_ai_clinician_instance, AI_Clinician
from models.record_predictor import get_instance as get_state_predictor_instance

records = Blueprint("records", __name__)

records_table = None


@records.route('/get_index', methods=['GET'])
def get_records_index():
    global records_table
    if records_table is None:
        records_table = get_ai_clinician_instance().MIMICtable
        records_table = records_table.loc[~get_ai_clinician_instance().train, :].reset_index(drop=True)

    start_bloc = np.where(records_table['bloc'] == 1)[0].tolist()

    # print(start_bloc)
    # print(records_table)
    # print(records_table.loc[1:20, :])

    return {'succeed': True, 'records_index': [{
        'start_pos': pos,
        'gender': 'Male' if records_table.loc[pos, 'gender'] else 'Female',
        'age': records_table.loc[pos, 'age'],
        'weight': records_table.loc[pos, 'Weight_kg']
    } for pos in start_bloc]}


@records.route('/filter_init', methods=['GET'])
def get_filtr_init():
    global records_table
    if records_table is None:
        records_table = get_ai_clinician_instance().MIMICtable
        records_table = records_table.loc[~get_ai_clinician_instance().train, :].reset_index(drop=True)

    start_bloc = np.where(records_table['bloc'] == 1)[0].tolist()
    lengths = [start_bloc[i + 1] - start_bloc[i] if i < len(start_bloc) - 1 else len(records_table) - start_bloc[i] for
               i in range(len(start_bloc))]

    return {'succeed': True, 'filter_index': {
        'gender': [0, 1],
        'age': [records_table.age.min(), records_table.age.max()],
        'weight': [records_table.Weight_kg.min(), records_table.Weight_kg.max()],
        'length': [min(lengths), max(lengths)]
    }}


filter_records_index = None


@records.route('/filter', methods=['GET'])
def get_filter_records():
    gender = int(request.args.get('gender'))
    age = list(map(float, request.args.get('age').split(',')))
    print(age)
    weight = list(map(float, request.args.get('weight').replace("%2C", ",").split(',')))
    print(weight)
    length = list(map(float, request.args.get('length').replace("%2C", ",").split(',')))

    random_limit = int(request.args.get('random'))

    global records_table
    if records_table is None:
        records_table = get_ai_clinician_instance().MIMICtable
        records_table = records_table.loc[~get_ai_clinician_instance().train, :].reset_index(drop=True)

    start_bloc = np.where(records_table['bloc'] == 1)[0].tolist()

    global filter_records_index
    filter_records_index = [
        i for i, pos in enumerate(start_bloc) if records_table.loc[pos, 'gender'] == gender \
                                                 and age[0] <= records_table.loc[pos, 'age'] <= age[1] \
                                                 and weight[0] <= records_table.loc[pos, 'Weight_kg'] <= weight[1] \
                                                 and length[0] <= (start_bloc[i + 1] - start_bloc[i]
                                                                   if (i < len(start_bloc) - 1)
                                                                   else len(records_table) - start_bloc[i]) <= length[1]
    ]

    filter_sample = random.sample(filter_records_index,
                                  random_limit if random_limit < len(filter_records_index) else len(
                                      filter_records_index))

    return {'succeed': True, 'records_index': filter_sample}


@records.route('/filter_refresh', methods=['GET'])
def get_filter_refresh():
    random_limit = int(request.args.get('random'))

    global filter_records_index
    if filter_records_index is None:
        return {'succeed': False, 'info': 'No initial filter index pool built.'}

    filter_sample = random.sample(filter_records_index,
                                  random_limit if random_limit < len(filter_records_index) else len(
                                      filter_records_index))

    return {'succeed': True, 'records_index': filter_sample}


@records.route('/init', methods=['GET'])
def get_init():
    restore_filename = './restore/' + 'records_state0' + '.json'
    if os.path.isfile(restore_filename):
        with open(restore_filename, 'r') as f:
            return json.loads(f.read())
    global records_table
    if records_table is None:
        records_table = get_ai_clinician_instance().MIMICtable
        records_table = records_table.loc[~get_ai_clinician_instance().train, :].reset_index(drop=True)

    start_bloc = np.where(records_table['bloc'] == 1)[0].tolist()

    ai_clinician_instance = get_ai_clinician_instance()

    records_state = [int(ai_clinician_instance.cluster_state(records_table.loc[i:i, :])) for i in
                     range(len(records_table))]

    records_action = []
    for i in range(len(records_table)):
        vc_action = records_table.loc[i, 'max_dose_vaso']
        io_action = records_table.loc[i, 'input_4hourly']

        vc_action = [i for i in range(len(ai_clinician_instance.vc_bins)) if
                     ai_clinician_instance.vc_bins[i][0] <= vc_action and vc_action <= ai_clinician_instance.vc_bins[i][
                         2]][0]
        io_action = [i for i in range(len(ai_clinician_instance.io_bins)) if
                     ai_clinician_instance.io_bins[i][0] <= io_action and io_action <= ai_clinician_instance.io_bins[i][
                         2]][0]

        records_action.append(io_action * 5 + vc_action)

    records_status = [ai_clinician_instance.state_statics(state) for state in records_state]

    result = {'succeed': True, 'records': [{
        'states': records_state[pos:start_bloc[i + 1]] \
            if i != len(start_bloc) - 1 else \
            records_state[pos:],
        'actions': records_action[pos:start_bloc[i + 1]] \
            if i != len(start_bloc) - 1 else \
            records_action[pos:],
        'status': records_status[pos:start_bloc[i + 1]] \
            if i != len(start_bloc) - 1 else \
            records_status[pos:],
    } for i, pos in enumerate(start_bloc)]}

    with open(restore_filename, 'w') as f:
        f.write(json.dumps(result))
    return result


@records.route('/get_projection', methods=['GET'])
def get_records_projection():
    restore_filename = './restore/' + 'projection0' + '.json'
    if os.path.isfile(restore_filename):
        with open(restore_filename, 'r') as f:
            return json.loads(f.read())
    global records_table
    ai_clinician_instance = get_ai_clinician_instance()
    if records_table is None:
        records_table = ai_clinician_instance.MIMICtable
        records_table = records_table.loc[~ai_clinician_instance.train, :].reset_index(drop=True)

    start_bloc = np.where(records_table['bloc'] == 1)[0].tolist()

    columns = AI_Clinician.colbin + AI_Clinician.collog + AI_Clinician.colnorm + ['mortality_90d']

    print('cp 1')

    records_zs = np.hstack(
        ai_clinician_instance.trans_zs(records_table) + [records_table.loc[:, ['mortality_90d']] - 0.5])
    print(records_zs.shape)

    # for i, pos in enumerate(start_bloc):
    #     print(pos, i)
    #     if i != len(start_bloc):
    #         records_zs[pos:start_bloc[i + 1], :]
    #     else:
    #         records_zs[pos:, :]

    records_zs = [records_zs[pos:start_bloc[i + 1], :] \
                      if i != len(start_bloc) - 1 else \
                      records_zs[pos:, :] for i, pos in enumerate(start_bloc)]
    print('cp 2')

    max_records_len = max([len(record) for record in records_zs])

    target_len = max_records_len // 5
    print(target_len)

    records_for_mds = []
    print('cp 3')

    for record_zs in records_zs:
        record_for_mds = np.empty([target_len, record_zs.shape[1]])
        if len(record_zs) <= target_len:
            target_pos = np.array_split(range(target_len), len(record_zs))
            for i, rang in enumerate(target_pos):
                record_for_mds[rang] = np.tile(record_zs[i], (len(rang), 1))
            records_for_mds.append(record_for_mds)
        else:
            target_pos = np.array_split(range(len(record_zs)), target_len)
            for i, rang in enumerate(target_pos):
                record_for_mds[i] = np.mean(record_zs[rang], axis=0)
            records_for_mds.append(record_for_mds)

    print('cp 4')

    records_for_mds = np.array(records_for_mds)
    records_for_mds[np.isnan(records_for_mds)] = 0

    mds = MDS(n_components=2, verbose=1)
    X = records_for_mds.reshape((len(records_for_mds), -1))
    print(X, len(X))
    Y = mds.fit_transform(X)

    with open(restore_filename, 'w') as f:
        f.write(json.dumps({'succeed': True, 'records_index': [Y[i].tolist() for i in range(len(start_bloc))]}))

    return {'succeed': True, 'points': [Y[i].tolist() for i in range(len(start_bloc))]}


@records.route('/state_pred_record', methods=['GET'])
def get_state_pred_record():
    def check_int(s):
        if s[0] in ('-', '+'):
            return s[1:].isdigit()
        return s.isdigit()

    ai_clinician = get_ai_clinician_instance()
    state_id = request.args.get('state_id')

    if check_int(state_id):
        record = ai_clinician.predict_record(int(state_id)).to_numpy().tolist()
        status = ai_clinician.state_statics(int(state_id))
        return {'succeed': True, 'record': record, 'status': status}
    else:
        state_id_list = list(map(int, state_id.split(',')))
        records_ = []
        status_ = []
        for state_id in state_id_list:
            record = ai_clinician.predict_record(state_id).to_numpy().tolist()
            status = ai_clinician.state_statics(state_id)
            records_.append(record)
            status_.append(status)
        return {'succeed': True, 'record': records_, 'status': status_}


@records.route('/detail_index', methods=['GET'])
def get_record_sumerization_index():
    global records_table
    if records_table is None:
        records_table = get_ai_clinician_instance().MIMICtable
        records_table = records_table.loc[~get_ai_clinician_instance().train, :].reset_index(drop=True)

    value_records = records_table.loc[:, AI_Clinician.colbin + AI_Clinician.colnorm + AI_Clinician.collog]

    with open('models/data/colindex.json', 'r') as f:
        normal_index = json.loads(f.read())['data']

    # normal_index = [{
    #     'column_name': 'PaO2_FiO2',
    #     'column_id': value_records.columns.tolist().index('PaO2_FiO2'),
    #     'unit': 'mmHg',
    #     'type': 'bin',
    #     'bins': [200, 400],
    #     'range': [0.0, 2342.857142857143],
    #     'labels': ['Severe', 'Moderate', 'Mild'],
    #     'colors': [0, 1, 2]
    # }, {
    #     'column_name': 'SysBP',
    #     'column_id': value_records.columns.tolist().index('SysBP'),
    #     'unit': 'mmHg',
    #     'type': 'bin',
    #     'bins': [80, 120],
    #     'range': [0, 340.5],
    #     'labels': ['Normal', 'Prehypertension', 'Hypertension'],
    #     'colors': [2, 1, 0]
    # }, {
    #     'column_name': 'GCS',
    #     'column_id': value_records.columns.tolist().index('GCS'),
    #     'unit': '',
    #     'type': 'bin',
    #     'bins': [13, 15],
    #     'range': [0, 15.0],
    #     'labels': ['Severe', 'Moderate', 'Mild'],
    #     'colors': [0, 1, 2]
    # }, {
    #     'column_name': 'WBC_count',
    #     'column_id': value_records.columns.tolist().index('WBC_count'),
    #     'unit': '/L',
    #     'type': 'range',
    #     'bins': [4.5, 11],
    #     'range': [0.1, 50],
    #     'labels': ['Low', 'Normal', 'High'],
    #     'colors': [4, 5, 6],
    # }, {
    #     'column_name': 'Shock_Index',
    #     'column_id': value_records.columns.tolist().index('Shock_Index'),
    #     'unit': '',
    #     'type': 'range',
    #     'bins': [0.5, 0.7],
    #     'range': [0, 2],
    #     'labels': ['Low', 'Normal', 'High'],
    #     'colors': [4, 5, 6],
    # }, {
    #     'column_name': 'Arterial_BE',
    #     'column_id': value_records.columns.tolist().index('Arterial_BE'),
    #     'unit': 'mEq/L',
    #     'type': 'range',
    #     'bins': [-2, 2],
    #     'range': [-50.0, 100.0],
    #     'labels': ['Low', 'Normal', 'High'],
    #     'colors': [4, 5, 6],
    # }, {
    #     'column_name': 'Creatinine',
    #     'column_id': value_records.columns.tolist().index('Creatinine'),
    #     'unit': 'mg/dL',
    #     'type': 'range',
    #     'bins': [0.6, 1.2],
    #     'range': [0.0, 5.0],
    #     'labels': ['Low', 'Normal', 'High'],
    #     'colors': [4, 5, 6],
    # }, {
    #     'column_name': 'Calcium',
    #     'column_id': value_records.columns.tolist().index('Calcium'),
    #     'unit': 'mg/dL',
    #     'type': 'range',
    #     'bins': [8.3, 10.3],
    #     'range': [0.0, 14.6],
    #     'labels': ['Low', 'Normal', 'High'],
    #     'colors': [4, 5, 6],
    # }, {
    #     'column_name': 'Platelets_count',
    #     'column_id': value_records.columns.tolist().index('Platelets_count'),
    #     'unit': 'K/uL',
    #     'type': 'range',
    #     'bins': [150, 450],
    #     'range': [5.0, 1220.0],
    #     'labels': ['Low', 'Normal', 'High'],
    #     'colors': [4, 5, 6],
    # }, {
    #     'column_name': 'input4hourly',
    #     'column_id': value_records.columns.tolist().index('input4hourly'),
    #     'unit': 'mL',
    #     'type': 'range',
    #     'bins': [],
    #     'range': [0, 600.0],
    #     'labels': [],
    # }, {
    #     'column_name': 'output4hourly',
    #     'column_id': value_records.columns.tolist().index('output4hourly'),
    #     'unit': 'mL',
    #     'type': 'range',
    #     'bins': [],
    #     'range': [0, 600.0],
    #     'labels': [],
    # }]

    return {'succeed': True, 'details_index': normal_index}
