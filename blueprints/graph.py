import copy
import json
import sys
from numpy import number
import pandas as pd
import numpy as np
from functools import partial

from flask import Blueprint, request

from icecream import ic
import os

# sys.path.append("..")
from models.ai_clinician import get_instance as get_ai_clinician_instance
from models.ai_clinician import AI_Clinician
from models.load_state_autoencoder import get_instance as get_state_predictor_instance
from models.load_state_autoencoder import RecordPredictor
from models.submodels.AE_model import AE

graph_bp = Blueprint("graph", __name__)

save_state = {'graph': None, 'ai_clinician': None, 'state_predictor': None, 'node_id_top': 0}


def init_graph_one_node(init_node, init_record, ai, state_predictor, MAX_LAYER_NUM, lim_action, node_id_top, layer_id, prev_records, prev_actions, real_action, branch_root):

    record = init_record
    graph = []
    current_node_id = init_node['node_id']
    current_record = record

    current_state = ai.cluster_state(current_record)

    graph.append(init_node)

    end_states = [AI_Clinician.ncl + 1, AI_Clinician.ncl]


    predict_action_fn = partial(ai.predict_action, n_branch=lim_action)
    # predict_state_fn = partial(ai.predict_state, n_branch=lim_state)

    def find_node_by_id(Id):
        return [x for x in graph if x['node_id'] == Id]
    
    branch_id = 0
    branch_layer = 0
    
    is_first = True
    while layer_id < MAX_LAYER_NUM:
        layer_id += 1

        # actions, actions_poss = predict_action_fn(state_idx=current_state)
        current_node = find_node_by_id(current_node_id)[0]

        # current_node['actions'] += [{
        #         'action': int(actions[i]),
        #         'possibility': float(actions_poss[i]),
        #         'open': True if i == 0 else False,
        #         'next_nodes': [],
        #     } for i in range(len(actions)) if actions_poss[i] > 0]

        if len(current_node['actions']) == 0:
            break
        best_action = current_node['actions'][0]['action']
        prev_records = pd.concat([prev_records, current_record], ignore_index=True)
        prev_actions.append(best_action)
        
        new_record = state_predictor.predict(prev_records, prev_actions)



        current_state = ai.cluster_state(new_record)

        branch_id += lim_action << branch_layer
        branch_layer += 1

        next_actions, next_actions_poss = predict_action_fn(state_idx=current_state)

        graph.append({
            'node_id': node_id_top,
            'layer_id': layer_id,
            'branch_id': branch_id,
            'branch_root': branch_root,
            'open': True if layer_id != MAX_LAYER_NUM else False,
            'mortality': ai.state_statics(current_state)['mortality'],
            'prev_node': current_node_id,
            'prev_action': int(best_action),
            'record': new_record.to_numpy().tolist(),
            'actions': [{
                'action': int(next_actions[i]),
                'possibility': float(next_actions_poss[i]),
                'open': True if i == 0 else False,
                'next_nodes': [],
            } for i in range(len(next_actions)) if next_actions_poss[i] > 0]
        })


        current_node['actions'][0]['next_nodes'].append(node_id_top)

        node_id_top += 1

        current_node_id = node_id_top - 1
        current_record = new_record
    
    graph.pop(0)

    return graph, node_id_top

def load_AI_clinician():
    global save_state
    if save_state['ai_clinician'] is None:
        save_state['ai_clinician'] = get_ai_clinician_instance()
    return save_state['ai_clinician']

def load_state_autoencoder():
    global save_state
    if save_state['state_predictor'] is None:
        save_state['state_predictor'] = get_state_predictor_instance()
    return save_state['state_predictor']

@graph_bp.route('/init', methods=['GET'])
def get_initial_graph():

    global save_state
    restore_filename = './restore/' + request.args.get('record')+'.json'
    if os.path.isfile(restore_filename):
        with open(restore_filename, 'r') as f:
            res = json.loads(f.read())
            save_state['graph'] = res['graph']
            save_state['node_id_top'] = len(res['graph'])

            return res

    ai = load_AI_clinician()
    state_predictor = load_state_autoencoder()

    records_table = ai.MIMICtable.loc[~ai.train, :].reset_index(drop=True)


    start_bloc = np.where(records_table['bloc'] == 1)[0].tolist()


    # ic.disable()
    lim_action = int(request.args.get('actionslimit'))
    # lim_state = int(request.args.get('stateslimit'))

    # columns = request.args.get('columns').split(',')
    record_raw = int(request.args.get('record'))
    # record_raw = request.args.get('record').split(';')
    # record_raw = [r.split(',') for r in record_raw]

    records = records_table.loc[start_bloc[record_raw]:start_bloc[record_raw + 1] - 1, :].reset_index(drop=True)
    print(records_table, records,start_bloc[record_raw])

    MAX_LAYER_NUM = len(records) - 1

    # records = pd.DataFrame([map(float, r) for r in record_raw], columns=columns)



    graph = []
    node_id_top = 0
    layer_id = 0

    prev_records = pd.DataFrame([], columns=records.columns)
    prev_actions = []

    last_real_chain_id = -1
    last_real_chain_action = -1
    predict_action_fn = partial(ai.predict_action, n_branch=lim_action)
    

    for i in range(len(records)):
        current_state = ai.cluster_state(records.loc[i:i, :])
        # ic(current_state)
        # print(records)
        # print(records.loc[i, :])

        vc_action = records.loc[i, 'max_dose_vaso']
        io_action = records.loc[i, 'input_4hourly']

        vc_action = [i for i in range(len(ai.vc_bins)) if ai.vc_bins[i][0] <= vc_action and vc_action <= ai.vc_bins[i][2]][0]
        io_action = [i for i in range(len(ai.io_bins)) if ai.io_bins[i][0] <= io_action and io_action <= ai.io_bins[i][2]][0]

        actions, actions_poss = predict_action_fn(state_idx=current_state)

        if io_action * 5 + vc_action in actions.tolist():
            del_idx = actions.tolist().index(io_action * 5 + vc_action)
            actions = np.array(actions.tolist()[:del_idx] + actions.tolist()[del_idx + 1:])
            actions_poss = np.array(actions_poss.tolist()[:del_idx] + actions_poss.tolist()[del_idx + 1:])

        graph.append({
            'node_id': node_id_top,
            'layer_id': layer_id,
            'branch_id': 0,
            'branch_root': node_id_top,
            'real_record': True,
            'open': True,
            'mortality': ai.state_statics(current_state)['mortality'],
            'prev_node': last_real_chain_id if last_real_chain_id != -1 else 'no_prev',
            'prev_action': last_real_chain_action if last_real_chain_action != -1 else 'no_prev',
            'record': records.loc[i:i, RecordPredictor.colbin + RecordPredictor.colnorm + RecordPredictor.collog].to_numpy().tolist(),
            'actions': [{
                'action': int(actions[i]),
                'possibility': float(actions_poss[i]),
                'open': True if i == 0 else False,
                'next_nodes': [],
            } for i in range(len(actions)) if actions_poss[i] > 0]
        })

        mark_pos = node_id_top

        node_id_top += 1

        new_graph, node_id_top = init_graph_one_node(graph[len(graph) - 1], records.loc[i:i, :], ai, state_predictor, MAX_LAYER_NUM, lim_action, node_id_top, layer_id, copy.deepcopy(prev_records), copy.deepcopy(prev_actions), io_action * 5 + vc_action, node_id_top-1)

        graph += new_graph

        real_action = {
            'action': io_action * 5 + vc_action,
            'possibility': 0,
            'real_action': True,
            'open': True,
            'next_nodes': [node_id_top] if i < len(records) -1 else []
        }
        graph[mark_pos]['actions'] = [real_action] + graph[mark_pos]['actions']

        
        prev_records = pd.concat([prev_records, records.loc[i:i, :]], ignore_index=True)
        prev_actions.append(io_action * 5 + vc_action)

        last_real_chain_id = mark_pos
        last_real_chain_action = io_action * 5 + vc_action
        layer_id += 1

        # if i == len(records) - 1:
        #     graph.append({
        #         'node_id': node_id_top,
        #         'layer_id': layer_id,
        #         'branch_id': 0,
        #         'branch_root': node_id_top,
        #         'real_record': True,
        #         'open': False,
        #         'mortality': records.loc[i, 'mortality_90d'],
        #         'prev_node': last_real_chain_id if last_real_chain_id != -1 else 'no_prev',
        #         'prev_action': last_real_chain_action if last_real_chain_action != -1 else 'no_prev',
        #         'record': None,
        #         'actions': []
        #     })
        #     node_id_top += 1
    
    graph.sort(key=lambda node: (-node['branch_root'], node['branch_id']))
    graph.sort(key=lambda node: node['layer_id'])


    save_state['graph'] = graph
    save_state['node_id_top'] = node_id_top

    with open(restore_filename, 'w') as f:
        f.write(json.dumps({'succeed': True, 'graph': graph}))

    return {'succeed': True, 'graph': graph}

@graph_bp.route('/expand', methods=['GET'])
def get_one_node():
    lim_action = int(request.args.get('actionslimit'))

    expand_node_id = int(request.args.get('expandnodeid'))

    expand_node_action_id = int(request.args.get('expandnodeactionid'))

    if save_state['graph'] is None:
        return {'succeed': False, 'info': 'graph need to be initialized first.'}

    graph = save_state['graph']

    node_id_top = save_state['node_id_top']

    def find_node_by_id(Id):
        return [x for x in graph if x['node_id'] == Id]


    node = find_node_by_id(expand_node_id)
    if len(node) == 0:
        return {'succeed': False, 'info': 'no node with this id.'}
    node = node[0]

    if node['actions'][expand_node_action_id]['open']:
        return {'succeed': False, 'info': 'this node has been expanded.'}


    ai = load_AI_clinician()
    state_predictor = load_state_autoencoder()

    records_table = ai.MIMICtable.loc[~ai.train, :].reset_index(drop=True)


    prev_records = pd.DataFrame([], columns=records_table.columns)
    prev_actions = []

    cur_node = node
    cur_next_action = node['actions'][expand_node_action_id]['action']



    while(True):
        prev_records = pd.concat([pd.DataFrame(cur_node['record'], columns=RecordPredictor.colbin + RecordPredictor.colnorm + RecordPredictor.collog), prev_records], ignore_index=True)
        prev_actions.append(cur_next_action)

        if cur_node['prev_node'] == 'no_prev':
            break

        cur_next_action = cur_node['prev_action']
        cur_node = find_node_by_id(cur_node['prev_node'])[0]



    new_record = state_predictor.predict(prev_records, prev_actions)

    current_state = ai.cluster_state(new_record)

    graph.append({
        'node_id': node_id_top,
        'layer_id': node['layer_id'] + 1,
        'branch_id': node['branch_id'] * lim_action + 1 + expand_node_action_id,
        'branch_root': node['branch_root'],
        'open': True,
        'new': True,
        'mortality': ai.state_statics(current_state)['mortality'],
        'prev_node': expand_node_id,
        'prev_action': node['actions'][expand_node_action_id]['action'],
        'record': new_record.to_numpy().tolist(),
        'actions': []
    })
    node['actions'][expand_node_action_id]['next_nodes'].append(node_id_top)
    node['actions'][expand_node_action_id]['open'] = True
    node_id_top += 1

    predict_action_fn = partial(ai.predict_action, n_branch=lim_action)

    actions, actions_poss = predict_action_fn(state_idx=current_state)

    graph[len(graph) - 1]['actions'] += [{
            'action': int(actions[i]),
            'possibility': float(actions_poss[i]),
            'open': False,
            'new': True,
            'next_nodes': [],
        } for i in range(len(actions)) if actions_poss[i] > 0]

    graph.sort(key=lambda node: (-node['branch_root'], node['branch_id']))
    graph.sort(key=lambda node: node['layer_id'])

    save_state['graph'] = [{i:node[i] for i in node if i!='new'} for node in graph]
    save_state['node_id_top'] = node_id_top

    return {'succeed': True, 'graph': graph}


def delete_node(graph, node_id):
    def find_node_by_id(Id):
        return [x for x in graph if x['node_id'] == Id]
    node = find_node_by_id(node_id)[0]
    next_node_id = [action['next_nodes'] for action in node['actions']]
    node_idx = [i for i in range(len(graph)) if graph[i]['node_id'] == node_id][0]
    graph = graph[:node_idx] + graph[node_idx + 1:]
    for next_nodes in next_node_id:
        for next_node in next_nodes:
            graph = delete_node(graph, next_node)
    return graph

@graph_bp.route('/unexpand', methods=['GET'])
def get_unexpand_one_node():
    lim_action = int(request.args.get('actionslimit'))

    expand_node_id = int(request.args.get('expandnodeid'))

    expand_node_action_id = int(request.args.get('expandnodeactionid'))


    if save_state['graph'] is None:
        return {'succeed': False, 'info': 'graph need to be initialized first.'}
    graph = save_state['graph']

    def find_node_by_id(Id):
        return [x for x in graph if x['node_id'] == Id]

    node = find_node_by_id(expand_node_id)
    if len(node) == 0:
        return {'succeed': False, 'info': 'no node with this id.'}
    node = node[0]


    if not node['actions'][expand_node_action_id]['open']:
        return {'succeed': False, 'info': 'action need to be open.'}

    if len(node['actions'][expand_node_action_id]['next_nodes']) == 0:
        return {'succeed': True, 'graph': graph}

    for next_node in node['actions'][expand_node_action_id]['next_nodes']:
        graph = delete_node(graph, next_node)

    node['actions'][expand_node_action_id]['next_nodes'] = []
    node['actions'][expand_node_action_id]['open'] = False

    save_state['graph'] = graph
    return {'succeed': True, 'graph': graph}




@graph_bp.route('/show_actions', methods=['GET'])
def get_actions():
    lim_action = int(request.args.get('actionslimit'))

    expand_node_id = int(request.args.get('expandnodeid'))

    if save_state['graph'] is None:
        return {'succeed': False, 'info': 'graph need to be initialized first.'}

    graph = save_state['graph']

    def find_node_by_id(Id):
        return [x for x in graph if x['node_id'] == Id]
    
    node = find_node_by_id(expand_node_id)
    if len(node) == 0:
        return {'succeed': False, 'info': 'no node with this id.'}
    node = node[0]

    if node['open']:
        return {'succeed': False, 'info': 'this node has been expanded.'}

    ai = load_AI_clinician()
    state_predictor = load_state_autoencoder()

    current_state = ai.cluster_state(pd.DataFrame(node['record'], columns=StateAutoencoder.colbin + StateAutoencoder.colnorm + StateAutoencoder.collog))
    predict_action_fn = partial(ai.predict_action, n_branch=lim_action)

    actions, actions_poss = predict_action_fn(state_idx=current_state)

    node['actions'] += [{
            'action': int(actions[i]),
            'possibility': float(actions_poss[i]),
            'open': False,
            'new': True,
            'next_nodes': [],
        } for i in range(len(actions)) if actions_poss[i] > 0]
    
    node['open'] = True

    save_state['graph'] = [{i:node[i] for i in node if i!='new'} for node in graph]
    return {'succeed': True, 'graph': graph}


@graph_bp.route('/hide_actions', methods=['GET'])
def get_hide_actions():
    lim_action = int(request.args.get('actionslimit'))

    expand_node_id = int(request.args.get('expandnodeid'))

    if save_state['graph'] is None:
        return {'succeed': False, 'info': 'graph need to be initialized first.'}
    graph = save_state['graph']

    def find_node_by_id(Id):
        return [x for x in graph if x['node_id'] == Id]

    node = find_node_by_id(expand_node_id)
    if len(node) == 0:
        return {'succeed': False, 'info': 'no node with this id.'}
    node = node[0]

    if not node['open']:
        return {'succeed': False, 'info': 'node need to be open.'}


    for action in node['actions']:

        if len(action['next_nodes']) == 0:
            continue

        for next_node in action['next_nodes']:
            graph = delete_node(graph, next_node)

        action['next_nodes'] = []
    
    node['actions'] = []
    node['open'] = False

    save_state['graph'] = graph
    return {'succeed': True, 'graph': graph}
