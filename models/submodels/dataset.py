import torch.utils.data as data
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
import joblib
import sys
import torch.nn.functional as F

from models.ai_clinician import get_instance, get_policy_bins

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def move_to_cuda(sample):
    if len(sample) == 0:
        return {}

    def _move_to_cuda(maybe_tensor):
        if torch.is_tensor(maybe_tensor):
            return maybe_tensor.cuda()
        elif isinstance(maybe_tensor, dict):
            return {
                key: _move_to_cuda(value)
                for key, value in maybe_tensor.items()
            }
        elif isinstance(maybe_tensor, list):
            return [_move_to_cuda(x) for x in maybe_tensor]
        else:
            return maybe_tensor

    return _move_to_cuda(sample)


class Dataset(data.Dataset):
    def __init__(self, ae_model, opt):
        # if not os.path.isfile(dump_sequences_path):
        MIMICtable = pd.read_csv('../data/final_table.csv')
        MIMICarray = MIMICtable.to_numpy()

        ai_clinician = get_instance('../data/')

        with open('../data/MIMICzs.npy', 'rb') as f:
            MIMICzs = np.load(f, allow_pickle=True)

        start_bloc = np.where(MIMICarray[:, 0] == 1)[0]

        embedding = []

        _, _, vc_bins, io_bins = get_policy_bins(MIMICtable)

        states = ai_clinician.cluster_state(MIMICtable)

        with torch.no_grad():
            for i in range(len(MIMICzs)):
                # input_features = torch.from_numpy(MIMICzs[i]).to(device, dtype=torch.float)
                # input_features = input_features.to(device, dtype=torch.float)
                # hidden, _ = ae_model.encode(input_features)

                # embedding.append(hidden.reshape([1, -1]))
                # embedding.append(input_features.reshape(1,-1))
                io = [j for j in range(len(io_bins))
                      if io_bins[j][0] <= MIMICtable.loc[i, 'input_4hourly'] <= io_bins[j][2]][0]
                vc = [j for j in range(len(vc_bins))
                      if vc_bins[j][0] <= MIMICtable.loc[i, 'max_dose_vaso'] <= vc_bins[j][2]][0]

                embedding.append(torch.tensor([io, vc]).type(torch.FloatTensor).reshape(1, -1))

        embedding = torch.cat(embedding)
        self.mean = torch.mean(embedding, dim=0)
        self.std = torch.std(embedding, dim=0)
        embedding = (embedding - self.mean) / self.std

        states = torch.tensor(states, dtype=torch.int64)
        states_one_hot = F.one_hot(states)
        embedding = torch.cat([embedding, states_one_hot], dim=1)

        self.sequences = [embedding[start_bloc[i]:start_bloc[i + 1]] \
                              if i < len(start_bloc) - 1 else \
                              embedding[start_bloc[i]:] \
                          for i in range(len(start_bloc))]
        self.sequences = [seq for seq in self.sequences if seq.shape[0] > 1]

        self.states = [states[start_bloc[i]:start_bloc[i + 1]] \
                           if i < len(start_bloc) - 1 else \
                           states[start_bloc[i]:] \
                       for i in range(len(start_bloc))]
        self.states = [state for state in self.states if state.shape[0] > 1]
        #     joblib.dump(self.sequences, dump_sequences_path)
        # else:
        #     self.sequences = joblib.load(dump_sequences_path)

        self.max_len = max([seq.shape[0] for seq in self.sequences])
        self.min_len = min([seq.shape[0] for seq in self.sequences])

    def __getitem__(self, idx):
        input_seq = self.sequences[idx][:-1]
        # target_seq = self.sequences[idx][1:]
        target_seq = self.states[idx][1:]

        return torch.cat((input_seq, torch.zeros(self.max_len - 1 - input_seq.shape[0], input_seq.shape[1]))), \
               input_seq.shape[0], \
               torch.cat((target_seq, torch.zeros(self.max_len - 1 - target_seq.shape[0], dtype=torch.int64))), \
               target_seq.shape[0]

    def __len__(self):
        return len(self.sequences)
