from models.submodels.RNN_model import RNN
from models.submodels.dataset import Dataset
import torch
from torch.utils import data
import torch.nn as nn
import numpy as np
import os
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

HIDDEN_DIMS = 12
N_LAYERS = 1
IN_FEATURES = 47
IN_WINDOW_SIZE = 4


def get_dataset(ae_model, opt):
    data = Dataset(ae_model, opt)

    data_len = len(data)

    train_dst, valid_dst = torch.utils.data.random_split(
        dataset=data,
        lengths=[int(0.8 * data_len), data_len - int(0.8 * data_len)],
        generator=torch.Generator().manual_seed(0)
    )

    return train_dst, valid_dst, data.max_len, data.mean, data.std


def train_rnn(opt):
    # if not os.path.isfile(opt['ae_path']) or not os.path.isfile(opt['study_pkl']):
    #     train_ae(opt)
    # save_pth = torch.load(opt['ae_path'])
    # ae_study = joblib.load(opt['study_pkl'])
    # ae_model = AE(ae_study.best_trial, ae_study.best_trial.params['latent_n_units'])
    # ae_model.load_state_dict(save_pth['model_state'])

    # model = RNN(ae_model.latent_features, ae_model.latent_features, HIDDEN_DIMS, N_LAYERS)
    ae_model = None
    model = RNN(752, 752, HIDDEN_DIMS, N_LAYERS)

    if os.path.isfile(opt['rnn_path']):
        print('Loading checkpoint ', end='')
        checkpoint = torch.load(opt['rnn_path'])
        model.load_state_dict(checkpoint['model_state'])
        print('[DONE]')

    batch_size = 16

    epoch_num = 150

    criterion = nn.CrossEntropyLoss() #nn.MSELoss()

    optimizer = torch.optim.SGD(params=model.parameters(), lr=0.0001, momentum=0.9)

    best_score = 0

    train_dst, valid_dst, max_seq_len, data_mean, data_std = get_dataset(ae_model, opt)
    train_loader = data.DataLoader(train_dst, batch_size=batch_size, shuffle=True)
    valid_loader = data.DataLoader(valid_dst, batch_size=batch_size, shuffle=True)

    torch.save(data_mean, opt['mean_save'])
    torch.save(data_std, opt['std_save'])

    def save_model(path):
        torch.save({
            'model_state': model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "best_score": best_score
        }, path)
        print(f"Model save as {path}")

    for epoch in range(epoch_num):
        model.train()
        for (input_seq, input_len, target_seq, target_len) in tqdm(train_loader, desc=f'Epoch {epoch}'):
            input_seq, target_seq = input_seq.to(device, dtype=torch.float), target_seq.to(device, dtype=torch.int64)

            _, indices = torch.sort(input_len, descending=True)

            input_seq, input_len, target_seq, target_len = input_seq[indices.tolist()], input_len[indices.tolist()], \
                                                           target_seq[indices.tolist()], target_len[indices.tolist()]
            for i in range(1, input_len.max() + 1):
                # s_i = i - IN_WINDOW_SIZE if i >= IN_WINDOW_SIZE else 0
                input_seq_sub = input_seq[:, :i, :]
                input_len_sub = input_len.clone()

                # nonzero_i = torch.count_nonzero(input_seq_sub, dim=2)[:, 0] != 0
                #
                # input_len_sub = input_len_sub[nonzero_i]
                # target_len_sub = target_len[nonzero_i]
                # input_seq_sub = input_seq_sub[nonzero_i]
                input_len_sub[input_len_sub > i] = i
                # input_pack = pack_padded_sequence(input_seq_sub, input_len_sub, batch_first=True)

                optimizer.zero_grad()
                output, hidden = model(input_seq_sub)
                # output = output[:, -1, :]
                # output, _ = pad_packed_sequence(output, batch_first=True)
                # output = output.reshape(-1)
                output = output.to(device)
                loss = criterion(output[target_len >= i], target_seq[target_len >= i, i - 1])
                loss.backward(retain_graph=True)
                optimizer.step()

        model.eval()

        loss_total = 0
        iter_num = 0
        with torch.no_grad():
            for batch_idx, (input_seq, input_len, target_seq, target_len) in enumerate(valid_loader):
                input_seq, target_seq = input_seq.to(device, dtype=torch.float), target_seq.to(device,
                                                                                               dtype=torch.int64)

                _, indices = torch.sort(input_len, descending=True)

                input_seq, input_len, target_seq, target_len = input_seq[indices.tolist()], input_len[indices.tolist()], \
                                                               target_seq[indices.tolist()], target_len[
                                                                   indices.tolist()]
                for i in range(1, input_len.max() + 1):
                    # s_i = i - IN_WINDOW_SIZE if i >= IN_WINDOW_SIZE else 0
                    input_seq_sub = input_seq[:, :i, :]
                    input_len_sub = input_len.clone()

                    # nonzero_i = torch.count_nonzero(input_seq_sub, dim=2)[:, 0] != 0
                    #
                    # input_len_sub = input_len_sub[nonzero_i]
                    # target_len_sub = target_len[nonzero_i]
                    # input_seq_sub = input_seq_sub[nonzero_i]
                    input_len_sub[input_len_sub > i] = i

                    output, hidden = model(input_seq_sub)
                    # output = output[:, -1, :]
                    # output, _ = pad_packed_sequence(output, batch_first=True)
                    # output = output.reshape(-1)
                    output = output.to(device)
                    loss = criterion(output[target_len >= i], target_seq[target_len >= i, i - 1])

                    loss_total += loss.item()
                    iter_num += 1
        print(loss_total, iter_num, loss_total / iter_num)

    save_model(opt['rnn_path'])

    pass


if __name__ == '__main__':
    opt = {'ae_path': 'ae_models.pth', 'rnn_path': 'rnn_models.pth', 'study_pkl': 'ae_study.pkl',
           'ae_model': 'ae_models.pickle', 'mean_save': 'mean_tensor.pt', 'std_save': 'std_tensor.pt'}

    train_rnn(opt)
