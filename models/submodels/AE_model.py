import joblib
import torch
import torch.nn as nn
import optuna
import torch.utils.data
import numpy as np
import pickle
from torch.nn import functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IN_FEATURES = 47
BATCH_SIZE = 16
EPOCHS = 20
M_N = 0.005


class Dataset(torch.utils.data.Dataset):
    def __init__(self):
        with open('../data/MIMICzs.npy', 'rb') as f:
            MIMICzs = np.load(f, allow_pickle=True)

        self._data = np.array(MIMICzs)

    def __getitem__(self, idx):
        return self._data[idx], self._data[idx]

    def __len__(self):
        return len(self._data)


def get_dataset():
    data = Dataset()

    data_len = len(data)

    train_dst, valid_dst = torch.utils.data.random_split(
        dataset=data,
        lengths=[int(0.8 * data_len), data_len - int(0.8 * data_len)],
        generator=torch.Generator().manual_seed(0)
    )

    return train_dst, valid_dst


class AE(nn.Module):
    def __init__(self, trial, latent_features):
        super().__init__()
        self.n_layers = trial.suggest_int("n_layers", 3, 5)
        # self.decoder_n_layers = trial.suggest_int("dec_n_layers", 1, 5)

        # encoder
        layers = []

        hidden_dims = []

        in_features = IN_FEATURES
        for i in range(self.n_layers):
            out_features = trial.suggest_int("n_units_l{}".format(i), latent_features, in_features)
            layers.append(nn.Linear(in_features, out_features))
            layers.append(nn.ReLU())
            p = trial.suggest_float("enc_dropout_l{}".format(i), 0.2, 0.5)
            layers.append(nn.Dropout(p))

            hidden_dims.append(out_features)

            in_features = out_features

        self.latent_features = latent_features

        self.encoder = nn.Sequential(*layers)

        self.mu = nn.Linear(hidden_dims[-1], latent_features)
        self.log_var = nn.Linear(hidden_dims[-1], latent_features)

        # decoder
        layers = []

        hidden_dims.reverse()

        layers.append(nn.Linear(latent_features, hidden_dims[0]))
        layers.append(nn.ReLU())

        in_features = latent_features

        for i in range(self.n_layers - 1):
            # out_features = trial.suggest_int("dec_n_units_l{}".format(i), in_features, IN_FEATURES)
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            layers.append(nn.ReLU())
            p = trial.suggest_float("dec_dropout_l{}".format(i), 0.2, 0.5)
            layers.append(nn.Dropout(p))

        self.decoder = nn.Sequential(*layers)

        self.final_layer = nn.Sequential(
            nn.Linear(hidden_dims[-1], IN_FEATURES),
            nn.Tanh()
        )

    def encode(self, features):

        result = self.encoder(features)

        mu = self.mu(result)
        log_var = self.log_var(result)

        return mu, log_var

    def decode(self, features):
        result = self.decoder(features)
        return self.final_layer(result)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, features):
        mu, log_var = self.encode(features)
        z = self.reparameterize(mu, log_var)
        return self.decode(z), features, mu, log_var

    def loss_function(self, output, target, mu, log_var):
        recons_loss = F.mse_loss(output, target)

        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)

        loss = recons_loss + M_N * kld_loss

        return {'loss': loss, 'reconstruction_loss': recons_loss, 'kld': kld_loss}


def objective(trial):
    latent_features = trial.suggest_int("latent_n_units", 20, IN_FEATURES)
    model = AE(trial, latent_features).to(device)

    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])
    lr = trial.suggest_loguniform("lr", 1e-7, 1e-5)
    optimizer = getattr(torch.optim, optimizer_name)(model.parameters(), lr=lr)

    criterion = nn.MSELoss()

    train_dst, valid_dst = get_dataset()
    train_loader = torch.utils.data.DataLoader(train_dst, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_dst, batch_size=BATCH_SIZE, shuffle=True)

    for epoch in range(EPOCHS):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device, dtype=torch.float), target.to(device, dtype=torch.float)

            optimizer.zero_grad()
            output, _, mu, log_var = model(data)
            # loss = criterion(output, target)
            loss = model.loss_function(output, target, mu, log_var)['loss']
            loss.backward()
            optimizer.step()

        model.eval()
        loss_total = 0
        iter_num = 0
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(valid_loader):
                data, target = data.to(device, dtype=torch.float), target.to(device, dtype=torch.float)

                output, _, mu, log_var = model(data)
                # loss = criterion(output, target)
                loss = model.loss_function(output, target, mu, log_var)['loss']
                loss_total += loss.item()
                iter_num += 1

        loss_avg = loss_total / iter_num
        print(f'Epoch {epoch + 1}/{EPOCHS}:', loss_avg)
        trial.report(loss_avg, epoch)

        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    trial.set_user_attr(key="best_booster", value={'model': model, 'optimizer': optimizer, 'loss': loss_avg})

    return loss_avg


def callback(study, trial):
    if study.best_trial.number == trial.number:
        study.set_user_attr(key="best_booster", value=trial.user_attrs["best_booster"])


def save_model(path, booster):
    torch.save({
        'model_state': booster['model'].state_dict(),
        "optimizer_state": booster['optimizer'].state_dict(),
        "best_loss": booster['loss']
    }, path)
    print(f"Model save as {path}")


def train_ae(opt):
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=100, timeout=6000, callbacks=[callback])

    trial = study.best_trial

    print("Best trial:")
    print("  Value: ", trial.value)

    best_model = study.user_attrs["best_booster"]

    # with open(opt['ae_model'], "wb") as fout:
    #     pickle.dump(best_model, fout)

    save_model(opt['ae_path'], best_model)
    joblib.dump(study, opt['study_pkl'])


if __name__ == '__main__':
    train_ae({'ae_path': 'ae_models.pth', 'study_pkl': 'ae_study.pkl', 'ae_model': 'ae_models.pickle'})
