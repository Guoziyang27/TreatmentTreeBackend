from dataset import Dataset
import torch
from torch.utils import data
import torch.nn as nn
from model import AE
import numpy as np


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def validate(model, loader):
    with torch.no_grad():
        for i, (prev_rec, end_rec) in enumerate(loader):
            prev_rec = prev_rec.to(device, dtype=torch.float)
            end_rec = end_rec.to(device, dtype=torch.float)

            output = model(prev_rec)
            preds = output.detach().max(dim=1)[1].cpu().numpy()

            targets = end_rec.cpu().numpy()



def get_dataset():
    data = Dataset(forward_num=4)

    data_len = len(data)

    train_dst, test_dst = torch.utils.data.random_split(
        dataset=data,
        lengths=[int(0.8 * data_len), data_len - int(0.8 * data_len)],
        generator=torch.Generator().manual_seed(0)
    )

    return train_dst, test_dst


def main():

    model = AE()

    batch_size = 16

    epoch_num = 20

    criterion = nn.MSELoss()

    optimizer = torch.optim.SGD(params=model.parameters(), lr=0.001, momentum=0.9)

    best_score = 0

    train_dst, test_dst = get_dataset()
    train_loader = data.DataLoader(train_dst, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = data.DataLoader(test_dst, batch_size=batch_size, shuffle=True, num_workers=2)

    def save_model(path):
        torch.save({
            'model_state': model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "best_score": best_score
        }, path)
        print(f"Model save as {path}")

    for epoch in range(epoch_num):
        model.train()
        loss_total = 0
        iter_num = 0
        for prev_rec, end_rec in train_loader:
            prev_rec = prev_rec.to(device, dtype=torch.float)
            end_rec = end_rec.to(device, dtype=torch.float)
            # print(prev_rec)
            # input('pause')

            optimizer.zero_grad()
            output = model(prev_rec)
            # if epoch > 0:
            #     print(output)
            #     input('pause')
            loss = criterion(output, end_rec)
            loss.backward()
            optimizer.step()
            # if not np.isnan(loss.item()):
            loss_total += loss.item()
            iter_num += 1
        print(loss_total, iter_num, loss_total / iter_num)
        
    
    save_model('all_models.pth')

    pass

if __name__ == '__main__':
    main()