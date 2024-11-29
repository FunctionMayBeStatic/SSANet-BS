import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import HSI
from model import SSANetBS

if __name__ == '__main__':

    # Configs.
    name_dataset = "IP220"
    dir_dataset = r'./data'
    patch_size = (7, 7)
    sparse = 0.1  # hyper parameter.
    nbs = 5  # num bands to select.

    # Train SSANet-BS.
    device = torch.device("cuda:0")
    hsi = HSI(name=name_dataset, dir_dataset=dir_dataset, patch_shape=patch_size)
    loader = DataLoader(dataset=hsi, shuffle=True, batch_size=16)
    ssa_net_bs = SSANetBS(num_bands=hsi.num_bands, patch_size_=patch_size).to(device)
    optimizer = torch.optim.SGD(params=ssa_net_bs.parameters(), lr=0.05)
    mse = nn.MSELoss().to(device)
    for epoch in tqdm.tqdm(range(100)):
        epoch_att_weights = []
        for idx_loader, (coord, patch) in enumerate(loader):
            patch = patch.to(device)
            pred, w1, w2 = ssa_net_bs(patch)
            loss = mse(pred, patch) + sparse * torch.sum(w1.norm(1, 1))  # w1&w2: (batch-size, 220)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            for w in w2.detach().cpu().numpy():
                epoch_att_weights.append(w)

        if epoch == 99:
            epoch_att_weights = np.average(np.array(epoch_att_weights), axis=0)
            band_rank = list(np.argsort(epoch_att_weights))[::-1]
            str_selected_bands = [str(int(b)) for b in band_rank[0:nbs]]  # int() for np.int64->int.
            print(f'Selected bands: [{",".join(str_selected_bands)}]')
