"""
Train a base VAE model with custom dataset and custom 1d convolutional encoder and decoder.
"""
import os, time, tqdm
import numpy as np
import torch
from models.dmp import CanonicalSystem, SingleDMP
from models.vae import CVAE
from torch.utils.data import DataLoader
from utils.data_loader import DataLoader as num_dataset
from utils.early_stop import EarlyStop
from matplotlib import pyplot as plt
plt.ion()

#%% trajectory steps and dof
shape = (2, 100)

# create dataset for training and testing
cs = CanonicalSystem(dt=0.01, ax=1)
dmp = SingleDMP(n_bfs=50, cs=cs, run_time=1.0, dt=0.01)
train_dataset = num_dataset(run_time=1, dmp=dmp, dt=0.01, dof=2)
test_dataset = num_dataset(run_time=1, dmp=dmp, dt=0.01, dof=2)
train_dataset.load_data('../data/train_paths.npz')
test_dataset.load_data('../data/test_paths.npz')
print("train dataset size: ", len(train_dataset))
print("test dataset size: ", len(test_dataset))

# create dataloader for training and testing
train_iter = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_iter = DataLoader(test_dataset, batch_size=64, shuffle=True)

# create model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
vae_net = CVAE(shape=shape, nclass=10, nhid=8, ncond=8)
vae_net.to(device)

#%% train model
lr = 1e-4
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, vae_net.parameters()), lr=lr, weight_decay=0.0001)

def adjust_lr(optimizer, decay_rate=0.95):
    for param_group in optimizer.param_groups:
        param_group['lr'] *= decay_rate

#%% if load
save_name = "cVAE_base.pt"
retrain = True
if os.path.exists(save_name):
    print("Model parameters have already been trained before. Retrain ? (y/n)")
    ans = input()
    if not (ans == 'y'):
        checkpoint = torch.load(save_name, map_location=device)
        vae_net.load_state_dict(checkpoint["net"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        for g in optimizer.param_groups:
            g['lr'] = lr


max_epochs = 1000
early_stop = EarlyStop(patience=20, save_name=save_name)

print("training on ", device)
for epoch in range(max_epochs):

    train_loss, n, start = 0.0, 0, time.time()
    for X, y in tqdm.tqdm(train_iter, ncols=50):
        X = X.to(device)
        # only use the first dimension of y
        y = y[:, 0].to(device)
        X_hat, mean, logvar = vae_net(X, y)

        l = vae_net.loss(X, X_hat, mean, logvar).to(device)
        optimizer.zero_grad()
        l.backward()
        optimizer.step()

        train_loss += l.cpu().item()
        n += X.shape[0]

    train_loss /= n
    print('epoch %d, train loss %.4f , time %.1f sec'
          % (epoch, train_loss, time.time() - start))

    adjust_lr(optimizer)

    # ------------ plot the training result ------------
    check_number = 5
    if epoch % 10 == 0:
        for i in range(X.shape[0]):
            if y[i] == check_number:
                data = X[i, :, :].cpu().numpy()
                data1 = X_hat[i, :, :].detach().cpu().numpy()
                plt.plot(data[0,:], data[1,:])
                plt.plot(data1[0, :], data1[1, :])
            plt.show()

    if (early_stop(train_loss, vae_net, optimizer)):
        break

#%% ----------------------------------- test -----------------------------------
checkpoint = torch.load(early_stop.save_name)
vae_net.load_state_dict(checkpoint["net"])
vae_net.eval()

#%%
# number = 8
# with torch.no_grad():
#     x = vae_net.generate(number)

# data = x.cpu().numpy()
# plt.plot(data[0, :], data[1, :])
# plt.show()