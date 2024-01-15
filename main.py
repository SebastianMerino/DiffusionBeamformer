import torch
import os
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader
from guided_diffusion import *
from datetime import datetime
import torch.nn.functional as func
from model5 import UNETv11
import torch.nn as nn

def main():
    # network hyperparameters
    device = torch.device("cuda:0" if torch.cuda.is_available() else torch.device('cpu'))
    save_dir = r'.\weights_v11'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    # training hyperparameters
    batch_size = 8  # 4 for testing, 16 for training
    n_epoch = 100
    l_rate = 1e-6  # changing from 1e-5 to 1e-6

    # Loading Data
    # input_folder = r'C:\Users\sebas\Documents\Data\DiffusionBeamformer\input_overfit'
    # output_folder = r'C:\Users\sebas\Documents\Data\DiffusionBeamformer\target_overfit'
    input_folder = r'C:\Users\u_imagenes\Documents\smerino\new_training\input'
    output_folder = r'C:\Users\u_imagenes\Documents\smerino\new_training\target_enh'
    dataset = CustomDataset(input_folder, output_folder, transform=True)
    print(f'Dataset length: {len(dataset)}')
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    print(f'Dataloader length: {len(train_loader)}')

    # DDPM noise schedule
    time_steps = 1000
    beta, gamma = linear_beta_schedule(time_steps, start=1e-4, end=0.03, device=device)
    # beta, gamma = cosine_schedule(time_steps, device)
    
    # Model and optimizer
    nn_model = UNETv11(rrdb_blocks=1).to(device)
    optim = torch.optim.Adam(nn_model.parameters(), lr=l_rate)

    trained_epochs = 0
    if trained_epochs > 0:
        nn_model.load_state_dict(torch.load(f"{save_dir}\\model_{trained_epochs}.pth", map_location=device))  # From last model
        loss_arr = np.load(f"{save_dir}\\loss_{trained_epochs}.npy").tolist()  # From last model
    else:
        loss_arr = []

    # Training
    nn_model.train()
    # pbar = tqdm(range(trained_epochs+1,n_epoch+1), mininterval=2)
    print(f' Epoch 0/{n_epoch}, {datetime.now()}')
    for ep in range(trained_epochs+1, n_epoch+1):
        # pbar = tqdm(train_loader, mininterval=2)
        for x, y in train_loader:  # x: images
            optim.zero_grad()
            x = x.to(device)
            y = y.to(device)

            # perturb data
            noise = torch.randn_like(y)
            t = torch.randint(0, time_steps, (x.shape[0],)).to(device)
            y_pert = forward_process(y, t, gamma, noise)
            # input_model = torch.cat((x, y_pert), 1)

            # use network to recover noise
            predicted_noise = nn_model(x, y_pert, t)

            # loss is mean squared error between the predicted and true noise
            loss = func.mse_loss(predicted_noise, noise)
            loss.backward()
            # nn.utils.clip_grad_norm_(nn_model.parameters(),0.5)
            loss_arr.append(loss.item())
            optim.step()

        print(f' Epoch {ep:03}/{n_epoch}, loss: {loss_arr[-1]:.2f}, {datetime.now()}')
        # save model every x epochs
        if ep % 10 == 0 or ep == int(n_epoch - 1):
            torch.save(nn_model.state_dict(), save_dir + f"\\model_{ep}.pth")
            np.save(save_dir + f"\\loss_{ep}.npy", np.array(loss_arr))

if __name__ == '__main__':
    main()
