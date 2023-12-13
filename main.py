import torch
import os
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader
from diffusion_utils import *
import torch.nn.functional as func
from model import UNETv5


def main():
    # network hyperparameters
    device = torch.device("cuda:0" if torch.cuda.is_available() else torch.device('cpu'))
    save_dir = r'.\weights'

    # training hyperparameters
    batch_size = 4  # 4 for testing, 16 for training
    n_epoch = 50
    l_rate = 1e-5

    # Loading Data
    # input_folder = r'C:\Users\smerino.C084288\Documents\simulatedCystDataset\downs800_0.0Att\input_id'
    # output_folder = r'C:\Users\smerino.C084288\Documents\simulatedCystDataset\downs800_0.0Att\target_enh'
    input_folder = r'C:\Users\sebas\Documents\MATLAB\DataProCiencia\Attenuation\DiffusionBeamformer\input_id'
    output_folder = r'C:\Users\sebas\Documents\MATLAB\DataProCiencia\Attenuation\DiffusionBeamformer\target_enh'
    dataset = CustomDataset(input_folder, output_folder)
    print(f'Dataset length: {len(dataset)}')
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    print(f'Dataloader length: {len(train_loader)}')

    # DDPM noise schedule
    timesteps = 500
    beta1 = 1e-4
    beta2 = 0.02
    b_t = (beta2 - beta1) * torch.linspace(0, 1, timesteps + 1, device=device) + beta1
    a_t = 1 - b_t
    ab_t = torch.cumsum(a_t.log(), dim=0).exp()
    ab_t[0] = 1

    # Model and optimizer
    nn_model = UNETv5(in_channels=3, out_channels=1)
    optim = torch.optim.Adam(nn_model.parameters(), lr=l_rate)
    loss_arr = []

    # Training
    nn_model.train()
    for ep in range(n_epoch):
        print(f' Epoch {ep}/{n_epoch}')

        pbar = tqdm(train_loader, mininterval=2)
        for x, y in pbar:  # x: images
            optim.zero_grad()
            x = x.to(device)
            y = y.to(device)

            # perturb data
            noise = torch.randn_like(y)
            t = torch.randint(1, timesteps + 1, (x.shape[0],)).to(device)
            y_pert = perturb_input(y, ab_t[t], noise)
            input_model = torch.cat((x, y_pert), 1)

            # use network to recover noise
            predicted_noise = nn_model(input_model, t)

            # loss is mean squared error between the predicted and true noise
            loss = func.mse_loss(predicted_noise, noise)
            loss.backward()
            loss_arr.append(loss.item())

            optim.step()

        # save model every 1 epochs
        if ep % 1 == 0 or ep == int(n_epoch - 1):
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            torch.save(nn_model.state_dict(), save_dir + f"\\model_{ep}.pth")
            np.save(save_dir + f"\\loss_{ep}.npy", np.array(loss_arr))
            print("Saved model and loss")


if __name__ == '__main__':
    main()
