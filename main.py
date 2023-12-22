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
    n_epoch = 1000
    l_rate = 1e-5  # changing from 1e-5 to 1e-6

    # Loading Data
    # input_folder = r'C:\Users\smerino.C084288\Documents\simulatedCystDataset\downs800_0.0Att\input_id'
    # output_folder = r'C:\Users\smerino.C084288\Documents\simulatedCystDataset\downs800_0.0Att\target_enh'
    input_folder = r'C:\Users\sebas\Documents\Data\DiffusionBeamformer\input_overfit'
    output_folder = r'C:\Users\sebas\Documents\Data\DiffusionBeamformer\target_overfit'
    dataset = CustomDataset(input_folder, output_folder)
    print(f'Dataset length: {len(dataset)}')
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    print(f'Dataloader length: {len(train_loader)}')

    # DDPM noise schedule
    time_steps = 500
    beta1 = 1e-4
    beta2 = 0.02
    b_t = (beta2 - beta1) * torch.linspace(0, 1, time_steps + 1, device=device) + beta1
    a_t = 1 - b_t
    ab_t = torch.cumsum(a_t.log(), dim=0).exp()
    ab_t[0] = 1

    # Model and optimizer
    nn_model = UNETv5(in_channels=3, out_channels=1).to(device)
    optim = torch.optim.Adam(nn_model.parameters(), lr=l_rate)

    trained_epochs = 0
    if trained_epochs > 0:
        nn_model.load_state_dict(torch.load(f"{save_dir}\\model_{trained_epochs}.pth", map_location=device))  # From last model
        loss_arr = np.load(f"{save_dir}\\loss_{trained_epochs}.npy").tolist()  # From last model
    else:
        loss_arr = []

    # Training
    nn_model.train()
    pbar = tqdm(range(trained_epochs+1,n_epoch+1), mininterval=2)
    for ep in pbar:
        # print(f' Epoch {ep}/{n_epoch}')
        # linearly decay learning rate
        optim.param_groups[0]['lr'] = l_rate*(0.1)**(ep/100)

        # pbar = tqdm(train_loader, mininterval=2)
        for x, y in train_loader:  # x: images
            optim.zero_grad()
            x = x.to(device)
            y = y.to(device)

            # perturb data
            noise = torch.randn_like(y)
            t = torch.randint(1, time_steps + 1, (x.shape[0],)).to(device)
            y_pert = perturb_input(y, ab_t[t], noise)
            input_model = torch.cat((x, y_pert), 1)

            # use network to recover noise
            predicted_noise = nn_model(input_model, t)

            # loss is mean squared error between the predicted and true noise
            loss = func.mse_loss(predicted_noise, noise)
            loss.backward()
            loss_arr.append(loss.item())

            optim.step()

        # save model every x epochs
        if ep % 100 == 0 or ep == int(n_epoch - 1):
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            torch.save(nn_model.state_dict(), save_dir + f"\\model_{ep}.pth")
            np.save(save_dir + f"\\loss_{ep}.npy", np.array(loss_arr))
            print("Saved model and loss")


if __name__ == '__main__':
    main()
