import torch
import os
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as func
from model import UNETv4


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
    nn_model = UNETv4(in_channels=3, out_channels=1)
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


def perturb_input(x, ab_t, noise):
    """ Perturbs an image to a specified noise level """
    ab_t_unsq = ab_t.unsqueeze(1).unsqueeze(1).unsqueeze(1)
    return ab_t_unsq.sqrt() * x + (1 - ab_t_unsq) * noise


def plot_sample(input_us, output_us):
    fig, axs = plt.subplots(1, 3, figsize=(10, 5))
    axs[0].imshow(input_us[0, :, :], cmap='gray')  # Assuming the first channel is grayscale
    axs[1].imshow(input_us[1, :, :], cmap='gray')  # Assuming the second channel is grayscale
    axs[2].imshow(output_us[0, :, :], cmap='gray')
    axs[0].set_title('I')
    axs[1].set_title('Q')    # plt.imshow(np.squeeze(cuec[:][:][:]))
    axs[2].set_title('Beamformed')  #
    plt.show()


class CustomDataset(Dataset):
    def __init__(self, input_folder, output_folder, transform=None):
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.transform = transform
        self.input_file_list = sorted(os.listdir(input_folder))
        self.output_file_list = sorted(os.listdir(output_folder))

    def __len__(self):
        return len(self.input_file_list)

    def __getitem__(self, idx):
        file_path = os.path.join(self.input_folder, self.input_file_list[idx])
        x = np.load(file_path)
        x = torch.Tensor(x)
        x = x.permute(2, 0, 1)

        file_path = os.path.join(self.output_folder, self.output_file_list[idx])
        y = np.load(file_path)
        y = torch.Tensor(y)
        y = y.unsqueeze(0)

        if self.transform:
            x = self.transform(x)
            y = self.transform(y)
        return x, y


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
