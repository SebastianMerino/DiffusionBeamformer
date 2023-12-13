import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import numpy as np
import torch
import os

def perturb_input(x, ab_t, noise):
    """ Perturbs an image to a specified noise level """
    ab_t_unsq = ab_t.unsqueeze(1).unsqueeze(1).unsqueeze(1)
    return ab_t_unsq.sqrt() * x + (1 - ab_t_unsq) * noise


def plot_sample(input_us, output_us):
    fig, axs = plt.subplots(1, 3, figsize=(12, 3))
    im0 = axs[0].imshow(input_us[0, :, :], cmap='gray', extent=[-20,20,50,0])
    im0.set_clim(-0.1,0.1)
    axs[0].set_title('I')
    plt.colorbar(mappable=im0,ax=axs[0])

    im1 = axs[1].imshow(input_us[1, :, :], cmap='gray', extent=[-20,20,50,0]) 
    im1.set_clim(-0.1,0.1)
    axs[1].set_title('Q') 
    plt.colorbar(mappable=im1,ax=axs[1])

    im2 = axs[2].imshow(output_us[0, :, :], cmap='gray', extent=[-20,20,50,0])
    axs[2].set_title('Beamformed')
    plt.colorbar(mappable=im2,ax=axs[2])

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