from matplotlib.animation import FuncAnimation, PillowWriter
from IPython.display import HTML
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
import torch
import math
import os

def perturb_input(x, ab_t, noise):
    """ Perturbs an image to a specified noise level
     CORRECTION: a sqrt was missing on the noise term """
    ab_t_unsq = ab_t.unsqueeze(1).unsqueeze(1).unsqueeze(1)
    return ab_t_unsq.sqrt() * x + (1 - ab_t_unsq).sqrt() * noise

def denoise_ddim(x, ab, ab_prev, pred_noise):   
    """ define sampling function for DDIM, removes the noise using ddim """ 
    x0_pred = ab_prev.sqrt() / ab.sqrt() * (x - (1 - ab).sqrt() * pred_noise)
    dir_xt = (1 - ab_prev).sqrt() * pred_noise

    return x0_pred + dir_xt

def linear_time_schedule(time_steps, beta1, beta2, device):
    b_t = (beta2 - beta1) * torch.linspace(0, 1, time_steps + 1, device=device) + beta1
    a_t = 1 - b_t
    ab_t = torch.cumsum(a_t.log(), dim=0).exp()
    ab_t[0] = 1
    return ab_t

def simple_time_schedule(time_steps, device):
    ab_t =  1 - torch.linspace(0, 1, time_steps + 1, device=device)
    return ab_t

def cosine_time_schedule(time_steps,s,device):
    t = torch.arange(time_steps).to(device)
    ab_t = torch.cos((t / time_steps + s) / (1 + s) * math.pi / 2) ** 2
    return ab_t

def plot_sample(input_us, output_us):
    fig, axs = plt.subplots(1, 3, figsize=(12, 3))
    im0 = axs[0].imshow(input_us[0, :, :], cmap='gray', extent=[-20,20,50,0])
    im0.set_clim(-1,1)
    axs[0].set_title('I')
    plt.colorbar(mappable=im0,ax=axs[0])

    im1 = axs[1].imshow(input_us[1, :, :], cmap='gray', extent=[-20,20,50,0]) 
    im1.set_clim(-1,1)
    axs[1].set_title('Q') 
    plt.colorbar(mappable=im1,ax=axs[1])

    im2 = axs[2].imshow(output_us[0, :, :], cmap='gray', extent=[-20,20,50,0])
    axs[2].set_title('Beamformed')
    plt.colorbar(mappable=im2,ax=axs[2])

    plt.show()


def animate_reverse_process(intermediate):
    fig, axs = plt.subplots(2, 2, figsize=(6, 6))
    for iSample in range(4):
        current_ax = axs[iSample//2][iSample%2]
        current_ax.set_title(f'Sample {iSample}')
        artist_im = current_ax.imshow(intermediate[0,iSample,0,:,:], extent=[-20,20,50,0], cmap='gray')
        artist_im.set_clim(-1,1)
        plt.colorbar(mappable=artist_im,ax=current_ax)
    plt.tight_layout()

    def update(frame):
        artist_arr = []
        for iSample in range(4):
            current_ax = axs[iSample//2][iSample%2]
            # current_ax.clear()
            # current_ax.set_title(f'Sample {iSample}')
            artist_im = current_ax.imshow(intermediate[frame,iSample,0,:,:], extent=[-20,20,50,0], cmap='gray')
            artist_im.set_clim(0,1)
            # plt.colorbar(mappable=artist_im,ax=current_ax)
            artist_arr.append(artist_im)
        return artist_arr

    num_frames = intermediate.shape[0]
    print("Creating animation...")
    ani = FuncAnimation(fig=fig, func=update, frames=num_frames, interval=30)
    plt.close()

    return HTML(ani.to_jshtml())


def plot_minibatch(samples, title="Sample"):
    fig, axs = plt.subplots(2, 2, figsize=(6, 6))
    for iSample in range(4):
        im = axs[iSample//2][iSample%2].imshow(samples[iSample,0,:,:].to('cpu'), extent=[-20,20,50,0],cmap='gray')
        im.set_clim(-1,1)
        axs[iSample//2][iSample%2].set_title(f'{title} {iSample}')
        plt.colorbar(mappable=im,ax=axs[iSample//2][iSample%2])
    plt.tight_layout()
    plt.show()


class CustomDataset(Dataset):
    def __init__(self, input_folder, output_folder, transform=True):
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.transform = transform
        self.transform_input = transforms.Normalize([0,0],[0.1172,0.1172])
        self.transform_output = transforms.Lambda(lambda t: (t * 2) - 1)
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
            x = self.transform_input(x)
            y = self.transform_output(y)
        return x, y