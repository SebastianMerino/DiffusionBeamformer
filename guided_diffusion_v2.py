import os
import math
import torch
import numpy as np
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import Dataset


def linear_beta_schedule(timesteps, start=0.0001, end=0.02, device="cpu"):
    """ 
    Generates schedule of linearly increasing variance.
    The variance at t=1 will be beta[0] 
    """
    beta = torch.linspace(start, end, timesteps, device=device)
    alpha = 1. - beta
    gamma = torch.cumprod(alpha, axis=0)
    return beta, gamma

def cosine_schedule(num_diffusion_timesteps, device):
    beta,gamma = betas_for_alpha_bar(
        num_diffusion_timesteps,
        lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
    )
    return beta.to(device), gamma.to(device)

def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].

    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    gammas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
        gammas.append(alpha_bar(t2))
    return torch.Tensor(np.array(betas)), torch.Tensor(np.array(gammas)), 

def get_index_from_list(vals, t, x_shape):
    """
    Returns a specific index t of a passed list of values vals
    while considering the batch dimension.
    """
    batch_size = t.shape[0]
    out = vals.gather(-1, t)
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(vals.device)

def forward_process(x0, t, gamma, noise):
    """
    Takes an image and a timestep as input and
    returns the noisy version of it
    """
    gamma = get_index_from_list(gamma,t,x0.shape)
    # mean + variance
    return gamma.sqrt() * x0 + (1-gamma).sqrt() * noise

@torch.no_grad()
def sample_timestep_cond(x, y_gen, t, model, beta):
    """
    Calls the model to predict the noise in the image and returns
    the denoised image.
    Applies noise to this image, if we are not in the last step yet.
    """
    device = beta.device
    alpha = 1. - beta
    gamma = torch.cumprod(alpha, axis=0)
    gamma_prev = torch.cat((torch.Tensor([1]).to(device), gamma[:-1]))

    beta_t = get_index_from_list(beta, t, y_gen.shape)
    alpha_t = get_index_from_list(alpha, t, y_gen.shape)
    gamma_t = get_index_from_list(gamma, t, y_gen.shape)
    gamma_t_prev = get_index_from_list(gamma_prev, t, y_gen.shape)

    #input = torch.cat((x, y_gen), 1)
    eps = model(x, y_gen, t)
    # Call model (current image - noise prediction)
    model_mean = (1/alpha_t.sqrt()) * (y_gen - beta_t * eps / (1 - gamma_t).sqrt())
    posterior_variance_t = beta_t * (1. - gamma_t_prev) / (1. - gamma_t)

    noise = torch.randn_like(y_gen)
    return model_mean + torch.sqrt(posterior_variance_t) * noise

@torch.no_grad()
def sample_image_cond(x, model, beta, num_intermediate = 5, clamp=True):
    """ Samples beamformed image y from x """
    # Sample noise
    device = x.device
    T = len(beta)
    y_shape = list(x.shape)
    y_shape[1] = 1
    y_gen = torch.randn(y_shape, device=device)
    stepsize = int(T/num_intermediate)
    intermediate = []
    for i in tqdm(range(T,0,-1)):
        t = torch.full((x.shape[0],), i-1, device=device, dtype=torch.long)
        y_gen = sample_timestep_cond(x, y_gen, t, model, beta)
        # Edit: This is to maintain the natural range of the distribution
        if clamp:
            y_gen = torch.clamp(y_gen, -1.0, 1.0)
        if i%stepsize == 0:
            intermediate.append(y_gen.detach().cpu())
    # y_norm = normalize_image(y_gen.detach().cpu())
    return y_gen.detach().cpu(), intermediate

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
