import os
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

def normalize_image(img):
    result = (img - torch.min(img)) / (torch.max(img) - torch.min(img))
    result = (result*2)-1
    return result

@torch.no_grad()
def sample_timestep_cond(x, y_gen, t, model, beta):
    """
    Calls the model to predict the noise in the image and returns
    the denoised image.
    Applies noise to this image, if we are not in the last step yet.
    """
    alpha = 1. - beta
    gamma = torch.cumprod(alpha, axis=0)

    beta_t = get_index_from_list(beta, t, y_gen.shape)
    alpha_t = get_index_from_list(alpha, t, y_gen.shape)
    gamma_t = get_index_from_list(gamma, t, y_gen.shape)
    gamma_prev = get_index_from_list(gamma, t-1, y_gen.shape) if t != 0 else 1

    input = torch.cat((x, y_gen), 1)
    eps = model(input, t)
    # Call model (current image - noise prediction)
    model_mean = (1/alpha_t.sqrt()) * (y_gen - beta_t * eps / (1 - gamma_t).sqrt())
    posterior_variance_t = beta_t * (1. - gamma_prev) / (1. - gamma_t)

    noise = torch.randn_like(y_gen)
    return model_mean + torch.sqrt(posterior_variance_t) * noise

@torch.no_grad()
def sample_image_cond(x, model, beta, num_intermediate = 5, clamp=True):
    """ Samples beamformed image y from x """
    # Sample noise
    device = torch.device("cuda:0" if torch.cuda.is_available() else torch.device('cpu'))
    x = x.to(device)
    T = len(beta)
    y_shape = list(x.shape)
    y_shape[1] = 1
    y_gen = torch.randn(y_shape, device=device)
    stepsize = int(T/num_intermediate)
    intermediate = []
    for i in tqdm(range(T,0,-1)):
        t = torch.full((1,), i-1, device=device, dtype=torch.long)
        y_gen = sample_timestep_cond(x, y_gen, t, model, beta)
        # Edit: This is to maintain the natural range of the distribution
        if clamp:
            y_gen = torch.clamp(y_gen, -1.0, 1.0)
        if i%stepsize == 0:
            intermediate.append(y_gen.detach().cpu())
    y_norm = normalize_image(y_gen.detach().cpu())
    return y_norm, intermediate

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
