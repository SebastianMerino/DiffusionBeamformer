import torch
from torchvision import transforms
import matplotlib.pyplot as plt
from diffusion_utils import CustomDataset
from tqdm import tqdm

def load_transformed_dataset():
    input_folder = r'C:\Users\sebas\Documents\Data\DiffusionBeamformer\input_id'
    output_folder = r'C:\Users\sebas\Documents\Data\DiffusionBeamformer\target_enh'
    return CustomDataset(input_folder, output_folder, transform=True)

def show_tensor_image(image):
    reverse_transforms = transforms.Compose([
        transforms.Lambda(lambda t: (t + 1) / 2),
        transforms.Lambda(lambda t: t.permute(1, 2, 0)), # CHW to HWC
        transforms.Lambda(lambda t: (t * 60) - 60.),
        transforms.Lambda(lambda t: t.numpy())
    ])
    # Take first image of batch
    if len(image.shape) == 4:
        image = image[0, :, :, :]
    plt.imshow(reverse_transforms(image), cmap='gray', extent=[-20,20,50,0])

def show_reverse_process(intermediate):
    num_intermediate = len(intermediate)
    plt.figure(figsize=(15,2))
    plt.axis('off')
    for id, y_gen in enumerate(intermediate):
        plt.subplot(1, num_intermediate, id+1)
        show_tensor_image(y_gen)
    plt.show()

def linear_beta_schedule(timesteps, start=0.0001, end=0.02):
    beta = torch.linspace(start, end, timesteps)
    alpha = 1. - beta
    gamma = torch.cumprod(alpha, axis=0)
    return beta, gamma

def get_index_from_list(vals, t, x_shape):
    """
    Returns a specific index t of a passed list of values vals
    while considering the batch dimension.
    """
    batch_size = t.shape[0]
    out = vals.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

def forward_diffusion_sample(x0, t, gamma, device="cpu"):
    """
    Takes an image and a timestep as input and
    returns the noisy version of it
    """
    x0 = x0.to(device)
    noise = torch.randn_like(x0).to(device)
    gamma = get_index_from_list(gamma,t,x0.shape).to(device)
    # mean + variance
    return gamma.sqrt() * x0 + (1-gamma).sqrt() * noise, noise


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
    gamma_prev = get_index_from_list(gamma, t-1, y_gen.shape)

    input = torch.cat((x, y_gen), 1)
    eps = model(input, t)
    # Call model (current image - noise prediction)
    model_mean = (1/alpha_t.sqrt()) * (y_gen - beta_t * eps / (1 - gamma_t).sqrt())
    posterior_variance_t = beta_t * (1. - gamma_prev) / (1. - gamma_t)

    noise = torch.randn_like(y_gen)
    return model_mean + torch.sqrt(posterior_variance_t) * noise

@torch.no_grad()
def sample_image_cond(x, model, beta, num_intermediate = 5):
    # Sample noise
    device = torch.device("cuda:0" if torch.cuda.is_available() else torch.device('cpu'))
    x = x.to(device)
    T = len(beta)
    y_shape = list(x.shape)
    y_shape[1] = 1
    y_gen = torch.randn(y_shape, device=device)
    stepsize = int(T/(num_intermediate-1))
    intermediate = [y_gen.detach().cpu()]
    for i in tqdm(range(T-1,0,-1)):
        t = torch.full((1,), i, device=device, dtype=torch.long)
        y_gen = sample_timestep_cond(x, y_gen, t, model, beta)
        # Edit: This is to maintain the natural range of the distribution
        y_gen = torch.clamp(y_gen, -1.0, 1.0)
        if (i-1)%stepsize == 0:
            intermediate.append(y_gen.detach().cpu())
    return y_gen.detach().cpu(), intermediate