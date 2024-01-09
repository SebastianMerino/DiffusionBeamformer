import matplotlib.pyplot as plt
import numpy as np
import h5py
import os
from scipy.signal import hilbert
import torch
from torch.nn.functional import grid_sample
import copy

PI = 3.14159265359

def main():
    h5_dir = 'E:/Itamar_LIM/datasets/simulatedCystDataset/raw_0.0Att'
    P = LoadDataParams(h5_dir=h5_dir, simu_name='simu00014')

    depths = np.linspace(P.grid_zlims[0], P.grid_xlims[1], num=800)
    laterals = np.linspace(P.grid_xlims[0], P.grid_xlims[1], num=128)
    grid = make_pixel_grid_from_pos(x_pos=laterals, z_pos=depths)

    bmode_DAS, _ = make_bimg_das1(copy.deepcopy(P), grid, device='cpu')


    extent = [laterals[0] * 1e3, laterals[-1] * 1e3, depths[-1] * 1e3, depths[0] * 1e3]
    fig, ax = plt.subplots(nrows=1, ncols=1, sharex=True, sharey=True)
    ax.imshow(bmode_DAS, cmap="gray", vmin=-60, vmax=0, extent=extent, origin="upper")
    ax.set_xlabel('Lateral [mm]')
    ax.set_ylabel('Axial [mm]')
    ax.set_title('DAS (fc: %.1f MHz)' % (P.fc/1e6))
    plt.savefig(os.path.join(h5_dir,'my_image.png'))

def get_data_params(h5_dir):
    import pandas as pd
    r = []
    cx = []
    cz = []
    c = []
    for id_simu in range(12500):
        P = LoadDataParams(h5_dir=h5_dir, simu_name=f'simu{id_simu:05d}')
        r.append(P.radius)
        c.append(P.c)
        cx.append(P.pos_lat)
        cz.append(P.pos_ax)
    dataset_params = {"r": r, "cx": cx, "cz": cz, "c": c}
    params_df = pd.DataFrame(dataset_params)
    params_df.to_csv('dataset_params.csv')

    P = LoadDataParams(h5_dir=h5_dir, simu_name=f'simu00001')
    common_params = [P.angles, P.ele_pos, P.fc, P.fdemod, P.fs, P.grid_xlims, P.grid_zlims, P.time_zero]
    common_params_keys = ["angles","ele_pos","fc","fdemod","fs","grid_xlims","grid_zlims", "time_zero"]
    common_params_dict = dict(zip(common_params_keys,common_params))
    common_params_df = pd.DataFrame(common_params_dict)
    common_params_df.to_csv("common_params.csv")



class PlaneWaveData:
    def __init__(self):
        """Dummy init. Do not USE"""
        raise NotImplementedError
        nangles, nchans, nsamps = 2, 3, 4
        # Initialize the parameters that *must* be populated by child classes.
        self.idata = np.zeros((nangles, nchans, nsamps), dtype="float32")
        self.qdata = np.zeros((nangles, nchans, nsamps), dtype="float32")
        self.angles = np.zeros((nangles,), dtype="float32")
        self.ele_pos = np.zeros((nchans, 3), dtype="float32")
        self.fc = 5e6
        self.fs = 20e6
        self.fdemod = 0
        self.c = 1540
        self.time_zero = np.zeros((nangles,), dtype="float32")

    def validate(self):
        """ Check to make sure that all information is loaded and valid. """
        # Check size of idata, qdata, angles, ele_pos
        assert self.idata.shape == self.qdata.shape
        assert self.idata.ndim == self.qdata.ndim == 3
        nangles, nchans, nsamps = self.idata.shape
        assert self.angles.ndim == 1 and self.angles.size == nangles
        assert self.ele_pos.ndim == 2 and self.ele_pos.shape == (nchans, 3)
        # Check frequencies (expecting more than 0.1 MHz)
        assert self.fc > 1e5
        assert self.fs > 1e5
        assert self.fdemod > 1e5 or self.fdemod == 0
        # Check speed of sound (should be between 1000-2000 for medical imaging)
        assert 1000 <= self.c <= 2000
        # Check that a separate time zero is provided for each transmit
        assert self.time_zero.ndim == 1 and self.time_zero.size == nangles
        # print("Dataset successfully loaded")


class LoadDataParams(PlaneWaveData):
    def __init__(self, h5_dir, simu_name):
        simu_number = int(simu_name[4:])
        lim_inf = 1000 * ((simu_number - 1) // 1000) + 1
        lim_sup = lim_inf + 999
        h5_name = 'simus_%.5d-%.5d.h5' % (lim_inf, lim_sup)
        h5filename = os.path.join(h5_dir, h5_name)

        with h5py.File(h5filename, "r") as g:
            f = g[simu_name]
            self.idata = np.expand_dims(np.array(f["signal"], dtype="float32"), 0)
            self.qdata = np.imag(hilbert(self.idata, axis=-1))
            self.angles = np.array([0])
            self.fc = np.array(f['fc']).item()
            self.fs = np.array(f['fs']).item()
            self.c = np.array(f['c']).item()
            self.time_zero = np.array([np.array(f['time_zero']).item()])
            self.fdemod = 0
            xs = np.squeeze(np.array(f['ele_pos']))
            self.grid_xlims = [xs[0], xs[-1]]
            self.grid_zlims = [30 * 1e-3, 80 * 1e-3]
            self.ele_pos = np.array([xs, np.zeros_like(xs), np.zeros_like(xs)]).T
            self.pos_lat = np.array(f['lat_pos']).item()
            self.pos_ax = np.array(f['ax_pos']).item()
            self.radius = np.array(f['r']).item()
        super().validate()


def delay_plane(grid, angles):
    # Use broadcasting to simplify computations
    x = grid[:, 0].unsqueeze(0)
    z = grid[:, 2].unsqueeze(0)
    # For each element, compute distance to pixels
    dist = x * torch.sin(angles) + z * torch.cos(angles)
    # Output has shape [nangles, npixels]
    return dist


def delay_focus(grid, ele_pos):
    # Compute distance to user-defined pixels from elements
    # Expects all inputs to be torch tensors specified in SI units.
    # grid    Pixel positions in x,y,z    [npixels, 3]
    # ele_pos Element positions in x,y,z  [nelems, 3]
    # Get norm of distance vector between elements and pixels via broadcasting
    dist = torch.norm(grid - ele_pos.unsqueeze(0), dim=-1)
    # Output has shape [nelems, npixels]
    return dist


# Simple phase rotation of I and Q component by complex angle theta
def complex_rotate(I, Q, theta):
    Ir = I * torch.cos(theta) - Q * torch.sin(theta)
    Qr = Q * torch.cos(theta) + I * torch.sin(theta)
    return Ir, Qr


def make_pixel_grid_from_pos(x_pos, z_pos):
    zz, xx = np.meshgrid(z_pos, x_pos, indexing="ij") # 'ij' -> rows: z, columns: x
    yy = xx * 0
    grid = np.stack((xx, yy, zz), axis=-1)  # [nrows, ncols, 3]
    return grid


class DAS_PW(torch.nn.Module):
    def __init__(
        self,
        P,
        grid,
        ang_list=None,
        ele_list=None,
        rxfnum=2,
        dtype=torch.float,
        device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    ):
        super().__init__()
        # If no angle or element list is provided, delay-and-sum all
        if ang_list is None:
            ang_list = range(P.angles.shape[0])
        elif not hasattr(ang_list, "__getitem__"):
            ang_list = [ang_list]
        if ele_list is None:
            ele_list = range(P.ele_pos.shape[0])
        elif not hasattr(ele_list, "__getitem__"):
            ele_list = [ele_list]

        # Convert plane wave data to tensors
        self.angles = torch.tensor(P.angles, dtype=dtype, device=device)
        self.ele_pos = torch.tensor(P.ele_pos, dtype=dtype, device=device)
        self.fc = torch.tensor(P.fc, dtype=dtype, device=device)
        self.fs = torch.tensor(P.fs, dtype=dtype, device=device)
        self.fdemod = torch.tensor(P.fdemod, dtype=dtype, device=device)
        self.c = torch.tensor(P.c, dtype=dtype, device=device)
        self.time_zero = torch.tensor(P.time_zero, dtype=dtype, device=device)

        # Convert grid to tensor
        self.grid = torch.tensor(grid, dtype=dtype, device=device).reshape(-1, 3)
        self.out_shape = grid.shape[:-1]

        # Store other information as well
        self.ang_list = torch.tensor(ang_list, dtype=torch.long, device=device)
        self.ele_list = torch.tensor(ele_list, dtype=torch.long, device=device)
        self.dtype = dtype
        self.device = device

    def forward(self, x, accumulate=False):
        dtype, device = self.dtype, self.device

        # Load data onto device as a torch tensor
        idata, qdata = x
        idata = torch.tensor(idata, dtype=dtype, device=device)
        qdata = torch.tensor(qdata, dtype=dtype, device=device)

        # Compute delays in meters
        nangles = len(self.ang_list)
        nelems = len(self.ele_list)
        npixels = self.grid.shape[0]
        xlims = (self.ele_pos[0, 0], self.ele_pos[-1, 0])  # Aperture width
        txdel = torch.zeros((nangles, npixels), dtype=dtype, device=device)
        rxdel = torch.zeros((nelems, npixels), dtype=dtype, device=device)
        txapo = torch.ones((nangles, npixels), dtype=dtype, device=device)
        rxapo = torch.ones((nelems, npixels), dtype=dtype, device=device)
        for i, tx in enumerate(self.ang_list):
            txdel[i] = delay_plane(self.grid, self.angles[[tx]])
            # txdel[i] += self.time_zero[tx] * self.c   # ORIGINAL
            txdel[i] -= self.time_zero[tx] * self.c     # IT HAS TO BE "-"
            # txapo[i] = apod_plane(self.grid, self.angles[tx], xlims)
        for j, rx in enumerate(self.ele_list):
            rxdel[j] = delay_focus(self.grid, self.ele_pos[[rx]])
            # rxapo[i] = apod_focus(self.grid, self.ele_pos[rx])

        # Convert to samples
        txdel *= self.fs / self.c
        rxdel *= self.fs / self.c

        # Initialize the output array
        idas = torch.zeros(npixels, dtype=self.dtype, device=self.device)
        qdas = torch.zeros(npixels, dtype=self.dtype, device=self.device)
        iq_cum = None
        if accumulate:
            iq_cum = torch.zeros(nangles, npixels, nelems, 2, dtype=dtype, device='cpu')

        for idx1, (t, td, ta) in enumerate(zip(self.ang_list, txdel, txapo)):
            for idx2, (r, rd, ra) in enumerate(zip(self.ele_list, rxdel, rxapo)):

                i_iq = idata[t, r].view(1, 1, 1, -1)
                q_iq = qdata[t, r].view(1, 1, 1, -1)
                # Convert delays to be used with grid_sample
                delays = td + rd
                dgs = (delays.view(1, 1, -1, 1) * 2 + 1) / idata.shape[-1] - 1
                dgs = torch.cat((dgs, 0 * dgs), axis=-1)
                # Interpolate using grid_sample and vectorize using view(-1)
                # ifoc, qfoc = grid_sample(iq, dgs, align_corners=False).view(2, -1)
                ifoc = grid_sample(i_iq, dgs, align_corners=False).view(-1)
                qfoc = grid_sample(q_iq, dgs, align_corners=False).view(-1)
                # torch.Size([144130])
                # Apply phase-rotation if focusing demodulated data
                if self.fdemod != 0:
                    tshift = delays.view(-1) / self.fs - self.grid[:, 2] * 2 / self.c
                    theta = 2 * PI * self.fdemod * tshift
                    ifoc, qfoc = complex_rotate(ifoc, qfoc, theta)
                # Apply apodization, reshape, and add to running sum
                # apods = ta * ra
                # idas += ifoc * apods
                # qdas += qfoc * apods
                idas += ifoc
                qdas += qfoc
                # torch.Size([355*406])
                if accumulate:
                    # 1, npixels, nelems, 2
                    iq_cum[idx1, :, idx2, 0] = ifoc.cpu()
                    iq_cum[idx1, :, idx2, 1] = qfoc.cpu()

        # Finally, restore the original pixel grid shape and convert to numpy array
        idas = idas.view(self.out_shape)
        qdas = qdas.view(self.out_shape)

        env = torch.sqrt(idas**2 + qdas**2)
        bimg = 20 * torch.log10(env + torch.tensor(1.0*1e-25))
        bimg = bimg - torch.max(bimg)
        return bimg, env, idas, qdas, iq_cum


def make_bimg_das1(P, grid, device):
    P.idata = P.idata / np.amax(P.idata)
    P.qdata = P.qdata / np.amax(P.qdata)

    id_angle = len(P.angles) // 2
    dasNet = DAS_PW(P, grid, ang_list=id_angle, device=device)
    bimg, env, _, _, _ = dasNet((P.idata, P.qdata), accumulate=False)
    bimg = bimg.detach().cpu().numpy()
    env = env.detach().cpu().numpy()
    # env = np.abs(idas+1j*qdas)
    # env_normalized = 10 ** (bimg / 20)  # Normalize by max value
    return bimg, env



