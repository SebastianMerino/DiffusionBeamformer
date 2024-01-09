import numpy as np

# Metricas: Contraste, gcnr, SNR
def compute_metrics(cx, cz, r, bmode_output, grid):
    env_output = 10 ** (bmode_output / 20)

    r0 = r - 1 / 1000
    r1 = r + 1 / 1000
    r2 = np.sqrt(r0 ** 2 + r1 ** 2)

    dist = np.sqrt((grid[:, :, 0] - cx) ** 2 + (grid[:, :, 2] - cz) ** 2)
    roi_i = dist <= r0
    roi_o = (r1 <= dist) * (dist <= r2)

    # Compute metrics
    env_inner = env_output[roi_i]
    env_outer = env_output[roi_o]

    contrast_value = contrast(env_inner, env_outer)
    snr_value = snr(env_outer)
    gcnr_value = gcnr(env_inner, env_outer)
    cnr_value = cnr(env_inner, env_outer)

    return contrast_value, cnr_value, gcnr_value, snr_value

def contrast(img1, img2):
    return 20 * np.log10(img1.mean() / img2.mean())

# Compute contrast-to-noise ratio
def cnr(img1, img2):
    return (img1.mean() - img2.mean()) / np.sqrt(img1.var() + img2.var())

# Compute the generalized contrast-to-noise ratio
def gcnr(img1, img2):
    _, bins = np.histogram(np.concatenate((img1, img2)), bins=256)
    f, _ = np.histogram(img1, bins=bins, density=True)
    g, _ = np.histogram(img2, bins=bins, density=True)
    f /= f.sum()
    g /= g.sum()
    return 1 - np.sum(np.minimum(f, g))

def snr(img):
    return img.mean() / img.std()