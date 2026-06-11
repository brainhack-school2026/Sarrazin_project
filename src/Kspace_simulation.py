from pathlib import Path

import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

### 1. Load one slice fron the dataset — ds005616 ---------------------------------------------------------------------------------------
def load_slice(data, slice_idx=None, axis=0):
    """
    Extract a 2D slice from a 3D (or 4D) MRI volume and apply preprocessing.

    Parameters
    ----------
    axis : int
        axis=0 : coupe sagittal
        axis=1 : coupe coronal 
        axis=2 : coupe axial

    """

    # default: middle slice
    if slice_idx is None:
        slice_idx = data.shape[axis] // 2

    # slice extraction
    if axis == 0:
        sl = data[slice_idx, :, :]
    elif axis == 1:
        sl = data[:, slice_idx, :]
    else:
        sl = data[:, :, slice_idx]

    sl = sl.astype(np.float32)

    # remove saturation artifacts
    p_high_raw = np.percentile(sl, 99.5)
    sl = np.clip(sl, 0, p_high_raw)

    # intensity normalization
    p_low  = np.percentile(sl, 2)
    p_high = np.percentile(sl, 99)
    sl = np.clip(sl, p_low, p_high)
    sl = (sl - p_low) / (p_high - p_low + 1e-8)
    sl = np.clip(sl, 0, 1)

    # remove annotation zones (scanner markers) on sagittal slice
    if axis==0:
        sl[sl.shape[0]-12:, :] = 0
        sl[:12, :] = 0

    return sl

### 2. Plot a slice in the image space --------------------------------------------------------------------------
def plot_image(ax, image, title, cmap='gray'):
    """Display a magnitude image with robust contrast."""
    image = np.rot90(image, k=-1)
    vmin, vmax = np.percentile(image, (2, 98))
    ax.imshow(image, cmap=cmap, origin='lower', vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.axis('off')
    return None

### 3. Coinversion image <-> k-space -----------------------------------------------------------------------------
def image_to_kspace(image):
    """ 
    Convert image domain → k-space domain using 2D FFT.

    The zero-frequency component is shifted to the center
    to match MRI acquisition conventions.
    """
    return np.fft.fftshift(np.fft.fft2(image))

def kspace_to_image(kspace):
    """
    Convert k-space → image domain using inverse FFT
    """
    return np.abs(np.fft.ifft2(np.fft.ifftshift(kspace)))

### 4. Plot a k-space  ---------------------------------------------------------------------------------------------
def plot_kspace(ax, kspace, title):
    """Display log-magnitude of k-space."""
    # kspace = np.rot90(kspace, k=-1)
    log_mag = np.log1p(np.abs(kspace))
    ax.imshow(log_mag, cmap='gray', origin='lower')
    ax.set_title(title)
    ax.axis('off')
    return None


### 5. Breathing signal simulation ---------------------------------------------------------------------------------------
import neurokit2 as nk
from scipy.interpolate import interp1d

def get_breathing_signal(ny, TR, TE, respiratory_rate=15, method="sinusoidal", seed=42):
    '''
    Generate a respiratory motion signal aligned with k-space acquisition.
    Each k-space line is assigned a motion value based on acquisition time.
    '''

    duration      = ny * TR          # total duration (s)
    sampling_rate = 100              # always 100 Hz

    rsp = nk.rsp_simulate(
        duration         = int(duration),
        sampling_rate    = sampling_rate,
        respiratory_rate = respiratory_rate,
        method           = method,
        random_state     = seed,
    )

    # Temps du signal neurokit
    t_nk = np.linspace(0, duration, len(rsp))

    # Temps des lignes ky (1 point par TR)
    t_ky = np.arange(ny) * TR + TE

    # Interpolation propre — pas de saut, pas de distorsion
    interp    = interp1d(t_nk, rsp, kind='linear', fill_value='extrapolate')
    rsp_resampled = interp(t_ky)

    # Normaliser à [-1, 1]
    rsp_norm  = rsp_resampled / (np.max(np.abs(rsp_resampled)) + 1e-8)
    

    return rsp_norm, t_ky

def breathing_simu_comparison (TR, sampling_rate=100, duration=30, respiratory_rate=15, method="sinusoidal"):
    '''Compare continuous respiratory signal with k-space sampled signal'''

    rsp = nk.rsp_simulate(
        duration=duration,
        sampling_rate=sampling_rate,
        respiratory_rate=respiratory_rate,
        method=method,
        random_state = 42
    )
    t_full = np.linspace(0, duration, len(rsp))
    interp = interp1d(t_full, rsp)

    t_ky = np.arange(0, duration, TR)
    rsp_ky = interp(t_ky)

    return rsp, t_full, t_ky, rsp_ky


### 6. Breathing motion corruption ----------------------------------------------------------------------------------------
def apply_motion_line_by_line(image, A, TR, TE, respiratory_rate=15, method="breathmetrics", seed=42):
    '''
    Simulate respiratory motion in k-space using a vectorized phase ramp model.

    Each k-space line is modulated by a phase term corresponding to
    a time-dependent spatial displacement.
    '''
    ny, nx = image.shape

    # Respiratory motion signal
    rsp_norm, t_seconds = get_breathing_signal(
        ny, TR, TE, respiratory_rate, method, seed
    )
    displacements = A * rsp_norm  # (ny,)

    # k-space
    clean_kspace = np.fft.fftshift(np.fft.fft2(image))  # (ny, nx)

    # Spatial frequency axis (nx, )
    kx = np.fft.fftshift(np.fft.fftfreq(nx))

    # Phase ramp: shiffting d[ky] pixels ↔ exp(-j2π · kx · d[ky])
    phase_ramps = np.exp(
        -1j * 2 * np.pi
        * displacements[:, np.newaxis]  # (ny, 1)
        * kx[np.newaxis, :]             # (1, nx)
    )

    # Apply motion corruption
    motion_kspace = clean_kspace * phase_ramps  # (ny, nx)

    return motion_kspace, clean_kspace, displacements, t_seconds

### 7. Cartesian undersampling in k-space -----------------------------------------------------------------------------
def make_undersampling_mask(shape, R, center_fraction=0.08, seed=42):
    """
    Cartesian undersampling mask along ky.

    center_fraction : fraction of central ky lines always acquired
    R : acceleration factor
    """
    ny, nx = shape
    rng  = np.random.default_rng(seed)
    mask = np.zeros(ny, dtype=bool)

    n_center   = max(1, int(ny * center_fraction))
    center_start = ny // 2 - n_center // 2
    mask[center_start : center_start + n_center] = True

    outer = np.where(~mask)[0]
    n_keep = len(outer) // R
    mask[rng.choice(outer, size=n_keep, replace=False)] = True

    return np.tile(mask[:, np.newaxis], (1, nx))


def apply_undersampling(kspace, mask):
    return kspace * mask


### 8. Complex Gaussian noise in k-space -----------------------------------------------------------------------------
def add_noise(kspace, mask, snr, seed=0):
    """
    Add complex Gaussian noise to acquired k-space lines.

    Noise std = RMS(acquired signal) / SNR
    """
    rng        = np.random.default_rng(seed)
    acquired   = kspace[mask.astype(bool)]
    signal_rms = np.sqrt(np.mean(np.abs(acquired) ** 2))
 
    snr_linear = 10**(snr/20)
    noise_std = signal_rms / snr_linear

    noise = noise_std * (
        rng.standard_normal(kspace.shape)
        + 1j * rng.standard_normal(kspace.shape)
    ) / np.sqrt(2)

    return kspace + noise * mask   # noise only on acquired lines