
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import hashlib

def compute_metrics(ref, img):
    
    s = ssim(ref, img, data_range=1.0)
    p = psnr(ref, img, data_range=1.0)
    nrmse = np.sqrt(np.mean((ref - img)**2)) / (ref.max() - ref.min())
    return {'SSIM': s, 'PSNR (dB)': p, 'NRMSE': nrmse}

def make_seed(sub_id, sl_idx, A, f, R, snr):
    key = f"{sub_id}_{sl_idx}_{A}_{f}_{R}_{snr}".encode()
    return int(hashlib.sha256(key).hexdigest()[:8], 16)