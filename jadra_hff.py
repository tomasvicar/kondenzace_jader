
from glob import glob
from cellpose import models
from scipy.ndimage import gaussian_filter
from scipy.ndimage import median_filter
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from scipy.interpolate import RegularGridInterpolator
from scipy.interpolate import interp1d
# from skimage.measure import EllipseModel
from skimage.measure import label, regionprops, regionprops_table
from scipy.signal import resample

from utils.read_ics_file import read_ics_file_ordered
from utils.mat2gray import mat2gray_rgb_percentile
from utils.reinterpolate_contour import reinterpolate_contour


data_path = r"D:\kondenzace_jader\data\Chromatin Architecture Analyses Python\Mikroskop horní\zz_Original data files"
output_path = r"D:\kondenzace_jader\data\results"





import numpy as np
from scipy import ndimage as ndi
from skimage import morphology

def robust_rescale(x, mask, p_low=1, p_high=99):
    v = x[mask]
    lo, hi = np.percentile(v, [p_low, p_high])
    if hi <= lo:
        y = np.zeros_like(x, float)
    else:
        y = np.clip((x - lo) / (hi - lo), 0, 1)
    return y

def soft_taper_from_mask(mask, taper_px=5):
    """
    Build W in [0,1]: 1 in the interior; smoothly decays to 0 within ~taper_px from the mask boundary.
    """
    # distance to background inside mask
    dist_in = ndi.distance_transform_edt(mask)
    # distance to mask outside (not used but shown for completeness)
    W = np.clip(dist_in / float(taper_px), 0, 1)
    # optional smooth nonlinearity (cosine ramp) for C^1 continuity
    W = 0.5 - 0.5 * np.cos(np.pi * W)   # equals 0 at 0, 1 at >=1; smooth ramp
    W[~mask] = 0.0
    return W

def radial_average(P):
    """Return radial profile (ring average) and corresponding normalized frequencies x in (0,1]."""
    if P.ndim == 2:
        H, W = P.shape
        yy, xx = np.indices((H, W))
        cy, cx = (H-1)/2, (W-1)/2
        r = np.sqrt((yy-cy)**2 + (xx-cx)**2)
    else:  # 3D
        Z, H, W = P.shape
        zz, yy, xx = np.indices((Z, H, W))
        cz, cy, cx = (Z-1)/2, (H-1)/2, (W-1)/2
        r = np.sqrt((zz-cz)**2 + (yy-cy)**2 + (xx-cx)**2)

    r_norm = r / (r.max() + 1e-12)
    nbins = 50
    bins = np.linspace(0, 1, nbins + 1)
    which = np.digitize(r_norm, bins) - 1
    radial = np.array([P[which == i].mean() if np.any(which == i) else 0.0 for i in range(nbins)])
    x = 0.5 * (bins[:-1] + bins[1:])
    # drop the DC bin
    return x[1:], radial[1:]

def hff_mask_only(raw_img, mask, f_cut=0.25, taper_px=5, eps=1e-3):
    """
    High-Frequency Fraction computed strictly from the masked region.
    - raw_img: 2D (H,W) or 3D (Z,H,W)
    - mask: boolean same shape
    Steps: robust rescale within mask -> soft taper -> FFT -> |F(I*W)|^2
           divide by |F(W)|^2 + eps (mask-spectrum compensation)
           radial average -> fraction above f_cut
    """
    assert raw_img.shape == mask.shape
    img = robust_rescale(raw_img, mask)
    W = soft_taper_from_mask(mask, taper_px=int(taper_px))

    # windowed signal
    X = img * W

    # FFT and spectra
    if img.ndim == 2:
        F = np.fft.fftshift(np.fft.fft2(X))
        FW = np.fft.fftshift(np.fft.fft2(W))
    else:
        F = np.fft.fftshift(np.fft.fftn(X))
        FW = np.fft.fftshift(np.fft.fftn(W))

    P_meas = np.abs(F)**2
    P_win  = np.abs(FW)**2

    # compensate window/mask blur (ratio PSD)
    P_comp = P_meas / (P_win + eps)

    # radial average
    x, radial = radial_average(P_comp)

    # normalize to obtain a proper fraction
    total = radial.sum() + 1e-12
    hf = radial[x > f_cut].sum()
    return float(hf / total)










percentile_r = [2 , 99.8]
percentile_g = [2, 99.8]
percentile_b = [2, 98.0]

gaus_sigma = (1, 1, 0.3)
med_size = (5, 5, 1)


model_name = 'nuc3_20250801_1520'
model = models.CellposeModel(gpu=True, pretrained_model=model_name)



fnames = glob(data_path + '/**/*01.ics', recursive=True)

for fname in fnames:
    data = read_ics_file_ordered(fname)



    data = gaussian_filter(data, sigma=gaus_sigma, axes=(0, 1, 2))
    data = median_filter(data, size=med_size, axes=(0, 1, 2))

    data_2d = np.max(data, axis=2)  
    data_2d_dapi = data_2d[:, :, 0]

    percentiles = [percentile_r, percentile_g, percentile_b]
    data_2d, percetiles_values = mat2gray_rgb_percentile(data_2d, percentiles, get_percitile=True)
    data_2d = np.round(data_2d * 255).astype(np.uint8)

    data_2d_dapi_norm = data_2d[:, :, 0]

    mask, flow, style = model.eval(data_2d)

    # plt.imshow(data_2d)
    # plt.show()

    # plt.imshow(mask)
    # plt.show()


    u = np.unique(mask)
    u = u[u > 0] 






    for nuc_num in range(1, u.max() + 1):
        nuc_mask = (mask == nuc_num).astype(np.uint8)


        dff_value= hff_mask_only(data_2d_dapi, nuc_mask)


        print(f"Nucleus {nuc_num}: DFF = {dff_value}")

        break
    break

