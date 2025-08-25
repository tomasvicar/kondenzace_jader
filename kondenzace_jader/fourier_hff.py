import numpy as np
from skimage.io import imread
import matplotlib.pyplot as plt
import cv2



import numpy as np
from scipy import ndimage as ndi
from skimage import morphology

# def robust_rescale(x, mask, p_low=1, p_high=99):
#     v = x[mask]
#     lo, hi = np.percentile(v, [p_low, p_high])
#     if hi <= lo:
#         y = np.zeros_like(x, float)
#     else:
#         y = np.clip((x - lo) / (hi - lo), 0, 1)
#     return y

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

# def radial_average(P):
#     """Return radial profile (ring average) and corresponding normalized frequencies x in (0,1]."""
#     if P.ndim == 2:
#         H, W = P.shape
#         yy, xx = np.indices((H, W))
#         cy, cx = (H-1)/2, (W-1)/2
#         r = np.sqrt((yy-cy)**2 + (xx-cx)**2)
#     else:  # 3D
#         Z, H, W = P.shape
#         zz, yy, xx = np.indices((Z, H, W))
#         cz, cy, cx = (Z-1)/2, (H-1)/2, (W-1)/2
#         r = np.sqrt((zz-cz)**2 + (yy-cy)**2 + (xx-cx)**2)

#     r_norm = r / (r.max() + 1e-12)
#     nbins = 50
#     bins = np.linspace(0, 1, nbins + 1)
#     which = np.digitize(r_norm, bins) - 1
#     radial = np.array([P[which == i].mean() if np.any(which == i) else 0.0 for i in range(nbins)])
#     x = 0.5 * (bins[:-1] + bins[1:])
#     # drop the DC bin
#     return x[1:], radial[1:]

# def hff_mask_only(raw_img, mask, f_cut=0.25, taper_px=5, eps=1e-3):
#     """
#     High-Frequency Fraction computed strictly from the masked region.
#     - raw_img: 2D (H,W) or 3D (Z,H,W)
#     - mask: boolean same shape
#     Steps: robust rescale within mask -> soft taper -> FFT -> |F(I*W)|^2
#            divide by |F(W)|^2 + eps (mask-spectrum compensation)
#            radial average -> fraction above f_cut
#     """
#     assert raw_img.shape == mask.shape
#     img = raw_img.astype(float)
#     # img = robust_rescale(raw_img, mask)
#     W = soft_taper_from_mask(mask, taper_px=int(taper_px))

#     # windowed signal
#     X = img * W

#     # FFT and spectra
#     if img.ndim == 2:
#         F = np.fft.fftshift(np.fft.fft2(X))
#         FW = np.fft.fftshift(np.fft.fft2(W))
#     else:
#         F = np.fft.fftshift(np.fft.fftn(X))
#         FW = np.fft.fftshift(np.fft.fftn(W))

#     P_meas = np.abs(F)**2
#     P_win  = np.abs(FW)**2

#     # compensate window/mask blur (ratio PSD)
#     P_comp = P_meas / (P_win + eps)

#     # radial average
#     x, radial = radial_average(P_comp)

#     # normalize to obtain a proper fraction
#     total = radial.sum() + 1e-12
#     hf = radial[x > f_cut].sum()
#     return float(hf / total)





def hff_mask_only(img, mask, taper_px=5, f_cut=0.25, eps=1e-3):

    img = img.astype(float)

    W = soft_taper_from_mask(mask, taper_px=int(taper_px))

    X = img * W


    grid_x, grid_y = np.meshgrid(
        np.linspace(-1, 1, img.shape[1]),
        np.linspace(-1, 1, img.shape[0])
    )

    dist = np.sqrt(grid_x**2 + grid_y**2)

    inner_mask = dist < 0.25
    outer_mask = dist >= 0.25



    F = np.fft.fftshift(np.fft.fft2(X))
    FW = np.fft.fftshift(np.fft.fft2(W))

    P_meas = np.abs(F)**2
    P_win  = np.abs(FW)**2

    P_comp = P_meas / (P_win + eps)

    # inner_power = P_comp[inner_mask].sum()
    outer_power = P_comp[outer_mask].sum()

    hff = outer_power / (P_comp.sum()+ 1e-12)

    return hff


if __name__ == "__main__":

    img = imread('image_test.tif')
    mask = imread('mask_test.png')

    mask = mask == 1

    bbox = cv2.boundingRect(mask.astype(np.uint8))
    img_crop = img[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]]
    mask_crop = mask[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]]


    hff = hff_mask_only(img_crop, mask_crop)
    print(hff)

    