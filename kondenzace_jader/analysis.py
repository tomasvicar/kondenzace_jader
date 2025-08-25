


from email.mime import base, image
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import os
import tifffile
import skimage.io
from IPython.display import clear_output
import cv2
from skimage.morphology import binary_erosion, disk
from skimage import exposure
import json
from skimage.filters import threshold_otsu

from utils.visboundaries import visboundaries
from utils.mat2gray import mat2gray
from fractal_dim import thresholding_fractal, compute_fractal_dimension
from fourier_hff import hff_mask_only
from radial_analysis import radial_analysis
from total_perimeter_domains_tpd import compute_total_perimeter_domains_tpd, normalize_tpd
from skimage.morphology import remove_small_objects
from scipy import ndimage as ndi
from skimage import morphology

from scipy.ndimage import generic_filter
from scipy.signal import convolve2d
from scipy.stats import entropy
from skimage import filters



data_path = r"D:\kondenzace_jader\data\Chromatin Architecture Analyses Python\Mikroskop horní\zz_Original data files  reduced"
results_path = data_path + '_results'


plots = False

fnames = glob(results_path + '/**/*01_focus.tif', recursive=True)


for fname_ind, fname in enumerate(fnames):
    print(f'{fname_ind+1}/{len(fnames)}')
    print(f'{fname}')



    mask_name = fname.replace('_focus.tif', '') + '_seg.tif'
    mask = tifffile.imread(mask_name)


    border = np.zeros_like(mask, dtype=bool)
    border_size = 5
    border[border_size:-border_size, border_size:-border_size] = True
    border = (border == False)

    # remove cells touching border
    u_cells = np.unique(mask)
    u_cells = u_cells[u_cells>0]
    for u in u_cells:
        if np.any(border[mask == u]):
            mask[mask == u] = 0



    mask = remove_small_objects(mask, min_size=500, connectivity=1)






    fname_tif = fname
    img = tifffile.imread(fname_tif)[:,:,2]

    fname_png = fname_tif.replace('_focus.tif', '') + '_focus_norm.png'
    img_png = skimage.io.imread(fname_png)[:,:,2]
    # img =  skimage.io.imread(fname)


    u_cells = np.unique(mask)
    u_cells = u_cells[u_cells>0]

    features = dict()
    for cell_num in u_cells:
        print(f'Cell {cell_num}')


        mask_cell = mask == cell_num

        bbox = cv2.boundingRect(mask_cell.astype(np.uint8))
        img_bbox = img[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]]
        img_bbox_png = img_png[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]]
        mask_bbox = mask_cell[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]]

        img_bbox = img_bbox.astype(np.float32)
        img_bbox_std = (img_bbox - np.mean(img_bbox[mask_bbox>0])) / np.std(img_bbox[mask_bbox>0])


        # binarized_fractal = thresholding_fractal(img_bbox, mask_bbox)

        saturation = 0.35
        p_low, p_high = np.percentile(img_bbox, (saturation, 100 - saturation))
        img_bbox_norm = exposure.rescale_intensity(img_bbox, in_range=(p_low, p_high))



        # th =  np.mean(img_bbox_norm[mask_bbox])

        # th = threshold_otsu(img_bbox_norm[mask_bbox])

        # threshold with largest contour:
        contour_sizes = []
        ths = np.arange(0.3, 0.9, 0.02)
        for t in ths:
            tmp = img_bbox_norm > t
            contour = tmp & (binary_erosion(tmp, disk(1)) == 0)
            # contour_size = np.sum(contour) / np.sum(tmp)
            contour_size = np.sum(contour)
            contour_sizes.append(contour_size)
        th = ths[np.argmax(contour_sizes)]

        # 


        # fds = []
        # ths = np.arange(0.3, 0.9, 0.02)
        # for th in ths:
        #     tmp = img_bbox_norm > th
        #     fractal_dim, scales, counts = compute_fractal_dimension(tmp, mask_bbox, plot=False)
        #     fds.append(fractal_dim)
        #     plt.imshow(tmp, cmap='gray')
        #     plt.title(f'Threshold: {th:.2f}  Fractal dim: {fractal_dim:.3f}')
        #     plt.show()
        # print(fds)
        # th = ths[np.argmin(fds)]


        binarized_fractal = img_bbox_norm > th

        fractal_dim, scales, counts = compute_fractal_dimension(binarized_fractal, mask_bbox, plot=False)
        features['fractal_dim'] = fractal_dim

        save_name_fractal = fname.replace('_focus.tif', '') + f'_cell_{cell_num}_fractal.png'
        binarized_fractal[mask_bbox == 0] = False


        if plots:
            plt.subplot(1, 2, 1)
            plt.imshow(img_bbox, cmap='gray')
            visboundaries(binarized_fractal)
            plt.subplot(1, 2, 2)
            plt.plot(np.log(scales), np.log(counts), marker='o')
            plt.xlabel('Log(Scale)')
            plt.ylabel('Log(Count)')
            plt.title(f'Fractal dim = {fractal_dim:.3f}')
            plt.savefig(save_name_fractal)
            plt.show()
            plt.close()



        hff = hff_mask_only(img_bbox_std, mask_bbox)
        save_name_hff = fname.replace('_focus.tif', '') + f'_cell_{cell_num}_fourier.png'

        if plots:
            plt.imshow(img_bbox, cmap='gray')
            visboundaries(mask_bbox>0)
            plt.title(f'HFF = {hff:.3f}')
            plt.savefig(save_name_hff)
            plt.show()
            plt.close()

        features['fourier_hff'] = hff



        img_norm = normalize_tpd(img_bbox)
        tpd, skeleton, edges_bin, edges = compute_total_perimeter_domains_tpd(img_norm, mask_bbox)

        save_name_tpd = fname.replace('_focus.tif', '') + f'_cell_{cell_num}_tpd.png'


        # skeleton_2show = np.repeat(img_bbox[:,:,np.newaxis], 3, axis=2)
        # skeleton_2show[skeleton] = [255, 0, 0]
        tmp = np.repeat(img_norm[:,:,np.newaxis], 3, axis=2)
        tmp[skeleton] = [255, 0, 0]


        if plots:
            plt.figure()
            plt.subplot(1, 2, 1)
            plt.imshow(edges, cmap='gray')
            visboundaries(edges_bin>0, linewidth=0.5)
            plt.subplot(1, 2, 2)
            plt.imshow(tmp)
            plt.title(f'TPD = {tpd:.3f}')
            plt.savefig(save_name_tpd)
            plt.show()
            plt.close()

        features['tpd'] = tpd




        radial_signal, center_of_mass = radial_analysis(img_bbox, mask_bbox)

        split_position = [0.45, 0.9] # skip end part
        radial_signal_norm = mat2gray(radial_signal)
        inner = np.mean(radial_signal_norm[:int(len(radial_signal_norm)*split_position[0])])
        outer = np.mean(radial_signal_norm[int(len(radial_signal_norm)*split_position[0]):int(len(radial_signal_norm)*split_position[1])])
        not_used_border = np.mean(radial_signal_norm[int(len(radial_signal_norm)*split_position[1]):])

        radial_in_out_fraction = inner / outer
        max_position = np.argmax(radial_signal) / len(radial_signal)

        features['radial_in_out_fraction'] = radial_in_out_fraction
        features['radial_max_position'] = max_position


        save_name_radial = fname.replace('_focus.tif', '') + f'_cell_{cell_num}_radial.png'

        if plots:
            plt.figure()
            plt.subplot(1, 2, 1)
            plt.imshow(img_bbox, cmap='gray')
            visboundaries(mask_bbox>0)
            plt.scatter(center_of_mass[1], center_of_mass[0])
            plt.subplot(1, 2, 2)
            x = np.arange(len(radial_signal))
            plt.plot(x[:int(len(radial_signal)*split_position[1])], radial_signal[:int(len(radial_signal)*split_position[1])])
            plt.plot(x[int(len(radial_signal)*split_position[1]):], radial_signal[int(len(radial_signal)*split_position[1]):], color='orange')
            plt.title(f'Radial in out frac = {radial_in_out_fraction:.3f}   Max position = {max_position:.3f}')
            plt.savefig(save_name_radial)
            plt.show()
            plt.close()



        std_sizes = [3, 5, 11, 15, 21]
        stds = []
        total_std = np.std(img_bbox[mask_bbox>0])
        for std_size in std_sizes:
            tmp = generic_filter(img_bbox, np.std, size=std_size)
            stds.append(tmp)
            features[f'localstd_{std_size}'] = np.sum(tmp[mask_bbox>0]) / total_std

        max_std = np.max(stds, axis=0)
        argmax_std = np.argmax(stds, axis=0)    

        features['localstd_max'] = np.mean(max_std[mask_bbox>0]) / total_std
        features['localstd_argmax'] = np.mean(argmax_std[mask_bbox>0]) / total_std


        if plots:
            plt.figure(figsize=(12, 6))
            plt.subplot(1, 2, 1)
            plt.imshow(max_std, cmap='gray')
            plt.title(f'Local Std Max = {features["localstd_max"]:.3f}')
            plt.subplot(1, 2, 2)
            plt.imshow(argmax_std, cmap='gray')
            plt.title(f'Local Std Argmax = {features["localstd_argmax"]:.3f}')
            plt.savefig(fname.replace('_focus.tif', '') + f'_cell_{cell_num}_localstd.png')
            plt.show()

        def fspecial_log(filter_size: int, sigma: float) -> np.ndarray:
            """

            The continuous LoG:
                LoG(x, y; σ) = ((x^2 + y^2 - 2σ^2) / (σ^4)) * exp(-(x^2 + y^2) / (2σ^2))

            The discrete kernel is sampled on a square grid of size `filter_size`,
            centered at zero. We also subtract the mean to enforce near-zero sum,
            matching the behavior of `fspecial('log')` (minor numerical deviations may occur).
            """
            # grid
            rad = filter_size // 2
            y, x = np.mgrid[-rad:rad+1, -rad:rad+1]
            r2 = x*x + y*y

            # raw LoG
            s2 = sigma * sigma
            kernel = (r2 - 2.0 * s2) / (s2 * s2) * np.exp(-r2 / (2.0 * s2))

            # normalize to zero mean (fspecial('log') has sum ~ 0)
            kernel -= kernel.mean()

            return kernel
        


        sigmas = np.asarray([2, 4, 6, 8], dtype=float)
        H, W = img_bbox.shape
        K = sigmas.size
        responses = np.zeros((H, W, K), dtype=np.float32)
        gamma = 2.0

        

        for idx, sig in enumerate(sigmas):
            filter_size = int(2 * np.ceil(3.0 * sig) + 1)
            hn = (sig ** gamma) * fspecial_log(filter_size, sig)

            # symmetric boundary, 'same' output, like your conv2_spec_symetric


            resp = convolve2d(img_bbox_std, hn, mode='same', boundary='symm')
            # plt.imshow(resp, cmap='gray')
            # plt.title(f'Sigma: {sig}')
            # plt.axis('off')
            # plt.show()
            responses[..., idx] = resp.astype(np.float32, copy=False)

            tmp = np.mean(responses[..., idx][mask_bbox>0])
            features[f'log_map_{sig}_mean'] = tmp

            tmp = np.mean(np.abs(responses[..., idx][mask_bbox>0]))
            features[f'log_map_{sig}_mean_abs'] = tmp

            tmp = responses[..., idx][mask_bbox>0]
            tmp = np.mean( tmp [tmp > 0])
            features[f'log_map_{sig}_mean_pos'] = tmp

            tmp = responses[..., idx][mask_bbox>0]
            tmp = np.mean( tmp [tmp < 0])
            features[f'log_map_{sig}_mean_neg'] = tmp


        # aggregate across scales: -min over k
        log_map_min = np.min(responses, axis=2)
        features['log_map_min_mean'] = np.mean(log_map_min[mask_bbox>0])
        log_map_max = np.max(responses, axis=2)
        features['log_map_max_mean'] = np.mean(log_map_max[mask_bbox>0])
        

        # max positive
        tmp = log_map_max[mask_bbox>0]
        tmp = np.mean( tmp [tmp > 0])
        features['log_map_max_mean_pos'] = tmp


        # min negative
        tmp = log_map_min[mask_bbox>0]
        tmp = np.mean( tmp [tmp < 0])
        features['log_map_min_mean_neg'] = tmp


        tmp = np.max(np.abs(responses), axis=2)
        # max_abs
        tmp = np.mean(tmp[mask_bbox>0])
        features['log_map_max_mean_abs'] = tmp

        log_map_argmin = np.argmin(responses, axis=2)
        features['log_map_argmin_mean'] = np.mean(log_map_argmin[mask_bbox>0])
        log_map_argmax = np.argmax(responses, axis=2)
        features['log_map_argmax_mean'] = np.mean(log_map_argmax[mask_bbox>0])


        # min
        tmp = log_map_argmin[mask_bbox>0]
        tmp = np.mean( tmp [tmp < 0])
        features['log_map_argmin_mean_neg'] = tmp

        # max
        tmp = log_map_argmax[mask_bbox>0]
        tmp = np.mean( tmp [tmp > 0])
        features['log_map_argmax_mean_pos'] = tmp


        # argmax abs
        tmp = log_map_argmax[mask_bbox>0]
        tmp = np.mean(np.abs(tmp [tmp > 0]))
        features['log_map_argmax_mean_abs'] = tmp



        
        lap = filters.laplace(img_bbox_png.astype(np.float32))   # Laplacian filter
        focus = lap[mask_bbox>0].var()
        features['focus_laplacian'] = focus



        if plots:
            plt.figure(figsize=(12, 6))
            plt.subplot(1, 2, 1)
            plt.imshow(log_map_argmin, cmap='gray')
            plt.title(f'Log Map Argmin = {features["log_map_argmin_mean"]:.3f}')
            plt.subplot(1, 2, 2)
            plt.imshow(log_map_argmax, cmap='gray')
            plt.title(f'Log Map Argmax = {features["log_map_argmax_mean"]:.3f}')
            plt.savefig(fname.replace('_focus.tif', '') + f'_cell_{cell_num}_logmap.png')
            plt.show()




        def shannon_entropy_masked(img: np.ndarray,
                                mask: np.ndarray,
                                base: float = 2.0,
                                nbins: int | None = None,
                                vmin=None,
                                vmax=None) -> float:
            """Shannon entropy H(X) of intensity distribution inside `mask`.

            Parameters
            ----------
            img : 2D float array
            mask : 2D bool array
            base : log base (2 => bits, e => nats)
            nbins : if None, use Freedman–Diaconis; else fixed number of bins.

            Returns
            -------
            H : float
                Entropy in units of `base` (e.g., bits).
            """
            vals = img[mask > 0].ravel()
            if vals.size == 0:
                return np.nan

            if vmin is None or vmax is None:
                vmin, vmax = np.nanmin(vals), np.nanmax(vals)
                if vmin == vmax:
                    return 0.0

            if nbins is None:
                # Freedman–Diaconis bin width
                q25, q75 = np.percentile(vals, [25, 75])
                iqr = max(q75 - q25, np.finfo(vals.dtype).eps)
                bin_width = 2.0 * iqr / np.cbrt(vals.size)
                nbins = int(np.clip(np.ceil((vmax - vmin) / bin_width), 16, 512))

            hist, _ = np.histogram(vals, bins=nbins, range=(vmin, vmax), density=False)
            p = hist.astype(np.float64)
            p /= p.sum()

            # Remove zeros to avoid log problems
            p = p[p > 0]
            H = entropy(p, base=base)
            return float(H)
        
        entropy_whole = shannon_entropy_masked(img_bbox_std, mask_bbox, nbins=64, vmin=0, vmax=255)

        features['entropy'] = entropy_whole

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

        def spectral_entropy_2d(img: np.ndarray,
                        mask: np.ndarray,
                        base: float = 2.0,
                        eps: float = 1e-12) -> float:
            """Shannon entropy of the normalized 2-D power spectrum within the masked crop.

            Notes
            -----
            - We multiply by the mask to avoid boundary leakage, and apply a Hanning window
            to reduce edge effects before the FFT.
            - Entropy is computed over the nonnegative frequencies (full PSD is symmetric);
            here we simply use the full PSD after fftshift and normalize it.
            """
            if np.count_nonzero(mask) < 4:
                return np.nan

            # windowing to reduce leakage
            h, w = img.shape
            win_y = np.hanning(max(h, 2))
            win_x = np.hanning(max(w, 2))
            window = np.outer(win_y, win_x)[:h, :w]

            x = img.astype(np.float64) * window
            x *= (mask > 0)  # zero outside mask to suppress background energy

            F = np.fft.fft2(x)
            S = np.abs(F) ** 2
            S = np.fft.fftshift(S)

            # restrict to masked region’s convex hull could be better; here we keep all and
            # guard normalization by eps
            P = S.ravel().astype(np.float64)
            P_sum = P.sum()
            if P_sum <= eps:
                return 0.0
            P /= P_sum
            P = P[P > 0]
            return float(entropy(P, base=base))
        

        spectral_entropy = spectral_entropy_2d(img_bbox_std, mask_bbox, base=2.0)
        features['spectral_entropy'] = spectral_entropy



        for key, value in features.items():
            features[key] = float(value) if np.isscalar(value) else value.tolist()

        savename_features = fname.replace('_focus.tif', '') + f'_cell_{cell_num}_features.json'
        with open(savename_features, 'w') as f:
            json.dump(features, f, indent=2)









    if ((fname_ind  + 1) % 3) == 0:
        clear_output()















