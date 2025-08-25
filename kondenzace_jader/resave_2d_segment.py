
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
from tifffile import imwrite
from skimage.io import imsave
from IPython.display import clear_output
from skimage.morphology import binary_erosion, disk

from utils.visboundaries import visboundaries
from utils.read_ics_file import read_ics_file_ordered
from utils.mat2gray import mat2gray_rgb_percentile
from utils.reinterpolate_contour import reinterpolate_contour
from get_focus import get_focus


data_path = r"D:\kondenzace_jader\data\Chromatin Architecture Analyses Python\Mikroskop horní\zz_Original data files  reduced"
results_path = data_path + '_results'



percentile_r = [2 , 99.8]
percentile_g = [2, 99.8]
percentile_b = [2, 98.0]

gaus_sigma = (1, 1, 0.3)
med_size = (5, 5, 1)


model_name = 'nuc3_20250801_1520'
model = models.CellposeModel(gpu=True, pretrained_model=model_name)



fnames = glob(data_path + '/**/*01.ics', recursive=True)
print(f'Number of files: {len(fnames)}')

for fname_ind, fname in enumerate(fnames):
    print(f'{fname_ind+1}/{len(fnames)}')
    print(f'{fname}')

    save_name = fname.replace(data_path, results_path)
    save_name_mip = save_name.replace('.ics', '') + '_mip.tif'
    save_name_mip_png = save_name.replace('.tif', '') + '_mip_norm.png'
    save_name_focus = save_name.replace('.ics', '_focus.tif')
    save_name_focus_png = save_name.replace('.ics', '_focus_norm.png')
    save_name_seg = save_name.replace('.ics', '_seg.tif')
    save_name_seg_example = save_name.replace('.ics', '_seg_example.png')



    if all([os.path.exists(fname) for fname in [save_name_mip, save_name_mip_png, save_name_focus, save_name_focus_png, save_name_seg,save_name_seg_example]]):
        print('File already exists. Continue to next.')
        continue
    else:
        print('Creating file.')

    data = read_ics_file_ordered(fname)



    data = gaussian_filter(data, sigma=gaus_sigma, axes=(0, 1, 2))
    data = median_filter(data, size=med_size, axes=(0, 1, 2))

    data_2d_mip = np.max(data, axis=2)  

    os.makedirs(os.path.dirname(save_name_mip), exist_ok=True)
    imwrite(save_name_mip, data_2d_mip)


    percentiles = [percentile_r, percentile_g, percentile_b]
    data_2d_mip, percentiles_values = mat2gray_rgb_percentile(data_2d_mip, percentiles, get_percitile=True)
    data_2d_mip = np.round(data_2d_mip * 255).astype(np.uint8)

    imsave(save_name_mip_png, data_2d_mip)


    focus_ind = get_focus(data)
    data_2d_focus = data[:,:,focus_ind,:]

    os.makedirs(os.path.dirname(save_name_focus), exist_ok=True)
    imwrite(save_name_focus, data_2d_focus)

    data_2d_focus, percetiles_values = mat2gray_rgb_percentile(data_2d_focus, percentiles, get_percitile=True)
    data_2d_focus = np.round(data_2d_focus * 255).astype(np.uint8)

    imsave(save_name_focus_png, data_2d_focus)

    mask, flow, style = model.eval(data_2d_focus)


    imwrite(save_name_seg, mask)


    plt.imshow(data_2d_focus)
    visual_mask = np.zeros_like(mask)
    uu = np.unique(mask)
    uu = uu[uu > 0]
    for u in uu:
        visual_mask[binary_erosion(mask == u, disk(1))] = u
    visboundaries(visual_mask>0, linewidth=0.5)
    plt.savefig(save_name_seg_example)
    plt.show()
    plt.close()

    if ((fname_ind  + 1) % 10) == 0:
        clear_output()


