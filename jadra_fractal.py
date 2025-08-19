
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
from fractal_dim import compute_fractal_dimension


data_path = r"D:\kondenzace_jader\data\Chromatin Architecture Analyses Python\Mikroskop horní\zz_Original data files"
output_path = r"D:\kondenzace_jader\data\results"





import numpy as np
from scipy import ndimage as ndi
from skimage import morphology

from skimage.io import imread, imsave







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

    imsave('image_test.png', data_2d_dapi)

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


        bbox = cv2.boundingRect(nuc_mask)
        img_crop = data_2d_dapi[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]]
        mask_crop = nuc_mask[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]]

        img_crop = img_crop * mask_crop

        img_crop_binary = img_crop > 170

        fd = compute_fractal_dimension(img_crop_binary, mask_crop, plot=False)

        print(fd)

        

        break
    break

