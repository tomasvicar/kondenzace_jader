
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

        
        # contours, _ = cv2.findContours(nuc_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # contour = contours[0]
        # points_orig = contour[:, 0, :]


        # ## distance to ellipse
        # points = reinterpolate_contour(points_orig, point_distance=0.5)
        # model_ellipse = EllipseModel()
        # model_ellipse.estimate(points)
        # xc, yc, a, b, theta = model_ellipse.params


        centroid = regionprops(nuc_mask)[0].centroid

        # plt.imshow(nuc_mask, cmap='gray')
        # plt.scatter(centroid[1], centroid[0], color='red', s=50, label='Center')
        # plt.axis('image')
        # plt.legend()
        # plt.show()

        
        x = np.arange(nuc_mask.shape[1])
        y = np.arange(nuc_mask.shape[0])
        X, Y = np.meshgrid(x, y)
        mask_interpolator = RegularGridInterpolator((y, x), nuc_mask, bounds_error=False)
        img_interpolator = RegularGridInterpolator((y, x), data_2d_dapi, bounds_error=False)
        
        line_final_len = 600
        lines_values = []



        for angle in range(360):

            # generate line from center 500 length with this angel
            line_max_len = 300
            
            x0, y0 = centroid[1], centroid[0]
            x1 = int(x0 + line_max_len * np.cos(np.deg2rad(angle)))
            y1 = int(y0 + line_max_len * np.sin(np.deg2rad(angle)))
            linex = np.linspace(x0, x1, num=3*line_max_len)
            liney = np.linspace(y0, y1, num=3*line_max_len)

            values_mask = mask_interpolator((liney, linex))
            values_img = img_interpolator((liney, linex))

            keep = values_mask > 0.5

            linex = linex[keep]
            liney = liney[keep]
            values_mask = values_mask[keep]
            values_img = values_img[keep]


            f = interp1d(np.linspace(0, 1, len(values_img)), values_img, kind='linear')
            x_resampled = np.linspace(0, 1, line_final_len)
            values_img_interpolated = f(x_resampled)

            lines_values.append(values_img_interpolated)

            # plt.plot(linex, liney, color='red', linewidth=0.5)




        # plt.show()

        # plt.plot(np.arange(line_final_len), np.mean(lines_values, axis=0), label=f'Nucleus {nuc_num}', linewidth=1.5)

        lines_values_avg = np.mean(lines_values, axis=0)



        break
    break

