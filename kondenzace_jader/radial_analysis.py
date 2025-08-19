

import numpy as np
from skimage.io import imread
import matplotlib.pyplot as plt
import scipy.ndimage as ndi
import cv2
from scipy.interpolate import RegularGridInterpolator
from scipy.interpolate import interp1d
from skimage.measure import label, regionprops, regionprops_table

def radial_analysis(img, mask):

    
    centroid = regionprops(mask.astype(np.uint8))[0].centroid


    
    x = np.arange(mask.shape[1])
    y = np.arange(mask.shape[0])
    X, Y = np.meshgrid(x, y)
    mask_interpolator = RegularGridInterpolator((y, x), mask, bounds_error=False)
    img_interpolator = RegularGridInterpolator((y, x), img, bounds_error=False)
    
    line_final_len = 600
    lines_values = []



    for angle in range(360):

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



    lines_values_avg = np.mean(lines_values, axis=0)

    return lines_values_avg, centroid






if __name__ == "__main__":

    img = imread('image_test.tif')
    mask = imread('mask_test.png')

    mask = mask == 1

    bbox = cv2.boundingRect(mask.astype(np.uint8))
    img_crop = img[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]]
    mask_crop = mask[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]]


    radial_signal = radial_analysis(img_crop, mask_crop)

    plt.plot(radial_signal, label='Radial Signal', linewidth=1.5)
    plt.show()

    