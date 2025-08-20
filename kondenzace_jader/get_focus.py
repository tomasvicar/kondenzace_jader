
from glob import glob

from scipy.ndimage import gaussian_filter
from scipy.ndimage import median_filter
import numpy as np
from skimage import filters
import matplotlib.pyplot as plt
import os
import cv2
from scipy.interpolate import RegularGridInterpolator
from scipy.interpolate import interp1d
# from skimage.measure import EllipseModel
from skimage.measure import label, regionprops, regionprops_table
from scipy.signal import resample







def focus_measure_variance_of_laplacian(image):
    """
    Sharpness measure: variance of Laplacian using skimage.
    """
    lap = filters.laplace(image.astype(float))   # Laplacian filter
    return lap.var()



def get_focus(zstack_rgb, plot=False):
    sharpnesses = []
    for slice in range(zstack_rgb.shape[2]):
        sharpness = focus_measure_variance_of_laplacian(zstack_rgb[:, :, slice, 2])
        sharpnesses.append(sharpness)

    sharp_ind = np.argmax(sharpnesses)

    if plot:
        plt.plot(sharpnesses)
        plt.xlabel("Slice")
        plt.ylabel("Sharpness (Variance of Laplacian)")
        plt.title("Sharpness Variation Across Slices")
        plt.show()

    return sharp_ind




if __name__ == "__main__":
    from utils.read_ics_file import read_ics_file_ordered
    from utils.mat2gray import mat2gray_rgb_percentile
    from utils.reinterpolate_contour import reinterpolate_contour
    from cellpose import models


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

        # data_2d = np.max(data, axis=2)  
        # data_2d_dapi = data_2d[:, :, 0]

        # percentiles = [percentile_r, percentile_g, percentile_b]
        # data_2d, percetiles_values = mat2gray_rgb_percentile(data_2d, percentiles, get_percitile=True)
        # data_2d = np.round(data_2d * 255).astype(np.uint8)

        # mask, flow, style = model.eval(data_2d)

        focus_slice = get_focus(data, plot=True)
