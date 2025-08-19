

import numpy as np
from skimage.io import imread
import matplotlib.pyplot as plt
import scipy.ndimage as ndi
import cv2
from scipy.interpolate import RegularGridInterpolator
from scipy.interpolate import interp1d
from skimage.measure import label, regionprops, regionprops_table
from skimage import exposure, img_as_float
from scipy.ndimage import sobel
from skimage.morphology import skeletonize
from scipy import stats




def normalize_tpd(img):

    saturation = 0.35
    p_low, p_high = np.percentile(img, (saturation, 100 - saturation))
    img = exposure.rescale_intensity(img, in_range=(p_low, p_high))
 
    img = exposure.equalize_hist(img)

    return img





def compute_total_perimeter_domains_tpd(img, mask):

     

    edges_x = sobel(img, axis=0)
    edges_y = sobel(img, axis=1)
    edges = np.hypot(edges_x, edges_y)




    th = np.mean(edges[mask])


    # values = edges[mask]
    # kde = stats.gaussian_kde(values, bw_method='scott')
    # xmin, xmax = values.min(), values.max()
    # xs = np.linspace(xmin, xmax, 2048)
    # pdf = kde(xs)
    # th =  xs[np.argmax(pdf)]

    edges_bin = edges > th
    edges_bin[mask == 0] = 0



    skeleton = skeletonize(edges_bin)

    skeleton_area = np.sum(skeleton)
    nuc_area = np.sum(mask)

    tpd = skeleton_area / nuc_area


    # plt.imshow(edges, cmap='gray')
    # plt.title('Sobel Edges')
    # plt.axis('off')
    # plt.show()

    # plt.imshow(edges_bin, cmap='gray')
    # plt.title('Binary Sobel Edges')
    # plt.axis('off')
    # plt.show()

    # plt.imshow(skeleton, cmap='gray')
    # plt.title('Skeleton of Binary Sobel Edges')
    # plt.axis('off')
    # plt.show()


    # plt.hist(edges[mask].ravel(), bins=100, color='gray', alpha=0.7)
    # # plot th
    # plt.axvline(th, color='red', linestyle='--')
    # plt.title('Histogram of Sobel Edges')
    # plt.xlabel('Edge Strength')
    # plt.ylabel('Frequency')
    # plt.show()

    return tpd, skeleton, edges_bin



if __name__ == "__main__":
   
    img = imread('image_test.tif')
    mask = imread('mask_test.png')

    mask = mask == 1


    bbox = cv2.boundingRect(mask.astype(np.uint8))
    img = img[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]]
    mask = mask[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]]

    img = normalize_tpd(img)

    tpd, skeleton, edges_bin = compute_total_perimeter_domains_tpd(img, mask)

    print(tpd)
