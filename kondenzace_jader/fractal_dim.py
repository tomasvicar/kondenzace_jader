import numpy as np
from skimage.io import imread
import matplotlib.pyplot as plt
import scipy.ndimage as ndi
from skimage import exposure, img_as_float
import cv2







def compute_fractal_dimension(img, mask, plot=False):

    img = (img > 0).astype(np.float32)
    mask = (mask > 0).astype(np.float32)
    img[mask == 0] = 0

    scales = np.round(2 ** np.arange(1, 4.5, 0.3)).astype(int)



    counts = []
    for scale_ind, scale in enumerate(scales):
        # Apply the box filter
        # img_filtered = ndi.uniform_filter(img, size=scale, mode='reflect')
        # mask_filtered = ndi.uniform_filter(mask, size=scale, mode='constant') 

        img_filtered = ndi.uniform_filter(img, size=scale, mode='constant')
        # mask_filtered = ndi.uniform_filter(mask, size=scale, mode='constant') 
        #crop valid part

        # img_filtered = img_filtered[scale//2:-scale//2, scale//2:-scale//2]

        # print(f'Scale: {scale}')
        # plt.imshow(img_filtered, cmap='gray')
        # plt.show()


        positions_occupied = (img_filtered > 0)
        positions_occupied = positions_occupied * mask
        # positions_in_mask = (mask_filtered > 0.5)



        # plt.imshow(positions_edge, cmap='gray')
        # plt.show()


        # total_positions = np.prod(img_filtered.shape)
        positions_edge_norm = np.sum(positions_occupied) / (scale  ** 2)

        counts.append(positions_edge_norm)


    xs, ys_log = np.log(np.array(1/scales)), np.log(np.array(counts))
    A = np.vstack([xs, np.ones_like(xs)]).T
    slope, intercept = np.linalg.lstsq(A, ys_log, rcond=None)[0]


    

    if plot:
        plt.figure()
        plt.plot(np.log(scales), np.log(counts), marker='o')
        plt.xlabel('Log(Scale)')
        plt.ylabel('Log(Count)')
        plt.title(f'Log-Log Plot  FD = {slope:.2f}')
        plt.grid()
        plt.show()


    return slope



def thresholding_fractal(img, mask):
    
    # img_orig = img.copy()

    saturation = 0.35
    p_low, p_high = np.percentile(img, (saturation, 100 - saturation))
    img = exposure.rescale_intensity(img, in_range=(p_low, p_high))

 
    img = exposure.equalize_hist(img)

    # plt.imshow(img, cmap='gray')
    # plt.show()

    # hist = np.histogram(img[mask].ravel(), bins=100)
    # print(np.mean(img[mask]), np.std(img[mask]))


    # plt.hist(img[mask].ravel(), bins=100, alpha=0.5, label='Image', )
    # plt.legend()
    # plt.show()

    binarized = img > np.mean(img[mask])

    # plt.imshow(img)
    # plt.show()
    # plt.imshow(binarized, cmap='gray')
    # plt.show()

    return binarized

if __name__ == "__main__":
   
    img = imread('image_test.tif')
    mask = imread('mask_test.png')

    mask = mask == 1

    bbox = cv2.boundingRect(mask.astype(np.uint8))
    img = img[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]]
    mask = mask[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]]

    binarized = thresholding_fractal(img, mask)
    fd = compute_fractal_dimension(binarized, mask, plot=True)

    print(fd)





