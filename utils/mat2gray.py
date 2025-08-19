import numpy as np

def mat2gray(img,min_max=None):
    
    if not min_max:
        
        min_max = [np.min(img),np.max(img)]
        
    min_ = min_max[0]
    max_ = min_max[1]
        
    img = (img-min_)/(max_-min_)
    
    img[img<0] = 0
    img[img>1] = 1
    
    return img

def mat2gray_rgb(img, min_max=None, color_channel=2):
    """
    Same as mat2gray, but for RGB images.
    Converts an RGB image to a normalized format where each channel is scaled independently.
    """
    if not min_max:
        min_max = [np.min(img, axis=(0, 1)), np.max(img, axis=(0, 1))]
    min_ = min_max[0]
    max_ = min_max[1]

    img_rgb = np.zeros_like(img, dtype=np.float32)
    for i in range(img.shape[color_channel]):
        channel_data = img[:, :, i]
        channel_min = min_[i]
        channel_max = max_[i]
        
        # Normalize the channel data
        normalized_channel = (channel_data - channel_min) / (channel_max - channel_min)
        
        # Clip values to [0, 1]
        normalized_channel[normalized_channel < 0] = 0
        normalized_channel[normalized_channel > 1] = 1
        
        img_rgb[:, :, i] = normalized_channel


    return img_rgb

def mat2gray_rgb_percentile(img, percentile, get_percitile=False):
    
    percentile_r, percentile_g, percentile_b = percentile

    percentile_high_r = np.percentile(img[:, :, 0], percentile_r[1])
    percentile_high_g = np.percentile(img[:, :, 1], percentile_g[1])
    percentile_high_b = np.percentile(img[:, :, 2], percentile_b[1])

    percentile_low_r = np.percentile(img[:, :, 0], percentile_r[0])
    percentile_low_g = np.percentile(img[:, :, 1], percentile_g[0])
    percentile_low_b = np.percentile(img[:, :, 2], percentile_b[0])

    percentile_low = [percentile_low_r, percentile_low_g, percentile_low_b]
    percentile_high = [percentile_high_r, percentile_high_g, percentile_high_b]

    img_rgb = mat2gray_rgb(img, min_max=[percentile_low, percentile_high], color_channel=2)
    
    if get_percitile:
        return img_rgb, [percentile_low, percentile_high]
    else:
        return img_rgb

