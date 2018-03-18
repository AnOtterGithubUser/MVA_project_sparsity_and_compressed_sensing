from utils import check_if_rgb, K_prime, gaussian_kernel, make_hist
from scipy.signal import fftconvolve
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb
import numpy as np
import matplotlib.pyplot as plt


def closest_mode_filter(I, sigma, freq=1., neighborhood_size=9):
    """Apply closest mode filtering on image I

    Parameters
    ----------
    I: numpy.ndarray
        Image to filter (Height x Width x Channels)

    sigma: float
        Standard deviation of the gaussian kernel

    freq: float
        Sampling rate over the 256 values for intensity (RGB)

    neighborhood_size: int or iterable
        If int, the neighborhood is a square,
        otherwise it is a rectangle

    Returns
    -------
    I_closest: numpy.ndarray
        Filtered image (Height x Width x Channels)
    """
    import pdb
    pdb.set_trace()
    if I.shape[2] > 1:  # multi channel image
        if check_if_rgb(I):
            I = rgb_to_hsv(I/255)  # convert to HSV
        else:
            raise ValueError("Input image is not RGB")
    I_closest = np.zeros(I.shape)
    if I_closest.shape[2] > 1:
        I_closest[:, :, 0] = I[:, :, 0]
        I_closest[:, :, 1] = I[:, :, 1]
        canal_to_filter = I[:, :, 2]
    else:
        canal_to_filter = I[:, :, 0]

    canal_to_filter = canal_to_filter*255
    canal_to_filter = canal_to_filter.astype(int)
    ## Temporary: histogram of the image
    original_hist = make_hist(canal_to_filter)
    # plt.bar(original_hist.keys(), original_hist.values())
    # plt.title("Original histogram")
    # plt.show()
    
    # Compute lookup table
    samples = np.arange(int(freq * 256))
    lookup_K_prime = {i: K_prime(i - samples, sigma) for i in range(256)}

    # Build the spatial kernel
    W = gaussian_kernel(3)

    # Get Di for i in [0, 256]
    pdb.set_trace()
    D = {i: 0 for i in range(256)}
    for i in D.keys():
        # Map the intensity of each pixel
        mapping_i = np.zeros(canal_to_filter.shape)
        for x in range(mapping_i.shape[0]):
            for y in range(mapping_i.shape[1]):
                mapping_i[x, y] = lookup_K_prime[canal_to_filter[x, y]][i]
        # Convolve with spatial kernel
        D[i] = fftconvolve(mapping_i, W, mode='same')
    pdb.set_trace()

    # Find the closest mode for each pixel
    filtered_canal = np.zeros(canal_to_filter.shape)
    for x in range(filtered_canal.shape[0]):
        for y in range(filtered_canal.shape[1]):
            # Get the histogram derivative at the point of coordinate (x, y)
            D_p = np.array([D[i][x, y] for i in i.keys()])
            filtered_canal[x, y] = find_closest_mode(canal_to_filter[x, y], D_p)
    if I_closest.shape[2] > 1:
        I_closest[:, :, 2] = filtered_canal / 255
    else:
        I_closest[:, :, 0] = filtered_canal / 255

    # Convert back to RGB
    I_closest = hsv_to_rgb(I_closest)

    return I_closest