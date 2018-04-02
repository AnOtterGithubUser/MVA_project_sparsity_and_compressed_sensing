from utils import check_if_rgb, K_prime, K, gaussian_kernel, make_hist, find_closest_mode
from scipy.signal import fftconvolve
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb
import numpy as np
import matplotlib.pyplot as plt


def closest_mode_filter(I, sigma, freq=1., neighborhood_size=9, display_histograms=False):
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

    display_histograms: bool, default=False
        Display the histogram and histogram derivative of the image

    Returns
    -------
    I_closest: numpy.ndarray
        Filtered image (Height x Width x Channels)
    """
    message = "Closest mode filtering"
    print(message)
    print("-" * len(message))
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
    
    print("Computing lookup tables...")
    # Compute lookup table
    samples = np.arange(int(freq * 256))
    lookup_K = {i: K(i - samples, sigma) for i in range(256)}
    lookup_K_prime = {i: -K_prime(i - samples, sigma) for i in range(256)}

    # Build the spatial kernel
    W = gaussian_kernel(neighborhood_size)
    print("Mapping the image...")

    histograms = np.zeros((canal_to_filter.shape[0],
                           canal_to_filter.shape[1],
                           256))
    for x in range(canal_to_filter.shape[0]):
        for y in range(canal_to_filter.shape[1]):
            histograms[x, y] = lookup_K[canal_to_filter[x, y]]

    histogram_derivatives = np.zeros((canal_to_filter.shape[0],
                           canal_to_filter.shape[1],
                           256))
    for x in range(canal_to_filter.shape[0]):
        for y in range(canal_to_filter.shape[1]):
            histogram_derivatives[x, y] = lookup_K_prime[canal_to_filter[x, y]]

    for i in range(256):
        histograms[:, :, i] = fftconvolve(histograms[:, :, i], W, mode="same")
        histogram_derivatives[:, :, i] = fftconvolve(histogram_derivatives[:, :, i], W, mode="same")

    total_histogram = np.sum(histograms, axis=(0, 1))
    total_histogram_derivative = np.sum(histogram_derivatives, axis=(0, 1))

    if display_histograms:
        f = plt.figure()
        plt.subplot(311)
        plt.plot(make_hist(canal_to_filter).keys(), make_hist(canal_to_filter).values())
        plt.title("histogram")
        plt.subplot(312)
        plt.plot(total_histogram)
        plt.title("smoothed histogram")
        plt.subplot(313)
        plt.plot(total_histogram_derivative)
        plt.plot([0]*256)
        plt.title("histogram derivative")
        plt.show()

    print("Finding closest mode...")
    filtered_canal = np.zeros(canal_to_filter.shape)
    for x in range(filtered_canal.shape[0]):
        for y in range(filtered_canal.shape[1]):
            closest_intensity = find_closest_mode(canal_to_filter[x, y],
                                                  total_histogram_derivative)
            filtered_canal[x, y] = closest_intensity

    if I_closest.shape[2] > 1:
        I_closest[:, :, 2] = filtered_canal / 255
    else:
        I_closest[:, :, 0]  = filtered_canal / 255

    I_closest = hsv_to_rgb(I_closest) * 255
    I_closest = I_closest.astype(np.uint8)

    return I_closest