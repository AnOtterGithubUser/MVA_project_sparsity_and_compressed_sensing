import numpy as np
import math
import scipy.stats as st


def check_if_rgb(I):
    """Check if the input image is in RGB

    Parameters
    ----------
    I: numpy.ndarray
        Image (Height x Width x Channels)
    """
    if np.any(I > 255) or np.any(I < 0):
        return False
    return True


def is_iterable(a):
    try:
        _ = (e for e in a)
        return True
    except TypeError:
        return False


def K(intensity, sigma=1., mu=0.):
    """Gaussian function

    Parameters
    ----------
    intensity: int
        Intensity at point p

    sigma: float
        Standard deviation

    mu: float
        Mean
    """
    if is_iterable(intensity):  # input is iterable
        return np.array([K(i, sigma, mu) for i in intensity])
    return (1/(math.sqrt(2*math.pi)*sigma)) * math.exp(-(intensity - mu)**2 / 2*sigma**2)


def K_prime(intensity, sigma=1., mu=0.):
    """Derivative of gaussian function

    Parameters
    ----------
    intensity: int or iterable
        Intensity at point p

    sigma: float
        Standard deviation

    mu: float
        Mean
    """
    if is_iterable(intensity):  # input is iterable
        return np.array([K_prime(i, sigma, mu) for i in intensity])
    return -((intensity - mu) / (math.sqrt(2*math.pi)*sigma)) * math.exp(-(intensity - mu)**2 / 2*sigma**2)


def gaussian_kernel(neighborhood_size=3, sigma=1):
    """Build a gaussian kernel matrix

    Parameters
    ----------
    shape: int or tuple
        Shape of the output matrix

    Returns
    -------
    gkernel: numpy.ndarray
        Gaussian kernel matrix
    """
    interval = (2*sigma+1.)/(neighborhood_size)
    x = np.linspace(-sigma-interval/2., sigma+interval/2., neighborhood_size+1)
    kern1d = np.diff(st.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw/kernel_raw.sum()
    return kernel

def make_hist(I):
    """Returns the histogram of the image

    Parameters
    ----------
    I: numpy.ndarray
        Array of size HxWx1

    Returns
    -------
    histogram: dict
        Intensity -> number of pixels where I=Intensity
    """
    histogram = {i: 0 for i in range(256)}
    for intensity in I.astype(int).flatten():
        histogram[intensity] += intensity
    return histogram

def find_closest_mode(intensity, D):
    """Find the closest mode to the pixel value

    Parameters
    ----------
    intensity: int
        Intensity at the pixel

    D: numpy.ndarray
        Local histogram derivative at pixel

    Returns
    -------
    mode: int
        Closest mode
    """
    if D[intensity] > 0:
        # find the closest mode superior to pixel value
        for i in range(len(D[intensity:])):
            if D[i] * D[i+1] < 0: # crossed a mode
                mode = intensity + i  # Note: we will consider only integer intensities here
                return mode
        return intensity  # if no mode was found return identity
                          # Note: this is an arbitrary choice and may change
                          # depending on the results
    else:
        # find the closest mode inferior to pixel value
        for j in range(len(D[:intensity])):
            if D[intensity-j-1] * D[intensity-j] < 0:  # crossed a mode
                mode = intensity - j
                return mode
        return intensity



