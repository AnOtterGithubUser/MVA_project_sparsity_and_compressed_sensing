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
    if np.any(a > 255) or np.any(a < 0):
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