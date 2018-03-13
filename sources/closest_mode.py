from utils import check_if_rgb, K_prime, gaussian_kernel
from scipy.signal import fftconvolve


def closest_mode_filter(I, sigma, freq=1./256, neighborhood_size=9):
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
	if I.shape[2] > 1:  # multi channel image
		if check_if_rgb(I):
			# convert to HSV
		else:
			raise ValueError("Input image is not RGB")
	I_closest = np.array(I.shape)
	if I_closest.shape[2] > 1:
		I_closest[:, :, 0] = I[:, :, 0]
		I_closest[:, :, 1] = I[:, :, 1]
		canal_to_filter = I_closest[:, :, 2]
	else:
		canal_to_filter = I_closest[:, :, 0]
	
	# Compute lookup table
	lookup_K_prime = {i: K_prime(i - samples, sigma) for i in range(256)}

	# Map the image
	def lookup(a):
		return lookup_K_prime[int(a)]
	vlookup = np.vectorize(lookup)
	canal_to_filter = vlookup(canal_to_filter)

	# Convolve the result with the spatial kernel
	W = gaussian_kernel(9)
	canal_to_filter = fftconvolve(canal_to_filter, W)

	# Get the histogram derivative

	# Find the closest mode for each pixel


	return I_closest