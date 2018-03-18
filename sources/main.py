from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from utils import K, K_prime
from closest_mode import closest_mode_filter


im = np.array(Image.open("../images/chicago2.png"))
im_crop = im[200:210, 200:210, :]

# Get the local histogram
local_hist = {i: 0 for i in range(256)}
for intensity in im_crop.flatten():
	local_hist[intensity] += 1

# Compute the look up tables
freq = 1
samples = np.arange(int(freq * 256))
lookup_K = {i: K(i - samples) for i in range(256)}
lookup_K_prime = {i: K_prime(i - samples) for i in range(256)}

# Compute smoothed local histogram
smoothed_local_hist = None
for intensity in local_hist.keys():
	if smoothed_local_hist is None:
		smoothed_local_hist = local_hist[intensity] * lookup_K[intensity]
	else:
		smoothed_local_hist += local_hist[intensity] * lookup_K[intensity]


# Compute smoothed local histogram derivative
smoothed_local_hist_derivative = None
for intensity in local_hist.keys():
	if smoothed_local_hist_derivative is None:
		smoothed_local_hist_derivative = local_hist[intensity] * lookup_K_prime[intensity]
	else:
		smoothed_local_hist_derivative += local_hist[intensity] * lookup_K_prime[intensity]

# f = plt.figure(0)
# plt.imshow(im)
# plt.title('Original image')

# f = plt.figure(1)
# plt.subplot(311)
# plt.bar(local_hist.keys(), local_hist.values())
# plt.title("Local histogram")
# plt.subplot(312)
# plt.plot(smoothed_local_hist)
# plt.title("Smooth local histogram")
# plt.subplot(313)
# plt.plot(smoothed_local_hist_derivative)
# plt.title("Smooth local histogram derivative")
# plt.show()

I_closest = closest_mode_filter(im_crop, sigma=1)