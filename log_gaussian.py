import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import numpy as np
import matplotlib.pyplot as plt
from scipy import io
import numpy as np
import spectral as sp
import pandas as pd
import cv2

directory = ""
data = np.array(sp.io.envi.open(directory + 'data.hdr', directory + 'data.raw').load())
white_reference = np.array(sp.io.envi.open(directory + '/WHITEREF.hdr', directory + '/WHITEREF.raw').load())
dark_reference = np.array(sp.io.envi.open(directory + '/DARKREF.hdr', directory + '/DARKREF.raw').load())

dr = np.mean(dark_reference, axis=0)
wr = np.mean(white_reference, axis=0)
calibration = (data - dr) / (wr - dr)

if np.isnan(calibration).any() or (calibration <= 0).any():
    calibration = calibration[~np.isnan(calibration) & (calibration > 0)]

log_calibration = np.log(calibration)

mu = np.mean(log_calibration)
sigma = np.std(log_calibration)


# log_gaussian = np.random.lognormal(mu, sigma, len(log_calibration))
# log_gaussian = (log_calibration - mu)/sigma
log_gaussian = log_calibration

# count, bins, ignored = plt.hist(log_gaussian, 100, density=True, align='mid')
# x = np.linspace(min(bins), max(bins), len(log_calibration))
# pdf = (1 / (log_gaussian * np.sqrt(2 * np.pi))) * np.exp(-(x - mu)**2 / (2 * sigma**2))
# plt.plot(x, pdf, linewidth=2, color='b')
plt.hist(log_gaussian, 100, density=True, align='mid')
plt.grid(True)
plt.axis('tight')
plt.show()


# log_gaussian = np.random.lognormal(mean, std_dev, 10000)


# mean = np.mean(calibration)
# std_dev = np.std(calibration)
# threshold = 1 * std_dev
# outliers_mask = np.abs(calibration - mean) > threshold
# outliers = calibration[outliers_mask]

# # 시각화
# plt.figure(figsize=(10, 6))
# plt.hist(calibration.flatten(), bins=50, color='blue', alpha=0.7, label='normal')
# plt.hist(outliers, bins=50, color='red', alpha=0.7, label='Outliers')
# plt.xlabel('Pixel Value')
# plt.ylabel('Frequency')
# # plt.title('Histogram of Calibration Data with Outliers')
# plt.grid(True)
# plt.legend()
# plt.show()