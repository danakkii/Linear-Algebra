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

mean = np.mean(calibration)
std_dev = np.std(calibration)
threshold = 1 * std_dev
outliers_mask = np.abs(calibration - mean) > threshold
outliers = calibration[outliers_mask]

# 시각화
plt.figure(figsize=(10, 6))
plt.hist(calibration.flatten(), bins=50, color='blue', alpha=0.7, label='normal')
plt.hist(outliers, bins=50, color='red', alpha=0.7, label='Outliers')
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')
# plt.title('Histogram of Calibration Data with Outliers')
plt.grid(True)
plt.legend()
plt.show()