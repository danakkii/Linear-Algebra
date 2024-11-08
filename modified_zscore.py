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
print(np.shape(calibration)) #(400, 512, 224)

# Modified Z-score
median_calibration = np.median(calibration)
# print(median_calibration)
# print(np.shape(median_calibration))

mad_calibration = np.median(np.abs(calibration - median_calibration))
# print(mad_calibration)
# print(np.shape(mad_calibration))
modified_z_scores = 0.6745 * (calibration - median_calibration) / mad_calibration


# 평균보다 특정 표준 편차 범위를 벗어나는 데이터 찾기
threshold = 3  # 예시에서는 표준 편차의 두 배 이상 벗어나는 것으로 설정합니다.
outliers = np.abs(modified_z_scores) > threshold

# Z 점수의 분포를 히스토그램으로 시각화
plt.figure(figsize=(10, 6))
plt.hist(modified_z_scores.flatten(), bins=50, color='blue', alpha=0.7)
plt.hist(modified_z_scores.flatten()[outliers.flatten()], bins=50, color='red', alpha=0.7)
plt.axvline(x=threshold, color='r', linestyle='--', label='Upper Threshold')
plt.axvline(x=-threshold, color='r', linestyle='--', label='Lower Threshold')
plt.xlabel('Z Score')
plt.ylabel('Frequency')
plt.xlim([-30,30])
plt.title('Z Score Distribution')
plt.legend()
plt.grid(True)
plt.show()