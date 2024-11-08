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

# 무작위로 생성된 스펙트럼 데이터 생성
# np.random.seed(0)
# wavelengths = np.linspace(0, 200, 100)  # 파장 범위: 0부터 200까지
# spectral_data = np.abs(np.random.randn(100))  # 임의의 스펙트럼 데이터 생성

directory = ""
data = np.array(sp.io.envi.open(directory + 'data.hdr', directory + 'data.raw').load())
white_reference = np.array(sp.io.envi.open(directory + '/WHITEREF.hdr', directory + '/WHITEREF.raw').load())
dark_reference = np.array(sp.io.envi.open(directory + '/DARKREF.hdr', directory + '/DARKREF.raw').load())

dr = np.mean(dark_reference, axis=0)
wr = np.mean(white_reference, axis=0)
calibration = (data - dr) / (wr - dr)
print(np.shape(calibration)) #(400, 512, 224)

# 주요 대역 선택
start_wavelength = 40
end_wavelength = 80

# 주요 대역에서의 데이터 추출
selected_indices = np.where((calibration >= start_wavelength) & (calibration <= end_wavelength))
selected_wavelengths = calibration[selected_indices]
selected_spectral_data = calibration[selected_indices]

# 최소 제곱법을 이용하여 추출된 주요 대역의 선형 모델 파라미터 계산
A = np.vstack([selected_wavelengths, np.ones(len(selected_wavelengths))]).T
m, c = np.linalg.lstsq(A, selected_spectral_data, rcond=None)[0]

# 추출된 주요 대역의 선형 모델 시각화
plt.figure(figsize=(10, 6))
plt.plot(selected_wavelengths, selected_spectral_data, 'o', label='Selected Spectrum Data')
plt.plot(selected_wavelengths, m*selected_wavelengths + c, 'r', label='Fitted Line')
plt.title('Fitted Line for Selected Spectrum Data')
plt.xlabel('Wavelength')
plt.ylabel('Intensity')
plt.legend()
plt.grid(True)
plt.show()

# 변경 전의 스펙트럼 데이터 시각화
plt.figure(figsize=(10, 6))
plt.plot(calibration, calibration, label='Original Spectrum')
plt.title('Original Spectrum Data')
plt.xlabel('Wavelength')
plt.ylabel('Intensity')
plt.legend()
plt.grid(True)
plt.show()

# 변경 후의 스펙트럼 데이터 시각화
# 추출된 주요 대역의 데이터를 최소 제곱법으로 구한 선형 모델로 대체하는 것으로 가정
changed_spectral_data = calibration.copy()
changed_spectral_data[selected_indices] = m * selected_wavelengths + c

plt.figure(figsize=(10, 6))
plt.plot(calibration, calibration, label='Original Spectrum', color='blue')
plt.plot(calibration, changed_spectral_data, label='Modified Spectrum', color='red', linestyle='--')
plt.title('Comparison of Original and Modified Spectrum Data')
plt.xlabel('Wavelength')
plt.ylabel('Intensity')
plt.legend()
plt.grid(True)
plt.show()
