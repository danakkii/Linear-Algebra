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
import math

# w1 = np.array([[1.1, 0.2],
#                [0.8, 2.1],
#                [0.4, 1.7]])

# b1 = np.array([0.1, 3.1])

# w2 = np.array([[-0.1, 2.5],
#                [0.9, -1.6]])

b1_w2 = np.array([2.78, -4.71])
b2 = np.array([1.2, -0.5])

# f1 = x* w1 + b1
# f2 = x* w2 + b2

print(b1_w2 + b2)

print(0.9 * math.sqrt(2))