import numpy as np
A=1.5
A_1=2.5
B=40
x=20
#조명축이 원점보다 높을 시
def func(theta):
    return x*np.tan(theta)+np.tan(theta)/np.cos(theta)*A_1*np.tan(theta)-B

#조명축이 원점보다 낮을 시 
def func2(theta):
    return  (B-A_1/np.cos(theta))*(x+A/np.cos(theta))-B*x

from scipy.optimize import fsolve

print(fsolve(func,0)*180/np.pi) # f(theta)=0을 만족하는 theta값을 찾아줌
# radian은 degree랑 비슷한 각도 측정 단위임

print(fsolve(func2,0)*180/np.pi)
 