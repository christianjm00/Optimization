import numpy as np
def pru_1(x1,x2):
    return np.array()
def pru(x1,x2):
    H=np.array([[8*x1**2 + 4*(x1**2 + x2**2 - 1), 8*x1*x2],
    [8*x1*x2, 16*x2**2 + 4*(x1**2 + x2**2 - 1) + 4*(x2**2 - 1)]])
    return H
print(pru(0,1))
print(pru(0,-1))