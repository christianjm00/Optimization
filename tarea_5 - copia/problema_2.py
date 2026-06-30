import numpy as np
def error(pi,y):
    indi=pi>0.5
    indi=indi.astype(int)
    return np.mean(np.abs(indi-y))