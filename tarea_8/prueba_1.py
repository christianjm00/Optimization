import numpy as np

print(str("vmreioeringvols"))
a=np.arange(1,4)
b=np.arange(4,7)
print(a)
print(b)
print(np.dot(a,b))
print(np.dot(a,b.T))
print(np.dot(a.T,b))
print(a@b.T)
print(a.T@b)
print(np.outer(a,b))