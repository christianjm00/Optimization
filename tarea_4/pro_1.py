import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

x_1 = np.linspace(-2,2,800)
x_2 = np.linspace(-2,2,800)
x_1, x_2 = np.meshgrid(x_1, x_2)

def fun_e(x_1,x_2):
    return (x_1**2+x_2**2-1)**2+(x_2**2-1)**2

f = fun_e(x_1,x_2)

# Puntos críticos
pu_cr_00=fun_e(0,0)
pu_cr_01=fun_e(0,1)
pu_cr_0l1=fun_e(0,-1)
pu_cr_10=fun_e(1,0)
pu_cr_l10=fun_e(-1,0)

fig=plt.figure()
ax=fig.add_subplot(111, projection='3d')

# Graficar la superficie
ax.plot_surface(x_1,x_2,f,cmap='viridis',alpha=1)

# Graficar puntos críticos con mayor tamaño y color
ax.scatter(0,0,pu_cr_00,color='red',s=100,edgecolors='black',label="(0,0)")
ax.scatter(0,1,pu_cr_01,color='blue',s=100,edgecolors='black',label="(0,1)")
ax.scatter(0,-1,pu_cr_0l1,color='green',s=100,edgecolors='black',label="(0,-1)")
ax.scatter(1,0,pu_cr_10,color='blue',s=100,edgecolors='black',label="(1,0)")
ax.scatter(-1,0,pu_cr_l10,color='green',s=100,edgecolors='black',label="(-1,0)")
# Etiquetas
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()
plt.savefig("T4_p1_gra.pdf")
plt.show()