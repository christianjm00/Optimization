import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Genero las dos variables en un rango de -10 a 10, dividido en 200 intervalos
x_1 = np.linspace(-10,10,200)
x_2 = np.linspace(-10, 10, 200)

# Genero una malla de valores para evaluar las funciones
x_1, x_2 = np.meshgrid(x_1, x_2)

# Defino las funciones f_1 y f_2 restando los valores de evaluación
def fun(x_1,x_2):
    y=8*x_1+12*x_2+x_1**2-2*x_2**2
    return y

y=fun(x_1,x_2)
# Genero la figura 3D
fig=plt.figure()
ax=fig.add_subplot(111, projection='3d')

# Graficar las superficies
ax.plot_surface(x_1,x_2,y, cmap='viridis', alpha=0.7)
ax.scatter(-4, 3, fun(-4,3), color='red', s=100)

# Etiquetas
ax.set_xlabel(r'$X_1$')
ax.set_ylabel(r'$X_2$')
ax.set_zlabel(r'$Y$')
ax.set_title("Gráfica de la función " + r"$F(x_1,x_2).$" )
plt.savefig("problema_5_a_tarea_2.pdf")
# Mostrar la gráfica


# Graficar las líneas de nivel
plt.figure(figsize=(8, 6))
contours=plt.contour(x_1, x_2, y, levels=np.logspace(-1, 3, 20), cmap='viridis')
plt.clabel(contours, inline=True, fontsize=8)
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.title('Líneas de nivel de la función de Rosenbrock')
plt.scatter(-4, 3, color='red', marker='o', label='Mínimo en $(1,1)$')
plt.legend()
plt.grid(True)
plt.savefig("problema_5_b_tarea_2.pdf")
plt.show()

