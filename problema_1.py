import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Genero las dos variables en un rango de -10 a 10, dividido en 200 intervalos
x_1 = np.linspace(-10,10,200)
x_2 = np.linspace(-10, 10, 200)

# Genero una malla de valores para evaluar las funciones
x_1, x_2 = np.meshgrid(x_1, x_2)

# Defino las funciones f_1 y f_2 restando los valores de evaluación
y_1=x_1**2-x_2**2-12
y_2=2*x_1*x_2-16

# Genero la figura 3D
fig=plt.figure()
ax=fig.add_subplot(111, projection='3d')

# Graficar las superficies
ax.plot_surface(x_1,x_2,y_1, cmap='viridis', alpha=0.7)
ax.plot_surface(x_1,x_2,y_2, cmap='plasma', alpha=0.7)

# Etiquetas
ax.set_xlabel(r'$X_1$')
ax.set_ylabel(r'$X_2$')
ax.set_zlabel(r'$Y$')
ax.set_title("Gráfica de la función " + r"$F_{1}(x_1,x_2)$" + " y " + r"$F_{2}(x_1,x_2)$.")
plt.savefig("problema_1_tarea_1.pdf")
# Mostrar la gráfica
plt.show()

fig=plt.figure()
ax=fig.add_subplot(111, projection='3d')

# Graficar las superficies
ax.plot_surface(x_1,x_2,y_1+y_2, cmap='viridis', alpha=0.7)


# Etiquetas
ax.set_xlabel(r'$X_1$')
ax.set_ylabel(r'$X_2$')
ax.set_zlabel(r'$Y$')
ax.set_title("Gráfica de la función " + r"$F_{1}(x_1,x_2)$" + " y " + r"$F_{2}(x_1,x_2)$.")

# Mostrar la gráfica
plt.show()