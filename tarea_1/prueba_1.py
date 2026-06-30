import numpy as np
import matplotlib.pyplot as plt

# Definir el rango de valores para x1 y x2
x_1 = np.linspace(-10, 10, 400)
x_2 = np.linspace(-10, 10, 400)

# Crear una malla de valores para evaluar las funciones
X1, X2 = np.meshgrid(x_1, x_2)

# Definir las funciones f1 y f2
F1 = X1**2 - X2**2  # f1(x1, x2) = x1^2 - x2^2
F2 = 2 * X1 * X2    # f2(x1, x2) = 2x1x2

# Crear la figura
plt.figure(figsize=(8, 6))

# Graficar las curvas de nivel para los valores dados
contour1 = plt.contour(X1, X2, F1, levels=[12], colors='blue', linewidths=2, linestyles='dashed')
contour2 = plt.contour(X1, X2, F2, levels=[16], colors='red', linewidths=2)

# Etiquetas y título
plt.xlabel(r'$x_1$')
plt.ylabel(r'$x_2$')
plt.title("Curvas de Nivel de $f_1(x_1, x_2)$ y $f_2(x_1, x_2)$")

# Leyenda
plt.clabel(contour1, inline=True, fontsize=10, fmt=r"$f_1(x_1, x_2) = 12$")
plt.clabel(contour2, inline=True, fontsize=10, fmt=r"$f_2(x_1, x_2) = 16$")

# Mostrar la gráfica
plt.grid()
plt.show()