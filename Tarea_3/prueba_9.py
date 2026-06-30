import numpy as np
import matplotlib.pyplot as plt

# Función de costo y su derivada
def J(theta):
    return (theta - 5) ** 2

def gradiente_J(theta):
    return 2 * (theta - 5)

# Parámetros iniciales
theta_batch = 0  # Inicialización para Batch Gradient Descent
theta_sgd = 0    # Inicialización para SGD
eta = 0.1        # Tasa de aprendizaje
num_iteraciones = 50

# Historial de valores de theta
theta_batch_hist = [theta_batch]
theta_sgd_hist = [theta_sgd]

# Descenso de gradiente estándar
for _ in range(num_iteraciones):
    grad = gradiente_J(theta_batch)  # Usa todo el conjunto de datos (en este caso es solo una función)
    theta_batch -= eta * grad
    theta_batch_hist.append(theta_batch)

# Descenso de gradiente estocástico (simulado con ruido)
for _ in range(num_iteraciones):
    ruido = np.random.uniform(-1, 1)  # Agregamos ruido para simular variabilidad en SGD
    grad = gradiente_J(theta_sgd) + ruido
    theta_sgd -= eta * grad
    theta_sgd_hist.append(theta_sgd)

# Graficamos la convergencia
plt.plot(theta_batch_hist, label="Batch Gradient Descent", marker="o")
plt.plot(theta_sgd_hist, label="Stochastic Gradient Descent", marker="x", linestyle="dashed")
plt.axhline(y=5, color='r', linestyle='--', label="Mínimo Verdadero")
plt.xlabel("Iteraciones")
plt.ylabel("Valor de θ")
plt.legend()
plt.title("Comparación de Batch Gradient Descent vs SGD")
plt.show()
