import numpy as np

# Función logística
def fun_logistica(var, num_lo, x):
    ma_theta = var[:-num_lo].reshape((num_lo, -1))  # Parámetros de la regresión
    ve_sesgos = var[-num_lo:]  # Sesgos (interceptos)
    z = np.dot(ma_theta, x.T) + ve_sesgos[:, np.newaxis]  # Cálculo de z
    return 1 / (1 + np.exp(-z)).T  # Función logística

# Gradiente de la máxima verosimilitud
def gra_maxima_verosimilitud(var, num_lo, x, y):
    pi_i = fun_logistica(var, num_lo, x)  # Calcular pi_i
    error = pi_i - y  # Error entre predicción y valor real
    grad_theta = np.dot(error, x)  # Gradiente respecto a theta (vector de tamaño p)
    grad_sesgos = np.sum(error)  # Gradiente respecto a los sesgos (escalar)
    
    # Asegurarse de que grad_theta sea un arreglo 1-dimensional
    grad_theta = grad_theta.flatten()
    
    # Concatenar gradientes en un solo vector de tamaño (p + 1)
    return np.concatenate([grad_theta, [grad_sesgos]])

# Ejemplo de uso
np.random.seed(0)
x = np.random.rand(100, 2)  # 100 muestras, 2 características
y = np.random.randint(0, 2, 100)  # Etiquetas binarias
num_lo = 1  # Número de clases (1 para regresión logística binaria)
var = np.random.rand(3)  # Parámetros iniciales (2 para theta, 1 para sesgo)

# Calcular gradiente
gradiente = gra_maxima_verosimilitud(var, num_lo, x, y)
print("Gradiente:", gradiente)