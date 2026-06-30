import numpy as np

# Definir una función para la matriz H_f
def H_f_matrix(x, y):
    H_f_11 = (3*x**2 - 2*x - 2) / (1 + 4*y**2)
    H_f_12 = -((x**3 - x**2 - 2*x) * 8 * y) / (1 + 4*y**2)**2
    H_f_21 = H_f_12  # Ya que H_f_21 es igual a H_f_12
    H_f_22 = -( (3*x**4 - 4*x**3 - 12*x**2 + 18) * ( (2 / (3*(1 + 4*y**2)**2)) - (32*y**2 / (3*(1 + 4*y**2)**3)) ) )
    
    # Crear la matriz
    H_f = np.array([[H_f_11, H_f_12], [H_f_21, H_f_22]])
    
    return H_f

# Evaluar la matriz en un punto (x, y)
x_value = 2.5417  # Cambia este valor de x
y_value = 0  # Cambia este valor de y

# Evaluar la matriz en el punto dado
result_matrix = H_f_matrix(x_value, y_value)

# Imprimir el resultado
print("Matriz H_f evaluada en x = {}, y = {}:".format(x_value, y_value))
print(result_matrix)