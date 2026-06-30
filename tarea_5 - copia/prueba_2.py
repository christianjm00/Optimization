import numpy as np

def sele_cord(dim, datos_y):
    cord_ge = np.arange(dim)  # Vector de coordenadas [0, 1, ..., dim-1]
    true_fa = (datos_y == 0) | (datos_y == 1)  # Máscara booleana para valores 0 y 1
    cord = cord_ge[true_fa]  # Filtrar coordenadas donde datos_y es 0 o 1
    return cord

# Ejemplo de uso:
datos_y = np.array([5, 0, 4, 1, 8, 4, 8])  # Vector de etiquetas
dim = len(datos_y)  # Dimensión basada en la longitud del vector
cord_filtradas = sele_cord(dim, datos_y)

print(cord_filtradas)  # Debería imprimir las coordenadas donde datos_y es 0 o 1