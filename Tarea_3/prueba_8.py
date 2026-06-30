import pandas as pd

# Cargar el DataFrame correctamente
df = pd.read_csv("iris.csv")

# Filtrar las posiciones donde 'variety' es 'Setosa'
pos_Setosa = df.index[df["variety"] == "Setosa"].tolist()

print(pos_Setosa)  # Lista con las posiciones de las filas donde la variedad es "Setosa"