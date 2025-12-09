# Import de bibliotecas necessárias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from time import time

# Carregar o dataset 'red wines'
data = pd.read_csv("data/winequality-red.csv", sep=";")
 
# Mostrar as 5 primeiras linhas
print(data.head(n=5))

# Verificar informações do dataset
data.isnull().any()
data.info()

n_vinhos = data.shape[0]

# Número de vinhos com qualidade maior que 6
qualidade_acima_6 = data.loc[data["quality"] > 6]
n_acima_6 = qualidade_acima_6.shape[0]

# Número de vinhos com qualidade menor que 5
qualidade_abaixo_5 = data.loc[data["quality"] < 5]
n_abaixo_5 = qualidade_abaixo_5.shape[0]

# Número de vinhos com qualidade entre 5 e 6
qualidade_entre_5 = data.loc[(data["quality"] >= 5) & (data["quality"] <= 6)]
n_entre_5 = qualidade_entre_5.shape[0]

# Porcentagem de vinho com qualidade acima de 6
acima_6_porcentagem = n_acima_6*100/n_vinhos

# Imprimir os resultados
print(f"Número total de vinhos no dataset: {n_vinhos}")
print(f"Número de vinhos com qualidade maior que 6: {n_acima_6}")
print(f"Número de vinhos com qualidade menor que 5: {n_abaixo_5}")
print(f"Número de vinhos com qualidade entre 5 e 6: {n_entre_5}")
print(f"Porcentagem de vinhos com qualidade maior que 6: {acima_6_porcentagem:.2f}")
