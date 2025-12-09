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
