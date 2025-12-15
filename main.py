# Import de bibliotecas necessárias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from time import time
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

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

# Cria um histograma com a distribuição de vinhos em cada qualidade
plt.figure(figsize=(8, 5))
sns.histplot(data=data, x="quality", discrete=True, color="teal")
plt.title("Distribuiçao dos vinhos em cada qualidade")
plt.xlabel("Qualidade")
plt.ylabel("Número total")
plt.show()

# Mais informações úteis
print(np.round(data.describe()))

# Gerar um gráfico de dispersão para procurar correlações
pd.plotting.scatter_matrix(data, alpha=0.3, figsize=(40,40), diagonal="kde")
plt.show()

correlation = data.corr()
print(correlation)
plt.figure(figsize=(14,12))
heatmap = sns.heatmap(correlation, annot=True, linewidths=0, vmin=-1, cmap="RdBu_r")
plt.show()

# Visualizar a correlação entre ph e fixed acidity
fixedAcidity_pH = data[['pH', 'fixed acidity']]
gridA = sns.JointGrid(x="fixed acidity", y="pH", data=fixedAcidity_pH, height=6)
gridA = gridA.plot_joint(sns.regplot, scatter_kws={"s": 10})
gridA = gridA.plot_marginals(sns.distplot)
plt.show()

# Visualizar a correlação entre citric acid e fixed acidity
fixedAcidity_citricAcid = data[['citric acid', 'fixed acidity']]
g = sns.JointGrid(x="fixed acidity", y="citric acid", data=fixedAcidity_citricAcid, height=6)
g = g.plot_joint(sns.regplot, scatter_kws={"s": 10})
g = g.plot_marginals(sns.distplot)
plt.show()

# Visualizar a correlação entre volatile acidity e qualidade (para valores discretos, melhor usar um gráfico de barras)
volatileAcidity_quality = data[['volatile acidity', 'quality']]
fig, axs = plt.subplots(ncols=1,figsize=(10,6))
sns.barplot(x='quality', y='volatile acidity', data=volatileAcidity_quality, ax=axs)
plt.title('quality VS volatile acidity')

plt.tight_layout()
plt.show()
plt.gcf().clear()

# Alcohol vs quality
quality_alcohol = data[['alcohol', 'quality']]
fig, axs = plt.subplots(ncols=1,figsize=(10,6))
sns.barplot(x='quality', y='alcohol', data=quality_alcohol, ax=axs)
plt.title('quality VS alcohol')

plt.tight_layout()
plt.show()
plt.gcf().clear()

# Método de Tukey
# Para cada característica, encontrar os pontos de dados com valores extremos altos ou baixos
outliers = []
for caracteristica in data.keys():
    if caracteristica == 'quality':
        continue
    
    # Calcular Q1 (25º percentil dos dados) para a característica dada
    Q1 = np.percentile(data[caracteristica], q=25)

    # Calcular Q3 (75º percentil dos dados) para a característica dada
    Q3 = np.percentile(data[caracteristica], q=75)

    # Usar o intervalo interquartil para calcular um passo de outlier (1.5 vezes o intervalo interquartil)
    intervalo_interquartil = Q3 - Q1
    passo = 1.5 * intervalo_interquartil

    # Exibir os outliers
    print("Pontos de dados considerados outliers para a característica '{}':".format(caracteristica))
    outliers_caracteristica = data[~((data[caracteristica] >= Q1 - passo) & (data[caracteristica] <= Q3 + passo))]
    print(outliers_caracteristica)

    # Armazenar os índices dos outliers detectados
    outliers.extend(outliers_caracteristica.index.tolist())

# Remover duplicatas e os outliers
outliers = list(set(outliers))
dados_limpos = data.drop(data.index[outliers]).reset_index(drop=True)
print(f"\nTotal de outliers removidos: {len(outliers)}")
print(f"Dados restantes: {dados_limpos.shape[0]} linhas")

# Preparação dos dados
y = dados_limpos['quality'].apply(lambda x: 1 if x > 6 else 0)
X = dados_limpos.drop('quality', axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Padronização dos dados
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Treinar modelo Random Forest
modelo = RandomForestClassifier(n_estimators=100, random_state=42)
modelo.fit(X_train_scaled, y_train)

# Avaliação do modelo
previsoes = modelo.predict(X_test_scaled)
acuracia = accuracy_score(y_test, previsoes)
print(f"Acurácia: {acuracia:.2%}")
print(classification_report(y_test, previsoes))

# Salvar artefatos
joblib.dump(modelo, 'modelo_vinho.pkl')
joblib.dump(scaler, 'scaler_vinho.pkl')
