
# Importando as bibliotecas necessárias
import pandas as pd  # Para manipulação de dados
import numpy as np  # Para operações numéricas
import matplotlib.pyplot as plt  # Para visualização de dados
from sklearn.model_selection import train_test_split  # Para dividir os dados
from sklearn.cluster import KMeans  # Para o algoritmo K-Means
from matplotlib.colors import ListedColormap  # Para mapear cores
from correlation import correlations
from import_data import data

# Obtendo os dois atributos com os maiores coeficientes de correlação
atributo_maior_correlacao = "BMI" #correlations.iloc[2]['Feature']
atributo_segundo_maior_correlacao = "HeightInMeters"  #correlations.iloc[6]['Feature']

# Selecionando as colunas para o modelo
X = data[[atributo_maior_correlacao, atributo_segundo_maior_correlacao]].values

# Solicitando o valor de k ao usuário
k = 10

# Criando o modelo K-Means
kmeans = KMeans(n_clusters=k, random_state=42)
y_kmeans = kmeans.fit_predict(X)

# Visualizando os clusters formados
plt.figure(figsize=(12, 8))
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis', label='Clusters')
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.75, marker='X', label='Centros dos Clusters')
plt.title('Clusters K-Means')
plt.xlabel(atributo_maior_correlacao)
plt.ylabel(atributo_segundo_maior_correlacao)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()  # Exibe o gráfico dos clusters
