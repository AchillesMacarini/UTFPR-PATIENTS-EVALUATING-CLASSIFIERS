# -*- coding: utf-8 -*-

# Importando as bibliotecas necessárias
import pandas as pd  # Para manipulação de dados
import numpy as np  # Para operações numéricas
import matplotlib.pyplot as plt  # Para visualização de dados
from sklearn.model_selection import train_test_split  # Para dividir os dados
from sklearn.neighbors import KNeighborsClassifier  # Para o classificador KNN
from matplotlib.colors import ListedColormap  # Para mapear cores
import import_data
import correlation


data = import_data.data
data = data.drop(columns=['PatientID'])
data['GeneralHealth'] = data['GeneralHealth'].astype('category').cat.codes

correlation_df = correlation.correlations
print(correlation_df)
correlations_ft = {}

ft_alcoholDrinkers = correlation_df.iloc[0]['Feature']
ft_highRisk = correlation_df.iloc[1]['Feature']

X = data[[ft_alcoholDrinkers,ft_highRisk]].values 
y = data['GeneralHealth'].values

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25, random_state=42)
k = 5

knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train,y_train)

accuracy = knn.score(X_test,y_test)
print(f'Acurácia do modelo KNN com k={k}: {accuracy:.2f}')

x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, .1), np.arange(y_min, y_max, .1))

Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.figure(figsize=(12, 8))
plt.contourf(xx, yy, Z, alpha=0.3, cmap=ListedColormap(('red', 'blue', 'green', 'orange', 'purple')))
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, edgecolor='k', marker='o', label='Treinamento')
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, edgecolor='k', marker='x', label='Teste')
plt.title('Região de Aceitação do KNN')
plt.xlabel(ft_alcoholDrinkers)
plt.ylabel(ft_highRisk)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()