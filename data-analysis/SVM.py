from import_data import data as dataset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split 
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score
import pandas as pd

labels = dataset['GeneralHealth'].values
data = dataset.drop(columns=['GeneralHealth']).values
discrete = pd.read_csv('data-set/GeneralHealth_discretizacao.csv')
mapping = dict(zip(discrete['GeneralHealth'], discrete['ID']))


X_train, X_test, y_train, y_test = train_test_split(data,labels, test_size=.3, random_state=42)

svm_classifier = SVC(kernel='linear', random_state=42)
svm_classifier.fit(X_train,y_train)


y_pred = svm_classifier.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test,y_pred)

print(f"Acurácia: {accuracy:.2f}")  # Imprime a acurácia
print(f"Métrica F1: {f1:.2f}")  # Imprime a métrica F1
print("Matriz de Confusão:\n", conf_matrix)  # Imprime a matriz de confusão
print("Relatório de Classificação:\n", class_report)  # Imprime o relatório de classificação

plt.figure(figsize=(8, 6))  # Define o tamanho da figura para visualização
plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)  # Exibe a matriz de confusão
plt.title('Matriz de Confusão')  # Título do gráfico
plt.colorbar()  # Adiciona uma barra de cores
tick_marks = np.arange(len(mapping.keys()))  # Marca dos ticks para as classes
plt.xticks(tick_marks, mapping.keys(), rotation=45)  # Rótulos do eixo x
plt.yticks(tick_marks, mapping.keys())  # Rótulos do eixo y

# Adicionando os valores da matriz de confusão no gráfico
thresh = conf_matrix.max() / 2.  # Define um limite para a cor
for i, j in np.ndindex(conf_matrix.shape):  # Itera sobre cada elemento da matriz
    plt.text(j, i, format(conf_matrix[i, j], 'd'),  # Adiciona o valor no gráfico
             horizontalalignment="center",
             color="white" if conf_matrix[i, j] > thresh else "black")

plt.ylabel('Classe Verdadeira')  # Rótulo do eixo y
plt.xlabel('Classe Prevista')  # Rótulo do eixo x
plt.tight_layout()  # Ajusta o layout
plt.show()  # Exibe o gráfico
