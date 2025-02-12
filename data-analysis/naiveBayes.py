from import_data import data as dataset
import pandas as pd
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
import numpy as np

dataset
dataset.GeneralHealth.hist()

labels = dataset['GeneralHealth'].values
data = dataset.drop(columns=['GeneralHealth']).values


skf = StratifiedKFold(n_splits=10, random_state=42, shuffle=True)

L_acc_nb = []  # Lista para armazenar acurácias do Naive Bayes
L_f1_nb = []  # Lista para armazenar pontuações F1 do Naive Bayes
L_f1_macro_nb = []  # Lista para armazenar pontuações F1 macro do Naive Bayes

for train_index, test_index in skf.split(data, labels):
    # Separando exemplos de treino e teste
    data_train, data_test = data[train_index], data[test_index]  # Divide os dados
    labels_train, labels_test = labels[train_index], labels[test_index]  # Divide os rótulos

    # Classificador Naive Bayes
    clf_nb = GaussianNB()  # Inicializa o classificador Naive Bayes
    clf_nb.fit(data_train, labels_train)  # Treina o modelo com os dados de treino

    # Predições
    y_pred_nb = clf_nb.predict(data_test)  # Faz previsões com os dados de teste

    # Avaliando
    acc_nb = accuracy_score(labels_test, y_pred_nb)  # Calcula a acurácia
    f1_nb = f1_score(labels_test, y_pred_nb,average='weighted')  # Calcula a pontuação F1
    f1_macro_nb = f1_score(labels_test, y_pred_nb, average='macro')  # Calcula a pontuação F1 macro

    # Armazenando os resultados
    L_acc_nb.append(acc_nb)  # Adiciona a acurácia à lista
    L_f1_nb.append(f1_nb)  # Adiciona a pontuação F1 à lista
    L_f1_macro_nb.append(f1_macro_nb)  # Adiciona a pontuação F1 macro à lista

# Imprimindo as médias das métricas do Naive Bayes
print('Acurácia Naive Bayes: ', np.mean(L_acc_nb))  
print('F1 Score Naive Bayes: ', np.mean(L_f1_nb))  
print('F1 Score Macro Naive Bayes: ', np.mean(L_f1_macro_nb))  
print('Parâmetros do Naive Bayes:', clf_nb.get_params())  # Exibe os parâmetros do modelo
print('=======')

metrics = {
    'Acurácia': [np.mean(L_acc_nb)],
    'F1 Score': [np.mean(L_f1_nb)],
    'F1 Score Macro': [np.mean(L_f1_macro_nb)]
}

labels = ['Naive Bayes']
x = np.arange(len(labels))
width = .25

fig, ax = plt.subplots()
rects1 = ax.bar(x - width, metrics['Acurácia'], width, label='Acurácia')
rects2 = ax.bar(x, metrics['F1 Score'], width, label='F1 Score Weighted')
rects3 = ax.bar(x + width, metrics['F1 Score Macro'], width, label='F1 Score Macro')

ax.set_ylabel('Pontuações')
ax.set_title('Comparação entre Classificadores')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

# Função para adicionar rótulos nas barras
def autolabel(rects):
    """Adiciona rótulos nas barras."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(round(height, 2)),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 pontos acima da barra
                    textcoords="offset points",
                    ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)
autolabel(rects3)

fig.tight_layout()  # Ajusta o layout
plt.show()  # Exibe o gráfico
