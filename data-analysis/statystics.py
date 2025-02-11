import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import import_data
from scipy.stats import kurtosis, entropy

data = import_data.data
classes = import_data.classes.unique()
features = import_data.features

atributos_extraidos = {
    'Classe': [],
    'Média': [],
    'Desvio Padrão': [],
    'Coeficiente de Variação': [],
    'Curtose': [],
    'Entropia': [],
    'RMS': [],
    'Máximo': [],
    'Mínimo': [],
    'Mediana': []
}

for coluna in features.columns:
    for classe in classes:
        subset = import_data.data[import_data.data['GeneralHealth'] == classe][coluna]
        class_name = classe
        
        media = subset.mean()
        desvio_padrao = subset.std()
        coeficiente_variacao = desvio_padrao / media if media != 0 else 0
        curtose = kurtosis(subset)
        entropia_valor = entropy(np.histogram(subset, bins=30)[0])
        rms = np.sqrt(np.mean(np.square(subset)))
        maximo = subset.max()
        minimo = subset.min()
        mediana = subset.median()
        
        atributos_extraidos['Classe'].append(classe)
        atributos_extraidos['Média'].append(media)
        atributos_extraidos['Desvio Padrão'].append(desvio_padrao)
        atributos_extraidos['Coeficiente de Variação'].append(coeficiente_variacao)
        atributos_extraidos['Curtose'].append(curtose)
        atributos_extraidos['Entropia'].append(entropia_valor)
        atributos_extraidos['RMS'].append(rms)
        atributos_extraidos['Máximo'].append(maximo)
        atributos_extraidos['Mínimo'].append(minimo)
        atributos_extraidos['Mediana'].append(mediana)

atributos_df = pd.DataFrame(atributos_extraidos)
print(atributos_df)

fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(18, 15))
axes = axes.flatten()

atributos = ['Média', 'Desvio Padrão', 'Coeficiente de Variação', 'Curtose', 'Entropia', 'RMS', 'Máximo', 'Mínimo', 'Mediana']

for i, atributo in enumerate(atributos):
    for classe in classes:
        subset = atributos_df[atributos_df['Classe'] == classe]
        axes[i].plot(subset['Classe'], subset[atributo], marker='o', label=f'Classe {classe}')
        
    axes[i].set_title(f'{atributo} por Classe')
    axes[i].set_xlabel('Classe')
    axes[i].set_ylabel(atributo)
    axes[i].legend()
    axes[i].grid(True)

plt.tight_layout()
plt.show()