import import_data
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

data = import_data.data

data = data.drop(columns=['PatientID'])

data['GeneralHealth'] = data['GeneralHealth'].astype('category').cat.codes

correlations = pd.DataFrame(columns=['Feature', 'Coefficient'])

output_dir = r'.\figures\Coefficient_between'
os.makedirs(output_dir, exist_ok=True)

for coluna in data.columns:
    if coluna != 'GeneralHealth':
        correlacao = data[coluna].corr(data['GeneralHealth'])
        correlations = correlations._append({'Feature': coluna, 'Coefficient': correlacao}, ignore_index=True)

correlations['Coefficient'] = pd.to_numeric(correlations['Coefficient'], errors='coerce')
correlations = correlations.sort_values(by='Coefficient', ascending=False)

correlations = correlations[np.isfinite(correlations['Coefficient'])]

plt.figure(figsize=(12, 8))
plt.scatter(correlations['Feature'], correlations['Coefficient'], color='blue')

for i, row in correlations.iterrows():
    plt.text(row['Feature'], row['Coefficient'], row['Feature'], fontsize=9, ha='center')

plt.title('Coeficiente de Correlação dos Atributos em Relação à Coluna ID')
plt.xlabel('Atributos')
plt.ylabel('Coefficient')
plt.xticks(rotation=45)
plt.grid(True)

plt.tight_layout()
plt.show()

color_dict = {
    0: "Red",
    1: "Blue",
    2: "Green",
    3: "Yellow",
    4: "Purple",
    5: "Orange",
    6: "Pink",
    7: "Brown",
    8: "Gray",
    9: "Black"
}

i = 0

output_dir = r'.\figures\correlation'
os.makedirs(output_dir, exist_ok=True)

while i < 10:
    plt.figure(figsize=(12, 8))
    plt.scatter(data['GeneralHealth'], data[correlations.iloc[i]['Feature']], color=color_dict[i], label=correlations.iloc[i]['Feature'])
    plt.title('Plot Features with the biggest Correlation')
    plt.xlabel('Class')
    plt.ylabel('Features Value')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    output_path = os.path.join(output_dir, f'correlation_{i}.png')
    plt.savefig(output_path)
    plt.close()
    i+=1

print(f'Charts saved in: {output_dir}')