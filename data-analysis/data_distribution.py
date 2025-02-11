import import_data
import matplotlib.pyplot as plt
import os

output_dir = r'.\figures\distribution'
os.makedirs(output_dir, exist_ok=True)

features = import_data.features

for i, coluna in enumerate(features.columns, start=1):
    plt.figure(figsize=(9, 5))
    for classe in import_data.classes.unique():
        subset = import_data.data[import_data.classes == classe]
        class_name = classe
        plt.hist(subset[coluna], bins=30, alpha=0.5, label=f'{class_name}', density=True)
    
    plt.xlim(import_data.features[coluna].min(), import_data.features[coluna].max())
    plt.ylim(0, plt.gca().get_ylim()[1])
    plt.title(f'{coluna}')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, f'distribution_{i}_{coluna}.png')
    plt.savefig(output_path)
    plt.close()

print(f'Charts saved in: {output_dir}')
