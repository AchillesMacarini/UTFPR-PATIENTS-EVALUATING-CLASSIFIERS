from sklearn.datasets import load_iris
from sklearn import tree
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import import_data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import import_data
from scipy.stats import kurtosis, entropy
import os

output_dir = r'.\figures\decision_tree'
os.makedirs(output_dir, exist_ok=True)

imports = import_data.data
dataset = imports.drop(columns=['PatientID','GeneralHealth'])
X = dataset
y = imports['GeneralHealth']
class_names = import_data.general_health_data['GeneralHealth']
print(X)
print(y)

df = dataset.copy()
print(df.head())
depths = [2, 3, 5, 10]
for max_depth in depths:
    globals()[f'clf_dt{max_depth}'] = tree.DecisionTreeClassifier(criterion='entropy', max_depth=max_depth)
    globals()[f'clf_dt{max_depth}'].fit(X, y)
    plt.figure(figsize=(12, 8))
    tree.plot_tree(globals()[f'clf_dt{max_depth}'], filled=True, feature_names=X.columns, class_names=class_names)
    plt.title(f"Árvore de Decisão (Profundidade Máxima: {max_depth})")
    output_path = os.path.join(output_dir, f"decision_tree_depth_{max_depth}.png")
    plt.savefig(output_path)
    plt.close()

clf_dt1 = tree.DecisionTreeClassifier(criterion='entropy')
clf_dt1.fit(X, y)

plt.figure(figsize=(12, 8))
tree.plot_tree(clf_dt1, filled=True, feature_names=X.columns, class_names=class_names)
plt.title("Árvore de Decisão (Critério: Entropia)")
output_path = os.path.join(output_dir, "decision_tree_full.png")
plt.savefig(output_path)
plt.close()

skf = StratifiedKFold(n_splits=10, random_state=42, shuffle=True)
L_f1_dt1, L_f1_dt2, L_f1_dt3, L_f1_dt5, L_f1_dt10 = [], [], [], [], []

for train_index, test_index in skf.split(X, y):
    data_train, data_test = X.iloc[train_index], X.iloc[test_index]
    labels_train, labels_test = y.iloc[train_index], y.iloc[test_index]

    L_f1_dt1.append(f1_score(labels_test, clf_dt1.predict(data_test), average='macro'))

    L_f1_dt2.append(f1_score(labels_test, globals()[f'clf_dt2'].predict(data_test), average='macro'))

    L_f1_dt3.append(f1_score(labels_test, globals()[f'clf_dt3'].predict(data_test), average='macro'))

    L_f1_dt5.append(f1_score(labels_test, globals()[f'clf_dt5'].predict(data_test), average='macro'))

    L_f1_dt10.append(f1_score(labels_test, globals()[f'clf_dt10'].predict(data_test), average='macro'))

results = pd.DataFrame({
    'Árvore de Decisão (sem limite)': L_f1_dt1,
    'Árvore de Decisão (max_depth=2)': L_f1_dt2,
    'Árvore de Decisão (max_depth=3)': L_f1_dt3,
    'Árvore de Decisão (max_depth=5)': L_f1_dt5,
    'Árvore de Decisão (max_depth=10)': L_f1_dt10
})

print(results)
