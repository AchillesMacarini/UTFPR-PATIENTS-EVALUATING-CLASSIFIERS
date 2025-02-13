import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score
import import_data
import os

imports = import_data.data
dataset = imports.drop(columns=['PatientID', 'GeneralHealth'])
X = imports[['HadHeartAttack', 'BMI']]
y = imports['GeneralHealth']
class_names = import_data.general_health_data['GeneralHealth']
colors = ['red', 'blue', 'green', 'yellow', 'purple']
print(X)
print(y)

output_dir = r'.\figures\perceptron'
os.makedirs(output_dir, exist_ok=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

perceptron = Perceptron(max_iter=1000, random_state=42)
perceptron.fit(X_train, y_train)

y_pred = perceptron.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy:.2f}")
print(f"F1 Score: {f1:.2f}")
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", class_report)

plt.figure(figsize=(10, 6))

for i, classe in enumerate(class_names):
    subset = X_train[y_train == i]
    plt.scatter(subset.iloc[:, 0], subset.iloc[:, 1], color=colors[i], label=classe)

x_min, x_max = X.iloc[:, 0].min() - 1, X.iloc[:, 0].max() + 1
y_min, y_max = X.iloc[:, 1].min() - 1, X.iloc[:, 1].max() + 1

xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
Z = perceptron.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.3)
plt.xlabel(X.columns[0])
plt.ylabel(X.columns[1])
plt.title("Perceptron Classifier - Patients-Data Dataset")
plt.legend()
output_path = os.path.join(output_dir, "perceptron.png")
plt.savefig(output_path)
plt.close()
