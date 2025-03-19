# Importando bibliotecas necessárias
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Carregando o conjunto de dados Íris
data = load_iris()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target

# Exibir as primeiras linhas do dataset
print("Visualização inicial do dataset:")
print(df.head())

# Análise exploratória dos dados
sns.pairplot(df, hue='target', palette='Dark2')
plt.show()

# Teste 1: Alteração da proporção de treino/teste para 70% treino e 30% teste
X = df.iloc[:, :-1]  # Features
y = df['target']  # Target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Padronização dos dados
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Teste 2: Alteração do número de vizinhos (k) para analisar impacto na acurácia
k_values = [1, 3, 5, 10, 15]  # Testando diferentes valores de k
results = {}

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    results[k] = acc  # Armazena a acurácia para cada k
    print(f"Acurácia do modelo KNN com k={k}: {acc:.4f}")

# Plotando a acurácia para diferentes valores de k
plt.plot(results.keys(), results.values(), marker='o', linestyle='dashed', color='b')
plt.xlabel('Número de Vizinhos (k)')
plt.ylabel('Acurácia')
plt.title('Impacto de k no Modelo KNN')
plt.show()

# Teste 3: Remoção de um atributo (Comprimento da Sépala) e reavaliação do modelo
X_reduced = df.iloc[:, 1:-1]  # Removendo a primeira coluna
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_reduced, y, test_size=0.3, random_state=42)

# Padronização dos dados
X_train_r = scaler.fit_transform(X_train_r)
X_test_r = scaler.transform(X_test_r)

# Treinamento e avaliação do modelo sem o atributo removido
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train_r, y_train_r)
y_pred_r = knn.predict(X_test_r)

# Exibindo métricas
print("\nMétricas do modelo sem o atributo (Comprimento da Sépala):")
print(f"Acurácia: {accuracy_score(y_test_r, y_pred_r):.4f}")
print("Matriz de Confusão:")
print(confusion_matrix(y_test_r, y_pred_r))
print("Relatório de Classificação:")
print(classification_report(y_test_r, y_pred_r))
