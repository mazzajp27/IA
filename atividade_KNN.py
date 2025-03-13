# Etapa 0: Importando pacotes
#B2.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Etapa 1: Carregar o conjunto de dados Íris
#B5.
data = load_iris()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target

# Exibir as cinco primeiras linhas do conjunto de dados
print("Visualização inicial do dataset:")
print(df.head())

# Etapa 2: Análise exploratória dos dados
#B1.
sns.pairplot(df, hue='target', palette='Dark2')
plt.show()

# Etapa 3: Separação em Conjunto de Treino e Teste
#B7.
X = df.iloc[:, :-1]  # Features
y = df['target']      # Target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Etapa 4: Padronização dos Dados
#B6.
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Etapa 5: Treinamento e Avaliação do KNN
#B4.
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

# Avaliação
print("Acurácia do modelo KNN:", accuracy_score(y_test, y_pred))
print("Matriz de Confusão:")
print(confusion_matrix(y_test, y_pred))
print("Relatório de Classificação:")
print(classification_report(y_test, y_pred))


# Etapa 6: Testando Diferentes Valores de k
#B3.
accuracies = []
k_values = range(1, 21)
for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    accuracies.append(accuracy_score(y_test, y_pred))

# Plotando a acurácia para diferentes valores de k
plt.plot(k_values, accuracies, marker='o', linestyle='dashed', color='b')
plt.xlabel('Número de Vizinhos (k)')
plt.ylabel('Acurácia')
plt.title('Impacto de k no Modelo KNN')
plt.show()
