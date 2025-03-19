RELATÓRIO – Análise do Algoritmo KNN no Conjunto de Dados Íris
Código original:
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
Código modificado:
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
________________________________________
1. Introdução
Este relatório apresenta a análise da implementação do algoritmo K-Nearest Neighbors (KNN) para classificação do conjunto de dados Íris. O objetivo é entender como diferentes parâmetros afetam o desempenho do modelo, avaliar as métricas resultantes e interpretar os gráficos gerados.
Foram realizadas diversas modificações no código original para testar o impacto das seguintes alterações:
•	Mudança na proporção treino/teste (de 80/20 para 70/30);
•	Variação do número de vizinhos kkk no modelo KNN;
•	Remoção de um atributo do conjunto de dados para avaliar sua importância na predição.
A partir dessas alterações, métricas como acurácia e matriz de confusão foram analisadas para identificar mudanças no desempenho do modelo.
________________________________________
2. Estrutura do Código e Explicação das Bibliotecas
O código foi estruturado em etapas sequenciais para preparar os dados, treinar o modelo e avaliar os resultados. Abaixo está a explicação das bibliotecas utilizadas:
•	NumPy (numpy): Manipulação eficiente de arrays e operações numéricas.
•	Pandas (pandas): Estruturação e manipulação dos dados em DataFrames.
•	Matplotlib (matplotlib.pyplot): Visualização de gráficos.
•	Seaborn (seaborn): Visualização avançada de dados, usada na análise exploratória.
•	Scikit-learn (sklearn): Biblioteca para aprendizado de máquina, usada para carregar o dataset, dividir os dados, padronizar as variáveis e implementar o modelo KNN.
O código segue a seguinte estrutura:
1.	Carregamento do conjunto de dados: O dataset Íris é carregado e transformado em um DataFrame.
2.	Análise exploratória: O dataset é visualizado por meio de gráficos.
3.	Divisão dos dados em treino e teste: O conjunto de dados é separado para treinar e avaliar o modelo.
4.	Padronização: Os dados são normalizados para melhorar o desempenho do KNN.
5.	Treinamento e avaliação do modelo: O KNN é treinado com k=3k=3k=3, e sua performance é avaliada.
6.	Variação do parâmetro kkk: O impacto da escolha de kkk na acurácia do modelo é analisado.
________________________________________
3. Modificações e Análise dos Resultados
3.1. Alteração da Proporção Treino/Teste (70% Treino, 30% Teste)
O código original utilizava 80% dos dados para treino e 20% para teste. Alteramos essa proporção para 70% treino e 30% teste e avaliamos o impacto na acurácia do modelo.
Resultados
A acurácia variou entre 97,7% e 100% para diferentes valores de kkk, mas manteve um desempenho elevado, indicando que o modelo é robusto mesmo com uma mudança na divisão dos dados.
________________________________________
3.2. Impacto da Variação de kkk no Modelo
Testamos valores de kkk variando de 1 a 20, avaliando a acurácia do modelo em cada caso.
Resultados
Os testes mostraram que a acurácia foi estável e alta para a maioria dos valores de kkk, oscilando entre 97,7% e 100%. Isso indica que o modelo é eficiente na classificação mesmo com diferentes quantidades de vizinhos.
A curva de acurácia em função de kkk apresentou um padrão relativamente estável, sem grandes quedas, o que confirma que os dados são bem separáveis entre as classes.
________________________________________
3.3. Remoção de um Atributo (Comprimento da Sépala)
Para entender a importância das variáveis no modelo, removemos a primeira coluna (comprimento da sépala) e repetimos o treinamento e avaliação.
Resultados
A acurácia permaneceu em 100% com k=3k=3k=3, sugerindo que essa característica não é essencial para a classificação correta do conjunto de dados Íris. Isso indica que outras características, como largura da pétala, são mais relevantes para distinguir as espécies.
________________________________________
4. Interpretação dos Gráficos e Métricas
4.1. Gráficos da Análise Exploratória
A análise exploratória exibiu gráficos de dispersão entre as variáveis, coloridos por classe. Os gráficos indicam que as espécies do conjunto de dados Íris possuem uma separação bem definida, facilitando a classificação pelo KNN.
________________________________________
4.2. Matriz de Confusão e Relatório de Classificação
A matriz de confusão para k=3k=3k=3 demonstrou que todas as previsões foram corretas (acurácia de 100%), sem falsos positivos ou falsos negativos. O relatório de classificação confirmou um desempenho excelente do modelo.
________________________________________
5. Conclusão
O experimento com o algoritmo K-Nearest Neighbors (KNN) no conjunto de dados Íris demonstrou a eficiência do modelo para problemas de classificação.
Os principais aprendizados foram:
✅ Ajustes na proporção treino/teste não afetaram significativamente a acurácia.
✅ Variações no número de vizinhos kkk mantiveram a acurácia elevada, confirmando a boa separação das classes.
✅ A remoção de um atributo (comprimento da sépala) não impactou a acurácia, indicando que algumas variáveis são mais informativas do que outras.
Esses resultados mostram que o KNN é uma técnica eficiente para classificação de dados bem estruturados e podem ser aplicados a diversos problemas de machine learning.
________________________________________
6. Aplicações do Algoritmo
O algoritmo KNN pode ser utilizado em diversas aplicações práticas, incluindo:
•	Reconhecimento de padrões (ex: reconhecimento facial, detecção de anomalias em dados).
•	Sistemas de recomendação (ex: recomendação de produtos com base na similaridade entre clientes).
•	Diagnóstico médico (ex: classificação de células cancerígenas com base em medidas de tecidos).
