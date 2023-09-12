#Librerias
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Matriz de confusión
from sklearn.metrics import confusion_matrix
import seaborn as sns  # Para visualizar la matriz de confusión de una manera más clara

from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

# Función sigmoide: 
# Esta función mapea cualquier valor a un valor entre 0 y 1, lo que la hace útil para modelos de probabilidad.
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Función de hipótesis:
# Estima la probabilidad de que y=1 dado un input x.
def hyp(m_b, x):
    return sigmoid(np.dot(x, m_b))

# Función de costo para regresión logística:
# Esta función evalúa qué tan bien el modelo predice comparado con los valores reales de y.
# Buscamos minimizar esta función para mejorar el modelo.
def cost(m_b, x, y):
    m = len(y)
    total_cost = -(1 / m) * np.sum(y * np.log(hyp(m_b, x)) + (1 - y) * np.log(1 - hyp(m_b, x)))
    return total_cost

# Gradiente descendente para regresión logística:
# Es un algoritmo de optimización para minimizar la función de costo.
# Ajusta los coeficientes m_b de manera iterativa para reducir el costo.
def gradient_descent(m_b, x, y, alpha):
    m = len(y)
    h = hyp(m_b, x)
    gradient = np.dot(x.T, (h - y)) / m
    m_b -= alpha * gradient
    return m_b

# Adición de unos al vector de características (bias):
# Agrega un término constante de 1 a las características para representar el intercepto en la regresión.
def add_ones(x):
    ones = np.ones((x.shape[0], 1))
    return np.concatenate((ones, x), axis=1)

# Función manual para dividir en entrenamiento y prueba:
# Divide el dataset en un conjunto de entrenamiento y otro de prueba de forma aleatoria.
def manual_train_test_split(X, y, test_size=0.3, random_state=None):
    if random_state:
        np.random.seed(random_state)
    
    m = len(y)
    test_m = int(m * test_size)
    
    # Permutación aleatoria de índices
    permuted_indices = np.random.permutation(m)
    test_indices = permuted_indices[:test_m]
    train_indices = permuted_indices[test_m:]
    
    X_train = X[train_indices]
    X_test = X[test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]
    
    return X_train, X_test, y_train, y_test

# Carga del dataset Iris y pre-procesamiento básico
df = pd.read_csv('Iris.csv')
df = df.drop(['Id'], axis=1)
types = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
df['Species'] = df['Species'].replace(types)
df = df[df['Species'] != 2]
X = df.iloc[:, 0:-1].values
y = df.iloc[:, -1].values

# División del dataset en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = manual_train_test_split(X, y, test_size=0.3, random_state=42)

X_train = add_ones(X_train)
X_test = add_ones(X_test)

# Inicialización de parámetros
m_b = np.zeros(X_train.shape[1])
alpha = 0.05  # Tasa de aprendizaje
epochs = 5000 # Número de iteraciones

cost_history = []
cost_history_test = []  # Nueva lista para almacenar el historial de pérdida del conjunto de prueba

# Bucle principal de entrenamiento
for _ in range(epochs):
    m_b = gradient_descent(m_b, X_train, y_train, alpha)
    cost_history.append(cost(m_b, X_train, y_train))
    cost_history_test.append(cost(m_b, X_test, y_test))  # Calcula y almacena la pérdida del conjunto de prueba

# Predicciones para el conjunto de entrenamiento
y_train_pred = [1 if hyp(m_b, x) >= 0.5 else 0 for x in X_train]

# Predicciones para el conjunto de prueba
y_test_pred = [1 if hyp(m_b, x) >= 0.5 else 0 for x in X_test]

# Evaluación del modelo en conjunto de entrenamiento
train_accuracy = np.mean(y_train_pred == y_train)
train_error = 1 - train_accuracy

# Evaluación del modelo en conjunto de prueba
test_accuracy = np.mean(y_test_pred == y_test)
test_error = 1 - test_accuracy

# Impresión de métricas
print(f"Training Accuracy: {train_accuracy * 100:.6f}%")
print(f"Training Error: {train_error * 100:.6f}%")
print(f"Test Accuracy: {test_accuracy * 100:.6f}%")
print(f"Test Error: {test_error * 100:.6f}%")

print("Accuracy_Score: ", accuracy_score(y_test, y_test_pred))
print("Precision_score: ", precision_score(y_test, y_test_pred))
print("Recall_score: ", recall_score(y_test, y_test_pred))
print("F1_score: ", f1_score(y_test, y_test_pred))

# Gráfica de cómo cambia la función de pérdida durante el entrenamiento
plt.figure()
plt.plot(cost_history)
plt.title("Función de Pérdida a lo largo de las Épocas (TRAIN)")
plt.xlabel("Épocas")
plt.ylabel("Pérdida")
plt.show()

# Gráfica de cómo cambia la función de pérdida del conjunto de prueba durante el entrenamiento
plt.figure()
plt.plot(cost_history_test, color='red')  # Usaré el color rojo para diferenciar, pero puedes cambiarlo
plt.title("Función de Pérdida a lo largo de las Épocas (TEST)")
plt.xlabel("Épocas")
plt.ylabel("Pérdida")
plt.show()

cm = confusion_matrix(y_test, y_test_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='g', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Matriz de Confusión')
plt.show()

