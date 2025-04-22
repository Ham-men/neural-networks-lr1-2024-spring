import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


#задача: вывести точку эталона для 1 и 2 класса в консоль

def etalon(X_train, y_train, x, classes):

    # Найти эталонный объект для каждого класса
    prototypes = {}
    for c in classes:
        X_class = X_train[y_train == c]
        prototypes[c] = np.mean(X_class, axis=0)

    # Найти ближайший эталонный объект к точке x
    min_dist = float('inf')
    nearest_class = None
    for c, prototype in prototypes.items():
        dist = np.linalg.norm(x - prototype)
        if dist < min_dist:
            min_dist = dist
            nearest_class = c

    return nearest_class

# Генерация синтетических данных в форме буквы X
np.random.seed(2)
n_samples = 10
X = np.zeros((n_samples * 2, 2))
y = np.zeros(n_samples * 2)

for i in range(n_samples):
    if i < n_samples / 2:
        X[i, 0] = i / (n_samples / 2)
        X[i, 1] = 0.5
    else:
        X[i, 0] = 0.5
        X[i, 1] = (i - n_samples / 2) / (n_samples / 2)
    y[i] = 0

for i in range(n_samples):
    if i < n_samples / 2:
        X[i + n_samples, 0] = 0.5
        X[i + n_samples, 1] = (i / (n_samples / 2)) + 0.5
    else:
        X[i + n_samples, 0] = (i - n_samples / 2) / (n_samples / 2) + 0.5
        X[i + n_samples, 1] = 0.5
    y[i + n_samples] = 1


# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Предсказание классов на обучающей выборке с помощью метода эталонных образов
y_train_pred = [etalon(X_train, y_train, x, np.unique(y_train)) for x in X_train]
#print("эталон для обучающей = ",y_train_pred)
# Предсказание классов на тестовой выборке с помощью метода эталонных образов
y_test_pred = [etalon(X_train, y_train, x, np.unique(y_train)) for x in X_test]
#print("эталон для тестовой = ",y_test_pred)
# Обучение модели K ближайших соседей
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)


# Найти среднее значение для каждого класса
mean_class_values = {}
for c in np.unique(y_train):
    X_class = X_train[y_train == c]
    mean_class_values[c] = np.mean(X_class, axis=0)

# Вывести среднее значение для каждого класса
for c, mean_value in mean_class_values.items():
    print(f"Среднее значение для класса {c}: {mean_value}")


# Оценка точности модели на обучающей выборке
train_accuracy = accuracy_score(y_train, y_train_pred)

# Оценка точности модели на тестовой выборке
test_accuracy = accuracy_score(y_test, y_test_pred)

print(f"Точность модели на обучающей выборке: {train_accuracy}")
print(f"Точность модели на тестовой выборке: {test_accuracy}")

# Построение графика
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm)
plt.xlabel('Признак 1')
plt.ylabel('Признак 2')

# Разделяющая граница модели
#вывод эталона
h = .02  # Шаг сетки в графике
x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = [etalon(X_train, y_train, [x, y], np.unique(y_train)) for x, y in zip(xx.ravel(), yy.ravel())]
Z = np.array(Z).reshape(xx.shape)
#print("z эталон = ",Z)
#plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)

#======
#вывод ближайшего соседа
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)

#===
plt.show()

