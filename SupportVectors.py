import numpy as np  # Для работы с массивами данных
import matplotlib.pyplot as plt  # Для работы с графиками
from sklearn.datasets import make_circles  # Импортирование данных
from sklearn.neighbors import KNeighborsClassifier  # Реализация модели KNN
from sklearn.model_selection import GridSearchCV  # Гиперпараметрическая оптимизация
from plotDecisionRegions import plot_decision_regions

X, y = make_circles(n_samples=512, random_state=123, noise=0.22, factor=0.16)
plt.figure()
plt.scatter(X[y == 0, 0], X[y == 0, 1], color='red', marker='^', alpha=0.5, label='0')
plt.scatter(X[y == 1, 0], X[y == 1, 1], color='blue', marker='o', alpha=0.5, label='1')
plt.legend()
plt.title("Исходные данные")
plt.show()

from sklearn.model_selection import train_test_split  # Разделение данных на тренировочную и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

from sklearn.preprocessing import StandardScaler  # Масштабирование (стандартизация) признаков
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

param_grid=[{'n_neighbors': range(1, 100), 'metric': ['minkowski']}]
knn=KNeighborsClassifier()
grid=GridSearchCV(knn,param_grid)
grid.fit(X_train,y_train)
cvres=grid.cv_results_
best_params = grid.best_params_
print('\n best_params=', best_params)

# Обучение оптимизированной модели
knn.fit(X_train_std, y_train)

# Доли ошибок на обучающей и тестовой выборках
err_train = np.mean(y_train != knn.predict(X_train_std))
err_test = np.mean(y_test != knn.predict(X_test_std))
print("\n err_train = ", {err_train}, '# -> err_train = #' "\n err_test = ", {err_test})

X_combined = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))
plt.figure(figsize=(12, 8))
plot_decision_regions(X_combined, y_combined, classifier=knn)
plt.xlabel('Признак 1')
plt.ylabel('Признак 2')
plt.legend(loc='upper left')
plt.show()
