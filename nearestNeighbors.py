from initialData import *  # Входные данные
import numpy as np  # Для работы с массивами данных
from plotDecisionRegions import plot_decision_regions
from sklearn.model_selection import train_test_split  # Разделение данных на тренировочную и тестовую выборки
from sklearn.preprocessing import StandardScaler # Масштабирование (стандартизация) признаков
from sklearn.neighbors import KNeighborsClassifier  # Реализация модели KNN
from sklearn.model_selection import GridSearchCV  # Гиперпараметрическая оптимизация

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Масштабирование (стандартизация) признаков
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

# Реализация модели KNN
# Гиперпараметрическая оптимизация
param_grid=[{'n_neighbors':range(1,100),'metric':['minkowski']}]
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
print("\n err_train = ", err_train, "# -> err_train = ", "\n err_test = ", err_test)

#Визуализация решения задачи классификации
X_combined = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))
plt.figure(figsize=(12, 8))
plot_decision_regions(X_combined, y_combined, classifier=knn)
plt.xlabel('Признак 1')
plt.ylabel('Признак 2')
plt.legend(loc='upper left')
plt.show()
