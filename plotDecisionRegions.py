import numpy as np  # для работы с массивами данных
import matplotlib.pyplot as plt  # для работы с графиками
from matplotlib.colors import ListedColormap  # для создания цветовой карты

def plot_decision_regions (X, y, classifier, resolution=0.02, test_idx=None):
       markers=('s','x','o','^','v')  # обозначение образцов маркерами
       colors=('red','blue','lightgreen','gray','cyan')  # настройка цветовой палитры
       cmap=ListedColormap(colors[:len(np.unique(y))])  # Построение графика поверхности решения
       x1_min,x1_max = X[:, 0].min()-1, X[:, 0].max()+1
       x2_min,x2_max = X[:, 1].min()-1, X[:, 1].max()+1
       xx1,xx2 = np.meshgrid(np.arange(x1_min,x1_max,resolution), np.arange(x2_min,x2_max,resolution))
       Z = classifier.predict(np.array([xx1.ravel(),xx2.ravel()]).T)
       Z = Z.reshape(xx1.shape)
       plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
       plt.xlim(xx1.min(), xx1.max())  # настройка границ графика
       plt.ylim(xx2.min(), xx2.max())
       for idx, cl in enumerate(np.unique(y)):
              plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1], alpha=0.8, c=colors[idx], edgecolor='black', marker=markers[idx], label=cl)
              if test_idx:  # обозначение тестовых образцов
                     X_test=X[test_idx, :]
                     plt.scatter(X_test[:, 0], X_test[:, 1], c='w', alpha=0.3, edgecolor='black', linewidths=1, marker='o', s=120, label='test set')
