import matplotlib.pyplot as plt  # Для работы с графиками
from sklearn.datasets import make_circles  # Импортирование данных


X, y = make_circles(n_samples=512, random_state=123, noise=0.22, factor=0.16)
plt.figure()
plt.scatter(X[y == 0, 0], X[y == 0, 1], color='red', marker='^', alpha=0.5, label='0')
plt.scatter(X[y == 1, 0], X[y == 1, 1], color='blue', marker='o', alpha=0.5, label='1')
plt.legend()
plt.title("Исходные данные")
plt.show()
