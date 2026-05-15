import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# pd.read_csv — читаем CSV файл и загружаем в таблицу
data = pd.read_csv("Linear Regression - Sheet1.csv")

# Удаление аномалий
data = data[~data["X"].isin([299, 300])].copy()

# X — входные данные для модели, y — правильные ответы которые модель должна предсказывать
X = data[["X"]]
y = data["Y"]

# Делим данные: 80% на обучение, 20% на проверку. random_state=42 фиксирует случайность
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Создаём модель линейной регрессии
model = LinearRegression()
# fit — обучаем модель на train данных, она ищет лучшую прямую линию через точки

model.fit(X_train, y_train)

# coef_ — наклон линии (k), intercept_ — точка где линия пересекает ось Y (b)

k = model.coef_[0]
b = model.intercept_

# predict — модель делает предсказания для train и test данных
y_pred_train = model.predict(X_train)
y_pred_test  = model.predict(X_test)

# mean_squared_error — средняя ошибка (чем меньше тем лучше)
# r2_score — точность модели (1.0 = идеально, 0 = бесполезно)

print(f"Формула : Y = {k:.4f} * X + {b:.4f}")
print(f"Train  →  MSE: {mean_squared_error(y_train, y_pred_train):.6f}  |  R²: {r2_score(y_train, y_pred_train):.6f}")
print(f"Test   →  MSE: {mean_squared_error(y_test, y_pred_test):.6f}  |  R²: {r2_score(y_test, y_pred_test):.6f}")

# dump — сохраняем обученную модель в файл чтобы использовать в predict.py без повторного обучения
joblib.dump(model, "model.pkl")
print("Модель сохранена: model.pkl")

# График
## plot — рисует линию регрессии через предсказанные точки

# scatter — рисует точки на графике
plt.figure(figsize=(10, 5))
plt.scatter(X_train, y_train, color="steelblue", s=12, alpha=0.5, label="Train")
plt.scatter(X_test,  y_test,  color="orange",    s=12, alpha=0.8, label="Test")

# linspace — создаёт 200 равномерных точек от min до max для рисования линии
x_line = np.linspace(X["X"].min(), X["X"].max(), 200).reshape(-1, 1)

plt.plot(x_line, model.predict(x_line), color="red", linewidth=2, label="Regression line")
plt.title(f"Y = {k:.4f} * X + {b:.4f}  |  R² = {r2_score(y_test, y_pred_test):.6f}")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

## savefig — сохраняем график как картинку

plt.savefig("regression_plot.png", dpi=150)
plt.show()