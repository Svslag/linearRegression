import joblib
import numpy as np

# Загрузка модели
try:
    model = joblib.load("model.pkl")
    print("Модель загружена. Введите 'q' для выхода.\n")
except FileNotFoundError:
    print("Ошибка: model.pkl не найден. Сначала запустите train.py")
    exit()

# Ввод и предсказание
while True:
        # input — ждёт ввода с клавиатуры, strip — убирает лишние пробелы
    user_input = input("Введите X: ").strip()

    # Если ввели q — break выходит из цикла и программа завершается
    if user_input.lower() == "q":
        print("Выход.")
        break

    # Проверяем что ввод не пустой
    if user_input == "":
        print("Ошибка: вы ничего не ввели.\n")
        continue

    # float — превращаем текст в число. Если ввели слово — вызовет ошибку ValueError
    try:
        x_value = float(user_input)

            # except — перехватываем ошибку если ввели слово вместо числа
    except ValueError:
        print(f"Ошибка: '{user_input}' — это не число. Введите числовое значение.\n")
        continue
    
  # np.array — оборачиваем число в массив, модель принимает только массивы
    # [0] — берём первый элемент результата так как predict возвращает список
    prediction = model.predict(np.array([[x_value]]))[0]
    print(f"X = {x_value}  →  Y = {prediction:.4f}\n")