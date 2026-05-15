from flask import Flask, render_template_string, request, jsonify
import joblib, numpy as np, pandas as pd, base64, os
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

app = Flask(__name__)

# Загрузка модели и данных
model = joblib.load("model.pkl")
data  = pd.read_csv("Linear Regression - Sheet1.csv")
data  = data[~data["X"].isin([299, 300])].copy()

X = data[["X"]]
y = data["Y"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

y_pred = model.predict(X_test)
r2  = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
k   = model.coef_[0]
b   = model.intercept_
samples = data.head(10).to_dict(orient="records")

# График в base64
def get_plot():
    if os.path.exists("regression_plot.png"):
        with open("regression_plot.png", "rb") as f:
            return base64.b64encode(f.read()).decode()
    return None

HTML = """
<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="UTF-8">
<title>Linear Regression</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: 'Segoe UI', sans-serif; background: #f4f6f9; color: #333; }
  header { background: #1a1a2e; color: white; padding: 20px 40px; }
  header h1 { font-size: 22px; }
  header p  { font-size: 13px; color: #aaa; margin-top: 4px; }
  main { max-width: 900px; margin: 28px auto; padding: 0 20px; }
  section { background: white; border-radius: 10px; padding: 20px; margin-bottom: 20px; border: 1px solid #e2e8f0; }
  h2 { font-size: 15px; color: #555; margin-bottom: 14px; padding-bottom: 8px; border-bottom: 1px solid #eee; }
  .metrics { display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; }
  .card { background: #f8fafc; border-radius: 8px; padding: 14px; text-align: center; }
  .card .label { font-size: 12px; color: #888; margin-bottom: 4px; }
  .card .value { font-size: 20px; font-weight: 500; color: #1D9E75; }
  .card .value.dark { color: #1a1a2e; }
  table { width: 100%; border-collapse: collapse; font-size: 14px; }
  th { background: #f1f5f9; padding: 9px 12px; text-align: left; }
  td { padding: 8px 12px; border-top: 1px solid #f1f5f9; }
  img { width: 100%; border-radius: 8px; }
  .note { background: #fff5f5; border: 1px solid #fed7d7; border-radius: 8px; padding: 12px; font-size: 14px; color: #c53030; }
  input { border: 1px solid #cbd5e0; border-radius: 8px; padding: 9px 14px; font-size: 15px; width: 160px; }
  button { background: #1a1a2e; color: white; border: none; border-radius: 8px; padding: 9px 22px; font-size: 15px; cursor: pointer; margin-left: 10px; }
  button:hover { background: #378ADD; }
  .result { margin-top: 14px; font-size: 18px; font-weight: 500; color: #1D9E75; }
  .error  { margin-top: 8px; font-size: 14px; color: #c53030; }
</style>
</head>
<body>
<header>
  <h1>Linear Regression </h1>
  <p>Kaggle dataset · scikit-learn · Flask</p>
</header>
<main>

  <section>
    <h2>Метрики модели</h2>
    <div class="metrics">
      <div class="card"><div class="label">R²</div><div class="value">{{ "%.4f"|format(r2) }}</div></div>
      <div class="card"><div class="label">Точность</div><div class="value">{{ "%.2f"|format(r2*100) }}%</div></div>
      <div class="card"><div class="label">MSE</div><div class="value dark">{{ "%.6f"|format(mse) }}</div></div>
      <div class="card"><div class="label">Формула</div><div class="value dark" style="font-size:13px;">Y = {{ "%.4f"|format(k) }}·X + {{ "%.4f"|format(b) }}</div></div>
    </div>
  </section>

  {% if plot %}
  <section>
    <h2>График</h2>
    <img src="data:image/png;base64,{{ plot }}">
  </section>
  {% endif %}

  <section>
    <h2>Предобработка</h2>
    <div class="note">Удалены строки X=299 и X=300 — значение Y=1.89 вместо ожидаемых ~199–200 (ошибки в данных).</div>
  </section>

  <section>
    <h2>10 примеров из датасета</h2>
    <table>
      <thead><tr><th>#</th><th>X</th><th>Y</th></tr></thead>
      <tbody>
        {% for row in samples %}
        <tr><td>{{ loop.index }}</td><td>{{ row.X }}</td><td>{{ "%.4f"|format(row.Y) }}</td></tr>
        {% endfor %}
      </tbody>
    </table>
  </section>

  <section>
    <h2>Предсказание</h2>
    <input type="text" id="x-input" placeholder="Введите X">
    <button onclick="predict()">Предсказать</button>
    <div class="result" id="result"></div>
    <div class="error"  id="error"></div>
  </section>

</main>
<script>
async function predict() {
  const val = document.getElementById('x-input').value.trim();
  document.getElementById('result').textContent = '';
  document.getElementById('error').textContent  = '';

  if (!val) { document.getElementById('error').textContent = 'Введите значение X.'; return; }
  if (isNaN(val)) { document.getElementById('error').textContent = `"${val}" — не число.`; return; }

  const res  = await fetch('/predict', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({x: parseFloat(val)})
  });
  const data = await res.json();
  if (data.error) document.getElementById('error').textContent = data.error;
  else document.getElementById('result').textContent = `X = ${data.x}  →  Y = ${data.prediction}`;
}
document.getElementById('x-input').addEventListener('keydown', e => { if (e.key === 'Enter') predict(); });
</script>
</body>
</html>
"""

@app.route("/")
def index():
    return render_template_string(HTML, r2=r2, mse=mse, k=k, b=b, samples=samples, plot=get_plot())

@app.route("/predict", methods=["POST"])
def predict():
    try:
        x_val = float(request.get_json()["x"])
        y_val = model.predict(np.array([[x_val]]))[0]
        return jsonify({"x": x_val, "prediction": round(float(y_val), 4)})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    print("Сервер запущен: http://127.0.0.1:5000")
    app.run(debug=True)