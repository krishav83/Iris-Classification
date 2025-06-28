from flask import Flask, request, render_template, url_for
import joblib
import numpy as np

app = Flask(__name__)

# Load model and preprocessing tools
model = joblib.load("iris_model.pkl")
scaler = joblib.load("scaler.pkl")
le = joblib.load("label_encoder.pkl")

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        features = [float(request.form[i]) for i in ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
        scaled = scaler.transform([features])
        pred = model.predict(scaled)
        species = le.inverse_transform(pred)[0]

        image_filename = f"images/{species.lower()}.jpg"  # e.g., 'images/setosa.jpg'
        image_url = url_for('static', filename=image_filename)

        return render_template("result.html", species=species, image_url=image_url)
    except Exception as e:
        return f"<h2>Error: {e}</h2>"

if __name__ == '__main__':
    app.run(debug=True)
