from flask import Flask, render_template, request
import numpy as np
import tensorflow as tf
import joblib

app = Flask(__name__)

model = tf.keras.models.load_model('model.h5')
scaler_X = joblib.load('scaler_X.pkl')
scaler_y = joblib.load('scaler_y.pkl')
features = joblib.load('features.pkl')


@app.route('/')
def index():
    return render_template('index.html', features=features)


@app.route('/predict', methods=['POST'])
def predict():
    try:
        values = []

        for feature in features:
            value = float(request.form[feature])
            values.append(value)

        data = np.array([values])

        data_scaled = scaler_X.transform(data)
        prediction = model.predict(data_scaled)
        result = scaler_y.inverse_transform(prediction)

        hasil = round(result[0][0], 2)

        return render_template(
            'index.html',
            prediction_text=f'Hasil Prediksi Konsumsi Energi: {hasil}',
            features=features
        )

    except Exception as e:
        return render_template(
            'index.html',
            prediction_text=f'Error: {str(e)}',
            features=features
        )


import os

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))