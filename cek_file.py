import joblib
import tensorflow as tf

# Load model
model = tf.keras.models.load_model('model.h5')
print("Model berhasil dibuka!")

# Tampilkan struktur model
model.summary()

# Load scaler
scaler_X = joblib.load('scaler_X.pkl')
scaler_y = joblib.load('scaler_y.pkl')

print("\nScaler X:", scaler_X)
print("Scaler Y:", scaler_y)

# Load fitur
features = joblib.load('features.pkl')
print("\nNama fitur:")
print(features)