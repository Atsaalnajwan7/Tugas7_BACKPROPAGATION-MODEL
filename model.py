import pandas as pd
import numpy as np
import joblib
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

Sequential = tf.keras.models.Sequential
Dense = tf.keras.layers.Dense
EarlyStopping = tf.keras.callbacks.EarlyStopping

# Load dataset
data = pd.read_csv('Energy_consumption_dataset.csv')

print("Data awal:")
print(data.head())

# =========================
# 🔥 ENCODING DATA KATEGORI
# =========================

# DayOfWeek → angka
data['DayOfWeek'] = data['DayOfWeek'].astype('category').cat.codes

# Holiday → Yes/No jadi 1/0
data['Holiday'] = data['Holiday'].map({'No': 0, 'Yes': 1})

# HVACUsage → On/Off jadi 1/0
data['HVACUsage'] = data['HVACUsage'].map({'Off': 0, 'On': 1})

# LightingUsage → On/Off jadi 1/0
data['LightingUsage'] = data['LightingUsage'].map({'Off': 0, 'On': 1})

print("\nSetelah encoding:")
print(data.head())

# =========================

# Pisahkan fitur & target
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values.reshape(-1, 1)

# Normalisasi
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X = scaler_X.fit_transform(X)
y = scaler_y.fit_transform(y)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model ANN
model = Sequential([
    Dense(16, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(12, activation='relu'),
    Dense(8, activation='relu'),
    Dense(1, activation='linear')
])

model.compile(
    optimizer='adam',
    loss='mean_squared_error'
)

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

# Training
model.fit(
    X_train,
    y_train,
    epochs=100,
    batch_size=5,
    validation_data=(X_test, y_test),
    callbacks=[early_stopping]
)
# Evaluasi
loss = model.evaluate(X_test, y_test)
print(f"\nLoss model: {loss}")

# Simpan model & scaler
model.save('model.h5')
joblib.dump(scaler_X, 'scaler_X.pkl')
joblib.dump(scaler_y, 'scaler_y.pkl')

# Simpan nama fitur
feature_names = data.columns[:-1].tolist()
joblib.dump(feature_names, 'features.pkl')

print("\nModel berhasil dibuat!")