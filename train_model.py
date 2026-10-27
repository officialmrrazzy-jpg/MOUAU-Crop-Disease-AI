import os
import tensorflow as tf
from tensorflow.keras import layers, models

print("MOUAU Crop Disease Detection AI Initialization...")

# 1. Build Directory Matrix
healthy_dir = '/content/crop_dataset/healthy_leaves'
diseased_dir = '/content/crop_dataset/diseased_leaves'
os.makedirs(healthy_dir, exist_ok=True)
os.makedirs(diseased_dir, exist_ok=True)

# 2. Compile CNN Architecture
model = models.Sequential([
    layers.Input(shape=(150, 150, 3)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(2, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
print("[SUCCESS] AI Architecture Compiled Flawlessly.")
