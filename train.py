import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import os

from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout, Dense, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ---------------- PATHS ----------------
dataset_path = "D:/archive/real_and_fake_face"

# ---------------- VISUALIZATION ----------------
def load_img(path):
    image = cv2.imread(path)
    image = cv2.resize(image, (224, 224))
    return image[..., ::-1]

real = os.path.join(dataset_path, "training_real")
fake = os.path.join(dataset_path, "training_fake")

real_path = os.listdir(real)
fake_path = os.listdir(fake)

# Show sample real images
plt.figure(figsize=(10, 10))
for i in range(16):
    plt.subplot(4, 4, i + 1)
    plt.imshow(load_img(os.path.join(real, real_path[i])))
    plt.axis('off')
plt.suptitle("Real Faces")
plt.show()

# Show sample fake images
plt.figure(figsize=(10, 10))
for i in range(16):
    plt.subplot(4, 4, i + 1)
    plt.imshow(load_img(os.path.join(fake, fake_path[i])))
    plt.axis('off')
plt.suptitle("Fake Faces")
plt.show()

# ---------------- DATA AUGMENTATION ----------------
data_gen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    rotation_range=15,
    zoom_range=0.1,
    validation_split=0.2
)

train = data_gen.flow_from_directory(
    dataset_path,
    target_size=(96, 96),
    batch_size=32,
    class_mode="binary",
    subset="training"
)
print("Class mapping:", train.class_indices)

val = data_gen.flow_from_directory(
    dataset_path,
    target_size=(96, 96),
    batch_size=32,
    class_mode="binary",
    subset="validation"
)

# ---------------- MODEL ----------------
base_model = MobileNetV2(include_top=False, weights="imagenet", input_shape=(96, 96, 3))

for layer in base_model.layers[:-20]:
    layer.trainable = False

for layer in base_model.layers[-20:]:
    layer.trainable = True

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(512, activation="relu"),
    BatchNormalization(),
    Dropout(0.5),   # increased to reduce overfitting
    Dense(128, activation="relu"),
    Dropout(0.3),
    Dense(1, activation="sigmoid")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ---------------- CALLBACKS ----------------
def scheduler(epoch, lr):
    if epoch < 3:
        return 0.001
    elif epoch < 10:
        return 0.0001
    else:
        return 0.00001

lr_callback = tf.keras.callbacks.LearningRateScheduler(scheduler)

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    "best_model.keras",
    monitor='val_accuracy',
    save_best_only=True
)

# ---------------- TRAINING ----------------
hist = model.fit(
    train,
    epochs=5,   # reduced (enough for demo)
    validation_data=val,
    callbacks=[lr_callback, early_stop, checkpoint]
)

# ---------------- SAVE MODEL ----------------
model.save("final_model.keras")

# ---------------- PLOTTING ----------------
train_loss = hist.history['loss']
val_loss = hist.history['val_loss']
train_acc = hist.history['accuracy']
val_acc = hist.history['val_accuracy']

xc = range(len(train_loss))

# Loss graph
plt.figure(figsize=(7, 5))
plt.plot(xc, train_loss, label='Train Loss')
plt.plot(xc, val_loss, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss Graph')
plt.legend()
plt.grid(True)
plt.show()

# Accuracy graph
plt.figure(figsize=(7, 5))
plt.plot(xc, train_acc, label='Train Accuracy')
plt.plot(xc, val_acc, label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Accuracy Graph')
plt.legend()
plt.grid(True)
plt.show()