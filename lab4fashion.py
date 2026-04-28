import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import confusion_matrix
import seaborn as sns
import pandas as pd

keras.utils.set_random_seed(42)

DATA_PATH = "untitled folder/fashion_dataset"
IMG_SIZE = 128
BATCH_SIZE = 16

# =====================================
# LOAD DATA (RGB)
# =====================================

train_ds = keras.utils.image_dataset_from_directory(
    os.path.join(DATA_PATH, "train"),
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    color_mode="rgb",   
    label_mode="int"
)

test_ds = keras.utils.image_dataset_from_directory(
    os.path.join(DATA_PATH, "test"),
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    color_mode="rgb",   
    label_mode="int",
    shuffle=False
)

class_names = train_ds.class_names
num_classes = len(class_names)

print("Classes:", class_names)

# =====================================
# DATA AUGMENTATION (Light)
# =====================================

data_augmentation = keras.Sequential([
    keras.layers.RandomFlip("horizontal"),
    keras.layers.RandomRotation(0.05),
])

# =====================================
# NORMALIZATION
# =====================================

normalization_layer = keras.layers.Rescaling(1./255)

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.map(
    lambda x, y: (data_augmentation(normalization_layer(x), training=True), y)
).prefetch(AUTOTUNE)

test_ds = test_ds.map(
    lambda x, y: (normalization_layer(x), y)
).prefetch(AUTOTUNE)

# =====================================
# MODEL (RGB INPUT)
# =====================================

inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))  

x = keras.layers.Conv2D(32, (3,3), padding="same", activation="relu")(inputs)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.MaxPooling2D()(x)

x = keras.layers.Conv2D(64, (3,3), padding="same", activation="relu")(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.MaxPooling2D()(x)

x = keras.layers.Conv2D(128, (3,3), padding="same", activation="relu")(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.MaxPooling2D()(x)

x = keras.layers.GlobalAveragePooling2D()(x)

x = keras.layers.Dense(128, activation="relu")(x)
x = keras.layers.Dropout(0.5)(x)

outputs = keras.layers.Dense(num_classes, activation="softmax")(x)

model = keras.Model(inputs, outputs)

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print(model.summary())

# =====================================
# EARLY STOPPING
# =====================================

early_stop = keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

# =====================================
# TRAIN
# =====================================

history = model.fit(
    train_ds,
    epochs=30,
    validation_data=test_ds,
    callbacks=[early_stop]
)

# =====================================
# PLOT TRAINING CURVES
# =====================================

plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Val Loss")
plt.legend()
plt.title("Loss")

plt.subplot(1,2,2)
plt.plot(history.history["accuracy"], label="Train Acc")
plt.plot(history.history["val_accuracy"], label="Val Acc")
plt.legend()
plt.title("Accuracy")

plt.show()

# =====================================
# EVALUATE
# =====================================

test_loss, test_accuracy = model.evaluate(test_ds)
print("Test Accuracy:", test_accuracy)

# =====================================
# CONFUSION MATRIX
# =====================================

y_true = []
y_pred = []

for images, labels in test_ds:
    preds = model.predict(images)
    y_true.extend(labels.numpy())
    y_pred.extend(np.argmax(preds, axis=1))

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d",
            xticklabels=class_names,
            yticklabels=class_names,
            cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# =====================================
# SAVE MODEL
# =====================================

model.save("custom_model_rgb.keras")

print("RGB Model Saved Successfully!")