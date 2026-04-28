import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow import keras

keras.utils.set_random_seed(42)


df = pd.read_csv("heart.csv")

print(df.shape)
print(df["target"].value_counts(normalize=True, dropna=False))


categorical_variables = ['sex','cp','fbs','restecg','exang','ca','thal']
numerical_variables = ['age','trestbps','chol','thalach','oldpeak','slope']

# one-hot encoding
df = pd.get_dummies(df, columns=categorical_variables)


test_df = df.sample(frac=0.2, random_state=42)
train_df = df.drop(test_df.index)


mean = train_df[numerical_variables].mean()
std = train_df[numerical_variables].std()

train_df[numerical_variables] = (train_df[numerical_variables] - mean) / std
test_df[numerical_variables]  = (test_df[numerical_variables] - mean) / std

print(train_df.head())


train_y = train_df["target"].values
test_y  = test_df["target"].values

train_X = train_df.drop(columns=["target"])
test_X  = test_df.drop(columns=["target"])


train_X = train_X.astype(np.float32).values
test_X  = test_X.astype(np.float32).values
train_y = train_y.astype(np.float32)
test_y  = test_y.astype(np.float32)

num_columns = train_X.shape[1]


# inputs = keras.Input(shape=(num_columns,))
# h = keras.layers.Dense(16, activation="relu", name="Hidden")(inputs)
# outputs = keras.layers.Dense(1, activation="sigmoid", name="Output")(h)

inputs = keras.Input(shape=(num_columns,))

h1 = keras.layers.Dense(16, activation="relu", name="Hidden_1")(inputs)
h2 = keras.layers.Dense(16, activation="relu", name="Hidden_2")(h1)
h3 = keras.layers.Dense(16, activation="relu", name="Hidden_3")(h2)

outputs = keras.layers.Dense(1, activation="sigmoid", name="Output")(h3)


model = keras.Model(inputs, outputs)

print(model.summary())


try:
    keras.utils.plot_model(model, show_shapes=True)
except:
    print("Graphviz not installed, skipping model plot")

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)


history = model.fit(
    train_X,
    train_y,
    epochs=300,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)


history_dict = history.history

loss_values = history_dict["loss"]
val_loss_values = history_dict["val_loss"]
epochs = range(1, len(loss_values) + 1)

plt.plot(epochs, loss_values, "bo", label="Training loss")
plt.plot(epochs, val_loss_values, "b", label="Validation loss")
plt.title("Training and validation loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

plt.clf()
acc = history_dict["accuracy"]
val_acc = history_dict["val_accuracy"]

plt.plot(epochs, acc, "bo", label="Training accuracy")
plt.plot(epochs, val_acc, "b", label="Validation accuracy")
plt.title("Training and validation accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()


test_loss, test_acc = model.evaluate(test_X, test_y, verbose=0)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")