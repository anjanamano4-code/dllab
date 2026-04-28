# Implement a multi-class lyric genre classification model using Keras with TextVectorization (multi-hot encoding), Dense layers, and Softmax output. Train and evaluate the model.

import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import os

keras.utils.set_random_seed(42)



def load_data():
    print("Loading dataset from lab5_dataset")

    base_path = "lab5_dataset"

    train_path =  "untitled folder/lab5_dataset/lyric_genre_train.csv"
    val_path = "untitled folder/lab5_dataset/lyric_genre_val.csv"
    test_path = "untitled folder/lab5_dataset/lyric_genre_test.csv"

    # Check if files exist
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"{train_path} not found!")

    train_df = pd.read_csv(train_path, index_col=0)
    val_df = pd.read_csv(val_path, index_col=0)
    test_df = pd.read_csv(test_path, index_col=0)

    print(f"""
    Train samples: {train_df.shape[0]}
    Validation samples: {val_df.shape[0]}
    Test samples: {test_df.shape[0]}
    """)

    return train_df, val_df, test_df


def preprocess_labels(train_df, val_df, test_df):
    y_train = pd.get_dummies(train_df['Genre']).to_numpy()
    y_val = pd.get_dummies(val_df['Genre']).to_numpy()
    y_test = pd.get_dummies(test_df['Genre']).to_numpy()

    return y_train, y_val, y_test



def create_text_vectorizer(train_text, max_tokens=20000):
    text_vectorization = keras.layers.TextVectorization(
        ngrams=2,
        max_tokens=max_tokens,
        output_mode="multi_hot"
    )

    print("Adapting text vectorizer...")
    text_vectorization.adapt(train_text)

    return text_vectorization


def vectorize_data(text_vectorization, train_df, val_df, test_df):
    X_train = text_vectorization(train_df['Lyric'])
    X_val = text_vectorization(val_df['Lyric'])
    X_test = text_vectorization(test_df['Lyric'])

    return X_train, X_val, X_test



def build_model(input_dim):
    inputs = keras.Input(shape=(input_dim,))
    x = keras.layers.Dense(8, activation="relu")(inputs)
    x = keras.layers.Dropout(0.5)(x)
    outputs = keras.layers.Dense(3, activation="softmax")(x)

    model = keras.Model(inputs, outputs)

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


def lyric_predict(model, text_vectorization, phrase):
    vect_data = text_vectorization([phrase])
    predictions = model.predict(vect_data)

    print("\nPrediction:")
    print(f"{float(predictions[0,0] * 100):.2f}% Hip-Hop")
    print(f"{float(predictions[0,1] * 100):.2f}% Pop")
    print(f"{float(predictions[0,2] * 100):.2f}% Rock")


def main():
    train_df, val_df, test_df = load_data()

    y_train, y_val, y_test = preprocess_labels(train_df, val_df, test_df)

    text_vectorization = create_text_vectorizer(train_df['Lyric'])

    X_train, X_val, X_test = vectorize_data(
        text_vectorization, train_df, val_df, test_df
    )

    model = build_model(20000)

    model.summary()

    print("\nTraining model...")
    model.fit(
        x=X_train,
        y=y_train,
        validation_data=(X_val, y_val),
        epochs=3,
        batch_size=32
    )

    print("\nEvaluating model...")
    model.evaluate(x=X_test, y=y_test)

    # Example prediction
    test_phrase = "I got money and power in my hands"
    lyric_predict(model, text_vectorization, test_phrase)


if __name__ == "__main__":
    main()