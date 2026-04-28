import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

keras.utils.set_random_seed(42)

train_url = "untitled folder/lab6_dataset/atis_train_data.csv"
test_url  = "untitled folder/lab6_dataset/atis_test_data.csv"


df_train = pd.read_csv(train_url, index_col=0)
df_test = pd.read_csv(test_url, index_col=0)

pd.set_option('display.max_colwidth', None)

df_small = pd.DataFrame(columns=['query','intent','slot filling'])
j = 0
for i in df_train.intent.unique():
    df_small.loc[j] = df_train[df_train.intent==i].iloc[0]
    j += 1


query_data_train = df_train['query'].values
slot_data_train = df_train['slot filling'].values

query_data_test = df_test['query'].values
slot_data_test = df_test['slot filling'].values

max_query_length = 30


text_vectorization_query = keras.layers.TextVectorization(
    output_sequence_length=max_query_length
)

text_vectorization_query.adapt(query_data_train)
query_vocab_size = text_vectorization_query.vocabulary_size()

source_train = text_vectorization_query(query_data_train)
source_test = text_vectorization_query(query_data_test)


text_vectorization_slots = keras.layers.TextVectorization(
    output_sequence_length=max_query_length,
    standardize=None
)

text_vectorization_slots.adapt(slot_data_train)
slot_vocab_size = text_vectorization_slots.vocabulary_size()

target_train = text_vectorization_slots(slot_data_train)
target_test = text_vectorization_slots(slot_data_test)


class TransformerEncoder(layers.Layer):
    def __init__(self, embed_dim, dense_dim, num_heads):
        super().__init__()
        self.att = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embed_dim // num_heads
        )
        self.ffn = keras.Sequential(
            [
                layers.Dense(dense_dim, activation="relu"),
                layers.Dense(embed_dim)
            ]
        )
        self.layernorm1 = layers.LayerNormalization()
        self.layernorm2 = layers.LayerNormalization()

    def call(self, inputs):
        attn_output = self.att(inputs, inputs)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        return self.layernorm2(out1 + ffn_output)


class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super().__init__()
        self.token_emb = layers.Embedding(
            input_dim=vocab_size,
            output_dim=embed_dim
        )
        self.pos_emb = layers.Embedding(
            input_dim=maxlen,
            output_dim=embed_dim
        )

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions


embed_dim = 512
dense_dim = 64
num_heads = 5

embedding = TokenAndPositionEmbedding(
    max_query_length,
    query_vocab_size,
    embed_dim
)

te = TransformerEncoder(embed_dim, dense_dim, num_heads)

inputs = keras.layers.Input(shape=(max_query_length,))
x = embedding(inputs)
encoder_out = te(x)

x = keras.layers.Dense(128, activation='relu')(encoder_out)
x = keras.layers.Dropout(0.5)(x)
outputs = keras.layers.Dense(slot_vocab_size, activation="softmax")(x)

model = keras.Model(inputs, outputs)

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()


BATCH_SIZE = 64
epochs = 10

history = model.fit(
    source_train,
    target_train,
    batch_size=BATCH_SIZE,
    epochs=epochs
)

model.evaluate(source_test, target_test)


def slot_filling_accuracy(actual, predicted, only_slots=False):
    not_padding = np.not_equal(actual, 0)

    if only_slots:
        non_slot_token = text_vectorization_slots(['O']).numpy()[0, 0]
        slots = np.not_equal(actual, non_slot_token)
        correct_predictions = np.equal(actual, predicted)[not_padding & slots]
    else:
        correct_predictions = np.equal(actual, predicted)[not_padding]

    return np.mean(correct_predictions)

predicted = np.argmax(model.predict(source_test), axis=-1).reshape(-1)
actual = np.array(target_test).reshape(-1)

acc = slot_filling_accuracy(actual, predicted, only_slots=False)
acc_slots = slot_filling_accuracy(actual, predicted, only_slots=True)

print(f'Accuracy = {acc:.3f}')
print(f'Accuracy on slots = {acc_slots:.3f}')

# =============================
# Prediction Function
# =============================
def predict_slots_query(query):
    sentence = text_vectorization_query([query])
    prediction = np.argmax(model.predict(sentence), axis=-1)[0]

    inverse_vocab = dict(enumerate(text_vectorization_slots.get_vocabulary()))
    decoded_prediction = " ".join(inverse_vocab[int(i)] for i in prediction)
    return decoded_prediction

examples = [
            'from los angeles',
            'to los angeles',
            'cheapest flight from boston to los angeles tomorrow',
            'what is the airport at orlando',
            'what are the air restrictions on flights from pittsburgh to atlanta for the airfare of 416 dollars',
            'flight from boston to santiago',
]

for e in examples:
  print(e)
  print(predict_slots_query(e))
  print()


# Example
print(predict_slots_query("cheapest flight to fly from MIT to Mars"))