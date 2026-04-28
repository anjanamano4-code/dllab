import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# -----------------------------------------
# 1. Load Dataset
# -----------------------------------------

df = pd.read_csv("untitled folder/lab8_dataset/sms_spam.csv")

print("\n--- Dataset Preview ---")
print(df.head())


# -----------------------------------------
# 2. Convert labels (ham=0, spam=1)
# -----------------------------------------

df['label'] = df['status'].map({'ham': 0, 'spam': 1})


# -----------------------------------------
# 3. Train-Test Split
# -----------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    df['text'],
    df['label'],
    test_size=0.2,
    random_state=42
)


# -----------------------------------------
# 4. Text Vectorization
# -----------------------------------------

vectorizer = TfidfVectorizer()

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)


# -----------------------------------------
# 5. Train Model
# -----------------------------------------

model = LogisticRegression()
model.fit(X_train_vec, y_train)


# -----------------------------------------
# 6. Evaluation
# -----------------------------------------

y_pred = model.predict(X_test_vec)

print("\n--- Model Evaluation ---")
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))


# -----------------------------------------
# 7. Prediction on Dataset
# -----------------------------------------

print("\n=== Predictions ===")

all_text_vec = vectorizer.transform(df['text'])
predictions = model.predict(all_text_vec)

df['predicted'] = predictions
df['predicted_label'] = df['predicted'].map({0: 'ham', 1: 'spam'})

for i in range(len(df)):
    print(f"\nMessage: {df['text'][i]}")
    print(f"Actual: {df['status'][i]}")
    print(f"Predicted: {df['predicted_label'][i]}")


# -----------------------------------------
# 8. User Input Prediction
# -----------------------------------------

print("\n--- Test Your Own Message ---")

user_input = input("Enter a message: ")

user_vec = vectorizer.transform([user_input])
prediction = model.predict(user_vec)[0]

label = "spam" if prediction == 1 else "ham"

print("\nPrediction:", label)