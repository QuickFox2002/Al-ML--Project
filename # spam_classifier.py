
import pandas as pd
import numpy as np
import string
import re
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# Preprocessing function
def preprocess_text(text):
    text = text.lower()  
    text = re.sub(f"[{string.punctuation}]", " ", text)  
    text = re.sub(r"\d+", " ", text)  
    text = re.sub(r"\s+", " ", text).strip()  
    return text

# Load dataset from KaggleHub (SMS Spam Collection Dataset)
def load_sms_dataset():
    import kagglehub
    dataset_path = kagglehub.dataset_download("uciml/sms-spam-collection-dataset")
    df = pd.read_csv(dataset_path + "/spam.csv", encoding="latin-1")
    df = df.rename(columns={"v1": "label", "v2": "message"})
    df = df[["label", "message"]]
    return df

def main():
    print("Loading dataset...")
    df = load_sms_dataset()
    df = df.dropna().reset_index(drop=True)
    print("Total messages:", len(df))

    # Encode labels (ham=0, spam=1)
    df["label"] = df["label"].map({"ham": 0, "spam": 1})

    # Preprocess messages
    df["message_clean"] = df["message"].apply(preprocess_text)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        df["message_clean"], df["label"], test_size=0.2, random_state=42
    )

    # Vectorize text
    vectorizer = TfidfVectorizer(stop_words="english", max_features=3000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Train classifier
    model = MultinomialNB()
    model.fit(X_train_vec, y_train)

    # Predict
    y_pred = model.predict(X_test_vec)

    # Evaluation
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-score:  {f1:.4f}")

    # Confusion matrix (directly show in output cell)
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Ham", "Spam"])
    disp.plot(cmap="Blues")
    plt.title("Confusion Matrix - Spam SMS Classifier")
    plt.show()

    # Save model & vectorizer
    joblib.dump((model, vectorizer), "spam_model.pkl")
    print("Saved trained model to spam_model.pkl")

if __name__ == "__main__":
    main()
