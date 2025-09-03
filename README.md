## 📌 Task 1 — Spam SMS Classifier (NLP with Scikit-learn)

### 🎯 Goal
Build a classifier that predicts whether a given SMS message is **Spam** or **Ham** (Not Spam).

### 🔧 Tools
- pandas, numpy  
- scikit-learn  
- matplotlib / seaborn  

### 📂 Dataset
- [UCI SMS Spam Collection Dataset](https://archive.ics.uci.edu/ml/datasets/sms+spam+collection)  
- Download manually as `spam.csv`.

### 🚀 Steps
1. Load dataset with **pandas**.  
2. Explore spam vs ham message counts.  
3. Preprocess text: lowercase, remove punctuation & stopwords, tokenize.  
4. Convert text to vectors using **TF-IDF (TfidfVectorizer)**.  
5. Split dataset into **train/test**.  
6. Train classifier (**Logistic Regression / Naive Bayes / SVM**).  
7. Evaluate with:
   - Accuracy  
   - Precision, Recall, F1-score  
   - Confusion Matrix  
8. Save model with **joblib**.  
9. Write a script that:
   - Takes user input (`input()`)  
   - Predicts spam/ham using the trained model.  

### 📦 Deliverables
- `spam_classifier.py` → Training code  
- `spam_model.pkl` → Saved model  
- `predict.py` → Script for testing new messages