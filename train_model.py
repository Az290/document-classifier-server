import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib
import os

# Đọc dataset
df = pd.read_csv('dataset.csv')

print(f" Tổng số mẫu: {len(df)}")
print(f" Phân bố nhãn:")
print(df['label'].value_counts())

# Chia train/test
X_train, X_test, y_train, y_test = train_test_split(
    df['text'], 
    df['label'], 
    test_size=0.2, 
    random_state=42,
    stratify=df['label']
)

print(f"\n Train set: {len(X_train)} mẫu")
print(f" Test set: {len(X_test)} mẫu")

# TF-IDF Vectorization
print("\n Vectorizing text...")
vectorizer = TfidfVectorizer(
    max_features=2000,
    ngram_range=(1, 2),
    min_df=1,
    max_df=0.9,
    lowercase=True,
    strip_accents='unicode'
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

print(f" Feature vector shape: {X_train_vec.shape}")

# Train Random Forest (đơn giản, hiệu quả)
print("\n Training Random Forest...")
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train_vec, y_train)
y_pred = model.predict(X_test_vec)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n Accuracy: {accuracy * 100:.2f}%")

# Đánh giá chi tiết
print("\n Classification Report:")
print(classification_report(
    y_test, 
    y_pred,
    target_names=['HÓA ĐƠN', 'HỢP ĐỒNG', 'CMND', 'KHÁC'],
    zero_division=0
))

print("\n Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Lưu model
os.makedirs('models', exist_ok=True)
joblib.dump(model, 'models/document_classifier.pkl')
joblib.dump(vectorizer, 'models/tfidf_vectorizer.pkl')

print("\n Model saved successfully!")
print(f" Model size: {os.path.getsize('models/document_classifier.pkl') / 1024 / 1024:.2f} MB")