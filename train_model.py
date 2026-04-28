import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os

# Đọc dataset
df = pd.read_csv('dataset.csv')

print(f"Tổng số mẫu: {len(df)}")
print(f"Phân bố nhãn:\n{df['label'].value_counts()}")

# Chia train/test
X_train, X_test, y_train, y_test = train_test_split(
    df['text'], 
    df['label'], 
    test_size=0.2, 
    random_state=42,
    stratify=df['label']
)

# Vectorization
vectorizer = TfidfVectorizer(
    max_features=1000,
    ngram_range=(1, 3),
    min_df=1,
    max_df=0.8
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train model
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)

print("\n Đang train model...")
model.fit(X_train_vec, y_train)

# Đánh giá
y_pred = model.predict(X_test_vec)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n Accuracy: {accuracy * 100:.2f}%")
print("\n Classification Report:")
print(classification_report(
    y_test, 
    y_pred,
    target_names=['HÓA ĐƠN', 'HỢP ĐỒNG', 'CMND', 'KHÁC']
))

# Lưu model
os.makedirs('models', exist_ok=True)
joblib.dump(model, 'models/document_classifier.pkl')
joblib.dump(vectorizer, 'models/tfidf_vectorizer.pkl')

print("\n Model đã được lưu vào thư mục 'models/'")