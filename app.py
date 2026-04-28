from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import os

app = Flask(__name__)
CORS(app)

# Load model
MODEL_PATH = 'models/document_classifier.pkl'
VECTORIZER_PATH = 'models/tfidf_vectorizer.pkl'

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(" Model chưa được train! Chạy train_model.py trước")

model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)

LABELS = ['HÓA ĐƠN', 'HỢP ĐỒNG', 'CMND', 'KHÁC']

@app.route('/')
def home():
    return jsonify({
        'status': 'running',
        'message': 'Document Classifier API',
        'version': '1.0'
    })

@app.route('/classify', methods=['POST'])
def classify():
    try:
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({'error': 'Missing text field'}), 400
        
        text = data['text']
        
        if not text.strip():
            return jsonify({'error': 'Empty text'}), 400
        
        # Vectorize
        text_vector = vectorizer.transform([text])
        
        # Predict
        prediction = model.predict(text_vector)[0]
        probabilities = model.predict_proba(text_vector)[0]
        confidence = float(probabilities[prediction])
        
        result = {
            'label': LABELS[prediction],
            'confidence': confidence,
            'all_probabilities': {
                LABELS[i]: float(prob) 
                for i, prob in enumerate(probabilities)
            }
        }
        
        print(f" Classified: {LABELS[prediction]} ({confidence*100:.1f}%)")
        
        return jsonify(result)
    
    except Exception as e:
        print(f" Error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy'}), 200

if __name__ == '__main__':
    print("\n Server starting at http://localhost:5000")
    print("📡 Endpoints:")
    print("   - POST /classify")
    print("   - GET  /health")
    app.run(host='0.0.0.0', port=5000, debug=True)