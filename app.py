from flask import Flask, request, render_template, jsonify
import joblib
import numpy as np
from my_modul import preprocess_text

app = Flask(__name__)

# Liste des catégories
encoder = ['ENTERTAINMENT', 'POLITICS', 'STYLE & BEAUTY', 'TRAVEL', 'WELLNESS']

# Charger le modèle et le vectoriseur
try:
    model = joblib.load('model/model.pkl')
    vectorizer = joblib.load('model/tfidf_vectorizer.pkl')
    print("✅ Modèle et vectoriseur chargés avec succès!")
except Exception as e:
    print(f"❌ Erreur lors du chargement des modèles: {e}")
    model = None
    vectorizer = None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None or vectorizer is None:
        return jsonify({'error': 'Modèle non chargé'}), 500

    text = request.form.get('text') or (request.json.get('text') if request.is_json else '')
    if not text.strip():
        return jsonify({'error': 'Veuillez entrer un texte'}), 400

    try:
        clean_text = preprocess_text(text)
        text_vectorized = vectorizer.transform([clean_text])

        # 🔹 Prédiction des probabilités
        proba = model.predict_proba(text_vectorized)[0]
        
        # 🔹 Classe principale
        main_index = np.argmax(proba)
        main_category = encoder[main_index]

        # 🔹 Préparer les alternatives triées par score
        alternatives = [
            {'category': cat, 'score': float(f"{score:.4f}")}
            for cat, score in zip(encoder, proba)
        ]
        alternatives.sort(key=lambda x: x['score'], reverse=True)

        return jsonify({
            'success': True,
            'predicted_category': main_category,
            'alternatives': alternatives
        })
    except Exception as e:
        return jsonify({'error': f'Erreur lors de la prédiction: {str(e)}'}), 500
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'Aucun fichier sélectionné'}), 400

    file = request.files['file']
    if not file or file.filename == '':
        return jsonify({'error': 'Aucun fichier sélectionné'}), 400

    if not file.filename.endswith('.txt'):
        return jsonify({'error': 'Format non supporté. Utilisez .txt'}), 400

    try:
        text = file.read().decode('utf-8')

        # ⚠️ Ici, on NE fait PAS la prédiction
        return jsonify({'text': text})
        
    except Exception as e:
        return jsonify({'error': f'Erreur lecture fichier: {str(e)}'}), 500
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
