from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB


app = Flask(__name__)
CORS(app)

# Load the saved model and vectorizer
with open("spam_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("vectorizer.pkl", "rb") as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

@app.route('/classify', methods=['POST'])
def classify_message():
    data = request.json
    message = data.get('message', '')
    
    if not message:
        return jsonify({"error": "No message provided"}), 400

    # Transform the input message using the loaded vectorizer
    transformed_message = vectorizer.transform([message])
    
    # Predict using the loaded model
    prediction = model.predict(transformed_message)[0]
    result = "ham" if prediction == 0 else "spam"
    return jsonify({"message": message, "classification": result})

@app.route('/train', methods=['POST'])
def train_model():
    try:
        data = request.json.get('data', [])
        if not data:
            return jsonify({"error": "No data provided"}), 400

        # Load the data into a DataFrame
        df = pd.DataFrame(data, columns=['v1', 'v2'])
        df['v1'] = df['v1'].map({'ham': 0, 'spam': 1})

        # Reuse the vectorizer vocabulary
        if vectorizer.vocabulary_ is None:
            return jsonify({"error": "Vectorizer vocabulary is missing"}), 500
        new_vectorizer = TfidfVectorizer(stop_words='english', max_features=3000)
        new_vectorizer.vocabulary_ = vectorizer.vocabulary_
        
        # Transform and fit
        X = new_vectorizer.fit_transform(df['v2'])
        y = df['v1']
        model.fit(X, y)

        # Save the updated model and vectorizer
        with open('spam_model.pkl', 'wb') as model_file:
            pickle.dump(model, model_file)
        with open('vectorizer.pkl', 'wb') as vectorizer_file:
            pickle.dump(new_vectorizer, vectorizer_file)

        return jsonify({"message": "Model retrained successfully."})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
