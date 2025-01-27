import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import pickle
import kagglehub

# Step 1: Load the Dataset
def load_dataset(file_path):
    print(f"Loading dataset from {file_path}")
    # Use latin1 encoding to avoid Unicode errors
    data = pd.read_csv(file_path, encoding='latin1')
    return data


# Step 2: Preprocess the Data
def preprocess_data(data):
    # Encode labels (ham: 0, spam: 1)
    data['v1'] = data['v1'].map({'ham': 0, 'spam': 1})

    # Clean the text
    data['v2'] = data['v2'].str.replace(r'[^a-zA-Z\s]', '', regex=True)
    data['v2'] = data['v2'].str.lower()
    return data

# Step 3: Convert Text into Numerical Format
def vectorize_text(messages):
    vectorizer = TfidfVectorizer(stop_words='english', max_features=3000)
    X = vectorizer.fit_transform(messages)
    return X, vectorizer

# Step 4: Train a Machine Learning Model
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = MultinomialNB()
    model.fit(X_train, y_train)
    
    # Evaluate the model
    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))
    
    return model, X_test, y_test

# Step 5: Save the Model and Vectorizer
def save_model_and_vectorizer(model, vectorizer, model_path, vectorizer_path):
    with open(model_path, 'wb') as model_file:
        pickle.dump(model, model_file)
    with open(vectorizer_path, 'wb') as vectorizer_file:
        pickle.dump(vectorizer, vectorizer_file)

# Main Execution
def main():
    # File path to the dataset
    file_path = './spam.csv'  # Replace with the path to your dataset
    
    # Load and preprocess data
    data = load_dataset(file_path)
    data = preprocess_data(data)
    
    # Vectorize text and train the model
    X, vectorizer = vectorize_text(data['v2'])
    model, X_test, y_test = train_model(X, data['v1'])

    # Save the model and vectorizer
    save_model_and_vectorizer(model, vectorizer, 'spam_model.pkl', 'vectorizer.pkl')
    print("Model and vectorizer saved successfully.")

if __name__ == "__main__":
    main()
