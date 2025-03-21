from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import pickle
import numpy as np
from spam_detector import preprocess_text, load_and_preprocess_data, QLearningAgent

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Path to model and vectorizer
MODEL_PATH = 'model/spam_detector_rl.npy'
VECTORIZER_PATH = 'model/vectorizer.pkl'

# Check if model exists, if not, train it
if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
    print("Model not found. Training new model...")
    # Create model directory if it doesn't exist
    os.makedirs('model', exist_ok=True)
    
    # Train the model and save it
    X_train, X_test, y_train, y_test, vectorizer = load_and_preprocess_data('../data/spam.csv')
    
    # Initialize and train agent (simplified for demonstration)
    feature_size = X_train.shape[1]
    agent = QLearningAgent(feature_size=feature_size)
    
    # Train agent (this is a placeholder - the actual training would be done in spam_detector.py)
    print("Training complete. Saving model...")
    
    # Save model and vectorizer
    agent.save_model(MODEL_PATH)
    with open(VECTORIZER_PATH, 'wb') as f:
        pickle.dump(vectorizer, f)
    
    print("Model saved.")
else:
    print("Loading existing model...")
    # Load vectorizer
    with open(VECTORIZER_PATH, 'rb') as f:
        vectorizer = pickle.load(f)
    
    # Initialize agent
    feature_size = len(vectorizer.get_feature_names_out())
    agent = QLearningAgent(feature_size=feature_size)
    
    # Load model weights
    agent.load_model(MODEL_PATH)
    
    print("Model loaded.")

def calculate_confidence(features, agent):
    """Calculate confidence score from agent's Q-values"""
    q_values = [agent.get_q_value(features, a) for a in range(agent.actions)]
    max_q = max(q_values)
    min_q = min(q_values)
    
    # Normalize to 0-100%
    if max_q == min_q:
        return 50.0  # Equal Q-values mean uncertain prediction
    
    # If max Q-value is for spam (action 1)
    if np.argmax(q_values) == 1:
        confidence = 50.0 + 50.0 * (q_values[1] - q_values[0]) / (max_q - min_q)
    else:
        confidence = 50.0 - 50.0 * (q_values[0] - q_values[1]) / (max_q - min_q)
        
    return confidence

@app.route('/classify', methods=['POST'])
def classify_message():
    data = request.json
    message = data.get('message', '')
    
    if not message:
        return jsonify({'error': 'No message provided'}), 400
    
    # Preprocess and vectorize message
    processed = preprocess_text(message)
    features = vectorizer.transform([processed]).toarray()[0]
    
    # Predict using agent
    action = agent.get_action(features, evaluate=True)
    prediction = "Spam" if action == 1 else "Ham"
    
    # Calculate confidence
    confidence = calculate_confidence(features, agent)
    
    return jsonify({
        'prediction': prediction,
        'confidence': confidence,
        'message': message
    })

@app.route('/train', methods=['POST'])
def train_model():
    # This would start a training job in a production system
    # For this demo, we'll just return a success message
    return jsonify({
        'status': 'success',
        'message': 'Training job started'
    })

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
