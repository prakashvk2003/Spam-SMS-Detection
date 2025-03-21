import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Download NLTK resources
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)

class SMSSpamEnvironment:
    """
    Environment for SMS spam detection using reinforcement learning
    """
    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.current_state = None
        self.current_index = 0
        self.episode_ended = False
        self.total_reward = 0
        self.reset()
        
    def reset(self):
        """Reset environment to initial state"""
        self.current_index = 0
        self.episode_ended = False
        self.total_reward = 0
        self.current_state = self.X_train[self.current_index]
        return self.current_state
    
    def step(self, action):
        """
        Take action (0 for ham, 1 for spam) and return next state, reward, and done flag
        """
        if self.episode_ended:
            # If episode has ended, reset the environment
            return self.reset(), 0, True
        
        # Get true label
        true_label = self.y_train[self.current_index]
        
        # Calculate reward (1 for correct prediction, -1 for incorrect)
        reward = 1 if action == true_label else -1
        self.total_reward += reward
        
        # Move to next message
        self.current_index += 1
        
        # Check if episode has ended
        if self.current_index >= len(self.X_train):
            self.episode_ended = True
            next_state = None
            done = True
        else:
            next_state = self.X_train[self.current_index]
            done = False
            
        return next_state, reward, done
    
    def evaluate(self, agent):
        """Evaluate agent performance on test set"""
        predictions = []
        true_labels = []
        
        for i in range(len(self.X_test)):
            state = self.X_test[i]
            action = agent.get_action(state, evaluate=True)
            predictions.append(action)
            true_labels.append(self.y_test[i])
        
        accuracy = accuracy_score(true_labels, predictions)
        conf_matrix = confusion_matrix(true_labels, predictions)
        report = classification_report(true_labels, predictions)
        
        return accuracy, conf_matrix, report, predictions


class QLearningAgent:
    """
    Q-Learning agent for SMS spam detection
    """
    def __init__(self, feature_size, actions=2, learning_rate=0.1, discount_factor=0.99, exploration_rate=1.0, exploration_decay=0.999, min_exploration_rate=0.01):
        self.actions = actions  # 0 for ham, 1 for spam
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.min_exploration_rate = min_exploration_rate
        
        # Initialize Q-table - we'll use a linear approximation for high-dimensional state space
        self.weights = np.zeros((actions, feature_size))
        
    def get_q_value(self, state, action):
        """Calculate Q-value using linear approximation"""
        return np.dot(self.weights[action], state)
    
    def get_action(self, state, evaluate=False):
        """
        Select action using epsilon-greedy policy
        If evaluate is True, use greedy policy (no exploration)
        """
        if not evaluate and np.random.rand() < self.exploration_rate:
            # Explore: choose random action
            return np.random.randint(self.actions)
        else:
            # Exploit: choose best action
            q_values = [self.get_q_value(state, a) for a in range(self.actions)]
            return np.argmax(q_values)
    
    def update(self, state, action, reward, next_state, done):
        """Update Q-values using Q-learning update rule"""
        if done or next_state is None:
            max_next_q = 0
        else:
            max_next_q = max([self.get_q_value(next_state, a) for a in range(self.actions)])
            
        current_q = self.get_q_value(state, action)
        td_target = reward + self.discount_factor * max_next_q
        td_error = td_target - current_q
        
        # Update weights for linear approximation
        self.weights[action] += self.learning_rate * td_error * state
        
        # Decay exploration rate
        self.exploration_rate = max(self.min_exploration_rate, 
                                    self.exploration_rate * self.exploration_decay)
    
    def save_model(self, filepath):
        """Save model weights"""
        np.save(filepath, self.weights)
        
    def load_model(self, filepath):
        """Load model weights"""
        self.weights = np.load(filepath)


def preprocess_text(text):
    """Clean and preprocess text"""
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    
    # Remove numbers and special characters
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Tokenize text
    tokens = nltk.word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [t for t in tokens if t not in stop_words]
    
    # Stemming
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(t) for t in tokens]
    
    return ' '.join(tokens)


def load_and_preprocess_data(filepath):
    """Load SMS dataset and preprocess it"""
    # Load dataset
    df = pd.read_csv(filepath, encoding='latin-1')
    
    # Rename columns for clarity
    df.columns = ['label', 'message'] + list(df.columns[2:])
    
    # Convert labels to binary (0 for ham, 1 for spam)
    df['label_binary'] = df['label'].map({'ham': 0, 'spam': 1})
    
    # Preprocess messages
    df['processed_message'] = df['message'].apply(preprocess_text)
    
    # Create feature vectors using TF-IDF
    vectorizer = TfidfVectorizer(max_features=1000)
    X = vectorizer.fit_transform(df['processed_message']).toarray()
    y = df['label_binary'].values
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test, vectorizer


def train_agent(env, agent, num_episodes=100):
    """Train Q-learning agent on SMS data"""
    rewards_history = []
    
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            action = agent.get_action(state)
            next_state, reward, done = env.step(action)
            agent.update(state, action, reward, next_state, done)
            state = next_state
            episode_reward += reward
        
        rewards_history.append(episode_reward)
        
        if (episode + 1) % 10 == 0:
            print(f"Episode {episode + 1}/{num_episodes}, Total Reward: {episode_reward}")
    
    return rewards_history


def plot_training_progress(rewards_history, save_path=None):
    """Plot training progress"""
    plt.figure(figsize=(10, 6))
    plt.plot(rewards_history)
    plt.title('Training Progress')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()


def plot_confusion_matrix(conf_matrix, save_path=None):
    """Plot confusion matrix"""
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()


if __name__ == "__main__":
    # Create output directories
    os.makedirs('model', exist_ok=True)
    os.makedirs('plots', exist_ok=True)
    
    # Load and preprocess data
    print("Loading and preprocessing data...")
    X_train, X_test, y_train, y_test, vectorizer = load_and_preprocess_data('../data/spam.csv')
    
    # Save vectorizer
    import pickle
    with open('model/vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
    
    # Create environment
    env = SMSSpamEnvironment(X_train, y_train, X_test, y_test)
    
    # Create agent
    feature_size = X_train.shape[1]
    agent = QLearningAgent(feature_size=feature_size)
    
    # Train agent
    print("\nTraining agent...")
    rewards_history = train_agent(env, agent, num_episodes=100)
    
    # Plot training progress
    plot_training_progress(rewards_history, save_path='plots/training_progress.png')
    
    # Evaluate agent
    print("\nEvaluating agent...")
    accuracy, conf_matrix, report, _ = env.evaluate(agent)
    
    print(f"Accuracy: {accuracy:.4f}")
    print("\nConfusion Matrix:")
    print(conf_matrix)
    print("\nClassification Report:")
    print(report)
    
    # Plot confusion matrix
    plot_confusion_matrix(conf_matrix, save_path='plots/confusion_matrix.png')
    
    # Save model
    agent.save_model('model/spam_detector_rl.npy')
    
    print("\nTraining and evaluation complete!")
