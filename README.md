# SMS Spam Detection using Reinforcement Learning

A machine learning project that uses Q-learning (a reinforcement learning algorithm) to detect spam SMS messages.

## Project Overview

This project implements a system that can classify SMS messages as spam or legitimate ("ham") using a reinforcement learning approach instead of traditional supervised learning methods.

### Key Features

- Q-learning agent for SMS classification
- TF-IDF vectorization of text messages
- Interactive web interface for real-time classification
- Training progress visualization
- Classification history and statistics tracking

## Repository Structure

```
spam-detection-rl/
│
├── backend/           # Python backend with ML model
├── frontend/          # React frontend interface
├── data/              # Dataset directory (you need to add the dataset)
├── screenshots/       # Application screenshots
└── requirements.txt   # Python dependencies
```

## Setup Instructions

### Prerequisites

- Python 3.8+
- Node.js 14+
- npm or yarn

### Backend Setup

1. Create a virtual environment (recommended):
   ```
   cd backend
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```
   pip install -r ../requirements.txt
   ```

3. Download the dataset:
   - Get the "SMS Spam Collection Dataset" from Kaggle
   - Save it as `data/spam.csv`

4. Run the backend:
   ```
   python app.py
   ```

### Frontend Setup

1. Install dependencies:
   ```
   cd frontend
   npm install
   ```

2. Run the frontend:
   ```
   npm run dev
   ```

3. Open your browser to `http://localhost:3000`

## Dataset

This project uses the [SMS Spam Collection Dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset) from Kaggle, which contains 5,574 SMS messages labeled as spam or ham.

## How It Works

1. The system preprocesses text messages (lowercase conversion, stopword removal, stemming, etc.)
2. Messages are converted to TF-IDF feature vectors
3. A Q-learning agent learns to classify messages based on rewards (correct/incorrect classifications)
4. The agent improves over time as it processes more messages
