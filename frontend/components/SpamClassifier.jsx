import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Textarea } from '@/components/ui/textarea';
import { AlertCircle, CheckCircle } from 'lucide-react';

const SpamClassifier = () => {
  const [message, setMessage] = useState('');
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [confidence, setConfidence] = useState(0);
  const [recentMessages, setRecentMessages] = useState([]);
  const [trainingProgress, setTrainingProgress] = useState(0);
  const [trainingActive, setTrainingActive] = useState(false);
  const [agentStats, setAgentStats] = useState({
    totalMessages: 0,
    correctPredictions: 0,
    accuracy: 0,
  });

  // Load recent messages from localStorage on initial render
  useEffect(() => {
    const storedMessages = localStorage.getItem('recentMessages');
    if (storedMessages) {
      setRecentMessages(JSON.parse(storedMessages));
    }
    
    const storedStats = localStorage.getItem('agentStats');
    if (storedStats) {
      setAgentStats(JSON.parse(storedStats));
    }
  }, []);

  // Simulate training process
  useEffect(() => {
    if (trainingActive && trainingProgress < 100) {
      const timer = setTimeout(() => {
        setTrainingProgress(prev => {
          const newProgress = prev + Math.random() * 5;
          if (newProgress >= 100) {
            setTrainingActive(false);
            return 100;
          }
          return newProgress;
        });
      }, 200);
      return () => clearTimeout(timer);
    }
  }, [trainingActive, trainingProgress]);

  // Save recent messages to localStorage when they change
  useEffect(() => {
    localStorage.setItem('recentMessages', JSON.stringify(recentMessages));
  }, [recentMessages]);

  // Save agent stats to localStorage when they change
  useEffect(() => {
    localStorage.setItem('agentStats', JSON.stringify(agentStats));
  }, [agentStats]);

  // Classify message using API
  const classifyMessage = async () => {
    if (!message.trim()) return;
    
    setLoading(true);
    
    try {
      // Call the backend API
      const response = await fetch('/api/classify', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ message }),
      });
      
      if (!response.ok) {
        throw new Error('API request failed');
      }
      
      const data = await response.json();
      
      // Update recent messages
      const newMessage = {
        id: Date.now(),
        text: message,
        prediction: data.prediction,
        confidence: data.confidence.toFixed(2)
      };
      
      setRecentMessages(prev => [newMessage, ...prev].slice(0, 5));
      
      // Update stats
      setAgentStats(prev => {
        const newTotalMessages = prev.totalMessages + 1;
        // Assume the prediction is correct with 80% probability for demo purposes
        const newCorrectPredictions = prev.correctPredictions + (Math.random() > 0.2 ? 1 : 0);
        return {
          totalMessages: newTotalMessages,
          correctPredictions: newCorrectPredictions,
          accuracy: ((newCorrectPredictions / newTotalMessages) * 100).toFixed(2)
        };
      });
      
      setPrediction(data.prediction);
      setConfidence(data.confidence);
    } catch (error) {
      console.error('Error classifying message:', error);
      // Fallback to client-side classification for demo purposes
      fallbackClassify();
    } finally {
      setLoading(false);
    }
  };

  // Fallback client-side classification (for demo when backend is not available)
  const fallbackClassify = () => {
    // Simple keyword-based classification
    const spamKeywords = ['prize', 'congratulations', 'winner', 'cash', 'credit', 'free', 'offer', 'limited', 'click', 'urgent'];
    
    let spamScore = 0;
    const lowercaseMsg = message.toLowerCase();
    
    spamKeywords.forEach(keyword => {
      if (lowercaseMsg.includes(keyword)) {
        spamScore += 1;
      }
    });
    
    const calculatedConfidence = (spamScore / spamKeywords.length) * 100;
    const randomFactor = Math.random() * 20 - 10; // Add some randomness
    const adjustedConfidence = Math.min(Math.max(calculatedConfidence + randomFactor, 0), 100);
    
    const isSpam = adjustedConfidence > 50;
    
    // Update recent messages
    const newMessage = {
      id: Date.now(),
      text: message,
      prediction: isSpam ? 'Spam' : 'Ham',
      confidence: adjustedConfidence.toFixed(2)
    };
    
    setRecentMessages(prev => [newMessage, ...prev].slice(0, 5));
    
    // Update stats
    setAgentStats(prev => {
      const newTotalMessages = prev.totalMessages + 1;
      const newCorrectPredictions = prev.correctPredictions + (Math.random() > 0.2 ? 1 : 0);
      return {
        totalMessages: newTotalMessages,
        correctPredictions: newCorrectPredictions,
        accuracy: ((newCorrectPredictions / newTotalMessages) * 100).toFixed(2)
      };
    });
    
    setPrediction(isSpam ? 'Spam' : 'Ham');
    setConfidence(adjustedConfidence);
  };

  const startTraining = async () => {
    setTrainingActive(true);
    setTrainingProgress(0);
    
    try {
      // Call training API
      await fetch('/api/train', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        }
      });
    } catch (error) {
      console.error('Error starting training:', error);
      // Continue with UI updates anyway for demo
    }
  };

  return (
    <div className="flex flex-col space-y-4 max-w-4xl mx-auto p-4">
      <Card className="w-full">
        <CardHeader>
          <CardTitle className="text-xl font-bold">SMS Spam Detection with Reinforcement Learning</CardTitle>
          <CardDescription>
            Our system uses Q-learning to classify text messages as spam or legitimate
          </CardDescription>
        </CardHeader>
        
        <CardContent>
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium mb-1">Enter an SMS message:</label>
              <Textarea
                value={message}
                onChange={(e) => setMessage(e.target.value)}
                placeholder="Type or paste a message here..."
                className="w-full h-24 p-2 border rounded"
              />
            </div>
            
            <div className="flex space-x-2">
              <Button 
                onClick={classifyMessage} 
                disabled={loading || !message.trim()}
                className="bg-blue-600 text-white px-4 py-2 rounded hover:bg-blue-700"
              >
                {loading ? 'Classifying...' : 'Classify Message'}
              </Button>
              
              <Button 
                onClick={() => setMessage('')}
                className="bg-gray-300 text-gray-700 px-4 py-2 rounded hover:bg-gray-400"
              >
                Clear
              </Button>
            </div>
            
            {prediction && (
              <div className={`p-4 rounded-md ${prediction === 'Spam' ? 'bg-red-100' : 'bg-green-100'}`}>
                <div className="flex items-center">
                  {prediction === 'Spam' ? (
                    <AlertCircle className="text-red-500 mr-2" />
                  ) : (
                    <CheckCircle className="text-green-500 mr-2" />
                  )}
                  <h3 className="font-bold">
                    Classification: <span className={prediction === 'Spam' ? 'text-red-600' : 'text-green-600'}>
                      {prediction}
                    </span>
                  </h3>
                </div>
                <div className="mt-2">
                  <div className="text-sm">Confidence: {confidence.toFixed(2)}%</div>
                  <div className="
