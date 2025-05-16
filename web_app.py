#!/usr/bin/env python3
"""
NoahAI Web Application

This module provides a web interface for interacting with NoahAI.
"""

import os
import json
import time
import logging
import datetime
from flask import Flask, render_template, request, jsonify, session
from flask_socketio import SocketIO
from flask_cors import CORS
from dotenv import load_dotenv

# Import NoahAI modules
from models.deep_learning_model import DeepLearningModel
from models.simple_model import SimpleResponseModel
from src.utils.nlp_utils import NLPUtils
from src.utils.learning_utils import LearningUtils

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/web_app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("web_app")

# Initialize Flask app
app = Flask(__name__,
            static_folder='web/static',
            template_folder='web/templates')
app.secret_key = os.getenv('SECRET_KEY', 'noahaisecretkey')
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

# Check if TensorFlow is available
try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
    logger.info("TensorFlow is available. Using deep learning model.")
except ImportError:
    TENSORFLOW_AVAILABLE = False
    logger.warning("TensorFlow not available. Using simple model.")

# Global variables
ai_instances = {}  # Dictionary to store AI instances for each session

class WebNoahAI:
    """
    Web interface for NoahAI.
    """

    def __init__(self, session_id, use_deep_learning=True, name="Noah"):
        """
        Initialize the web interface for NoahAI.

        Args:
            session_id (str): Session ID for the current user
            use_deep_learning (bool): Whether to use the deep learning model
            name (str): Name of the AI assistant
        """
        self.session_id = session_id
        self.name = name
        self.use_deep_learning = use_deep_learning and TENSORFLOW_AVAILABLE
        self.user_name = None
        self.conversation_history = []

        # Initialize the model
        if self.use_deep_learning:
            logger.info(f"Session {session_id}: Using deep learning model")
            self.model = DeepLearningModel(responses_file="data/responses.json", use_advanced_nlp=True)
        else:
            logger.info(f"Session {session_id}: Using simple response model")
            self.model = SimpleResponseModel("data/responses.json")

        # Initialize NLP utilities with multilingual support
        self.nlp_utils = NLPUtils(use_advanced_nlp=True, use_multilingual=True)

        # Initialize learning utilities if using deep learning
        if self.use_deep_learning:
            self.learning_utils = LearningUtils(self.model)

    def process_input(self, user_input):
        """
        Process user input and generate a response.

        Args:
            user_input (str): The user's input text

        Returns:
            dict: Response data including the AI's response and metadata
        """
        if not user_input:
            return {
                "response": "I didn't receive any input. Could you please try again?",
                "timestamp": datetime.datetime.now().isoformat()
            }

        # Check for special commands
        if user_input.lower().startswith("my name is "):
            # Set user name
            self.user_name = user_input[11:].strip()
            response = f"Nice to meet you, {self.user_name}! How can I help you today?"
        elif user_input.lower().startswith("rate "):
            # Add feedback
            try:
                rating = int(user_input[5:].strip())
                if 1 <= rating <= 5:
                    if len(self.conversation_history) >= 2:
                        last_user_input = self.conversation_history[-2]["text"]
                        last_ai_response = self.conversation_history[-1]["text"]

                        if self.use_deep_learning:
                            self.learning_utils.add_feedback(last_user_input, last_ai_response, rating)

                        response = "Thank you for your feedback! I'll use it to improve."
                    else:
                        response = "I don't have enough conversation history to collect feedback."
                else:
                    response = "Please rate between 1 and 5."
            except ValueError:
                response = "Please provide a valid rating between 1 and 5."
        elif user_input.lower() == "train" and self.use_deep_learning:
            # Train the model
            stats = self.learning_utils.get_feedback_stats()
            count = stats["count"]

            if count < 10:
                response = f"I need more feedback to train. Currently have {count}/10 feedback entries."
            else:
                # Train in a separate thread to avoid blocking
                response = "Training in progress... This might take a moment."
                socketio.start_background_task(self._train_model)
        else:
            # Generate response using the model
            response = self.model.generate_response(user_input, self.user_name)

        # Add to conversation history
        user_message = {
            "role": "user",
            "text": user_input,
            "timestamp": datetime.datetime.now().isoformat()
        }

        ai_message = {
            "role": "ai",
            "text": response,
            "timestamp": datetime.datetime.now().isoformat()
        }

        self.conversation_history.append(user_message)
        self.conversation_history.append(ai_message)

        # Prepare response data
        response_data = {
            "response": response,
            "timestamp": ai_message["timestamp"]
        }

        # Add NLP analysis if available
        try:
            # Detect language
            language = self.nlp_utils.detect_language(user_input)

            # Extract keywords, sentiment, intent, and entities
            keywords = self.nlp_utils.extract_keywords(user_input)
            sentiment = self.nlp_utils.sentiment_analysis(user_input)
            intent = self.nlp_utils.recognize_intent(user_input)
            entities = self.nlp_utils.extract_entities(user_input)

            # Get topics
            topics = self.nlp_utils.analyze_topic(user_input, num_topics=1, num_words=5)

            # Add translation if not in English
            translation = None
            if language["lang_code"] != "en" and language["confidence"] > 0.7:
                translation = self.nlp_utils.translate(user_input, source_lang=language["lang_code"], target_lang="en")

            response_data["analysis"] = {
                "language": language,
                "keywords": keywords,
                "sentiment": sentiment,
                "intent": intent,
                "entities": entities,
                "topics": topics,
                "translation": translation
            }
        except Exception as e:
            logger.error(f"Error in NLP analysis: {e}")

        return response_data

    def _train_model(self, epochs=5, batch_size=32, learning_rate=0.001,
                  use_transfer_learning=False, use_reinforcement_learning=False):
        """
        Train the model in the background.

        Args:
            epochs (int): Number of training epochs
            batch_size (int): Batch size for training
            learning_rate (float): Learning rate for training
            use_transfer_learning (bool): Whether to use transfer learning
            use_reinforcement_learning (bool): Whether to use reinforcement learning
        """
        try:
            logger.info(f"Starting model training with epochs={epochs}, batch_size={batch_size}, "
                       f"learning_rate={learning_rate}, use_transfer_learning={use_transfer_learning}, "
                       f"use_reinforcement_learning={use_reinforcement_learning}")

            # Set learning rate if provided
            if hasattr(self.learning_utils, 'set_learning_rate'):
                self.learning_utils.set_learning_rate(learning_rate)

            # Set reinforcement learning if available
            if hasattr(self.learning_utils, 'set_reinforcement_learning'):
                self.learning_utils.set_reinforcement_learning(use_reinforcement_learning)

            # Emit initial progress
            socketio.emit('training_progress', {
                'progress': 0,
                'epoch': 0,
                'total_epochs': epochs,
                'accuracy': 0,
                'loss': 0
            }, room=self.session_id)

            # Define progress callback function
            def progress_callback(epoch, total_epochs, metrics):
                progress = int((epoch / total_epochs) * 100)
                accuracy = metrics.get('accuracy', 0)
                loss = metrics.get('loss', 0)

                # Emit progress update
                socketio.emit('training_progress', {
                    'progress': progress,
                    'epoch': epoch,
                    'total_epochs': total_epochs,
                    'accuracy': accuracy,
                    'loss': loss
                }, room=self.session_id)

                # Add small delay to avoid overwhelming the socket
                time.sleep(0.1)

            # Train the model with the specified parameters
            if use_transfer_learning and hasattr(self.learning_utils, 'train_with_transfer_learning'):
                result = self.learning_utils.train_with_transfer_learning(
                    epochs=epochs,
                    batch_size=batch_size,
                    callback=progress_callback if hasattr(self.learning_utils, 'set_callback') else None
                )
            else:
                result = self.learning_utils.train_model(
                    epochs=epochs,
                    batch_size=batch_size,
                    callback=progress_callback if hasattr(self.learning_utils, 'set_callback') else None
                )

            if result:
                stats = self.learning_utils.get_feedback_stats()
                count = stats["count"]
                message = f"Training complete! Processed {count} feedback entries with {epochs} epochs."

                # Emit final progress
                socketio.emit('training_progress', {
                    'progress': 100,
                    'epoch': epochs,
                    'total_epochs': epochs,
                    'accuracy': result.get('accuracy', 0) if isinstance(result, dict) else 0,
                    'loss': result.get('loss', 0) if isinstance(result, dict) else 0
                }, room=self.session_id)
            else:
                message = "Training failed. Please check the logs for details."

            # Emit the training result
            socketio.emit('training_result', {'message': message}, room=self.session_id)
        except Exception as e:
            logger.error(f"Error during training: {e}")
            socketio.emit('training_result', {'message': f"Error during training: {e}"}, room=self.session_id)

# Flask routes
@app.route('/')
def index():
    """
    Render the main page.
    """
    return render_template('index.html')

@app.route('/ai_training')
def ai_training():
    """
    Render the AI training interface page.
    """
    return render_template('ai_training.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    """
    Process chat messages.
    """
    data = request.json
    user_input = data.get('message', '')
    session_id = session.get('session_id')

    if not session_id:
        session_id = str(time.time())
        session['session_id'] = session_id

    # Get or create AI instance for this session
    if session_id not in ai_instances:
        use_deep_learning = data.get('use_deep_learning', True)
        ai_instances[session_id] = WebNoahAI(session_id, use_deep_learning)

    # Process the input
    noah = ai_instances[session_id]
    response_data = noah.process_input(user_input)

    return jsonify(response_data)

@app.route('/api/history', methods=['GET'])
def get_history():
    """
    Get conversation history.
    """
    session_id = session.get('session_id')

    if not session_id or session_id not in ai_instances:
        return jsonify([])

    noah = ai_instances[session_id]
    return jsonify(noah.conversation_history)

@app.route('/api/clear', methods=['POST'])
def clear_history():
    """
    Clear conversation history.
    """
    session_id = session.get('session_id')

    if session_id and session_id in ai_instances:
        noah = ai_instances[session_id]
        noah.conversation_history = []

    return jsonify({"status": "success"})

@app.route('/api/stop_training', methods=['POST'])
def stop_training():
    """
    Stop an ongoing training process.
    """
    try:
        session_id = session.get('session_id')
        if not session_id or session_id not in ai_instances:
            return jsonify({
                "success": False,
                "message": "No active training session found."
            }), 400

        # Set a flag to stop training if possible
        noah = ai_instances[session_id]
        if hasattr(noah.learning_utils, 'stop_training'):
            noah.learning_utils.stop_training()

        return jsonify({
            "success": True,
            "message": "Training stop signal sent."
        })

    except Exception as e:
        logger.error(f"Error stopping training: {e}")
        return jsonify({
            "success": False,
            "message": f"Error stopping training: {str(e)}"
        }), 500

@app.route('/api/train_model', methods=['POST'])
def train_model():
    """
    Train the AI model with specified parameters.
    """
    try:
        data = request.json or {}
        session_id = session.get('session_id')

        if not session_id:
            session_id = str(time.time())
            session['session_id'] = session_id

        # Get or create AI instance for this session
        if session_id not in ai_instances:
            use_deep_learning = True  # Always use deep learning for training
            ai_instances[session_id] = WebNoahAI(session_id, use_deep_learning)

        noah = ai_instances[session_id]

        # Check if deep learning is available
        if not noah.use_deep_learning:
            return jsonify({
                "success": False,
                "message": "Deep learning is not available. Cannot train the model."
            }), 400

        # Get training parameters
        epochs = int(data.get('epochs', 5))
        batch_size = int(data.get('batch_size', 32))
        learning_rate = float(data.get('learning_rate', 0.001))
        use_transfer_learning = data.get('use_transfer_learning', False)
        use_reinforcement_learning = data.get('use_reinforcement_learning', False)

        # Start training in background
        socketio.start_background_task(
            noah._train_model,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            use_transfer_learning=use_transfer_learning,
            use_reinforcement_learning=use_reinforcement_learning
        )

        return jsonify({
            "success": True,
            "message": "Training started in the background. You will be notified when it completes."
        })

    except Exception as e:
        logger.error(f"Error starting training: {e}")
        return jsonify({
            "success": False,
            "message": f"Error starting training: {str(e)}"
        }), 500

# SocketIO events
@socketio.on('connect')
def handle_connect():
    """
    Handle client connection.
    """
    session_id = session.get('session_id')

    if not session_id:
        session_id = str(time.time())
        session['session_id'] = session_id

    logger.info(f"Client connected: {session_id}")
    socketio.emit('welcome', {'message': 'Connected to NoahAI'})

@socketio.on('disconnect')
def handle_disconnect():
    """
    Handle client disconnection.
    """
    session_id = session.get('session_id')
    logger.info(f"Client disconnected: {session_id}")

@socketio.on('request_training_status')
def handle_training_status_request():
    """
    Handle request for training status.
    """
    session_id = session.get('session_id')
    if not session_id:
        session_id = str(time.time())
        session['session_id'] = session_id

    # Emit current training status if available
    socketio.emit('training_status', {
        'is_training': False,
        'message': 'Ready to start training'
    }, room=session_id)

if __name__ == '__main__':
    # Create necessary directories
    os.makedirs('logs', exist_ok=True)
    os.makedirs('web/templates', exist_ok=True)
    os.makedirs('web/static/css', exist_ok=True)
    os.makedirs('web/static/js', exist_ok=True)
    os.makedirs('data/model', exist_ok=True)
    os.makedirs('data/feedback', exist_ok=True)

    # Run the app
    port = int(os.getenv('PORT', 5000))
    socketio.run(app, host='0.0.0.0', port=port, debug=True)
