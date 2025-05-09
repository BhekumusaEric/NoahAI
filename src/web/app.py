"""
NoahAI Web Interface

This module provides a web-based interface for interacting with NoahAI using Flask.
"""

import os
import sys
import json
import logging
import threading
import time
from datetime import datetime
from flask import Flask, render_template, request, jsonify, session
from flask_socketio import SocketIO, emit

# Add parent directory to path to import NoahAI
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import NoahAI
from noah_ai import NoahAI

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/web_interface.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("web_interface")

# Create Flask app
app = Flask(__name__)
app.secret_key = os.urandom(24)
socketio = SocketIO(app, cors_allowed_origins="*")

# Create data directories
os.makedirs("logs", exist_ok=True)
os.makedirs("data/sessions", exist_ok=True)

# Store active NoahAI instances
ai_instances = {}

# Background thread for AI processing
processing_thread = None
thread_stop_event = threading.Event()

def get_or_create_ai_instance(session_id, user_name=None):
    """
    Get or create a NoahAI instance for the session.
    
    Args:
        session_id (str): Session ID
        user_name (str, optional): User name
        
    Returns:
        NoahAI: NoahAI instance
    """
    if session_id not in ai_instances:
        # Create a new NoahAI instance
        ai_instance = NoahAI(
            name="Noah",
            responses_file="data/responses.json",
            use_deep_learning=True,
            verbose=False,
            model_type="advanced_lstm",
            use_reinforcement_learning=True,
            use_advanced_rl=True,
            use_explainable_ai=True,
            use_safety_filter=True,
            safety_level="medium",
            user_id=session_id
        )
        
        # Set user name if provided
        if user_name:
            ai_instance.user_name = user_name
            if ai_instance.use_advanced_rl and ai_instance.advanced_rl_manager:
                ai_instance.advanced_rl_manager.set_user_id(session_id, user_name)
        
        ai_instances[session_id] = ai_instance
        logger.info(f"Created new NoahAI instance for session {session_id}")
    
    return ai_instances[session_id]

def background_processing(session_id, user_input):
    """
    Process user input in a background thread.
    
    Args:
        session_id (str): Session ID
        user_input (str): User input
    """
    try:
        # Get AI instance
        ai_instance = get_or_create_ai_instance(session_id)
        
        # Process input
        response = ai_instance.process_input(user_input)
        
        # Emit response
        socketio.emit('ai_response', {
            'response': response,
            'timestamp': datetime.now().isoformat()
        }, room=session_id)
        
        # Save conversation history
        save_conversation_history(session_id, ai_instance.conversation_history)
    except Exception as e:
        logger.error(f"Error processing input: {e}")
        socketio.emit('ai_response', {
            'response': f"Sorry, there was an error processing your input: {str(e)}",
            'timestamp': datetime.now().isoformat()
        }, room=session_id)

def save_conversation_history(session_id, conversation_history):
    """
    Save conversation history to disk.
    
    Args:
        session_id (str): Session ID
        conversation_history (list): Conversation history
    """
    try:
        # Create session directory
        session_dir = os.path.join("data/sessions", session_id)
        os.makedirs(session_dir, exist_ok=True)
        
        # Save conversation history
        history_path = os.path.join(session_dir, "conversation_history.json")
        with open(history_path, 'w') as f:
            json.dump(conversation_history, f, indent=2)
    except Exception as e:
        logger.error(f"Error saving conversation history: {e}")

def load_conversation_history(session_id):
    """
    Load conversation history from disk.
    
    Args:
        session_id (str): Session ID
        
    Returns:
        list: Conversation history
    """
    try:
        # Get history path
        history_path = os.path.join("data/sessions", session_id, "conversation_history.json")
        
        # Check if file exists
        if not os.path.exists(history_path):
            return []
        
        # Load conversation history
        with open(history_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading conversation history: {e}")
        return []

@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html')

@app.route('/api/session', methods=['POST'])
def create_session():
    """Create a new session."""
    try:
        # Get user name
        data = request.json
        user_name = data.get('user_name', 'User')
        
        # Create session ID
        session_id = f"session_{int(time.time())}_{os.urandom(4).hex()}"
        
        # Store session ID
        session['session_id'] = session_id
        
        # Create AI instance
        get_or_create_ai_instance(session_id, user_name)
        
        return jsonify({
            'success': True,
            'session_id': session_id,
            'message': f"Hello, {user_name}! I'm Noah, your AI assistant."
        })
    except Exception as e:
        logger.error(f"Error creating session: {e}")
        return jsonify({
            'success': False,
            'message': f"Error creating session: {str(e)}"
        }), 500

@app.route('/api/chat', methods=['POST'])
def chat():
    """Process a chat message."""
    try:
        # Get session ID
        session_id = session.get('session_id')
        if not session_id:
            return jsonify({
                'success': False,
                'message': "No active session. Please refresh the page."
            }), 400
        
        # Get user input
        data = request.json
        user_input = data.get('message', '')
        
        if not user_input:
            return jsonify({
                'success': False,
                'message': "Please enter a message."
            }), 400
        
        # Get AI instance
        ai_instance = get_or_create_ai_instance(session_id)
        
        # Process input
        response = ai_instance.process_input(user_input)
        
        # Save conversation history
        save_conversation_history(session_id, ai_instance.conversation_history)
        
        return jsonify({
            'success': True,
            'response': response
        })
    except Exception as e:
        logger.error(f"Error processing chat: {e}")
        return jsonify({
            'success': False,
            'message': f"Error processing chat: {str(e)}"
        }), 500

@app.route('/api/feedback', methods=['POST'])
def feedback():
    """Process feedback for the last response."""
    try:
        # Get session ID
        session_id = session.get('session_id')
        if not session_id:
            return jsonify({
                'success': False,
                'message': "No active session. Please refresh the page."
            }), 400
        
        # Get rating
        data = request.json
        rating = data.get('rating')
        
        if not rating or not isinstance(rating, int) or rating < 1 or rating > 5:
            return jsonify({
                'success': False,
                'message': "Please provide a valid rating (1-5)."
            }), 400
        
        # Get AI instance
        ai_instance = get_or_create_ai_instance(session_id)
        
        # Add feedback
        feedback_response = ai_instance._add_feedback(rating)
        
        # Save conversation history
        save_conversation_history(session_id, ai_instance.conversation_history)
        
        return jsonify({
            'success': True,
            'response': feedback_response
        })
    except Exception as e:
        logger.error(f"Error processing feedback: {e}")
        return jsonify({
            'success': False,
            'message': f"Error processing feedback: {str(e)}"
        }), 500

@app.route('/api/train', methods=['POST'])
def train():
    """Train the model."""
    try:
        # Get session ID
        session_id = session.get('session_id')
        if not session_id:
            return jsonify({
                'success': False,
                'message': "No active session. Please refresh the page."
            }), 400
        
        # Get training options
        data = request.json
        use_transfer_learning = data.get('use_transfer_learning', False)
        use_continual_learning = data.get('use_continual_learning', False)
        use_distributed_training = data.get('use_distributed_training', False)
        
        # Get AI instance
        ai_instance = get_or_create_ai_instance(session_id)
        
        # Train model
        training_response = ai_instance._train_model(
            use_transfer_learning=use_transfer_learning,
            use_continual_learning=use_continual_learning,
            use_distributed_training=use_distributed_training
        )
        
        return jsonify({
            'success': True,
            'response': training_response
        })
    except Exception as e:
        logger.error(f"Error training model: {e}")
        return jsonify({
            'success': False,
            'message': f"Error training model: {str(e)}"
        }), 500

@app.route('/api/history', methods=['GET'])
def history():
    """Get conversation history."""
    try:
        # Get session ID
        session_id = session.get('session_id')
        if not session_id:
            return jsonify({
                'success': False,
                'message': "No active session. Please refresh the page."
            }), 400
        
        # Get AI instance
        ai_instance = get_or_create_ai_instance(session_id)
        
        # Get conversation history
        conversation_history = ai_instance.conversation_history
        
        return jsonify({
            'success': True,
            'history': conversation_history
        })
    except Exception as e:
        logger.error(f"Error getting history: {e}")
        return jsonify({
            'success': False,
            'message': f"Error getting history: {str(e)}"
        }), 500

@socketio.on('connect')
def handle_connect():
    """Handle WebSocket connection."""
    logger.info(f"Client connected: {request.sid}")

@socketio.on('disconnect')
def handle_disconnect():
    """Handle WebSocket disconnection."""
    logger.info(f"Client disconnected: {request.sid}")

@socketio.on('join')
def handle_join(data):
    """Handle joining a room."""
    session_id = data.get('session_id')
    if session_id:
        socketio.join_room(session_id)
        logger.info(f"Client {request.sid} joined room {session_id}")

@socketio.on('chat')
def handle_chat(data):
    """Handle chat message via WebSocket."""
    try:
        # Get session ID and user input
        session_id = data.get('session_id')
        user_input = data.get('message')
        
        if not session_id or not user_input:
            emit('error', {
                'message': "Invalid request. Please provide session_id and message."
            })
            return
        
        # Start background processing
        threading.Thread(target=background_processing, args=(session_id, user_input)).start()
        
        # Acknowledge receipt
        emit('message_received', {
            'message': user_input,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Error handling WebSocket chat: {e}")
        emit('error', {
            'message': f"Error processing your message: {str(e)}"
        })

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)
