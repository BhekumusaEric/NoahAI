"""
Transformer-Based State Representation for NoahAI

This module provides advanced state representation for reinforcement learning
using transformer models and attention mechanisms.
"""

import os
import numpy as np
import logging
from typing import List, Dict, Tuple, Any, Optional, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/transformer_state.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("transformer_state")

# Try to import TensorFlow and Transformers
try:
    import tensorflow as tf
    from tensorflow.keras.layers import Dense, Dropout, LayerNormalization, MultiHeadAttention
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    logger.warning("TensorFlow not available. Transformer state representation will be limited.")

try:
    from transformers import TFAutoModel, AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("Transformers library not available. Using fallback state representation.")

class TransformerStateRepresentation:
    """
    Transformer-based state representation for reinforcement learning.
    
    This class uses transformer models to create rich state representations
    that capture the semantic meaning and context of conversations.
    """
    
    def __init__(self, state_size=128, model_name="distilbert-base-uncased", 
                 use_attention=True, max_sequence_length=512, cache_dir="data/models"):
        """
        Initialize the transformer state representation.
        
        Args:
            state_size (int): Size of the state vector
            model_name (str): Name of the pre-trained transformer model
            use_attention (bool): Whether to use attention mechanisms
            max_sequence_length (int): Maximum sequence length for tokenization
            cache_dir (str): Directory to cache models
        """
        self.state_size = state_size
        self.model_name = model_name
        self.use_attention = use_attention
        self.max_sequence_length = max_sequence_length
        self.cache_dir = cache_dir
        
        # Create cache directory if it doesn't exist
        os.makedirs(cache_dir, exist_ok=True)
        
        # Initialize transformer model and tokenizer
        self.model = None
        self.tokenizer = None
        self.attention_layer = None
        
        # Check if TensorFlow and Transformers are available
        if TENSORFLOW_AVAILABLE and TRANSFORMERS_AVAILABLE:
            try:
                logger.info(f"Loading transformer model: {model_name}")
                self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
                self.model = TFAutoModel.from_pretrained(model_name, cache_dir=cache_dir)
                
                # Initialize attention layer if requested
                if use_attention:
                    self.attention_layer = MultiHeadAttention(
                        num_heads=8, 
                        key_dim=64,
                        dropout=0.1
                    )
                    logger.info("Initialized multi-head attention layer")
                
                logger.info(f"Successfully loaded transformer model: {model_name}")
            except Exception as e:
                logger.error(f"Error loading transformer model: {e}")
                self.model = None
                self.tokenizer = None
        else:
            logger.warning("TensorFlow or Transformers not available. Using fallback state representation.")
    
    def get_state(self, user_input, ai_response, conversation_history):
        """
        Convert conversation data into a state representation using transformers.
        
        Args:
            user_input (str): User input
            ai_response (str): AI response
            conversation_history (list): Conversation history
            
        Returns:
            numpy.ndarray: State representation
        """
        if self.model is not None and self.tokenizer is not None:
            return self._get_transformer_state(user_input, ai_response, conversation_history)
        else:
            return self._get_fallback_state(user_input, ai_response, conversation_history)
    
    def _get_transformer_state(self, user_input, ai_response, conversation_history):
        """
        Get state representation using transformer model.
        
        Args:
            user_input (str): User input
            ai_response (str): AI response
            conversation_history (list): Conversation history
            
        Returns:
            numpy.ndarray: State representation
        """
        # Prepare input text
        # Format: [CLS] User: {user_input} [SEP] AI: {ai_response} [SEP] {history}
        input_text = f"User: {user_input} [SEP] AI: {ai_response}"
        
        # Add recent conversation history
        recent_history = conversation_history[-5:] if len(conversation_history) > 5 else conversation_history
        for i, (role, text) in enumerate(recent_history):
            input_text += f" [SEP] {role}: {text}"
        
        # Tokenize input
        inputs = self.tokenizer(
            input_text,
            return_tensors="tf",
            max_length=self.max_sequence_length,
            truncation=True,
            padding="max_length"
        )
        
        # Get model outputs
        outputs = self.model(inputs)
        
        # Get the hidden states
        hidden_states = outputs.last_hidden_state
        
        # Apply attention if requested
        if self.use_attention and self.attention_layer is not None:
            # Apply self-attention
            attention_output = self.attention_layer(
                hidden_states, hidden_states, hidden_states
            )
            
            # Get the mean of attention outputs as the state representation
            state_vector = tf.reduce_mean(attention_output, axis=1)[0]
        else:
            # Use the [CLS] token embedding or mean pooling
            if self.model_name.startswith("bert") or self.model_name.startswith("distilbert"):
                # For BERT-like models, use the [CLS] token
                state_vector = hidden_states[:, 0, :][0]
            else:
                # For other models, use mean pooling
                state_vector = tf.reduce_mean(hidden_states, axis=1)[0]
        
        # Resize to state_size
        if len(state_vector) > self.state_size:
            state_vector = state_vector[:self.state_size]
        elif len(state_vector) < self.state_size:
            padding = tf.zeros(self.state_size - len(state_vector))
            state_vector = tf.concat([state_vector, padding], axis=0)
        
        return state_vector.numpy()
    
    def _get_fallback_state(self, user_input, ai_response, conversation_history):
        """
        Get a fallback state representation without transformers.
        
        Args:
            user_input (str): User input
            ai_response (str): AI response
            conversation_history (list): Conversation history
            
        Returns:
            numpy.ndarray: State representation
        """
        # Initialize state vector
        state = np.zeros(self.state_size)
        
        # Simple features
        features = [
            len(user_input),                                # Length of user input
            len(ai_response),                               # Length of AI response
            len(conversation_history),                      # Conversation length
            user_input.count('?') > 0,                      # User asked a question
            user_input.count('!') > 0,                      # User used exclamation
            ai_response.count('?') > 0,                     # AI asked a question
            ai_response.count('!') > 0,                     # AI used exclamation
            len(user_input.split()) / 20,                   # Normalized word count (user)
            len(ai_response.split()) / 50,                  # Normalized word count (AI)
            sum(1 for c in user_input if c.isupper()) / max(1, len(user_input)),  # Uppercase ratio (user)
            sum(1 for c in ai_response if c.isupper()) / max(1, len(ai_response)),  # Uppercase ratio (AI)
        ]
        
        # Add features to state vector
        for i, feature in enumerate(features):
            if i < self.state_size:
                state[i] = feature
        
        # Add character frequency features
        char_offset = len(features)
        for i, c in enumerate("abcdefghijklmnopqrstuvwxyz"):
            if char_offset + i < self.state_size:
                state[char_offset + i] = user_input.lower().count(c) / max(1, len(user_input))
        
        # Add conversation history features
        history_offset = char_offset + 26
        if history_offset < self.state_size and conversation_history:
            # Average length of previous exchanges
            avg_length = sum(len(text) for _, text in conversation_history) / len(conversation_history)
            if history_offset < self.state_size:
                state[history_offset] = avg_length / 100  # Normalize
            
            # Number of turns
            if history_offset + 1 < self.state_size:
                state[history_offset + 1] = len(conversation_history) / 10  # Normalize
            
            # Sentiment indicators (very simple)
            positive_words = ["good", "great", "excellent", "happy", "thanks", "thank", "appreciate"]
            negative_words = ["bad", "wrong", "error", "issue", "problem", "sorry", "fail"]
            
            positive_count = sum(1 for word in positive_words if word in user_input.lower())
            negative_count = sum(1 for word in negative_words if word in user_input.lower())
            
            if history_offset + 2 < self.state_size:
                state[history_offset + 2] = positive_count / 5  # Normalize
            
            if history_offset + 3 < self.state_size:
                state[history_offset + 3] = negative_count / 5  # Normalize
        
        return state

class AttentionLayer(tf.keras.layers.Layer):
    """
    Custom attention layer for state representation.
    
    This layer implements a self-attention mechanism to focus on
    important parts of the input sequence.
    """
    
    def __init__(self, attention_dim=64, **kwargs):
        """
        Initialize the attention layer.
        
        Args:
            attention_dim (int): Dimension of the attention mechanism
        """
        super(AttentionLayer, self).__init__(**kwargs)
        self.attention_dim = attention_dim
        
    def build(self, input_shape):
        """
        Build the layer.
        
        Args:
            input_shape: Shape of the input tensor
        """
        self.W = self.add_weight(
            name="attention_weight",
            shape=(input_shape[-1], self.attention_dim),
            initializer="glorot_uniform",
            trainable=True
        )
        
        self.b = self.add_weight(
            name="attention_bias",
            shape=(self.attention_dim,),
            initializer="zeros",
            trainable=True
        )
        
        self.u = self.add_weight(
            name="context_vector",
            shape=(self.attention_dim,),
            initializer="glorot_uniform",
            trainable=True
        )
        
        super(AttentionLayer, self).build(input_shape)
    
    def call(self, inputs):
        """
        Apply the attention mechanism.
        
        Args:
            inputs: Input tensor
            
        Returns:
            tuple: (Context vector, attention weights)
        """
        # inputs shape: (batch_size, seq_len, features)
        
        # Apply the first weight matrix and bias
        uit = tf.tanh(tf.tensordot(inputs, self.W, axes=1) + self.b)
        
        # Calculate attention weights
        ait = tf.tensordot(uit, self.u, axes=1)
        attention_weights = tf.nn.softmax(ait, axis=1)
        
        # Apply attention weights to get the context vector
        context_vector = tf.reduce_sum(inputs * tf.expand_dims(attention_weights, -1), axis=1)
        
        return context_vector, attention_weights
