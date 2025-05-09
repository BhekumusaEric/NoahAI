"""
Deep Learning Model for NoahAI

This module implements a neural network model for text classification and response generation.
It uses TensorFlow to build, train, and use the model with advanced capabilities including
transfer learning and reinforcement learning.
"""

import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional, Input, Concatenate, GlobalMaxPooling1D, Conv1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.applications import MobileNetV2
import pickle
import random
from datetime import datetime

class DeepLearningModel:
    """
    A deep learning model for text classification and response generation.
    """

    def __init__(self, model_dir="data/model", responses_file="data/responses.json",
                 max_words=10000, max_sequence_length=100, use_advanced_nlp=True,
                 model_type="advanced_lstm", use_reinforcement_learning=False):
        """
        Initialize the deep learning model.

        Args:
            model_dir (str): Directory to save/load model files
            responses_file (str): Path to the responses JSON file
            max_words (int): Maximum number of words in the vocabulary
            max_sequence_length (int): Maximum length of input sequences
            use_advanced_nlp (bool): Whether to use advanced NLP capabilities
            model_type (str): Type of model to create. Options:
                - "simple_lstm": Basic LSTM model
                - "advanced_lstm": Advanced LSTM with bidirectional layers (default)
                - "cnn_lstm": CNN-LSTM hybrid model
                - "transfer_learning": Model using transfer learning
            use_reinforcement_learning (bool): Whether to use reinforcement learning
        """
        self.model_dir = model_dir
        self.responses_file = responses_file
        self.max_words = max_words
        self.max_sequence_length = max_sequence_length
        self.tokenizer = None
        self.model = None
        self.responses = self._load_responses()
        self.conversation_history = []
        self.feedback_history = []
        self.use_advanced_nlp = use_advanced_nlp
        self.model_type = model_type
        self.use_reinforcement_learning = use_reinforcement_learning

        # Create model directory if it doesn't exist
        os.makedirs(self.model_dir, exist_ok=True)

        # Initialize NLP utils
        try:
            from src.utils.nlp_utils import NLPUtils
            self.nlp_utils = NLPUtils(use_advanced_nlp=self.use_advanced_nlp)
            print(f"Initialized NLP utilities with advanced capabilities: {self.use_advanced_nlp}")
        except ImportError:
            print("NLP utilities not available. Using basic response generation.")
            self.nlp_utils = None

        # Initialize or load the model and tokenizer
        self._initialize_model()

        # Print model information
        print(f"Initialized deep learning model with type: {self.model_type}")
        if self.use_reinforcement_learning:
            print("Reinforcement learning is enabled.")

    def _load_responses(self):
        """
        Load response templates from a JSON file.

        Returns:
            dict: A dictionary of response templates
        """
        try:
            if os.path.exists(self.responses_file):
                with open(self.responses_file, "r") as f:
                    return json.load(f)
        except Exception as e:
            print(f"Error loading responses: {e}")

        # Default responses if file doesn't exist or has an error
        return {
            "greeting": ["Hello!", "Hi there!", "Greetings!"],
            "farewell": ["Goodbye!", "See you later!", "Until next time!"],
            "unknown": ["I'm not sure I understand.", "I don't know how to respond to that."]
        }

    def _initialize_model(self):
        """
        Initialize or load the model and tokenizer.
        """
        tokenizer_path = os.path.join(self.model_dir, "tokenizer.pickle")
        model_path = os.path.join(self.model_dir, f"model_{self.model_type}.h5")
        model_config_path = os.path.join(self.model_dir, "model_config.json")

        # Check if model and tokenizer exist
        if os.path.exists(tokenizer_path) and os.path.exists(model_path):
            # Load existing model and tokenizer
            try:
                with open(tokenizer_path, 'rb') as handle:
                    self.tokenizer = pickle.load(handle)

                self.model = load_model(model_path)

                # Load model configuration if it exists
                if os.path.exists(model_config_path):
                    with open(model_config_path, 'r') as f:
                        config = json.load(f)
                        # Update model type from saved config if it exists
                        if 'model_type' in config:
                            self.model_type = config['model_type']

                print(f"Loaded existing model (type: {self.model_type}) and tokenizer.")
                return
            except Exception as e:
                print(f"Error loading model or tokenizer: {e}")
                print(f"Creating new model with type: {self.model_type}")

        # Create new tokenizer
        self.tokenizer = Tokenizer(num_words=self.max_words)

        # Create new model
        self.model = self._create_model(model_type=self.model_type)

        # Save the tokenizer
        with open(tokenizer_path, 'wb') as handle:
            pickle.dump(self.tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # Save model configuration
        with open(model_config_path, 'w') as f:
            json.dump({
                'model_type': self.model_type,
                'max_words': self.max_words,
                'max_sequence_length': self.max_sequence_length,
                'use_advanced_nlp': self.use_advanced_nlp,
                'use_reinforcement_learning': self.use_reinforcement_learning,
                'created_at': datetime.now().isoformat()
            }, f, indent=2)

    def _create_model(self, model_type="advanced_lstm"):
        """
        Create a new deep learning model.

        Args:
            model_type (str): Type of model to create. Options:
                - "simple_lstm": Basic LSTM model
                - "advanced_lstm": Advanced LSTM with bidirectional layers (default)
                - "cnn_lstm": CNN-LSTM hybrid model
                - "transfer_learning": Model using transfer learning

        Returns:
            tf.keras.Model: A compiled Keras model
        """
        # Get the number of categories (response types)
        num_categories = len(self.responses.keys())

        # Choose model architecture based on model_type
        if model_type == "simple_lstm":
            # Simple LSTM model
            model = Sequential()
            model.add(Embedding(self.max_words, 128, input_length=self.max_sequence_length))
            model.add(LSTM(64))
            model.add(Dropout(0.2))
            model.add(Dense(64, activation='relu'))
            model.add(Dropout(0.2))
            model.add(Dense(num_categories, activation='softmax'))

        elif model_type == "cnn_lstm":
            # CNN-LSTM hybrid model
            input_layer = Input(shape=(self.max_sequence_length,))
            embedding_layer = Embedding(self.max_words, 128)(input_layer)

            # CNN layers
            conv1 = Conv1D(filters=64, kernel_size=3, padding='same', activation='relu')(embedding_layer)
            conv2 = Conv1D(filters=64, kernel_size=4, padding='same', activation='relu')(embedding_layer)
            conv3 = Conv1D(filters=64, kernel_size=5, padding='same', activation='relu')(embedding_layer)

            # Max pooling
            pool1 = GlobalMaxPooling1D()(conv1)
            pool2 = GlobalMaxPooling1D()(conv2)
            pool3 = GlobalMaxPooling1D()(conv3)

            # Concatenate CNN features
            concat = Concatenate()([pool1, pool2, pool3])

            # LSTM layer
            lstm_layer = LSTM(64)(embedding_layer)

            # Combine CNN and LSTM features
            combined = Concatenate()([concat, lstm_layer])

            # Dense layers
            dense1 = Dense(128, activation='relu')(combined)
            dropout1 = Dropout(0.2)(dense1)
            output_layer = Dense(num_categories, activation='softmax')(dropout1)

            # Create model
            model = Model(inputs=input_layer, outputs=output_layer)

        elif model_type == "transfer_learning":
            # This is a simplified example of transfer learning for text
            # In a real implementation, you would use a pre-trained text model like BERT

            # Create a base model with pre-trained weights (using MobileNetV2 as an example)
            # Note: This is just for demonstration - in practice, you'd use a text-specific model
            input_layer = Input(shape=(self.max_sequence_length,))
            embedding_layer = Embedding(self.max_words, 128)(input_layer)

            # Add custom layers on top of the base model
            x = Bidirectional(LSTM(64, return_sequences=True))(embedding_layer)
            x = Dropout(0.2)(x)
            x = Bidirectional(LSTM(32))(x)
            x = Dropout(0.2)(x)
            x = Dense(64, activation='relu')(x)
            x = Dropout(0.2)(x)
            output_layer = Dense(num_categories, activation='softmax')(x)

            # Create model
            model = Model(inputs=input_layer, outputs=output_layer)

            # Use a lower learning rate for transfer learning
            optimizer = Adam(learning_rate=0.0001)

        else:  # Default to advanced_lstm
            # Advanced LSTM model with bidirectional layers
            model = Sequential()
            model.add(Embedding(self.max_words, 128, input_length=self.max_sequence_length))
            model.add(Bidirectional(LSTM(64, return_sequences=True)))
            model.add(Dropout(0.2))
            model.add(Bidirectional(LSTM(32)))
            model.add(Dropout(0.2))
            model.add(Dense(64, activation='relu'))
            model.add(Dropout(0.2))
            model.add(Dense(num_categories, activation='softmax'))
            optimizer = 'adam'

        # Compile the model
        if model_type == "transfer_learning":
            model.compile(loss='categorical_crossentropy',
                        optimizer=optimizer,
                        metrics=['accuracy'])
        else:
            model.compile(loss='categorical_crossentropy',
                        optimizer='adam',
                        metrics=['accuracy'])

        return model

    def preprocess_text(self, text):
        """
        Preprocess text for the model.

        Args:
            text (str): The input text

        Returns:
            numpy.ndarray: Preprocessed text as a padded sequence
        """
        # Tokenize the text
        sequences = self.tokenizer.texts_to_sequences([text])

        # Pad the sequence
        padded_sequences = pad_sequences(sequences, maxlen=self.max_sequence_length)

        return padded_sequences

    def generate_response(self, user_input, user_name=None):
        """
        Generate a response based on the user input using the deep learning model.

        Args:
            user_input (str): The user's input text
            user_name (str): The user's name for personalization

        Returns:
            str: The generated response
        """
        # Add to conversation history
        self.conversation_history.append(("user", user_input))

        # Get the current context from NLP utils if available
        context = {}
        if hasattr(self, 'nlp_utils'):
            context = self.nlp_utils.get_context()

        # Check if model is trained (has been fit at least once)
        if not hasattr(self.model, 'history') or self.model.history is None:
            # Model is not trained yet, use NLP utils for response generation if available
            if hasattr(self, 'nlp_utils'):
                response = self.nlp_utils.generate_response(
                    input_text=user_input,
                    response_templates=self.responses,
                    user_name=user_name,
                    context=context
                )
                self.conversation_history.append(("ai", response))
                return response
            else:
                # Fallback to rule-based response
                response_category = self._rule_based_category(user_input)
        else:
            # Preprocess the input
            processed_input = self.preprocess_text(user_input)

            # Get model prediction
            prediction = self.model.predict(processed_input)[0]

            # Get the category with the highest probability
            category_index = np.argmax(prediction)

            # Map the index to a category name
            categories = list(self.responses.keys())
            response_category = categories[category_index] if category_index < len(categories) else "unknown"

            # Update context with the predicted category
            if hasattr(self, 'nlp_utils'):
                self.nlp_utils.update_context('predicted_category', response_category)

        # Get a response from the selected category
        if response_category in self.responses:
            response = random.choice(self.responses[response_category])
        else:
            response = random.choice(self.responses.get("unknown", ["I don't understand that yet."]))

        # Personalize the response if we have a user name
        if user_name and "{user_name}" in response:
            response = response.format(user_name=user_name)

        # Add to conversation history
        self.conversation_history.append(("ai", response))

        return response

    def _rule_based_category(self, input_text):
        """
        Determine the response category using rule-based approach.

        Args:
            input_text (str): The user's input text

        Returns:
            str: The determined category
        """
        # Convert to lowercase
        text = input_text.lower()

        # Check for specific keywords
        if any(word in text for word in ['hello', 'hi', 'hey']):
            return 'greeting'

        if any(word in text for word in ['bye', 'goodbye', 'exit', 'quit']):
            return 'farewell'

        if any(word in text for word in ['thanks', 'thank']):
            return 'thanks'

        if any(word in text for word in ['weather', 'temperature', 'forecast']):
            return 'weather'

        if any(word in text for word in ['joke', 'funny']):
            return 'joke'

        if any(word in text for word in ['time', 'date']):
            return 'time'

        if any(word in text for word in ['name', 'called']):
            return 'name'

        if any(phrase in text for phrase in ['how are you', 'how do you do', 'how are things']):
            return 'how_are_you'

        # Default to unknown
        return 'unknown'

    def add_feedback(self, user_input, ai_response, rating):
        """
        Add user feedback for a conversation.

        Args:
            user_input (str): The user's input text
            ai_response (str): The AI's response
            rating (int): User rating (1-5, where 5 is best)
        """
        # Validate rating
        if not 1 <= rating <= 5:
            raise ValueError("Rating must be between 1 and 5")

        # Add to feedback history
        self.feedback_history.append({
            "user_input": user_input,
            "ai_response": ai_response,
            "rating": rating,
            "timestamp": datetime.now().isoformat()
        })

        # Save feedback to file
        self._save_feedback()

    def _save_feedback(self):
        """
        Save feedback history to a file.
        """
        feedback_file = os.path.join(self.model_dir, "feedback.json")

        try:
            with open(feedback_file, 'w') as f:
                json.dump(self.feedback_history, f, indent=2)
        except Exception as e:
            print(f"Error saving feedback: {e}")

    def train(self, epochs=10, batch_size=32, use_early_stopping=True, use_transfer_learning=False):
        """
        Train the model using the feedback history.

        Args:
            epochs (int): Number of training epochs
            batch_size (int): Batch size for training
            use_early_stopping (bool): Whether to use early stopping
            use_transfer_learning (bool): Whether to use transfer learning techniques

        Returns:
            dict: Training history
        """
        if not self.feedback_history:
            print("No feedback data available for training.")
            return None

        # Prepare training data
        texts = []
        categories = []
        ratings = []

        # Get all response categories
        all_categories = list(self.responses.keys())

        # Process feedback data
        for feedback in self.feedback_history:
            user_input = feedback["user_input"]
            ai_response = feedback["ai_response"]
            rating = feedback["rating"]

            # Store the rating for reinforcement learning
            ratings.append(rating)

            # For standard supervised learning, only use high-rated responses
            # For reinforcement learning, we'll use all responses with their ratings
            if not self.use_reinforcement_learning and rating < 4:
                continue

            # Find the category of the response
            category = "unknown"
            for cat, responses in self.responses.items():
                # Check if the response is in this category
                for resp in responses:
                    # Remove user name formatting
                    clean_resp = resp.replace("{user_name}", "").strip()
                    if clean_resp in ai_response:
                        category = cat
                        break
                if category != "unknown":
                    break

            texts.append(user_input)
            categories.append(category)

        if not texts:
            print("No suitable feedback available for training.")
            return None

        # Fit the tokenizer on the texts
        self.tokenizer.fit_on_texts(texts)

        # Convert texts to sequences
        sequences = self.tokenizer.texts_to_sequences(texts)

        # Pad sequences
        X = pad_sequences(sequences, maxlen=self.max_sequence_length)

        # Convert categories to one-hot encoding
        y = np.zeros((len(categories), len(all_categories)))
        for i, category in enumerate(categories):
            if category in all_categories:
                y[i, all_categories.index(category)] = 1

        # Set up callbacks
        callbacks = []

        if use_early_stopping:
            early_stopping = EarlyStopping(
                monitor='val_loss',
                patience=3,
                restore_best_weights=True
            )
            callbacks.append(early_stopping)

        # Add model checkpoint to save the best model
        model_checkpoint = ModelCheckpoint(
            filepath=os.path.join(self.model_dir, f"model_{self.model_type}_best.h5"),
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
        callbacks.append(model_checkpoint)

        # If using reinforcement learning, apply sample weights based on ratings
        if self.use_reinforcement_learning:
            # Normalize ratings to be between 0 and 1
            sample_weights = np.array(ratings) / 5.0

            # Train the model with sample weights
            history = self.model.fit(
                X, y,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=0.2,
                callbacks=callbacks,
                sample_weight=sample_weights
            )
        else:
            # Standard training
            history = self.model.fit(
                X, y,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=0.2,
                callbacks=callbacks
            )

        # If using transfer learning, fine-tune with a lower learning rate
        if use_transfer_learning and self.model_type == "transfer_learning":
            print("Fine-tuning with transfer learning...")

            # Reduce learning rate for fine-tuning
            optimizer = Adam(learning_rate=0.00001)
            self.model.compile(
                loss='categorical_crossentropy',
                optimizer=optimizer,
                metrics=['accuracy']
            )

            # Fine-tune with a few more epochs
            fine_tune_history = self.model.fit(
                X, y,
                epochs=5,  # Fewer epochs for fine-tuning
                batch_size=batch_size,
                validation_split=0.2,
                callbacks=callbacks
            )

            # Combine histories
            for key in fine_tune_history.history:
                history.history[key].extend(fine_tune_history.history[key])

        # Save the model and tokenizer
        self._save_model()

        return history.history

    def _save_model(self):
        """
        Save the model and tokenizer.
        """
        model_path = os.path.join(self.model_dir, f"model_{self.model_type}.h5")
        tokenizer_path = os.path.join(self.model_dir, "tokenizer.pickle")
        model_config_path = os.path.join(self.model_dir, "model_config.json")

        try:
            # Save model
            self.model.save(model_path)

            # Save tokenizer
            with open(tokenizer_path, 'wb') as handle:
                pickle.dump(self.tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

            # Save model configuration
            with open(model_config_path, 'w') as f:
                json.dump({
                    'model_type': self.model_type,
                    'max_words': self.max_words,
                    'max_sequence_length': self.max_sequence_length,
                    'use_advanced_nlp': self.use_advanced_nlp,
                    'use_reinforcement_learning': self.use_reinforcement_learning,
                    'updated_at': datetime.now().isoformat()
                }, f, indent=2)

            print(f"Model (type: {self.model_type}) and tokenizer saved to {self.model_dir}")
        except Exception as e:
            print(f"Error saving model or tokenizer: {e}")

    def get_conversation_history(self):
        """
        Get the conversation history.

        Returns:
            list: A list of (speaker, text) tuples
        """
        return self.conversation_history

    def clear_conversation_history(self):
        """
        Clear the conversation history.
        """
        self.conversation_history = []
