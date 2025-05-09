#!/usr/bin/env python3
"""
Hyperparameter Optimizer for NoahAI

This script performs hyperparameter optimization for NoahAI's deep learning model.
It uses Keras Tuner to find the best hyperparameters for the model.
"""

import os
import sys
import json
import argparse
import logging
import numpy as np
import tensorflow as tf
from datetime import datetime
from typing import List, Dict, Tuple, Any, Optional, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/hyperparameter_optimization.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("hyperparameter_optimizer")

# Try to import NoahAI modules
try:
    from models.deep_learning_model import DeepLearningModel
    from src.utils.learning_utils import LearningUtils
    NOAHAI_AVAILABLE = True
except ImportError:
    NOAHAI_AVAILABLE = False
    logger.error("NoahAI modules not available. Please run this script from the NoahAI directory.")
    sys.exit(1)

# Try to import Keras Tuner
try:
    import keras_tuner as kt
    KERAS_TUNER_AVAILABLE = True
except ImportError:
    KERAS_TUNER_AVAILABLE = False
    logger.error("Keras Tuner not available. Please install it with: pip install keras-tuner")
    sys.exit(1)

class NoahAIHyperModel(kt.HyperModel):
    """
    Hypermodel for NoahAI's deep learning model.
    """
    
    def __init__(self, config, texts, categories, ratings):
        """
        Initialize the hypermodel.
        
        Args:
            config (dict): Configuration dictionary
            texts (list): List of input texts
            categories (list): List of categories
            ratings (list): List of ratings
        """
        self.config = config
        self.texts = texts
        self.categories = categories
        self.ratings = ratings
        
        # Get unique categories
        self.unique_categories = sorted(set(categories))
        self.num_categories = len(self.unique_categories)
        
        # Create tokenizer
        self.tokenizer = tf.keras.preprocessing.text.Tokenizer(
            num_words=config["data"]["max_words"]
        )
        self.tokenizer.fit_on_texts(texts)
        
        # Convert texts to sequences
        self.sequences = self.tokenizer.texts_to_sequences(texts)
        
        # Pad sequences
        self.padded_sequences = tf.keras.preprocessing.sequence.pad_sequences(
            self.sequences,
            maxlen=config["data"]["max_sequence_length"]
        )
        
        # Convert categories to one-hot encoding
        self.category_to_index = {cat: i for i, cat in enumerate(self.unique_categories)}
        self.category_indices = [self.category_to_index[cat] for cat in categories]
        self.one_hot_categories = tf.keras.utils.to_categorical(
            self.category_indices,
            num_classes=self.num_categories
        )
    
    def build(self, hp):
        """
        Build the model with hyperparameters.
        
        Args:
            hp (kt.HyperParameters): Hyperparameters
            
        Returns:
            tf.keras.Model: Compiled model
        """
        # Define hyperparameters to tune
        embedding_dim = hp.Int(
            "embedding_dim",
            min_value=64,
            max_value=256,
            step=64,
            default=128
        )
        
        lstm_units_1 = hp.Int(
            "lstm_units_1",
            min_value=32,
            max_value=128,
            step=32,
            default=64
        )
        
        lstm_units_2 = hp.Int(
            "lstm_units_2",
            min_value=16,
            max_value=64,
            step=16,
            default=32
        )
        
        dense_units = hp.Int(
            "dense_units",
            min_value=32,
            max_value=128,
            step=32,
            default=64
        )
        
        dropout_rate = hp.Float(
            "dropout_rate",
            min_value=0.1,
            max_value=0.5,
            step=0.1,
            default=0.2
        )
        
        learning_rate = hp.Choice(
            "learning_rate",
            values=[1e-4, 5e-4, 1e-3, 5e-3],
            default=1e-3
        )
        
        # Build model
        model_type = hp.Choice(
            "model_type",
            values=["simple_lstm", "advanced_lstm", "cnn_lstm"],
            default="advanced_lstm"
        )
        
        # Get vocabulary size
        vocab_size = min(len(self.tokenizer.word_index) + 1, self.config["data"]["max_words"])
        
        # Create model based on type
        if model_type == "simple_lstm":
            model = tf.keras.Sequential([
                tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=self.config["data"]["max_sequence_length"]),
                tf.keras.layers.LSTM(lstm_units_1),
                tf.keras.layers.Dropout(dropout_rate),
                tf.keras.layers.Dense(dense_units, activation='relu'),
                tf.keras.layers.Dropout(dropout_rate),
                tf.keras.layers.Dense(self.num_categories, activation='softmax')
            ])
        
        elif model_type == "cnn_lstm":
            model = tf.keras.Sequential([
                tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=self.config["data"]["max_sequence_length"]),
                tf.keras.layers.Conv1D(filters=64, kernel_size=5, padding='same', activation='relu'),
                tf.keras.layers.MaxPooling1D(pool_size=2),
                tf.keras.layers.LSTM(lstm_units_1),
                tf.keras.layers.Dropout(dropout_rate),
                tf.keras.layers.Dense(dense_units, activation='relu'),
                tf.keras.layers.Dropout(dropout_rate),
                tf.keras.layers.Dense(self.num_categories, activation='softmax')
            ])
        
        else:  # advanced_lstm
            model = tf.keras.Sequential([
                tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=self.config["data"]["max_sequence_length"]),
                tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(lstm_units_1, return_sequences=True)),
                tf.keras.layers.Dropout(dropout_rate),
                tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(lstm_units_2)),
                tf.keras.layers.Dropout(dropout_rate),
                tf.keras.layers.Dense(dense_units, activation='relu'),
                tf.keras.layers.Dropout(dropout_rate),
                tf.keras.layers.Dense(self.num_categories, activation='softmax')
            ])
        
        # Compile model
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def fit(self, hp, model, *args, **kwargs):
        """
        Fit the model with hyperparameters.
        
        Args:
            hp (kt.HyperParameters): Hyperparameters
            model (tf.keras.Model): Model to fit
            
        Returns:
            tf.keras.callbacks.History: Training history
        """
        # Get batch size
        batch_size = hp.Int(
            "batch_size",
            min_value=16,
            max_value=64,
            step=16,
            default=32
        )
        
        # Add early stopping
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        )
        
        # Train model
        return model.fit(
            self.padded_sequences,
            self.one_hot_categories,
            batch_size=batch_size,
            epochs=20,
            validation_split=0.2,
            callbacks=[early_stopping],
            *args,
            **kwargs
        )

class HyperparameterOptimizer:
    """
    Optimizer for finding the best hyperparameters for NoahAI's deep learning model.
    """
    
    def __init__(self, config_file=None, output_dir="data/hyperparameter_tuning"):
        """
        Initialize the optimizer.
        
        Args:
            config_file (str): Path to configuration file
            output_dir (str): Directory for output
        """
        self.output_dir = output_dir
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Load configuration
        self.config = self._load_config(config_file)
    
    def _load_config(self, config_file):
        """
        Load configuration from file or use defaults.
        
        Args:
            config_file (str): Path to configuration file
            
        Returns:
            dict: Configuration dictionary
        """
        default_config = {
            "training": {
                "batch_size": 32,
                "epochs": 20,
                "learning_rate": 0.001,
                "early_stopping_patience": 5,
                "validation_split": 0.2
            },
            "data": {
                "training_data_file": "data/massive_training_data.json",
                "max_sequence_length": 100,
                "max_words": 10000
            },
            "optimization": {
                "num_trials": 10,
                "search_algorithm": "bayesian",  # random, bayesian, hyperband
                "max_epochs": 20,
                "factor": 3,
                "hyperband_iterations": 2
            }
        }
        
        if config_file and os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    user_config = json.load(f)
                
                # Merge user config with defaults
                for section, settings in user_config.items():
                    if section in default_config:
                        default_config[section].update(settings)
                    else:
                        default_config[section] = settings
                
                logger.info(f"Loaded configuration from {config_file}")
            except Exception as e:
                logger.error(f"Error loading configuration: {e}")
        
        return default_config
    
    def load_training_data(self, data_file=None):
        """
        Load training data from file.
        
        Args:
            data_file (str): Path to training data file
            
        Returns:
            tuple: (texts, categories, ratings) for training
        """
        data_file = data_file or self.config["data"]["training_data_file"]
        
        if not os.path.exists(data_file):
            logger.error(f"Training data file not found: {data_file}")
            return None, None, None
        
        try:
            with open(data_file, 'r') as f:
                data = json.load(f)
            
            logger.info(f"Loaded {len(data)} training examples from {data_file}")
            
            # Extract texts, categories, and ratings
            texts = []
            categories = []
            ratings = []
            
            for entry in data:
                texts.append(entry["user_input"])
                categories.append(entry["category"])
                ratings.append(entry.get("rating", 5))  # Default to 5 if rating not provided
            
            return texts, categories, ratings
        
        except Exception as e:
            logger.error(f"Error loading training data: {e}")
            return None, None, None
    
    def optimize(self, data_file=None):
        """
        Run hyperparameter optimization.
        
        Args:
            data_file (str): Path to training data file
            
        Returns:
            dict: Best hyperparameters
        """
        # Load training data
        texts, categories, ratings = self.load_training_data(data_file)
        
        if texts is None:
            logger.error("Failed to load training data")
            return None
        
        # Create hypermodel
        hypermodel = NoahAIHyperModel(self.config, texts, categories, ratings)
        
        # Set up tuner
        optimization_config = self.config["optimization"]
        search_algorithm = optimization_config.get("search_algorithm", "bayesian")
        
        if search_algorithm == "random":
            tuner = kt.RandomSearch(
                hypermodel,
                objective="val_accuracy",
                max_trials=optimization_config["num_trials"],
                directory=self.output_dir,
                project_name="noahai_random_search"
            )
        elif search_algorithm == "hyperband":
            tuner = kt.Hyperband(
                hypermodel,
                objective="val_accuracy",
                max_epochs=optimization_config.get("max_epochs", 20),
                factor=optimization_config.get("factor", 3),
                hyperband_iterations=optimization_config.get("hyperband_iterations", 2),
                directory=self.output_dir,
                project_name="noahai_hyperband"
            )
        else:  # bayesian
            tuner = kt.BayesianOptimization(
                hypermodel,
                objective="val_accuracy",
                max_trials=optimization_config["num_trials"],
                directory=self.output_dir,
                project_name="noahai_bayesian"
            )
        
        # Start optimization
        logger.info(f"Starting hyperparameter optimization with {search_algorithm} search")
        logger.info(f"Running {optimization_config['num_trials']} trials")
        
        tuner.search()
        
        # Get best hyperparameters
        best_hp = tuner.get_best_hyperparameters(1)[0]
        
        # Log best hyperparameters
        logger.info("Best hyperparameters:")
        for param, value in best_hp.values.items():
            logger.info(f"  {param}: {value}")
        
        # Save best hyperparameters
        best_hp_file = os.path.join(self.output_dir, "best_hyperparameters.json")
        with open(best_hp_file, 'w') as f:
            json.dump(best_hp.values, f, indent=2)
        
        logger.info(f"Saved best hyperparameters to {best_hp_file}")
        
        return best_hp.values

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Hyperparameter optimization for NoahAI")
    parser.add_argument("--config", help="Path to configuration file")
    parser.add_argument("--data", help="Path to training data file")
    parser.add_argument("--output-dir", default="data/hyperparameter_tuning", help="Directory for output")
    args = parser.parse_args()
    
    # Create optimizer
    optimizer = HyperparameterOptimizer(
        config_file=args.config,
        output_dir=args.output_dir
    )
    
    # Run optimization
    optimizer.optimize(data_file=args.data)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
