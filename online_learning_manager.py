#!/usr/bin/env python3
"""
Online Learning Manager for NoahAI

This script implements online learning for NoahAI's deep learning model.
It allows the model to continuously learn from new data without retraining from scratch.
Features:
- Incremental model updates
- Continuous learning from user feedback
- Adaptive learning rates
- Concept drift detection
- Model versioning
"""

import os
import sys
import json
import time
import argparse
import logging
import numpy as np
import tensorflow as tf
from datetime import datetime
from typing import List, Dict, Tuple, Any, Optional, Union
from collections import deque

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/online_learning.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("online_learning")

# Try to import NoahAI modules
try:
    from models.deep_learning_model import DeepLearningModel
    from src.utils.learning_utils import LearningUtils
    NOAHAI_AVAILABLE = True
except ImportError:
    NOAHAI_AVAILABLE = False
    logger.error("NoahAI modules not available. Please run this script from the NoahAI directory.")
    sys.exit(1)

class OnlineLearningManager:
    """
    Manager for online learning of NoahAI's deep learning model.
    """
    
    def __init__(self, config_file=None, model_dir="data/model", output_dir="data/online_learning"):
        """
        Initialize the online learning manager.
        
        Args:
            config_file (str): Path to configuration file
            model_dir (str): Directory for model files
            output_dir (str): Directory for output
        """
        self.model_dir = model_dir
        self.output_dir = output_dir
        
        # Create directories
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)
        
        # Load configuration
        self.config = self._load_config(config_file)
        
        # Initialize model
        self.model = self._load_model()
        
        # Initialize learning buffer
        self.learning_buffer = deque(maxlen=self.config["online_learning"]["buffer_size"])
        
        # Initialize metrics
        self.metrics = {
            "updates": 0,
            "samples_processed": 0,
            "current_accuracy": 0.0,
            "accuracy_history": [],
            "loss_history": [],
            "drift_detected": False,
            "drift_score": 0.0,
            "last_update_time": None,
            "model_versions": []
        }
    
    def _load_config(self, config_file):
        """
        Load configuration from file or use defaults.
        
        Args:
            config_file (str): Path to configuration file
            
        Returns:
            dict: Configuration dictionary
        """
        default_config = {
            "model": {
                "model_type": "advanced_lstm",
                "max_words": 10000,
                "max_sequence_length": 100,
                "use_reinforcement_learning": True
            },
            "online_learning": {
                "buffer_size": 1000,
                "update_frequency": 50,  # Update after this many new samples
                "min_samples_for_update": 10,
                "learning_rate": 0.001,
                "adaptive_learning_rate": True,
                "batch_size": 16,
                "epochs_per_update": 1,
                "validation_split": 0.1,
                "detect_concept_drift": True,
                "drift_threshold": 0.1,
                "versioning": {
                    "enabled": True,
                    "max_versions": 5,
                    "save_frequency": 10  # Save a new version after this many updates
                }
            },
            "evaluation": {
                "enabled": True,
                "frequency": 5,  # Evaluate after this many updates
                "metrics": ["accuracy", "loss", "precision", "recall", "f1"]
            }
        }
        
        if config_file and os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    user_config = json.load(f)
                
                # Merge user config with defaults
                for section, settings in user_config.items():
                    if section in default_config:
                        if isinstance(settings, dict) and isinstance(default_config[section], dict):
                            default_config[section].update(settings)
                        else:
                            default_config[section] = settings
                    else:
                        default_config[section] = settings
                
                logger.info(f"Loaded configuration from {config_file}")
            except Exception as e:
                logger.error(f"Error loading configuration: {e}")
        
        return default_config
    
    def _load_model(self):
        """
        Load or create a model.
        
        Returns:
            DeepLearningModel: Model instance
        """
        model_config = self.config["model"]
        
        # Check if model exists
        model_file = os.path.join(self.model_dir, f"model_{model_config['model_type']}.h5")
        
        if os.path.exists(model_file):
            logger.info(f"Loading existing model from {model_file}")
            
            try:
                model = DeepLearningModel(
                    model_dir=self.model_dir,
                    responses_file="data/responses.json",
                    max_words=model_config["max_words"],
                    max_sequence_length=model_config["max_sequence_length"],
                    use_advanced_nlp=True,
                    model_type=model_config["model_type"],
                    use_reinforcement_learning=model_config["use_reinforcement_learning"]
                )
                
                # Load the model
                model.load_model()
                
                return model
            
            except Exception as e:
                logger.error(f"Error loading model: {e}")
                logger.info("Creating new model instead")
        
        # Create new model
        logger.info("Creating new model")
        
        model = DeepLearningModel(
            model_dir=self.model_dir,
            responses_file="data/responses.json",
            max_words=model_config["max_words"],
            max_sequence_length=model_config["max_sequence_length"],
            use_advanced_nlp=True,
            model_type=model_config["model_type"],
            use_reinforcement_learning=model_config["use_reinforcement_learning"]
        )
        
        return model
    
    def add_sample(self, text, category, rating=None):
        """
        Add a new sample to the learning buffer.
        
        Args:
            text (str): Input text
            category (str): Category
            rating (int, optional): Rating (1-5)
            
        Returns:
            bool: Whether an update was triggered
        """
        # Add sample to buffer
        sample = {
            "text": text,
            "category": category,
            "rating": rating if rating is not None else 5,
            "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
        }
        
        self.learning_buffer.append(sample)
        self.metrics["samples_processed"] += 1
        
        # Check if update is needed
        update_frequency = self.config["online_learning"]["update_frequency"]
        min_samples = self.config["online_learning"]["min_samples_for_update"]
        
        if (len(self.learning_buffer) >= min_samples and 
            self.metrics["samples_processed"] % update_frequency == 0):
            self.update_model()
            return True
        
        return False
    
    def add_samples(self, samples):
        """
        Add multiple samples to the learning buffer.
        
        Args:
            samples (list): List of samples, each with 'text', 'category', and optionally 'rating'
            
        Returns:
            bool: Whether an update was triggered
        """
        for sample in samples:
            self.add_sample(
                sample["text"],
                sample["category"],
                sample.get("rating")
            )
        
        # Check if update is needed
        update_frequency = self.config["online_learning"]["update_frequency"]
        min_samples = self.config["online_learning"]["min_samples_for_update"]
        
        if (len(self.learning_buffer) >= min_samples and 
            self.metrics["samples_processed"] % update_frequency == 0):
            self.update_model()
            return True
        
        return False
    
    def update_model(self):
        """
        Update the model with the current learning buffer.
        
        Returns:
            dict: Update metrics
        """
        if len(self.learning_buffer) < self.config["online_learning"]["min_samples_for_update"]:
            logger.info(f"Not enough samples for update. Have {len(self.learning_buffer)}, need {self.config['online_learning']['min_samples_for_update']}")
            return None
        
        logger.info(f"Updating model with {len(self.learning_buffer)} samples")
        
        # Extract data from buffer
        texts = []
        categories = []
        ratings = []
        
        for sample in self.learning_buffer:
            texts.append(sample["text"])
            categories.append(sample["category"])
            ratings.append(sample["rating"])
        
        # Check for concept drift if enabled
        if self.config["online_learning"]["detect_concept_drift"]:
            drift_score = self._detect_concept_drift(texts, categories)
            self.metrics["drift_score"] = drift_score
            
            if drift_score > self.config["online_learning"]["drift_threshold"]:
                logger.info(f"Concept drift detected (score: {drift_score:.4f})")
                self.metrics["drift_detected"] = True
                
                # Adjust learning rate if adaptive learning rate is enabled
                if self.config["online_learning"]["adaptive_learning_rate"]:
                    self._adjust_learning_rate(drift_score)
        
        # Prepare data for training
        self.model.prepare_training_data(texts, categories)
        
        # Set up training parameters
        batch_size = self.config["online_learning"]["batch_size"]
        epochs = self.config["online_learning"]["epochs_per_update"]
        validation_split = self.config["online_learning"]["validation_split"]
        
        # Update the model
        try:
            # Get current optimizer
            optimizer = self.model.model.optimizer
            
            # Set learning rate if specified
            if self.config["online_learning"]["learning_rate"] is not None:
                tf.keras.backend.set_value(optimizer.learning_rate, self.config["online_learning"]["learning_rate"])
            
            # Train for a few epochs
            history = self.model.model.fit(
                self.model.X_train,
                self.model.y_train,
                batch_size=batch_size,
                epochs=epochs,
                validation_split=validation_split,
                verbose=0
            )
            
            # Update metrics
            self.metrics["updates"] += 1
            self.metrics["last_update_time"] = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
            
            if history.history.get("val_accuracy"):
                self.metrics["current_accuracy"] = history.history["val_accuracy"][-1]
                self.metrics["accuracy_history"].append(self.metrics["current_accuracy"])
            
            if history.history.get("val_loss"):
                self.metrics["loss_history"].append(history.history["val_loss"][-1])
            
            # Save model version if enabled
            if (self.config["online_learning"]["versioning"]["enabled"] and 
                self.metrics["updates"] % self.config["online_learning"]["versioning"]["save_frequency"] == 0):
                self._save_model_version()
            
            # Evaluate model if enabled
            if (self.config["evaluation"]["enabled"] and 
                self.metrics["updates"] % self.config["evaluation"]["frequency"] == 0):
                self._evaluate_model()
            
            # Save metrics
            self._save_metrics()
            
            # Save model
            self.model._save_model()
            
            logger.info(f"Model updated successfully (update #{self.metrics['updates']})")
            
            return {
                "update_id": self.metrics["updates"],
                "samples_used": len(self.learning_buffer),
                "accuracy": self.metrics["current_accuracy"],
                "drift_detected": self.metrics["drift_detected"],
                "drift_score": self.metrics["drift_score"]
            }
        
        except Exception as e:
            logger.error(f"Error updating model: {e}")
            return None
    
    def _detect_concept_drift(self, texts, categories):
        """
        Detect concept drift in the new data.
        
        Args:
            texts (list): List of input texts
            categories (list): List of categories
            
        Returns:
            float: Drift score (0-1)
        """
        # Skip if no previous data
        if not hasattr(self.model, "X_train") or self.model.X_train is None:
            return 0.0
        
        try:
            # Preprocess new data
            X_new = self.model.preprocess_text(texts)
            
            # Get predictions on new data
            y_pred = self.model.model.predict(X_new)
            
            # Convert categories to one-hot encoding
            unique_categories = sorted(set(self.model.category_mapping.keys()))
            category_to_index = {cat: i for i, cat in enumerate(unique_categories)}
            y_indices = [category_to_index.get(cat, 0) for cat in categories]
            y_true = tf.keras.utils.to_categorical(y_indices, num_classes=len(unique_categories))
            
            # Calculate prediction error
            errors = np.mean(np.abs(y_pred - y_true), axis=1)
            mean_error = np.mean(errors)
            
            # Normalize to 0-1 range
            drift_score = min(1.0, mean_error * 2)  # Scale up for better sensitivity
            
            return drift_score
        
        except Exception as e:
            logger.error(f"Error detecting concept drift: {e}")
            return 0.0
    
    def _adjust_learning_rate(self, drift_score):
        """
        Adjust learning rate based on concept drift.
        
        Args:
            drift_score (float): Drift score (0-1)
        """
        try:
            # Get current optimizer
            optimizer = self.model.model.optimizer
            
            # Get current learning rate
            current_lr = float(tf.keras.backend.get_value(optimizer.learning_rate))
            
            # Calculate new learning rate based on drift score
            # Higher drift score = higher learning rate
            base_lr = self.config["online_learning"]["learning_rate"]
            max_lr = base_lr * 5  # Maximum 5x the base learning rate
            
            new_lr = base_lr + (max_lr - base_lr) * drift_score
            
            # Set new learning rate
            tf.keras.backend.set_value(optimizer.learning_rate, new_lr)
            
            logger.info(f"Adjusted learning rate from {current_lr:.6f} to {new_lr:.6f} (drift score: {drift_score:.4f})")
        
        except Exception as e:
            logger.error(f"Error adjusting learning rate: {e}")
    
    def _save_model_version(self):
        """
        Save a versioned copy of the model.
        """
        try:
            # Create version directory
            version = self.metrics["updates"]
            version_dir = os.path.join(self.output_dir, f"version_{version}")
            os.makedirs(version_dir, exist_ok=True)
            
            # Save model
            model_file = os.path.join(version_dir, f"model_{self.config['model']['model_type']}.h5")
            self.model.model.save(model_file)
            
            # Save tokenizer
            tokenizer_file = os.path.join(version_dir, "tokenizer.pkl")
            self.model._save_tokenizer(tokenizer_file)
            
            # Save category mapping
            mapping_file = os.path.join(version_dir, "category_mapping.json")
            with open(mapping_file, 'w') as f:
                json.dump(self.model.category_mapping, f, indent=2)
            
            # Add to version history
            self.metrics["model_versions"].append({
                "version": version,
                "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
                "accuracy": self.metrics["current_accuracy"],
                "samples_processed": self.metrics["samples_processed"],
                "path": version_dir
            })
            
            # Limit number of versions
            max_versions = self.config["online_learning"]["versioning"]["max_versions"]
            if len(self.metrics["model_versions"]) > max_versions:
                # Remove oldest versions (but keep at least one)
                versions_to_remove = len(self.metrics["model_versions"]) - max_versions
                for i in range(versions_to_remove):
                    old_version = self.metrics["model_versions"].pop(0)
                    logger.info(f"Removing old model version: {old_version['version']}")
                    # Note: We don't actually delete the files, just remove from the list
            
            logger.info(f"Saved model version {version}")
        
        except Exception as e:
            logger.error(f"Error saving model version: {e}")
    
    def _evaluate_model(self):
        """
        Evaluate the model on the current data.
        """
        try:
            # Skip if no data
            if not hasattr(self.model, "X_train") or self.model.X_train is None:
                return
            
            # Split data for evaluation
            from sklearn.model_selection import train_test_split
            
            X_train, X_eval, y_train, y_eval = train_test_split(
                self.model.X_train, self.model.y_train, 
                test_size=0.2, random_state=42
            )
            
            # Evaluate model
            evaluation = self.model.model.evaluate(X_eval, y_eval, verbose=0)
            
            # Get metrics
            metrics = {}
            for i, metric_name in enumerate(self.model.model.metrics_names):
                metrics[metric_name] = float(evaluation[i])
            
            # Make predictions for more detailed metrics
            y_pred = self.model.model.predict(X_eval)
            y_pred_classes = np.argmax(y_pred, axis=1)
            y_true_classes = np.argmax(y_eval, axis=1)
            
            # Calculate precision, recall, and F1 score
            from sklearn.metrics import precision_score, recall_score, f1_score
            
            metrics["precision"] = float(precision_score(y_true_classes, y_pred_classes, average='weighted'))
            metrics["recall"] = float(recall_score(y_true_classes, y_pred_classes, average='weighted'))
            metrics["f1"] = float(f1_score(y_true_classes, y_pred_classes, average='weighted'))
            
            # Update evaluation metrics
            self.metrics["evaluation"] = {
                "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
                "metrics": metrics
            }
            
            logger.info(f"Model evaluation: accuracy={metrics.get('accuracy', 0):.4f}, f1={metrics.get('f1', 0):.4f}")
        
        except Exception as e:
            logger.error(f"Error evaluating model: {e}")
    
    def _save_metrics(self):
        """
        Save metrics to file.
        """
        metrics_file = os.path.join(self.output_dir, "online_learning_metrics.json")
        
        try:
            with open(metrics_file, 'w') as f:
                json.dump(self.metrics, f, indent=2)
            
            logger.info(f"Saved metrics to {metrics_file}")
        except Exception as e:
            logger.error(f"Error saving metrics: {e}")
    
    def load_feedback_data(self, feedback_file="data/feedback.json"):
        """
        Load feedback data for online learning.
        
        Args:
            feedback_file (str): Path to feedback file
            
        Returns:
            list: List of samples
        """
        if not os.path.exists(feedback_file):
            logger.warning(f"Feedback file not found: {feedback_file}")
            return []
        
        try:
            with open(feedback_file, 'r') as f:
                feedback_data = json.load(f)
            
            samples = []
            for entry in feedback_data:
                sample = {
                    "text": entry.get("user_input"),
                    "category": entry.get("category", "unknown"),
                    "rating": entry.get("rating", 5)
                }
                
                if sample["text"]:
                    samples.append(sample)
            
            logger.info(f"Loaded {len(samples)} samples from {feedback_file}")
            return samples
        
        except Exception as e:
            logger.error(f"Error loading feedback data: {e}")
            return []

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Online learning for NoahAI")
    parser.add_argument("--config", help="Path to configuration file")
    parser.add_argument("--model-dir", default="data/model", help="Directory for model files")
    parser.add_argument("--output-dir", default="data/online_learning", help="Directory for output")
    parser.add_argument("--feedback", default="data/feedback.json", help="Path to feedback file")
    parser.add_argument("--update", action="store_true", help="Force model update")
    args = parser.parse_args()
    
    # Create online learning manager
    manager = OnlineLearningManager(
        config_file=args.config,
        model_dir=args.model_dir,
        output_dir=args.output_dir
    )
    
    # Load feedback data
    samples = manager.load_feedback_data(args.feedback)
    
    if samples:
        # Add samples to learning buffer
        manager.add_samples(samples)
    
    # Force update if requested
    if args.update:
        manager.update_model()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
