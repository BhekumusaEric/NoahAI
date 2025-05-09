#!/usr/bin/env python3
"""
Massive Training Manager for NoahAI

This script manages the massive automated training of NoahAI's deep learning model.
It supports:
- Distributed training across multiple machines
- Hyperparameter optimization
- Checkpointing and model versioning
- Training metrics tracking
- Automated evaluation
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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/massive_training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("massive_training")

# Try to import NoahAI modules
try:
    from models.deep_learning_model import DeepLearningModel
    from src.utils.learning_utils import LearningUtils
    NOAHAI_AVAILABLE = True
except ImportError:
    NOAHAI_AVAILABLE = False
    logger.error("NoahAI modules not available. Please run this script from the NoahAI directory.")
    sys.exit(1)

class MassiveTrainingManager:
    """
    Manager for massive automated training of NoahAI's deep learning model.
    """
    
    def __init__(self, config_file=None, model_dir="data/model", output_dir="data/training_output"):
        """
        Initialize the training manager.
        
        Args:
            config_file (str): Path to configuration file
            model_dir (str): Directory for model files
            output_dir (str): Directory for training output
        """
        self.model_dir = model_dir
        self.output_dir = output_dir
        
        # Create directories
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)
        
        # Load configuration
        self.config = self._load_config(config_file)
        
        # Initialize TensorFlow
        self._setup_tensorflow()
        
        # Initialize training metrics
        self.training_metrics = {
            "start_time": None,
            "end_time": None,
            "duration": 0,
            "epochs_completed": 0,
            "total_samples_processed": 0,
            "samples_per_second": 0,
            "loss_history": [],
            "accuracy_history": [],
            "validation_loss_history": [],
            "validation_accuracy_history": [],
            "best_model_epoch": 0,
            "best_model_accuracy": 0,
            "best_model_loss": float('inf')
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
            "training": {
                "batch_size": 32,
                "epochs": 20,
                "learning_rate": 0.001,
                "early_stopping_patience": 5,
                "validation_split": 0.2,
                "use_transfer_learning": True,
                "use_reinforcement_learning": True,
                "use_distributed_training": False
            },
            "distributed": {
                "strategy": "mirrored",  # mirrored, multi_worker, parameter_server
                "num_workers": 2,
                "worker_addresses": ["localhost:2222", "localhost:2223"],
                "parameter_server_addresses": ["localhost:2224"],
                "communication_options": {
                    "implementation": "ring",
                    "reduce_to_one_device": True
                }
            },
            "data": {
                "training_data_file": "data/massive_training_data.json",
                "max_sequence_length": 100,
                "max_words": 10000,
                "shuffle_buffer_size": 10000,
                "prefetch_buffer_size": tf.data.AUTOTUNE,
                "cache_dataset": True
            },
            "model": {
                "model_type": "advanced_lstm",  # simple_lstm, advanced_lstm, cnn_lstm, transfer_learning
                "embedding_dim": 128,
                "lstm_units": [64, 32],
                "dense_units": [64],
                "dropout_rate": 0.2
            },
            "optimization": {
                "enable_auto_optimization": False,
                "num_trials": 10,
                "parameters": {
                    "learning_rate": [0.0001, 0.001, 0.01],
                    "batch_size": [16, 32, 64],
                    "lstm_units": [[32, 16], [64, 32], [128, 64]],
                    "dropout_rate": [0.1, 0.2, 0.3, 0.4]
                }
            },
            "checkpointing": {
                "save_best_only": True,
                "save_frequency": 1,  # epochs
                "keep_checkpoint_max": 3
            },
            "evaluation": {
                "evaluate_after_training": True,
                "test_split": 0.1,
                "metrics": ["accuracy", "precision", "recall", "f1"]
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
    
    def _setup_tensorflow(self):
        """
        Set up TensorFlow for training.
        """
        # Enable memory growth to avoid allocating all GPU memory at once
        physical_devices = tf.config.list_physical_devices('GPU')
        if physical_devices:
            for device in physical_devices:
                try:
                    tf.config.experimental.set_memory_growth(device, True)
                    logger.info(f"Enabled memory growth for GPU: {device}")
                except Exception as e:
                    logger.warning(f"Error setting memory growth: {e}")
        
        # Set up distributed strategy if enabled
        self.strategy = None
        if self.config["training"]["use_distributed_training"]:
            strategy_type = self.config["distributed"]["strategy"]
            
            if strategy_type == "mirrored":
                self.strategy = tf.distribute.MirroredStrategy()
                logger.info(f"Using MirroredStrategy with {self.strategy.num_replicas_in_sync} replicas")
            
            elif strategy_type == "multi_worker":
                # Set up TF_CONFIG for multi-worker training
                worker_addresses = self.config["distributed"]["worker_addresses"]
                task_index = 0  # Assume this is the chief worker
                
                tf_config = {
                    "cluster": {
                        "worker": worker_addresses
                    },
                    "task": {"type": "worker", "index": task_index}
                }
                os.environ["TF_CONFIG"] = json.dumps(tf_config)
                
                self.strategy = tf.distribute.MultiWorkerMirroredStrategy()
                logger.info(f"Using MultiWorkerMirroredStrategy with {self.strategy.num_replicas_in_sync} replicas")
            
            elif strategy_type == "parameter_server":
                # Set up TF_CONFIG for parameter server strategy
                worker_addresses = self.config["distributed"]["worker_addresses"]
                ps_addresses = self.config["distributed"]["parameter_server_addresses"]
                task_index = 0  # Assume this is the chief worker
                
                tf_config = {
                    "cluster": {
                        "worker": worker_addresses,
                        "ps": ps_addresses
                    },
                    "task": {"type": "worker", "index": task_index}
                }
                os.environ["TF_CONFIG"] = json.dumps(tf_config)
                
                self.strategy = tf.distribute.experimental.ParameterServerStrategy()
                logger.info("Using ParameterServerStrategy")
            
            else:
                logger.warning(f"Unknown strategy type: {strategy_type}. Using default.")
                self.strategy = tf.distribute.get_strategy()
    
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
    
    def create_tf_dataset(self, texts, categories, ratings):
        """
        Create a TensorFlow dataset for training.
        
        Args:
            texts (list): List of input texts
            categories (list): List of categories
            ratings (list): List of ratings
            
        Returns:
            tf.data.Dataset: TensorFlow dataset
        """
        # Get unique categories
        unique_categories = sorted(set(categories))
        category_to_index = {cat: i for i, cat in enumerate(unique_categories)}
        
        # Convert categories to indices
        category_indices = [category_to_index[cat] for cat in categories]
        
        # Create dataset
        dataset = tf.data.Dataset.from_tensor_slices((
            texts,
            tf.keras.utils.to_categorical(category_indices, num_classes=len(unique_categories)),
            ratings
        ))
        
        # Shuffle, batch, and prefetch
        buffer_size = min(len(texts), self.config["data"]["shuffle_buffer_size"])
        batch_size = self.config["training"]["batch_size"]
        
        dataset = dataset.shuffle(buffer_size)
        
        if self.config["data"]["cache_dataset"]:
            dataset = dataset.cache()
        
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(self.config["data"]["prefetch_buffer_size"])
        
        return dataset, unique_categories
    
    def train_model(self, model=None, texts=None, categories=None, ratings=None):
        """
        Train the model with the provided data.
        
        Args:
            model (DeepLearningModel): Model to train
            texts (list): List of input texts
            categories (list): List of categories
            ratings (list): List of ratings
            
        Returns:
            dict: Training metrics
        """
        # Start timing
        self.training_metrics["start_time"] = datetime.now()
        
        # Load data if not provided
        if texts is None or categories is None or ratings is None:
            texts, categories, ratings = self.load_training_data()
            
            if texts is None:
                logger.error("Failed to load training data")
                return None
        
        # Create or load model
        if model is None:
            model = self._create_model()
        
        # Train the model
        training_config = self.config["training"]
        
        try:
            history = model.train(
                epochs=training_config["epochs"],
                batch_size=training_config["batch_size"],
                use_early_stopping=True,
                use_transfer_learning=training_config["use_transfer_learning"]
            )
            
            # Update training metrics
            if history:
                self.training_metrics["loss_history"] = history.get("loss", [])
                self.training_metrics["accuracy_history"] = history.get("accuracy", [])
                self.training_metrics["validation_loss_history"] = history.get("val_loss", [])
                self.training_metrics["validation_accuracy_history"] = history.get("val_accuracy", [])
                self.training_metrics["epochs_completed"] = len(history.get("loss", []))
                
                # Find best model
                val_acc = history.get("val_accuracy", [])
                if val_acc:
                    best_epoch = np.argmax(val_acc)
                    self.training_metrics["best_model_epoch"] = best_epoch
                    self.training_metrics["best_model_accuracy"] = val_acc[best_epoch]
                    self.training_metrics["best_model_loss"] = history.get("val_loss", [])[best_epoch]
            
            # End timing
            self.training_metrics["end_time"] = datetime.now()
            self.training_metrics["duration"] = (self.training_metrics["end_time"] - self.training_metrics["start_time"]).total_seconds()
            
            # Save training metrics
            self._save_training_metrics()
            
            return self.training_metrics
        
        except Exception as e:
            logger.error(f"Error during training: {e}")
            return None
    
    def _create_model(self):
        """
        Create a new model instance.
        
        Returns:
            DeepLearningModel: New model instance
        """
        model_config = self.config["model"]
        
        model = DeepLearningModel(
            model_dir=self.model_dir,
            responses_file="data/responses.json",
            max_words=self.config["data"]["max_words"],
            max_sequence_length=self.config["data"]["max_sequence_length"],
            use_advanced_nlp=True,
            model_type=model_config["model_type"],
            use_reinforcement_learning=self.config["training"]["use_reinforcement_learning"]
        )
        
        return model
    
    def _save_training_metrics(self):
        """
        Save training metrics to file.
        """
        metrics_file = os.path.join(self.output_dir, "training_metrics.json")
        
        try:
            with open(metrics_file, 'w') as f:
                json.dump(self.training_metrics, f, indent=2)
            
            logger.info(f"Saved training metrics to {metrics_file}")
        except Exception as e:
            logger.error(f"Error saving training metrics: {e}")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Massive automated training for NoahAI")
    parser.add_argument("--config", help="Path to configuration file")
    parser.add_argument("--data", help="Path to training data file")
    parser.add_argument("--model-dir", default="data/model", help="Directory for model files")
    parser.add_argument("--output-dir", default="data/training_output", help="Directory for training output")
    args = parser.parse_args()
    
    # Create training manager
    manager = MassiveTrainingManager(
        config_file=args.config,
        model_dir=args.model_dir,
        output_dir=args.output_dir
    )
    
    # Train model
    manager.train_model()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
