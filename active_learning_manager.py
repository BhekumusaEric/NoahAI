#!/usr/bin/env python3
"""
Active Learning Manager for NoahAI

This script implements active learning for NoahAI's deep learning model.
It identifies the most informative examples for labeling, reducing the amount
of labeled data needed for effective training.

Features:
- Uncertainty sampling
- Diversity sampling
- Query by committee
- Expected model change
- Batch mode active learning
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
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/active_learning.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("active_learning")

# Try to import NoahAI modules
try:
    from models.deep_learning_model import DeepLearningModel
    from src.utils.learning_utils import LearningUtils
    NOAHAI_AVAILABLE = True
except ImportError:
    NOAHAI_AVAILABLE = False
    logger.error("NoahAI modules not available. Please run this script from the NoahAI directory.")
    sys.exit(1)

class ActiveLearningManager:
    """
    Manager for active learning of NoahAI's deep learning model.
    """
    
    def __init__(self, config_file=None, model_dir="data/model", output_dir="data/active_learning"):
        """
        Initialize the active learning manager.
        
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
        
        # Initialize metrics
        self.metrics = {
            "queries": 0,
            "samples_labeled": 0,
            "current_accuracy": 0.0,
            "accuracy_history": [],
            "last_query_time": None,
            "query_history": []
        }
        
        # Initialize unlabeled pool
        self.unlabeled_pool = []
        
        # Initialize labeled pool
        self.labeled_pool = []
    
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
            "active_learning": {
                "strategy": "uncertainty",  # uncertainty, diversity, committee, expected_change, batch
                "batch_size": 10,  # Number of examples to query at once
                "uncertainty_threshold": 0.3,  # Minimum uncertainty to consider an example informative
                "diversity_weight": 0.5,  # Weight for diversity vs uncertainty (0-1)
                "committee_size": 3,  # Number of models in the committee
                "max_pool_size": 10000,  # Maximum size of the unlabeled pool
                "min_examples_per_category": 5,  # Minimum number of examples per category
                "retraining": {
                    "enabled": True,
                    "frequency": 50,  # Retrain after this many new labeled examples
                    "epochs": 5,
                    "batch_size": 32,
                    "validation_split": 0.2
                }
            },
            "evaluation": {
                "enabled": True,
                "frequency": 5,  # Evaluate after this many queries
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
    
    def load_unlabeled_data(self, data_file):
        """
        Load unlabeled data into the pool.
        
        Args:
            data_file (str): Path to data file
            
        Returns:
            int: Number of examples loaded
        """
        if not os.path.exists(data_file):
            logger.error(f"Data file not found: {data_file}")
            return 0
        
        try:
            with open(data_file, 'r') as f:
                data = json.load(f)
            
            # Extract texts
            for i, entry in enumerate(data):
                if isinstance(entry, dict):
                    text = entry.get("text") or entry.get("user_input")
                    
                    if text:
                        self.unlabeled_pool.append({
                            "id": len(self.unlabeled_pool) + i,
                            "text": text,
                            "embedding": None,  # Will be computed when needed
                            "uncertainty": None,  # Will be computed when needed
                            "diversity": None  # Will be computed when needed
                        })
                elif isinstance(entry, str):
                    self.unlabeled_pool.append({
                        "id": len(self.unlabeled_pool) + i,
                        "text": entry,
                        "embedding": None,
                        "uncertainty": None,
                        "diversity": None
                    })
            
            # Limit pool size
            max_pool_size = self.config["active_learning"]["max_pool_size"]
            if len(self.unlabeled_pool) > max_pool_size:
                self.unlabeled_pool = self.unlabeled_pool[:max_pool_size]
            
            logger.info(f"Loaded {len(self.unlabeled_pool)} unlabeled examples from {data_file}")
            return len(self.unlabeled_pool)
        
        except Exception as e:
            logger.error(f"Error loading unlabeled data: {e}")
            return 0
    
    def load_labeled_data(self, data_file):
        """
        Load labeled data into the pool.
        
        Args:
            data_file (str): Path to data file
            
        Returns:
            int: Number of examples loaded
        """
        if not os.path.exists(data_file):
            logger.error(f"Data file not found: {data_file}")
            return 0
        
        try:
            with open(data_file, 'r') as f:
                data = json.load(f)
            
            # Extract texts and labels
            for i, entry in enumerate(data):
                if isinstance(entry, dict):
                    text = entry.get("text") or entry.get("user_input")
                    category = entry.get("category") or entry.get("label")
                    
                    if text and category:
                        self.labeled_pool.append({
                            "id": len(self.labeled_pool) + i,
                            "text": text,
                            "category": category,
                            "embedding": None  # Will be computed when needed
                        })
            
            logger.info(f"Loaded {len(self.labeled_pool)} labeled examples from {data_file}")
            
            # Update model with labeled data
            if self.labeled_pool and self.config["active_learning"]["retraining"]["enabled"]:
                self._retrain_model()
            
            return len(self.labeled_pool)
        
        except Exception as e:
            logger.error(f"Error loading labeled data: {e}")
            return 0
    
    def compute_embeddings(self, texts):
        """
        Compute embeddings for a list of texts.
        
        Args:
            texts (list): List of texts
            
        Returns:
            np.ndarray: Array of embeddings
        """
        try:
            # Preprocess texts
            sequences = self.model.tokenizer.texts_to_sequences(texts)
            padded_sequences = tf.keras.preprocessing.sequence.pad_sequences(
                sequences, maxlen=self.config["model"]["max_sequence_length"]
            )
            
            # Get embedding layer
            embedding_layer = None
            for layer in self.model.model.layers:
                if isinstance(layer, tf.keras.layers.Embedding):
                    embedding_layer = layer
                    break
            
            if embedding_layer is None:
                logger.error("Embedding layer not found in model")
                return None
            
            # Create a model that outputs embeddings
            embedding_model = tf.keras.Model(
                inputs=self.model.model.inputs,
                outputs=embedding_layer.output
            )
            
            # Get embeddings
            embeddings = embedding_model.predict(padded_sequences)
            
            # Average over sequence length to get a single vector per text
            embeddings = np.mean(embeddings, axis=1)
            
            return embeddings
        
        except Exception as e:
            logger.error(f"Error computing embeddings: {e}")
            return None
    
    def compute_uncertainty(self, texts):
        """
        Compute uncertainty scores for a list of texts.
        
        Args:
            texts (list): List of texts
            
        Returns:
            np.ndarray: Array of uncertainty scores
        """
        try:
            # Preprocess texts
            X = self.model.preprocess_text(texts)
            
            # Get predictions
            predictions = self.model.model.predict(X)
            
            # Compute uncertainty (entropy)
            uncertainty = -np.sum(predictions * np.log(predictions + 1e-10), axis=1)
            
            # Normalize to [0, 1]
            uncertainty = uncertainty / np.log(predictions.shape[1])
            
            return uncertainty
        
        except Exception as e:
            logger.error(f"Error computing uncertainty: {e}")
            return None
    
    def compute_diversity(self, embeddings, labeled_embeddings=None):
        """
        Compute diversity scores for a list of embeddings.
        
        Args:
            embeddings (np.ndarray): Array of embeddings
            labeled_embeddings (np.ndarray, optional): Array of embeddings for labeled examples
            
        Returns:
            np.ndarray: Array of diversity scores
        """
        try:
            if labeled_embeddings is None or len(labeled_embeddings) == 0:
                # If no labeled examples, use K-means clustering
                n_clusters = min(10, len(embeddings))
                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                clusters = kmeans.fit_predict(embeddings)
                
                # Count examples in each cluster
                cluster_counts = np.bincount(clusters, minlength=n_clusters)
                
                # Compute diversity score (higher for examples in smaller clusters)
                diversity = 1.0 - (cluster_counts[clusters] / np.max(cluster_counts))
                
                return diversity
            
            else:
                # Compute similarity to labeled examples
                similarity = cosine_similarity(embeddings, labeled_embeddings)
                
                # Take the maximum similarity for each example
                max_similarity = np.max(similarity, axis=1)
                
                # Compute diversity score (higher for examples less similar to labeled examples)
                diversity = 1.0 - max_similarity
                
                return diversity
        
        except Exception as e:
            logger.error(f"Error computing diversity: {e}")
            return None
    
    def select_examples(self, strategy=None, batch_size=None):
        """
        Select the most informative examples for labeling.
        
        Args:
            strategy (str, optional): Selection strategy
            batch_size (int, optional): Number of examples to select
            
        Returns:
            list: List of selected examples
        """
        if len(self.unlabeled_pool) == 0:
            logger.warning("No unlabeled examples available")
            return []
        
        # Use configured values if not provided
        strategy = strategy or self.config["active_learning"]["strategy"]
        batch_size = batch_size or self.config["active_learning"]["batch_size"]
        
        # Limit batch size to available examples
        batch_size = min(batch_size, len(self.unlabeled_pool))
        
        logger.info(f"Selecting {batch_size} examples using {strategy} strategy")
        
        # Get texts from unlabeled pool
        texts = [example["text"] for example in self.unlabeled_pool]
        
        # Compute embeddings if needed
        if strategy in ["diversity", "batch"]:
            # Compute embeddings for unlabeled examples
            unlabeled_embeddings = self.compute_embeddings(texts)
            
            # Update embeddings in unlabeled pool
            for i, embedding in enumerate(unlabeled_embeddings):
                self.unlabeled_pool[i]["embedding"] = embedding
            
            # Compute embeddings for labeled examples if available
            labeled_embeddings = None
            if self.labeled_pool:
                labeled_texts = [example["text"] for example in self.labeled_pool]
                labeled_embeddings = self.compute_embeddings(labeled_texts)
                
                # Update embeddings in labeled pool
                for i, embedding in enumerate(labeled_embeddings):
                    self.labeled_pool[i]["embedding"] = embedding
        
        # Compute uncertainty if needed
        if strategy in ["uncertainty", "batch"]:
            uncertainty = self.compute_uncertainty(texts)
            
            # Update uncertainty in unlabeled pool
            for i, score in enumerate(uncertainty):
                self.unlabeled_pool[i]["uncertainty"] = score
        
        # Select examples based on strategy
        if strategy == "uncertainty":
            # Sort by uncertainty (descending)
            sorted_indices = np.argsort([-example["uncertainty"] for example in self.unlabeled_pool])
            selected_indices = sorted_indices[:batch_size]
        
        elif strategy == "diversity":
            # Compute diversity scores
            diversity = self.compute_diversity(
                np.array([example["embedding"] for example in self.unlabeled_pool]),
                np.array([example["embedding"] for example in self.labeled_pool]) if self.labeled_pool else None
            )
            
            # Update diversity in unlabeled pool
            for i, score in enumerate(diversity):
                self.unlabeled_pool[i]["diversity"] = score
            
            # Sort by diversity (descending)
            sorted_indices = np.argsort([-example["diversity"] for example in self.unlabeled_pool])
            selected_indices = sorted_indices[:batch_size]
        
        elif strategy == "batch":
            # Combine uncertainty and diversity
            uncertainty = np.array([example["uncertainty"] for example in self.unlabeled_pool])
            
            # Compute diversity scores
            diversity = self.compute_diversity(
                np.array([example["embedding"] for example in self.unlabeled_pool]),
                np.array([example["embedding"] for example in self.labeled_pool]) if self.labeled_pool else None
            )
            
            # Update diversity in unlabeled pool
            for i, score in enumerate(diversity):
                self.unlabeled_pool[i]["diversity"] = score
            
            # Normalize scores
            uncertainty = (uncertainty - np.min(uncertainty)) / (np.max(uncertainty) - np.min(uncertainty) + 1e-10)
            diversity = (diversity - np.min(diversity)) / (np.max(diversity) - np.min(diversity) + 1e-10)
            
            # Combine scores
            diversity_weight = self.config["active_learning"]["diversity_weight"]
            combined_scores = (1 - diversity_weight) * uncertainty + diversity_weight * diversity
            
            # Sort by combined score (descending)
            sorted_indices = np.argsort(-combined_scores)
            selected_indices = sorted_indices[:batch_size]
        
        elif strategy == "committee":
            # Not implemented yet
            logger.warning("Committee strategy not implemented yet, falling back to uncertainty")
            
            # Sort by uncertainty (descending)
            sorted_indices = np.argsort([-example["uncertainty"] for example in self.unlabeled_pool])
            selected_indices = sorted_indices[:batch_size]
        
        elif strategy == "expected_change":
            # Not implemented yet
            logger.warning("Expected change strategy not implemented yet, falling back to uncertainty")
            
            # Sort by uncertainty (descending)
            sorted_indices = np.argsort([-example["uncertainty"] for example in self.unlabeled_pool])
            selected_indices = sorted_indices[:batch_size]
        
        else:
            logger.warning(f"Unknown strategy: {strategy}, falling back to uncertainty")
            
            # Sort by uncertainty (descending)
            sorted_indices = np.argsort([-example["uncertainty"] for example in self.unlabeled_pool])
            selected_indices = sorted_indices[:batch_size]
        
        # Get selected examples
        selected_examples = [self.unlabeled_pool[i] for i in selected_indices]
        
        # Update metrics
        self.metrics["queries"] += 1
        self.metrics["last_query_time"] = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
        
        # Add query to history
        self.metrics["query_history"].append({
            "query_id": self.metrics["queries"],
            "timestamp": self.metrics["last_query_time"],
            "strategy": strategy,
            "batch_size": batch_size,
            "selected_examples": [example["id"] for example in selected_examples]
        })
        
        # Save metrics
        self._save_metrics()
        
        return selected_examples
    
    def add_labeled_example(self, example_id, text, category):
        """
        Add a labeled example to the pool.
        
        Args:
            example_id (int): Example ID
            text (str): Text
            category (str): Category
            
        Returns:
            bool: Whether the example was added
        """
        # Check if example is already labeled
        for example in self.labeled_pool:
            if example["id"] == example_id:
                logger.warning(f"Example {example_id} is already labeled")
                return False
        
        # Add to labeled pool
        self.labeled_pool.append({
            "id": example_id,
            "text": text,
            "category": category,
            "embedding": None  # Will be computed when needed
        })
        
        # Remove from unlabeled pool
        self.unlabeled_pool = [example for example in self.unlabeled_pool if example["id"] != example_id]
        
        # Update metrics
        self.metrics["samples_labeled"] += 1
        
        # Check if retraining is needed
        if (self.config["active_learning"]["retraining"]["enabled"] and 
            self.metrics["samples_labeled"] % self.config["active_learning"]["retraining"]["frequency"] == 0):
            self._retrain_model()
        
        # Check if evaluation is needed
        if (self.config["evaluation"]["enabled"] and 
            self.metrics["queries"] % self.config["evaluation"]["frequency"] == 0):
            self._evaluate_model()
        
        # Save metrics
        self._save_metrics()
        
        return True
    
    def add_labeled_examples(self, examples):
        """
        Add multiple labeled examples to the pool.
        
        Args:
            examples (list): List of examples, each with 'id', 'text', and 'category'
            
        Returns:
            int: Number of examples added
        """
        added_count = 0
        
        for example in examples:
            if self.add_labeled_example(example["id"], example["text"], example["category"]):
                added_count += 1
        
        return added_count
    
    def _retrain_model(self):
        """
        Retrain the model with the current labeled pool.
        
        Returns:
            dict: Training metrics
        """
        if len(self.labeled_pool) < self.config["active_learning"]["min_examples_per_category"]:
            logger.info(f"Not enough labeled examples for retraining. Have {len(self.labeled_pool)}, need at least {self.config['active_learning']['min_examples_per_category']} per category")
            return None
        
        logger.info(f"Retraining model with {len(self.labeled_pool)} labeled examples")
        
        # Extract data from labeled pool
        texts = [example["text"] for example in self.labeled_pool]
        categories = [example["category"] for example in self.labeled_pool]
        
        # Prepare data for training
        self.model.prepare_training_data(texts, categories)
        
        # Set up training parameters
        retraining_config = self.config["active_learning"]["retraining"]
        batch_size = retraining_config["batch_size"]
        epochs = retraining_config["epochs"]
        validation_split = retraining_config["validation_split"]
        
        # Train the model
        try:
            history = self.model.model.fit(
                self.model.X_train,
                self.model.y_train,
                batch_size=batch_size,
                epochs=epochs,
                validation_split=validation_split,
                verbose=1
            )
            
            # Update metrics
            if history.history.get("val_accuracy"):
                self.metrics["current_accuracy"] = history.history["val_accuracy"][-1]
                self.metrics["accuracy_history"].append(self.metrics["current_accuracy"])
            
            # Save model
            self.model._save_model()
            
            logger.info(f"Model retrained successfully (accuracy: {self.metrics['current_accuracy']:.4f})")
            
            return {
                "accuracy": self.metrics["current_accuracy"],
                "epochs": epochs,
                "samples_used": len(self.labeled_pool)
            }
        
        except Exception as e:
            logger.error(f"Error retraining model: {e}")
            return None
    
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
        metrics_file = os.path.join(self.output_dir, "active_learning_metrics.json")
        
        try:
            with open(metrics_file, 'w') as f:
                json.dump(self.metrics, f, indent=2)
            
            logger.info(f"Saved metrics to {metrics_file}")
        except Exception as e:
            logger.error(f"Error saving metrics: {e}")
    
    def save_pools(self):
        """
        Save the current unlabeled and labeled pools to files.
        
        Returns:
            tuple: (unlabeled_file, labeled_file)
        """
        unlabeled_file = os.path.join(self.output_dir, "unlabeled_pool.json")
        labeled_file = os.path.join(self.output_dir, "labeled_pool.json")
        
        try:
            # Save unlabeled pool
            with open(unlabeled_file, 'w') as f:
                # Remove embeddings to save space
                simplified_pool = [{k: v for k, v in example.items() if k != "embedding"} 
                                  for example in self.unlabeled_pool]
                json.dump(simplified_pool, f, indent=2)
            
            # Save labeled pool
            with open(labeled_file, 'w') as f:
                # Remove embeddings to save space
                simplified_pool = [{k: v for k, v in example.items() if k != "embedding"} 
                                  for example in self.labeled_pool]
                json.dump(simplified_pool, f, indent=2)
            
            logger.info(f"Saved pools to {unlabeled_file} and {labeled_file}")
            return unlabeled_file, labeled_file
        
        except Exception as e:
            logger.error(f"Error saving pools: {e}")
            return None, None

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Active learning for NoahAI")
    parser.add_argument("--config", help="Path to configuration file")
    parser.add_argument("--model-dir", default="data/model", help="Directory for model files")
    parser.add_argument("--output-dir", default="data/active_learning", help="Directory for output")
    parser.add_argument("--unlabeled", help="Path to unlabeled data file")
    parser.add_argument("--labeled", help="Path to labeled data file")
    parser.add_argument("--select", action="store_true", help="Select examples for labeling")
    parser.add_argument("--batch-size", type=int, help="Number of examples to select")
    parser.add_argument("--strategy", help="Selection strategy")
    args = parser.parse_args()
    
    # Create active learning manager
    manager = ActiveLearningManager(
        config_file=args.config,
        model_dir=args.model_dir,
        output_dir=args.output_dir
    )
    
    # Load data if provided
    if args.unlabeled:
        manager.load_unlabeled_data(args.unlabeled)
    
    if args.labeled:
        manager.load_labeled_data(args.labeled)
    
    # Select examples if requested
    if args.select:
        selected_examples = manager.select_examples(
            strategy=args.strategy,
            batch_size=args.batch_size
        )
        
        # Print selected examples
        print(f"Selected {len(selected_examples)} examples:")
        for i, example in enumerate(selected_examples):
            print(f"{i+1}. ID: {example['id']}, Text: {example['text'][:50]}...")
    
    # Save pools
    manager.save_pools()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
