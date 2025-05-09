#!/usr/bin/env python3
"""
Active Learning Feedback Loop for NoahAI

This script implements a tight feedback loop between active learning and model training.
It continuously selects the most informative examples for labeling, updates the model,
and repeats the process to maximize learning efficiency.

Features:
- Continuous active learning cycle
- Adaptive selection strategies
- Performance-based strategy switching
- Learning curve tracking
- Stopping criteria based on performance plateaus
- Batch optimization
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
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/active_learning_feedback.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("active_learning_feedback")

# Try to import NoahAI modules
try:
    from models.deep_learning_model import DeepLearningModel
    from src.utils.learning_utils import LearningUtils
    from automated_labeling_manager import AutomatedLabelingManager
    from active_learning_manager import ActiveLearningManager
    NOAHAI_AVAILABLE = True
except ImportError:
    NOAHAI_AVAILABLE = False
    logger.error("NoahAI modules not available. Please run this script from the NoahAI directory.")
    sys.exit(1)

class ActiveLearningFeedbackLoop:
    """
    Implements a tight feedback loop between active learning and model training.
    """
    
    def __init__(self, config_file=None, model_dir="data/model", output_dir="data/active_learning_feedback"):
        """
        Initialize the active learning feedback loop.
        
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
        
        # Initialize active learning manager
        self.active_learning_manager = ActiveLearningManager(
            config_file=config_file,
            model_dir=model_dir,
            output_dir=os.path.join(output_dir, "active_learning")
        )
        
        # Initialize automated labeling manager
        self.labeling_manager = AutomatedLabelingManager(
            config_file=config_file,
            model_dir=model_dir,
            output_dir=os.path.join(output_dir, "automated_labeling")
        )
        
        # Initialize feedback loop metrics
        self.metrics = {
            "cycles": 0,
            "total_examples_labeled": 0,
            "total_examples_processed": 0,
            "performance_history": [],
            "strategy_history": [],
            "batch_sizes": [],
            "learning_curve": {
                "examples": [],
                "accuracy": [],
                "precision": [],
                "recall": [],
                "f1": []
            },
            "start_time": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
            "end_time": None,
            "duration": 0
        }
        
        # Initialize pools
        self.unlabeled_pool = []
        self.labeled_pool = []
        self.validation_pool = []
    
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
            "feedback_loop": {
                "max_cycles": 10,
                "initial_batch_size": 100,
                "max_batch_size": 500,
                "adaptive_batch_size": True,
                "batch_growth_rate": 1.5,
                "initial_strategy": "uncertainty",
                "strategy_switching": True,
                "performance_plateau_threshold": 0.005,
                "plateau_patience": 3,
                "early_stopping": True,
                "target_accuracy": 0.95,
                "validation_split": 0.2,
                "human_in_the_loop": True,
                "auto_labeling_threshold": 0.9
            },
            "strategies": {
                "uncertainty": {
                    "enabled": True,
                    "weight": 1.0
                },
                "diversity": {
                    "enabled": True,
                    "weight": 0.7
                },
                "expected_model_change": {
                    "enabled": True,
                    "weight": 0.8
                },
                "expected_error_reduction": {
                    "enabled": True,
                    "weight": 0.9
                },
                "density_weighted": {
                    "enabled": True,
                    "weight": 0.6
                }
            },
            "training": {
                "epochs": 5,
                "batch_size": 32,
                "learning_rate": 0.001,
                "validation_split": 0.2
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
    
    def load_data(self, unlabeled_file, labeled_file=None, validation_file=None):
        """
        Load data for the feedback loop.
        
        Args:
            unlabeled_file (str): Path to unlabeled data file
            labeled_file (str, optional): Path to labeled data file
            validation_file (str, optional): Path to validation data file
            
        Returns:
            tuple: (num_unlabeled, num_labeled, num_validation)
        """
        # Load unlabeled data
        self.unlabeled_pool = self.active_learning_manager.load_unlabeled_data(unlabeled_file)
        
        # Load labeled data if provided
        if labeled_file and os.path.exists(labeled_file):
            self.labeled_pool = self.active_learning_manager.load_labeled_data(labeled_file)
        
        # Load validation data if provided
        if validation_file and os.path.exists(validation_file):
            try:
                with open(validation_file, 'r') as f:
                    self.validation_pool = json.load(f)
            except Exception as e:
                logger.error(f"Error loading validation data: {e}")
        
        # If no validation data provided, create from labeled data
        elif self.labeled_pool and self.config["feedback_loop"]["validation_split"] > 0:
            # Split labeled data into training and validation
            from sklearn.model_selection import train_test_split
            
            train_pool, validation_pool = train_test_split(
                self.labeled_pool,
                test_size=self.config["feedback_loop"]["validation_split"],
                random_state=42
            )
            
            self.labeled_pool = train_pool
            self.validation_pool = validation_pool
        
        return len(self.unlabeled_pool), len(self.labeled_pool), len(self.validation_pool)
    
    def run_feedback_loop(self):
        """
        Run the active learning feedback loop.
        
        Returns:
            dict: Feedback loop metrics
        """
        if not self.unlabeled_pool:
            logger.error("No unlabeled data available")
            return self.metrics
        
        # Initialize loop parameters
        max_cycles = self.config["feedback_loop"]["max_cycles"]
        batch_size = self.config["feedback_loop"]["initial_batch_size"]
        strategy = self.config["feedback_loop"]["initial_strategy"]
        performance_history = []
        plateau_counter = 0
        
        logger.info(f"Starting active learning feedback loop with {len(self.unlabeled_pool)} unlabeled examples")
        logger.info(f"Initial batch size: {batch_size}, Initial strategy: {strategy}")
        
        # Run feedback loop
        for cycle in range(1, max_cycles + 1):
            logger.info(f"Starting cycle {cycle}/{max_cycles}")
            
            # 1. Select examples for labeling
            selected_examples = self._select_examples(strategy, batch_size)
            
            if not selected_examples:
                logger.warning("No examples selected for labeling. Stopping loop.")
                break
            
            # 2. Label examples
            if self.config["feedback_loop"]["human_in_the_loop"]:
                # Use automated labeling with human verification for low-confidence examples
                high_confidence, low_confidence = self.labeling_manager.label_data(selected_examples)
                
                # Add high-confidence examples to labeled pool
                self.labeled_pool.extend(high_confidence)
                
                # For low-confidence examples, we would normally use human verification
                # For this simulation, we'll just use the model's prediction
                self.labeled_pool.extend(low_confidence)
            else:
                # Use fully automated labeling
                labeled_examples = []
                for example in selected_examples:
                    # Preprocess text
                    X = self.model.preprocess_text([example["text"]])
                    
                    # Get prediction
                    prediction = self.model.model.predict(X)[0]
                    
                    # Get predicted category
                    category_idx = np.argmax(prediction)
                    category = list(self.model.category_mapping.keys())[category_idx]
                    
                    # Create labeled example
                    labeled_example = example.copy()
                    labeled_example["category"] = category
                    labeled_examples.append(labeled_example)
                
                # Add labeled examples to labeled pool
                self.labeled_pool.extend(labeled_examples)
            
            # 3. Update model
            performance = self._update_model()
            
            # 4. Update metrics
            self.metrics["cycles"] = cycle
            self.metrics["total_examples_labeled"] += len(selected_examples)
            self.metrics["performance_history"].append(performance)
            self.metrics["strategy_history"].append(strategy)
            self.metrics["batch_sizes"].append(batch_size)
            
            # Update learning curve
            self.metrics["learning_curve"]["examples"].append(self.metrics["total_examples_labeled"])
            self.metrics["learning_curve"]["accuracy"].append(performance["accuracy"])
            self.metrics["learning_curve"]["precision"].append(performance["precision"])
            self.metrics["learning_curve"]["recall"].append(performance["recall"])
            self.metrics["learning_curve"]["f1"].append(performance["f1"])
            
            # 5. Check stopping criteria
            if self.config["feedback_loop"]["early_stopping"]:
                # Check if target accuracy reached
                if performance["accuracy"] >= self.config["feedback_loop"]["target_accuracy"]:
                    logger.info(f"Target accuracy {self.config['feedback_loop']['target_accuracy']} reached. Stopping loop.")
                    break
                
                # Check for performance plateau
                if len(performance_history) > 0:
                    improvement = performance["accuracy"] - performance_history[-1]["accuracy"]
                    
                    if improvement < self.config["feedback_loop"]["performance_plateau_threshold"]:
                        plateau_counter += 1
                    else:
                        plateau_counter = 0
                    
                    if plateau_counter >= self.config["feedback_loop"]["plateau_patience"]:
                        logger.info(f"Performance plateau detected for {plateau_counter} cycles. Stopping loop.")
                        break
            
            # 6. Adapt batch size if enabled
            if self.config["feedback_loop"]["adaptive_batch_size"]:
                # Increase batch size if performance is improving
                if len(performance_history) > 0 and performance["accuracy"] > performance_history[-1]["accuracy"]:
                    batch_size = min(
                        int(batch_size * self.config["feedback_loop"]["batch_growth_rate"]),
                        self.config["feedback_loop"]["max_batch_size"]
                    )
                    logger.info(f"Increased batch size to {batch_size}")
            
            # 7. Switch strategy if enabled
            if self.config["feedback_loop"]["strategy_switching"] and len(performance_history) > 0:
                strategy = self._select_best_strategy(performance, performance_history[-1])
            
            # Save current performance for next cycle
            performance_history.append(performance)
            
            # Remove selected examples from unlabeled pool
            selected_ids = [example["id"] for example in selected_examples]
            self.unlabeled_pool = [example for example in self.unlabeled_pool if example["id"] not in selected_ids]
            
            # Check if we've run out of unlabeled examples
            if not self.unlabeled_pool:
                logger.info("No more unlabeled examples available. Stopping loop.")
                break
        
        # Update final metrics
        self.metrics["end_time"] = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
        start_time = datetime.strptime(self.metrics["start_time"], "%Y-%m-%dT%H:%M:%S")
        end_time = datetime.strptime(self.metrics["end_time"], "%Y-%m-%dT%H:%M:%S")
        self.metrics["duration"] = (end_time - start_time).total_seconds()
        
        # Save metrics and learning curve
        self._save_metrics()
        self._plot_learning_curve()
        
        return self.metrics
    
    def _select_examples(self, strategy, batch_size):
        """
        Select examples for labeling using the specified strategy.
        
        Args:
            strategy (str): Selection strategy
            batch_size (int): Number of examples to select
            
        Returns:
            list: Selected examples
        """
        # Limit batch size to available examples
        batch_size = min(batch_size, len(self.unlabeled_pool))
        
        if strategy == "uncertainty":
            return self.active_learning_manager.select_examples("uncertainty", batch_size)
        elif strategy == "diversity":
            return self.active_learning_manager.select_examples("diversity", batch_size)
        elif strategy == "expected_model_change":
            return self._select_by_expected_model_change(batch_size)
        elif strategy == "expected_error_reduction":
            return self._select_by_expected_error_reduction(batch_size)
        elif strategy == "density_weighted":
            return self._select_by_density_weighted_uncertainty(batch_size)
        else:
            logger.warning(f"Unknown strategy: {strategy}. Using uncertainty.")
            return self.active_learning_manager.select_examples("uncertainty", batch_size)
    
    def _select_by_expected_model_change(self, batch_size):
        """
        Select examples by expected model change.
        
        Args:
            batch_size (int): Number of examples to select
            
        Returns:
            list: Selected examples
        """
        # This is a simplified implementation of expected model change
        # In a real implementation, we would compute the expected gradient length
        
        # Get texts from unlabeled pool
        texts = [example["text"] for example in self.unlabeled_pool]
        
        # Preprocess texts
        X = self.model.preprocess_text(texts)
        
        # Get predictions
        predictions = self.model.model.predict(X)
        
        # Compute entropy (uncertainty)
        entropy = -np.sum(predictions * np.log(predictions + 1e-10), axis=1)
        
        # Compute expected model change (simplified as entropy * prediction confidence)
        expected_change = entropy * np.max(predictions, axis=1)
        
        # Sort by expected change (descending)
        sorted_indices = np.argsort(-expected_change)
        selected_indices = sorted_indices[:batch_size]
        
        # Get selected examples
        selected_examples = [self.unlabeled_pool[i] for i in selected_indices]
        
        return selected_examples
    
    def _select_by_expected_error_reduction(self, batch_size):
        """
        Select examples by expected error reduction.
        
        Args:
            batch_size (int): Number of examples to select
            
        Returns:
            list: Selected examples
        """
        # This is a simplified implementation of expected error reduction
        # In a real implementation, we would compute the expected reduction in error
        
        # Get texts from unlabeled pool
        texts = [example["text"] for example in self.unlabeled_pool]
        
        # Preprocess texts
        X = self.model.preprocess_text(texts)
        
        # Get predictions
        predictions = self.model.model.predict(X)
        
        # Compute expected error reduction (simplified as 1 - max probability)
        expected_reduction = 1 - np.max(predictions, axis=1)
        
        # Sort by expected reduction (descending)
        sorted_indices = np.argsort(-expected_reduction)
        selected_indices = sorted_indices[:batch_size]
        
        # Get selected examples
        selected_examples = [self.unlabeled_pool[i] for i in selected_indices]
        
        return selected_examples
    
    def _select_by_density_weighted_uncertainty(self, batch_size):
        """
        Select examples by density-weighted uncertainty.
        
        Args:
            batch_size (int): Number of examples to select
            
        Returns:
            list: Selected examples
        """
        # This is a simplified implementation of density-weighted uncertainty
        # In a real implementation, we would compute the density in the feature space
        
        # Get texts from unlabeled pool
        texts = [example["text"] for example in self.unlabeled_pool]
        
        # Preprocess texts
        X = self.model.preprocess_text(texts)
        
        # Get predictions
        predictions = self.model.model.predict(X)
        
        # Compute uncertainty (entropy)
        uncertainty = -np.sum(predictions * np.log(predictions + 1e-10), axis=1)
        
        # Compute density (simplified as average similarity to other examples)
        # In a real implementation, we would use embeddings and compute cosine similarity
        density = np.ones_like(uncertainty)  # Placeholder
        
        # Compute density-weighted uncertainty
        density_weighted = uncertainty * density
        
        # Sort by density-weighted uncertainty (descending)
        sorted_indices = np.argsort(-density_weighted)
        selected_indices = sorted_indices[:batch_size]
        
        # Get selected examples
        selected_examples = [self.unlabeled_pool[i] for i in selected_indices]
        
        return selected_examples
    
    def _update_model(self):
        """
        Update the model with the labeled pool.
        
        Returns:
            dict: Performance metrics
        """
        if not self.labeled_pool:
            logger.warning("No labeled examples available for model update")
            return {"accuracy": 0, "precision": 0, "recall": 0, "f1": 0}
        
        try:
            # Extract texts and categories
            texts = [example["text"] for example in self.labeled_pool]
            categories = [example["category"] for example in self.labeled_pool]
            
            # Prepare data for training
            self.model.prepare_training_data(texts, categories)
            
            # Set up training parameters
            training_config = self.config["training"]
            
            # Train the model
            history = self.model.model.fit(
                self.model.X_train,
                self.model.y_train,
                batch_size=training_config["batch_size"],
                epochs=training_config["epochs"],
                validation_split=training_config["validation_split"],
                verbose=1
            )
            
            # Save the model
            self.model._save_model()
            
            # Evaluate on validation set if available
            if self.validation_pool:
                performance = self._evaluate_model()
            else:
                # Use validation metrics from training
                performance = {
                    "accuracy": history.history["val_accuracy"][-1],
                    "precision": 0,  # Not available from history
                    "recall": 0,     # Not available from history
                    "f1": 0          # Not available from history
                }
            
            logger.info(f"Model updated with {len(texts)} examples. Accuracy: {performance['accuracy']:.4f}")
            
            return performance
        
        except Exception as e:
            logger.error(f"Error updating model: {e}")
            return {"accuracy": 0, "precision": 0, "recall": 0, "f1": 0}
    
    def _evaluate_model(self):
        """
        Evaluate the model on the validation set.
        
        Returns:
            dict: Performance metrics
        """
        if not self.validation_pool:
            logger.warning("No validation examples available for evaluation")
            return {"accuracy": 0, "precision": 0, "recall": 0, "f1": 0}
        
        try:
            # Extract texts and categories
            texts = [example["text"] for example in self.validation_pool]
            true_categories = [example["category"] for example in self.validation_pool]
            
            # Preprocess texts
            X = self.model.preprocess_text(texts)
            
            # Get predictions
            predictions = self.model.model.predict(X)
            
            # Get predicted categories
            predicted_categories = []
            for prediction in predictions:
                category_idx = np.argmax(prediction)
                category = list(self.model.category_mapping.keys())[category_idx]
                predicted_categories.append(category)
            
            # Compute metrics
            accuracy = accuracy_score(true_categories, predicted_categories)
            precision, recall, f1, _ = precision_recall_fscore_support(
                true_categories, predicted_categories, average="weighted"
            )
            
            return {
                "accuracy": float(accuracy),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1)
            }
        
        except Exception as e:
            logger.error(f"Error evaluating model: {e}")
            return {"accuracy": 0, "precision": 0, "recall": 0, "f1": 0}
    
    def _select_best_strategy(self, current_performance, previous_performance):
        """
        Select the best strategy based on performance improvement.
        
        Args:
            current_performance (dict): Current performance metrics
            previous_performance (dict): Previous performance metrics
            
        Returns:
            str: Selected strategy
        """
        # Check which strategies are enabled
        enabled_strategies = []
        for strategy, config in self.config["strategies"].items():
            if config["enabled"]:
                enabled_strategies.append(strategy)
        
        if not enabled_strategies:
            logger.warning("No strategies enabled. Using uncertainty.")
            return "uncertainty"
        
        # If performance improved significantly, stick with current strategy
        improvement = current_performance["accuracy"] - previous_performance["accuracy"]
        if improvement >= self.config["feedback_loop"]["performance_plateau_threshold"]:
            return self.metrics["strategy_history"][-1]
        
        # Otherwise, try a different strategy
        current_strategy = self.metrics["strategy_history"][-1]
        other_strategies = [s for s in enabled_strategies if s != current_strategy]
        
        if not other_strategies:
            return current_strategy
        
        # Select strategy with highest weight
        strategy_weights = {s: self.config["strategies"][s]["weight"] for s in other_strategies}
        best_strategy = max(strategy_weights.items(), key=lambda x: x[1])[0]
        
        logger.info(f"Switching strategy from {current_strategy} to {best_strategy}")
        
        return best_strategy
    
    def _save_metrics(self):
        """
        Save metrics to file.
        """
        metrics_file = os.path.join(self.output_dir, "feedback_loop_metrics.json")
        
        try:
            with open(metrics_file, 'w') as f:
                json.dump(self.metrics, f, indent=2)
            
            logger.info(f"Saved metrics to {metrics_file}")
        except Exception as e:
            logger.error(f"Error saving metrics: {e}")
    
    def _plot_learning_curve(self):
        """
        Plot the learning curve.
        """
        try:
            plt.figure(figsize=(10, 6))
            
            # Plot accuracy
            plt.plot(
                self.metrics["learning_curve"]["examples"],
                self.metrics["learning_curve"]["accuracy"],
                'b-',
                label='Accuracy'
            )
            
            # Plot F1 score
            plt.plot(
                self.metrics["learning_curve"]["examples"],
                self.metrics["learning_curve"]["f1"],
                'r-',
                label='F1 Score'
            )
            
            # Add labels and title
            plt.xlabel('Number of Labeled Examples')
            plt.ylabel('Performance')
            plt.title('Active Learning Feedback Loop Learning Curve')
            plt.legend()
            plt.grid(True)
            
            # Save figure
            curve_file = os.path.join(self.output_dir, "learning_curve.png")
            plt.savefig(curve_file)
            plt.close()
            
            logger.info(f"Saved learning curve to {curve_file}")
        except Exception as e:
            logger.error(f"Error plotting learning curve: {e}")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Active learning feedback loop for NoahAI")
    parser.add_argument("--config", help="Path to configuration file")
    parser.add_argument("--model-dir", default="data/model", help="Directory for model files")
    parser.add_argument("--output-dir", default="data/active_learning_feedback", help="Directory for output")
    parser.add_argument("--unlabeled", required=True, help="Path to unlabeled data file")
    parser.add_argument("--labeled", help="Path to labeled data file")
    parser.add_argument("--validation", help="Path to validation data file")
    args = parser.parse_args()
    
    # Create feedback loop
    feedback_loop = ActiveLearningFeedbackLoop(
        config_file=args.config,
        model_dir=args.model_dir,
        output_dir=args.output_dir
    )
    
    # Load data
    feedback_loop.load_data(args.unlabeled, args.labeled, args.validation)
    
    # Run feedback loop
    feedback_loop.run_feedback_loop()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
