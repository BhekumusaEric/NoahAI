#!/usr/bin/env python3
"""
Automated Data Labeling Manager for NoahAI

This script implements automated data labeling for NoahAI's deep learning model.
It uses the trained model to automatically label new data with high confidence,
reducing the need for manual labeling and accelerating the training pipeline.

Features:
- Confidence-based labeling
- Consensus labeling with multiple models
- Human-in-the-loop verification for uncertain examples
- Active learning integration
- Quality assurance metrics
- Incremental model updating
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
from collections import Counter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/automated_labeling.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("automated_labeling")

# Try to import NoahAI modules
try:
    from models.deep_learning_model import DeepLearningModel
    from src.utils.learning_utils import LearningUtils
    NOAHAI_AVAILABLE = True
except ImportError:
    NOAHAI_AVAILABLE = False
    logger.error("NoahAI modules not available. Please run this script from the NoahAI directory.")
    sys.exit(1)

class AutomatedLabelingManager:
    """
    Manager for automated data labeling using NoahAI's deep learning model.
    """
    
    def __init__(self, config_file=None, model_dir="data/model", output_dir="data/automated_labeling"):
        """
        Initialize the automated labeling manager.
        
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
        
        # Initialize primary model
        self.primary_model = self._load_model(model_dir)
        
        # Initialize ensemble models if enabled
        self.ensemble_models = []
        if self.config["labeling"]["use_ensemble"]:
            self._initialize_ensemble()
        
        # Initialize metrics
        self.metrics = {
            "total_examples": 0,
            "labeled_examples": 0,
            "high_confidence_examples": 0,
            "low_confidence_examples": 0,
            "human_verified_examples": 0,
            "estimated_accuracy": 0.0,
            "confidence_distribution": {},
            "category_distribution": {},
            "start_time": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
            "end_time": None,
            "duration": 0
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
            "labeling": {
                "confidence_threshold": 0.9,  # Minimum confidence for automatic labeling
                "use_ensemble": True,  # Whether to use an ensemble of models
                "ensemble_size": 3,  # Number of models in the ensemble
                "ensemble_agreement_threshold": 0.7,  # Minimum agreement ratio among ensemble models
                "human_verification_threshold": 0.7,  # Confidence below which human verification is required
                "max_examples_per_batch": 1000,  # Maximum number of examples to label in one batch
                "save_confidence": True,  # Whether to save confidence scores with labels
                "incremental_update": {
                    "enabled": True,  # Whether to update the model incrementally with new labeled data
                    "update_frequency": 500,  # Update after this many new labeled examples
                    "min_examples_for_update": 100,  # Minimum number of examples needed for an update
                    "epochs": 3,  # Number of epochs for incremental update
                    "batch_size": 32  # Batch size for incremental update
                }
            },
            "quality_assurance": {
                "enabled": True,  # Whether to perform quality assurance
                "validation_split": 0.1,  # Portion of data to use for validation
                "min_accuracy_threshold": 0.8,  # Minimum accuracy required for the model
                "confusion_matrix": True,  # Whether to generate a confusion matrix
                "cross_validation": False  # Whether to use cross-validation
            },
            "active_learning_integration": {
                "enabled": True,  # Whether to integrate with active learning
                "strategy": "uncertainty",  # Active learning strategy
                "batch_size": 100  # Batch size for active learning
            },
            "output": {
                "format": "json",  # Output format (json, csv, etc.)
                "include_metadata": True,  # Whether to include metadata in output
                "separate_files_by_confidence": True  # Whether to separate high and low confidence examples
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
    
    def _load_model(self, model_dir=None):
        """
        Load a model from the specified directory.
        
        Args:
            model_dir (str, optional): Directory containing model files
            
        Returns:
            DeepLearningModel: Loaded model
        """
        model_dir = model_dir or self.model_dir
        model_config = self.config["model"]
        
        # Check if model exists
        model_file = os.path.join(model_dir, f"model_{model_config['model_type']}.h5")
        
        if os.path.exists(model_file):
            logger.info(f"Loading existing model from {model_file}")
            
            try:
                model = DeepLearningModel(
                    model_dir=model_dir,
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
            model_dir=model_dir,
            responses_file="data/responses.json",
            max_words=model_config["max_words"],
            max_sequence_length=model_config["max_sequence_length"],
            use_advanced_nlp=True,
            model_type=model_config["model_type"],
            use_reinforcement_learning=model_config["use_reinforcement_learning"]
        )
        
        return model
    
    def _initialize_ensemble(self):
        """
        Initialize an ensemble of models for consensus labeling.
        """
        ensemble_size = self.config["labeling"]["ensemble_size"]
        
        # Check if ensemble models exist
        for i in range(1, ensemble_size):
            ensemble_dir = os.path.join(self.model_dir, f"ensemble_{i}")
            os.makedirs(ensemble_dir, exist_ok=True)
            
            try:
                model = self._load_model(ensemble_dir)
                self.ensemble_models.append(model)
                logger.info(f"Loaded ensemble model {i} from {ensemble_dir}")
            except Exception as e:
                logger.error(f"Error loading ensemble model {i}: {e}")
        
        # If not enough ensemble models, use the primary model with different configurations
        while len(self.ensemble_models) < ensemble_size - 1:
            logger.info(f"Creating additional ensemble model {len(self.ensemble_models) + 1}")
            
            # Create a copy of the primary model with slightly different configuration
            model = DeepLearningModel(
                model_dir=os.path.join(self.model_dir, f"ensemble_{len(self.ensemble_models) + 1}"),
                responses_file="data/responses.json",
                max_words=self.config["model"]["max_words"],
                max_sequence_length=self.config["model"]["max_sequence_length"],
                use_advanced_nlp=True,
                model_type=self.config["model"]["model_type"],
                use_reinforcement_learning=self.config["model"]["use_reinforcement_learning"]
            )
            
            # If primary model has been trained, copy its weights with small variations
            if hasattr(self.primary_model, "model") and self.primary_model.model is not None:
                # Create a similar model architecture
                model.create_model()
                
                # Copy weights with small random variations
                for i, layer in enumerate(self.primary_model.model.layers):
                    if hasattr(layer, "get_weights") and i < len(model.model.layers):
                        weights = layer.get_weights()
                        
                        # Add small random variations to weights
                        for j in range(len(weights)):
                            weights[j] = weights[j] + np.random.normal(0, 0.01, weights[j].shape)
                        
                        model.model.layers[i].set_weights(weights)
            
            self.ensemble_models.append(model)
    
    def load_unlabeled_data(self, data_file):
        """
        Load unlabeled data from file.
        
        Args:
            data_file (str): Path to data file
            
        Returns:
            list: List of unlabeled examples
        """
        if not os.path.exists(data_file):
            logger.error(f"Data file not found: {data_file}")
            return []
        
        try:
            with open(data_file, 'r') as f:
                data = json.load(f)
            
            # Extract texts
            unlabeled_examples = []
            
            for i, entry in enumerate(data):
                if isinstance(entry, dict):
                    text = entry.get("text") or entry.get("user_input")
                    
                    if text:
                        unlabeled_examples.append({
                            "id": i,
                            "text": text
                        })
                elif isinstance(entry, str):
                    unlabeled_examples.append({
                        "id": i,
                        "text": entry
                    })
            
            # Limit to max examples per batch
            max_examples = self.config["labeling"]["max_examples_per_batch"]
            if len(unlabeled_examples) > max_examples:
                unlabeled_examples = unlabeled_examples[:max_examples]
            
            logger.info(f"Loaded {len(unlabeled_examples)} unlabeled examples from {data_file}")
            return unlabeled_examples
        
        except Exception as e:
            logger.error(f"Error loading unlabeled data: {e}")
            return []
    
    def predict_with_primary_model(self, texts):
        """
        Make predictions with the primary model.
        
        Args:
            texts (list): List of texts
            
        Returns:
            tuple: (predictions, confidences, categories)
        """
        try:
            # Preprocess texts
            X = self.primary_model.preprocess_text(texts)
            
            # Get predictions
            predictions = self.primary_model.model.predict(X)
            
            # Get predicted categories and confidences
            categories = []
            confidences = []
            
            for prediction in predictions:
                category_idx = np.argmax(prediction)
                confidence = float(prediction[category_idx])
                
                # Map index to category name
                category = list(self.primary_model.category_mapping.keys())[category_idx]
                
                categories.append(category)
                confidences.append(confidence)
            
            return predictions, confidences, categories
        
        except Exception as e:
            logger.error(f"Error predicting with primary model: {e}")
            return None, None, None
    
    def predict_with_ensemble(self, texts):
        """
        Make predictions with the ensemble of models.
        
        Args:
            texts (list): List of texts
            
        Returns:
            tuple: (consensus_categories, consensus_confidences, agreement_ratios)
        """
        if not self.ensemble_models:
            logger.warning("No ensemble models available")
            return None, None, None
        
        try:
            # Get predictions from primary model
            _, primary_confidences, primary_categories = self.predict_with_primary_model(texts)
            
            # Initialize lists for ensemble predictions
            all_categories = [primary_categories]
            all_confidences = [primary_confidences]
            
            # Get predictions from ensemble models
            for model in self.ensemble_models:
                # Preprocess texts
                X = model.preprocess_text(texts)
                
                # Get predictions
                predictions = model.model.predict(X)
                
                # Get predicted categories and confidences
                categories = []
                confidences = []
                
                for prediction in predictions:
                    category_idx = np.argmax(prediction)
                    confidence = float(prediction[category_idx])
                    
                    # Map index to category name
                    category = list(model.category_mapping.keys())[category_idx]
                    
                    categories.append(category)
                    confidences.append(confidence)
                
                all_categories.append(categories)
                all_confidences.append(confidences)
            
            # Calculate consensus
            consensus_categories = []
            consensus_confidences = []
            agreement_ratios = []
            
            for i in range(len(texts)):
                # Get all categories predicted for this example
                example_categories = [categories[i] for categories in all_categories]
                
                # Count occurrences of each category
                category_counts = Counter(example_categories)
                
                # Get the most common category
                most_common = category_counts.most_common(1)[0]
                consensus_category = most_common[0]
                
                # Calculate agreement ratio
                agreement_ratio = most_common[1] / len(all_categories)
                
                # Get confidences for the consensus category
                consensus_confidences_list = []
                for j, categories in enumerate(all_categories):
                    if categories[i] == consensus_category:
                        consensus_confidences_list.append(all_confidences[j][i])
                
                # Calculate average confidence
                consensus_confidence = np.mean(consensus_confidences_list) if consensus_confidences_list else 0.0
                
                consensus_categories.append(consensus_category)
                consensus_confidences.append(consensus_confidence)
                agreement_ratios.append(agreement_ratio)
            
            return consensus_categories, consensus_confidences, agreement_ratios
        
        except Exception as e:
            logger.error(f"Error predicting with ensemble: {e}")
            return None, None, None
    
    def label_data(self, unlabeled_examples):
        """
        Label data using the model.
        
        Args:
            unlabeled_examples (list): List of unlabeled examples
            
        Returns:
            tuple: (labeled_examples, low_confidence_examples)
        """
        if not unlabeled_examples:
            logger.warning("No unlabeled examples provided")
            return [], []
        
        # Extract texts
        texts = [example["text"] for example in unlabeled_examples]
        
        # Update metrics
        self.metrics["total_examples"] += len(texts)
        
        # Get predictions
        if self.config["labeling"]["use_ensemble"]:
            categories, confidences, agreement_ratios = self.predict_with_ensemble(texts)
        else:
            _, confidences, categories = self.predict_with_primary_model(texts)
            agreement_ratios = [1.0] * len(texts)  # No ensemble, so perfect agreement
        
        if categories is None:
            logger.error("Failed to get predictions")
            return [], []
        
        # Separate high and low confidence examples
        high_confidence_examples = []
        low_confidence_examples = []
        
        confidence_threshold = self.config["labeling"]["confidence_threshold"]
        agreement_threshold = self.config["labeling"]["ensemble_agreement_threshold"]
        human_verification_threshold = self.config["labeling"]["human_verification_threshold"]
        
        for i, example in enumerate(unlabeled_examples):
            # Check if confidence and agreement are high enough
            is_high_confidence = (confidences[i] >= confidence_threshold and 
                                 agreement_ratios[i] >= agreement_threshold)
            
            # Create labeled example
            labeled_example = example.copy()
            labeled_example["category"] = categories[i]
            labeled_example["confidence"] = confidences[i]
            labeled_example["agreement_ratio"] = agreement_ratios[i]
            labeled_example["is_high_confidence"] = is_high_confidence
            labeled_example["needs_human_verification"] = confidences[i] < human_verification_threshold
            labeled_example["labeling_time"] = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
            
            # Add to appropriate list
            if is_high_confidence:
                high_confidence_examples.append(labeled_example)
                self.metrics["high_confidence_examples"] += 1
            else:
                low_confidence_examples.append(labeled_example)
                self.metrics["low_confidence_examples"] += 1
            
            # Update category distribution
            category = categories[i]
            if category not in self.metrics["category_distribution"]:
                self.metrics["category_distribution"][category] = 0
            self.metrics["category_distribution"][category] += 1
            
            # Update confidence distribution
            confidence_bin = int(confidences[i] * 10) / 10  # Round to nearest 0.1
            if confidence_bin not in self.metrics["confidence_distribution"]:
                self.metrics["confidence_distribution"][str(confidence_bin)] = 0
            self.metrics["confidence_distribution"][str(confidence_bin)] += 1
        
        self.metrics["labeled_examples"] += len(unlabeled_examples)
        
        # Estimate accuracy based on confidence
        if self.metrics["labeled_examples"] > 0:
            self.metrics["estimated_accuracy"] = self.metrics["high_confidence_examples"] / self.metrics["labeled_examples"]
        
        # Update metrics
        self.metrics["end_time"] = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
        start_time = datetime.strptime(self.metrics["start_time"], "%Y-%m-%dT%H:%M:%S")
        end_time = datetime.strptime(self.metrics["end_time"], "%Y-%m-%dT%H:%M:%S")
        self.metrics["duration"] = (end_time - start_time).total_seconds()
        
        # Save metrics
        self._save_metrics()
        
        # Check if incremental update is needed
        if (self.config["labeling"]["incremental_update"]["enabled"] and 
            len(high_confidence_examples) >= self.config["labeling"]["incremental_update"]["min_examples_for_update"] and
            self.metrics["labeled_examples"] % self.config["labeling"]["incremental_update"]["update_frequency"] < len(unlabeled_examples)):
            self._update_model_incrementally(high_confidence_examples)
        
        return high_confidence_examples, low_confidence_examples
    
    def _update_model_incrementally(self, labeled_examples):
        """
        Update the model incrementally with new labeled data.
        
        Args:
            labeled_examples (list): List of labeled examples
            
        Returns:
            bool: Whether the update was successful
        """
        if not labeled_examples:
            logger.warning("No labeled examples provided for incremental update")
            return False
        
        logger.info(f"Updating model incrementally with {len(labeled_examples)} examples")
        
        try:
            # Extract texts and categories
            texts = [example["text"] for example in labeled_examples]
            categories = [example["category"] for example in labeled_examples]
            
            # Prepare data for training
            self.primary_model.prepare_training_data(texts, categories)
            
            # Set up training parameters
            update_config = self.config["labeling"]["incremental_update"]
            batch_size = update_config["batch_size"]
            epochs = update_config["epochs"]
            
            # Train the model
            history = self.primary_model.model.fit(
                self.primary_model.X_train,
                self.primary_model.y_train,
                batch_size=batch_size,
                epochs=epochs,
                validation_split=0.1,
                verbose=1
            )
            
            # Save the updated model
            self.primary_model._save_model()
            
            logger.info(f"Model updated incrementally with {len(labeled_examples)} examples")
            return True
        
        except Exception as e:
            logger.error(f"Error updating model incrementally: {e}")
            return False
    
    def save_labeled_data(self, high_confidence_examples, low_confidence_examples):
        """
        Save labeled data to files.
        
        Args:
            high_confidence_examples (list): List of high confidence examples
            low_confidence_examples (list): List of low confidence examples
            
        Returns:
            tuple: (high_confidence_file, low_confidence_file)
        """
        output_config = self.config["output"]
        
        # Create timestamp for filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save high confidence examples
        high_confidence_file = None
        if high_confidence_examples:
            high_confidence_file = os.path.join(self.output_dir, f"high_confidence_{timestamp}.json")
            
            try:
                with open(high_confidence_file, 'w') as f:
                    json.dump(high_confidence_examples, f, indent=2)
                
                logger.info(f"Saved {len(high_confidence_examples)} high confidence examples to {high_confidence_file}")
            except Exception as e:
                logger.error(f"Error saving high confidence examples: {e}")
        
        # Save low confidence examples
        low_confidence_file = None
        if low_confidence_examples:
            low_confidence_file = os.path.join(self.output_dir, f"low_confidence_{timestamp}.json")
            
            try:
                with open(low_confidence_file, 'w') as f:
                    json.dump(low_confidence_examples, f, indent=2)
                
                logger.info(f"Saved {len(low_confidence_examples)} low confidence examples to {low_confidence_file}")
            except Exception as e:
                logger.error(f"Error saving low confidence examples: {e}")
        
        # Save all examples together if not separating by confidence
        if not output_config["separate_files_by_confidence"]:
            all_examples = high_confidence_examples + low_confidence_examples
            all_file = os.path.join(self.output_dir, f"all_labeled_{timestamp}.json")
            
            try:
                with open(all_file, 'w') as f:
                    json.dump(all_examples, f, indent=2)
                
                logger.info(f"Saved {len(all_examples)} labeled examples to {all_file}")
                return all_file, None
            except Exception as e:
                logger.error(f"Error saving all labeled examples: {e}")
        
        return high_confidence_file, low_confidence_file
    
    def _save_metrics(self):
        """
        Save metrics to file.
        """
        metrics_file = os.path.join(self.output_dir, "labeling_metrics.json")
        
        try:
            with open(metrics_file, 'w') as f:
                json.dump(self.metrics, f, indent=2)
            
            logger.info(f"Saved metrics to {metrics_file}")
        except Exception as e:
            logger.error(f"Error saving metrics: {e}")
    
    def integrate_with_active_learning(self, low_confidence_examples):
        """
        Integrate with active learning to prioritize uncertain examples.
        
        Args:
            low_confidence_examples (list): List of low confidence examples
            
        Returns:
            str: Path to the active learning file
        """
        if not self.config["active_learning_integration"]["enabled"]:
            logger.info("Active learning integration is disabled")
            return None
        
        if not low_confidence_examples:
            logger.warning("No low confidence examples for active learning")
            return None
        
        try:
            # Create active learning directory
            active_learning_dir = os.path.join(self.output_dir, "active_learning")
            os.makedirs(active_learning_dir, exist_ok=True)
            
            # Create file for active learning
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            active_learning_file = os.path.join(active_learning_dir, f"active_learning_{timestamp}.json")
            
            # Sort examples by confidence (ascending)
            sorted_examples = sorted(low_confidence_examples, key=lambda x: x["confidence"])
            
            # Limit to batch size
            batch_size = self.config["active_learning_integration"]["batch_size"]
            if len(sorted_examples) > batch_size:
                sorted_examples = sorted_examples[:batch_size]
            
            # Save examples for active learning
            with open(active_learning_file, 'w') as f:
                json.dump(sorted_examples, f, indent=2)
            
            logger.info(f"Saved {len(sorted_examples)} examples for active learning to {active_learning_file}")
            return active_learning_file
        
        except Exception as e:
            logger.error(f"Error integrating with active learning: {e}")
            return None
    
    def perform_quality_assurance(self, labeled_examples):
        """
        Perform quality assurance on labeled examples.
        
        Args:
            labeled_examples (list): List of labeled examples
            
        Returns:
            dict: Quality assurance metrics
        """
        if not self.config["quality_assurance"]["enabled"]:
            logger.info("Quality assurance is disabled")
            return None
        
        if not labeled_examples:
            logger.warning("No labeled examples for quality assurance")
            return None
        
        try:
            # Split examples into high and low confidence
            high_confidence = [ex for ex in labeled_examples if ex.get("is_high_confidence", False)]
            low_confidence = [ex for ex in labeled_examples if not ex.get("is_high_confidence", False)]
            
            # Calculate basic metrics
            qa_metrics = {
                "total_examples": len(labeled_examples),
                "high_confidence_examples": len(high_confidence),
                "low_confidence_examples": len(low_confidence),
                "high_confidence_ratio": len(high_confidence) / len(labeled_examples) if labeled_examples else 0,
                "average_confidence": np.mean([ex.get("confidence", 0) for ex in labeled_examples]),
                "category_distribution": {}
            }
            
            # Calculate category distribution
            for example in labeled_examples:
                category = example.get("category")
                if category:
                    if category not in qa_metrics["category_distribution"]:
                        qa_metrics["category_distribution"][category] = 0
                    qa_metrics["category_distribution"][category] += 1
            
            # Convert counts to percentages
            for category in qa_metrics["category_distribution"]:
                qa_metrics["category_distribution"][category] /= len(labeled_examples)
            
            # Save QA metrics
            qa_metrics_file = os.path.join(self.output_dir, "quality_assurance_metrics.json")
            with open(qa_metrics_file, 'w') as f:
                json.dump(qa_metrics, f, indent=2)
            
            logger.info(f"Saved quality assurance metrics to {qa_metrics_file}")
            return qa_metrics
        
        except Exception as e:
            logger.error(f"Error performing quality assurance: {e}")
            return None

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Automated data labeling for NoahAI")
    parser.add_argument("--config", help="Path to configuration file")
    parser.add_argument("--model-dir", default="data/model", help="Directory for model files")
    parser.add_argument("--output-dir", default="data/automated_labeling", help="Directory for output")
    parser.add_argument("--data", required=True, help="Path to unlabeled data file")
    args = parser.parse_args()
    
    # Create automated labeling manager
    manager = AutomatedLabelingManager(
        config_file=args.config,
        model_dir=args.model_dir,
        output_dir=args.output_dir
    )
    
    # Load unlabeled data
    unlabeled_examples = manager.load_unlabeled_data(args.data)
    
    if not unlabeled_examples:
        logger.error("No unlabeled examples loaded")
        return 1
    
    # Label data
    high_confidence_examples, low_confidence_examples = manager.label_data(unlabeled_examples)
    
    # Save labeled data
    manager.save_labeled_data(high_confidence_examples, low_confidence_examples)
    
    # Integrate with active learning
    manager.integrate_with_active_learning(low_confidence_examples)
    
    # Perform quality assurance
    manager.perform_quality_assurance(high_confidence_examples + low_confidence_examples)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
