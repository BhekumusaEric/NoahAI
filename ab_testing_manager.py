#!/usr/bin/env python3
"""
A/B Testing Manager for NoahAI

This script manages A/B testing of different model architectures and training approaches.
It allows you to:
- Define multiple model variants
- Train them in parallel
- Compare their performance
- Select the best model based on metrics
"""

import os
import sys
import json
import time
import argparse
import logging
import numpy as np
import tensorflow as tf
import multiprocessing as mp
from datetime import datetime
from typing import List, Dict, Tuple, Any, Optional, Union
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/ab_testing.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("ab_testing")

# Try to import NoahAI modules
try:
    from models.deep_learning_model import DeepLearningModel
    from src.utils.learning_utils import LearningUtils
    NOAHAI_AVAILABLE = True
except ImportError:
    NOAHAI_AVAILABLE = False
    logger.error("NoahAI modules not available. Please run this script from the NoahAI directory.")
    sys.exit(1)

class ModelVariant:
    """
    Represents a model variant for A/B testing.
    """
    
    def __init__(self, variant_id, name, config):
        """
        Initialize a model variant.
        
        Args:
            variant_id (str): Unique identifier for the variant
            name (str): Human-readable name for the variant
            config (dict): Configuration for the variant
        """
        self.variant_id = variant_id
        self.name = name
        self.config = config
        self.model = None
        self.metrics = {}
        self.training_history = None
    
    def create_model(self):
        """
        Create a model instance for this variant.
        
        Returns:
            DeepLearningModel: Model instance
        """
        model_config = self.config.get("model", {})
        training_config = self.config.get("training", {})
        
        model = DeepLearningModel(
            model_dir=f"data/model/{self.variant_id}",
            responses_file="data/responses.json",
            max_words=self.config.get("data", {}).get("max_words", 10000),
            max_sequence_length=self.config.get("data", {}).get("max_sequence_length", 100),
            use_advanced_nlp=True,
            model_type=model_config.get("model_type", "advanced_lstm"),
            use_reinforcement_learning=training_config.get("use_reinforcement_learning", True)
        )
        
        self.model = model
        return model
    
    def train(self, texts, categories, ratings):
        """
        Train the model for this variant.
        
        Args:
            texts (list): List of input texts
            categories (list): List of categories
            ratings (list): List of ratings
            
        Returns:
            dict: Training metrics
        """
        if self.model is None:
            self.create_model()
        
        training_config = self.config.get("training", {})
        
        # Prepare training data
        self.model.prepare_training_data(texts, categories)
        
        # Train the model
        try:
            history = self.model.train(
                epochs=training_config.get("epochs", 20),
                batch_size=training_config.get("batch_size", 32),
                use_early_stopping=training_config.get("use_early_stopping", True),
                use_transfer_learning=training_config.get("use_transfer_learning", True)
            )
            
            self.training_history = history
            
            # Extract metrics
            if history:
                self.metrics["loss"] = history.get("loss", [])[-1] if history.get("loss") else None
                self.metrics["accuracy"] = history.get("accuracy", [])[-1] if history.get("accuracy") else None
                self.metrics["val_loss"] = history.get("val_loss", [])[-1] if history.get("val_loss") else None
                self.metrics["val_accuracy"] = history.get("val_accuracy", [])[-1] if history.get("val_accuracy") else None
                self.metrics["epochs_completed"] = len(history.get("loss", []))
                
                # Find best epoch
                val_acc = history.get("val_accuracy", [])
                if val_acc:
                    best_epoch = np.argmax(val_acc)
                    self.metrics["best_epoch"] = best_epoch
                    self.metrics["best_accuracy"] = val_acc[best_epoch]
                    self.metrics["best_loss"] = history.get("val_loss", [])[best_epoch]
            
            return self.metrics
        
        except Exception as e:
            logger.error(f"Error training variant {self.variant_id}: {e}")
            self.metrics["error"] = str(e)
            return self.metrics
    
    def evaluate(self, texts, categories):
        """
        Evaluate the model on test data.
        
        Args:
            texts (list): List of input texts
            categories (list): List of categories
            
        Returns:
            dict: Evaluation metrics
        """
        if self.model is None:
            logger.error(f"Model for variant {self.variant_id} not trained yet")
            return {}
        
        try:
            # Prepare evaluation data
            X_test = self.model.preprocess_text(texts)
            
            # Convert categories to one-hot encoding
            unique_categories = sorted(set(self.model.category_mapping.keys()))
            category_to_index = {cat: i for i, cat in enumerate(unique_categories)}
            y_indices = [category_to_index.get(cat, 0) for cat in categories]
            y_test = tf.keras.utils.to_categorical(y_indices, num_classes=len(unique_categories))
            
            # Evaluate model
            evaluation = self.model.model.evaluate(X_test, y_test, verbose=0)
            
            # Get metrics
            metrics = {}
            for i, metric_name in enumerate(self.model.model.metrics_names):
                metrics[metric_name] = float(evaluation[i])
            
            # Make predictions for more detailed metrics
            y_pred = self.model.model.predict(X_test)
            y_pred_classes = np.argmax(y_pred, axis=1)
            y_true_classes = np.argmax(y_test, axis=1)
            
            # Calculate precision, recall, and F1 score
            from sklearn.metrics import precision_score, recall_score, f1_score
            
            metrics["precision"] = float(precision_score(y_true_classes, y_pred_classes, average='weighted'))
            metrics["recall"] = float(recall_score(y_true_classes, y_pred_classes, average='weighted'))
            metrics["f1"] = float(f1_score(y_true_classes, y_pred_classes, average='weighted'))
            
            # Update metrics
            self.metrics.update(metrics)
            
            return metrics
        
        except Exception as e:
            logger.error(f"Error evaluating variant {self.variant_id}: {e}")
            self.metrics["evaluation_error"] = str(e)
            return {}
    
    def save_metrics(self, output_dir):
        """
        Save metrics to file.
        
        Args:
            output_dir (str): Output directory
        """
        os.makedirs(output_dir, exist_ok=True)
        metrics_file = os.path.join(output_dir, f"{self.variant_id}_metrics.json")
        
        try:
            with open(metrics_file, 'w') as f:
                json.dump(self.metrics, f, indent=2)
            
            logger.info(f"Saved metrics for variant {self.variant_id} to {metrics_file}")
        except Exception as e:
            logger.error(f"Error saving metrics for variant {self.variant_id}: {e}")

class ABTestingManager:
    """
    Manager for A/B testing of NoahAI models.
    """
    
    def __init__(self, config_file=None, output_dir="data/ab_testing"):
        """
        Initialize the A/B testing manager.
        
        Args:
            config_file (str): Path to configuration file
            output_dir (str): Directory for output
        """
        self.output_dir = output_dir
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Load configuration
        self.config = self._load_config(config_file)
        
        # Initialize variants
        self.variants = self._initialize_variants()
        
        # Initialize testing metrics
        self.testing_metrics = {
            "start_time": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
            "end_time": None,
            "duration": 0,
            "variants_tested": len(self.variants),
            "best_variant": None,
            "variant_metrics": {}
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
            "data": {
                "training_data_file": "data/massive_training_data.json",
                "test_split": 0.2,
                "max_sequence_length": 100,
                "max_words": 10000
            },
            "variants": [
                {
                    "id": "baseline",
                    "name": "Baseline LSTM",
                    "model": {
                        "model_type": "simple_lstm"
                    },
                    "training": {
                        "batch_size": 32,
                        "epochs": 20,
                        "use_transfer_learning": False,
                        "use_reinforcement_learning": False
                    }
                },
                {
                    "id": "advanced",
                    "name": "Advanced Bidirectional LSTM",
                    "model": {
                        "model_type": "advanced_lstm"
                    },
                    "training": {
                        "batch_size": 32,
                        "epochs": 20,
                        "use_transfer_learning": True,
                        "use_reinforcement_learning": True
                    }
                },
                {
                    "id": "cnn_lstm",
                    "name": "CNN-LSTM Hybrid",
                    "model": {
                        "model_type": "cnn_lstm"
                    },
                    "training": {
                        "batch_size": 32,
                        "epochs": 20,
                        "use_transfer_learning": True,
                        "use_reinforcement_learning": True
                    }
                }
            ],
            "testing": {
                "parallel": True,
                "num_processes": None,  # None means use all available cores
                "primary_metric": "val_accuracy",
                "save_all_models": True,
                "detailed_comparison": True
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
    
    def _initialize_variants(self):
        """
        Initialize model variants from configuration.
        
        Returns:
            list: List of ModelVariant instances
        """
        variants = []
        
        for variant_config in self.config.get("variants", []):
            variant_id = variant_config.get("id")
            variant_name = variant_config.get("name")
            
            if not variant_id:
                logger.warning("Variant without ID found in configuration, skipping")
                continue
            
            # Create full config for the variant
            full_config = {
                "data": self.config.get("data", {}),
                "model": variant_config.get("model", {}),
                "training": variant_config.get("training", {})
            }
            
            # Create variant
            variant = ModelVariant(variant_id, variant_name, full_config)
            variants.append(variant)
        
        logger.info(f"Initialized {len(variants)} model variants")
        return variants
    
    def load_data(self):
        """
        Load and split data for training and testing.
        
        Returns:
            tuple: (train_texts, train_categories, train_ratings, test_texts, test_categories, test_ratings)
        """
        data_file = self.config.get("data", {}).get("training_data_file")
        test_split = self.config.get("data", {}).get("test_split", 0.2)
        
        if not os.path.exists(data_file):
            logger.error(f"Training data file not found: {data_file}")
            return None, None, None, None, None, None
        
        try:
            with open(data_file, 'r') as f:
                data = json.load(f)
            
            logger.info(f"Loaded {len(data)} examples from {data_file}")
            
            # Extract texts, categories, and ratings
            texts = []
            categories = []
            ratings = []
            
            for entry in data:
                texts.append(entry["user_input"])
                categories.append(entry["category"])
                ratings.append(entry.get("rating", 5))  # Default to 5 if rating not provided
            
            # Split data
            from sklearn.model_selection import train_test_split
            
            train_texts, test_texts, train_categories, test_categories, train_ratings, test_ratings = train_test_split(
                texts, categories, ratings, test_size=test_split, random_state=42, stratify=categories
            )
            
            logger.info(f"Split data into {len(train_texts)} training and {len(test_texts)} testing examples")
            
            return train_texts, train_categories, train_ratings, test_texts, test_categories, test_ratings
        
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return None, None, None, None, None, None
    
    def _train_variant(self, variant, train_texts, train_categories, train_ratings, test_texts, test_categories):
        """
        Train and evaluate a single variant.
        
        Args:
            variant (ModelVariant): Variant to train
            train_texts (list): Training texts
            train_categories (list): Training categories
            train_ratings (list): Training ratings
            test_texts (list): Testing texts
            test_categories (list): Testing categories
            
        Returns:
            ModelVariant: Updated variant with metrics
        """
        logger.info(f"Training variant: {variant.name} ({variant.variant_id})")
        
        # Train the variant
        variant.train(train_texts, train_categories, train_ratings)
        
        # Evaluate the variant
        variant.evaluate(test_texts, test_categories)
        
        # Save metrics
        variant.save_metrics(self.output_dir)
        
        return variant
    
    def run_testing(self):
        """
        Run A/B testing on all variants.
        
        Returns:
            dict: Testing metrics
        """
        # Load data
        train_texts, train_categories, train_ratings, test_texts, test_categories, test_ratings = self.load_data()
        
        if train_texts is None:
            logger.error("Failed to load data")
            return None
        
        # Start timing
        start_time = time.time()
        
        # Check if parallel testing is enabled
        if self.config.get("testing", {}).get("parallel", True):
            # Determine number of processes
            num_processes = self.config.get("testing", {}).get("num_processes")
            if num_processes is None:
                num_processes = min(mp.cpu_count(), len(self.variants))
            else:
                num_processes = min(num_processes, len(self.variants))
            
            logger.info(f"Running parallel testing with {num_processes} processes")
            
            # Create process pool
            with mp.Pool(processes=num_processes) as pool:
                # Create tasks
                tasks = [(variant, train_texts, train_categories, train_ratings, test_texts, test_categories) 
                         for variant in self.variants]
                
                # Run tasks in parallel
                results = list(tqdm(
                    pool.starmap(self._train_variant_wrapper, tasks),
                    total=len(tasks),
                    desc="Testing variants"
                ))
                
                # Update variants with results
                for i, result in enumerate(results):
                    if result is not None:
                        self.variants[i] = result
        
        else:
            logger.info("Running sequential testing")
            
            # Train and evaluate each variant sequentially
            for i, variant in enumerate(self.variants):
                self.variants[i] = self._train_variant(
                    variant, train_texts, train_categories, train_ratings, test_texts, test_categories
                )
        
        # End timing
        end_time = time.time()
        duration = end_time - start_time
        
        # Find best variant
        primary_metric = self.config.get("testing", {}).get("primary_metric", "val_accuracy")
        best_variant = None
        best_metric_value = -float('inf')
        
        for variant in self.variants:
            metric_value = variant.metrics.get(primary_metric)
            
            if metric_value is not None and metric_value > best_metric_value:
                best_variant = variant
                best_metric_value = metric_value
        
        # Update testing metrics
        self.testing_metrics["end_time"] = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
        self.testing_metrics["duration"] = duration
        
        if best_variant:
            self.testing_metrics["best_variant"] = {
                "id": best_variant.variant_id,
                "name": best_variant.name,
                "metrics": best_variant.metrics
            }
        
        # Add metrics for all variants
        for variant in self.variants:
            self.testing_metrics["variant_metrics"][variant.variant_id] = {
                "name": variant.name,
                "metrics": variant.metrics
            }
        
        # Save testing metrics
        self._save_testing_metrics()
        
        return self.testing_metrics
    
    def _train_variant_wrapper(self, variant, train_texts, train_categories, train_ratings, test_texts, test_categories):
        """
        Wrapper for _train_variant to use with multiprocessing.
        """
        try:
            return self._train_variant(variant, train_texts, train_categories, train_ratings, test_texts, test_categories)
        except Exception as e:
            logger.error(f"Error in training variant {variant.variant_id}: {e}")
            return None
    
    def _save_testing_metrics(self):
        """
        Save testing metrics to file.
        """
        metrics_file = os.path.join(self.output_dir, "ab_testing_metrics.json")
        
        try:
            with open(metrics_file, 'w') as f:
                json.dump(self.testing_metrics, f, indent=2)
            
            logger.info(f"Saved testing metrics to {metrics_file}")
        except Exception as e:
            logger.error(f"Error saving testing metrics: {e}")
    
    def generate_comparison_report(self):
        """
        Generate a detailed comparison report of all variants.
        
        Returns:
            str: Path to the report file
        """
        if not self.config.get("testing", {}).get("detailed_comparison", True):
            return None
        
        report_file = os.path.join(self.output_dir, "variant_comparison_report.md")
        
        try:
            with open(report_file, 'w') as f:
                f.write("# NoahAI Model Variant Comparison Report\n\n")
                f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                # Write summary
                f.write("## Summary\n\n")
                f.write(f"- **Number of variants tested:** {len(self.variants)}\n")
                f.write(f"- **Testing duration:** {self.testing_metrics.get('duration', 0):.2f} seconds\n")
                
                if self.testing_metrics.get("best_variant"):
                    best = self.testing_metrics["best_variant"]
                    f.write(f"- **Best variant:** {best['name']} ({best['id']})\n")
                    f.write(f"- **Best {self.config.get('testing', {}).get('primary_metric', 'val_accuracy')}:** {best['metrics'].get(self.config.get('testing', {}).get('primary_metric', 'val_accuracy'), 0):.4f}\n")
                
                f.write("\n## Variant Comparison\n\n")
                
                # Create comparison table
                f.write("| Variant ID | Name | Model Type | Val Accuracy | Val Loss | Precision | Recall | F1 Score |\n")
                f.write("|------------|------|------------|--------------|----------|-----------|--------|----------|\n")
                
                for variant in self.variants:
                    metrics = variant.metrics
                    model_type = variant.config.get("model", {}).get("model_type", "unknown")
                    
                    f.write(f"| {variant.variant_id} | {variant.name} | {model_type} | ")
                    f.write(f"{metrics.get('val_accuracy', 0):.4f} | {metrics.get('val_loss', 0):.4f} | ")
                    f.write(f"{metrics.get('precision', 0):.4f} | {metrics.get('recall', 0):.4f} | {metrics.get('f1', 0):.4f} |\n")
                
                # Write detailed information for each variant
                f.write("\n## Detailed Variant Information\n\n")
                
                for variant in self.variants:
                    f.write(f"### {variant.name} ({variant.variant_id})\n\n")
                    
                    # Model configuration
                    f.write("#### Model Configuration\n\n")
                    f.write("```json\n")
                    f.write(json.dumps(variant.config.get("model", {}), indent=2))
                    f.write("\n```\n\n")
                    
                    # Training configuration
                    f.write("#### Training Configuration\n\n")
                    f.write("```json\n")
                    f.write(json.dumps(variant.config.get("training", {}), indent=2))
                    f.write("\n```\n\n")
                    
                    # Metrics
                    f.write("#### Metrics\n\n")
                    f.write("```json\n")
                    f.write(json.dumps(variant.metrics, indent=2))
                    f.write("\n```\n\n")
            
            logger.info(f"Generated comparison report at {report_file}")
            return report_file
        
        except Exception as e:
            logger.error(f"Error generating comparison report: {e}")
            return None

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="A/B testing for NoahAI models")
    parser.add_argument("--config", help="Path to configuration file")
    parser.add_argument("--output-dir", default="data/ab_testing", help="Directory for output")
    args = parser.parse_args()
    
    # Create A/B testing manager
    manager = ABTestingManager(
        config_file=args.config,
        output_dir=args.output_dir
    )
    
    # Run testing
    manager.run_testing()
    
    # Generate comparison report
    manager.generate_comparison_report()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
