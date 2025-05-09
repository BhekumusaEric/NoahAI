#!/usr/bin/env python3
"""
Explainable AI Manager for NoahAI

This script implements explainable AI techniques for NoahAI's deep learning model.
It helps understand and interpret model predictions, making the AI more transparent.

Features:
- Feature importance visualization
- Attention visualization
- LIME (Local Interpretable Model-agnostic Explanations)
- SHAP (SHapley Additive exPlanations)
- Counterfactual explanations
- Decision boundary visualization
"""

import os
import sys
import json
import time
import argparse
import logging
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from datetime import datetime
from typing import List, Dict, Tuple, Any, Optional, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/explainable_ai.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("explainable_ai")

# Try to import NoahAI modules
try:
    from models.deep_learning_model import DeepLearningModel
    from src.utils.learning_utils import LearningUtils
    NOAHAI_AVAILABLE = True
except ImportError:
    NOAHAI_AVAILABLE = False
    logger.error("NoahAI modules not available. Please run this script from the NoahAI directory.")
    sys.exit(1)

# Try to import explainability libraries
try:
    import lime
    import lime.lime_text
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False
    logger.warning("LIME not available. Install with: pip install lime")

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logger.warning("SHAP not available. Install with: pip install shap")

class ExplainableAIManager:
    """
    Manager for explainable AI techniques for NoahAI's deep learning model.
    """
    
    def __init__(self, config_file=None, model_dir="data/model", output_dir="data/explainable_ai"):
        """
        Initialize the explainable AI manager.
        
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
        
        # Initialize explainers
        self.lime_explainer = None
        self.shap_explainer = None
        
        if LIME_AVAILABLE:
            self._initialize_lime()
        
        if SHAP_AVAILABLE:
            self._initialize_shap()
    
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
            "explainable_ai": {
                "lime": {
                    "num_samples": 1000,
                    "num_features": 10
                },
                "shap": {
                    "num_samples": 100,
                    "max_examples": 50
                },
                "attention": {
                    "enabled": True,
                    "layer_name": "attention"
                },
                "counterfactual": {
                    "enabled": True,
                    "num_examples": 5
                },
                "visualization": {
                    "save_format": "png",
                    "dpi": 300,
                    "figsize": [10, 6]
                }
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
    
    def _initialize_lime(self):
        """
        Initialize LIME explainer.
        """
        if not LIME_AVAILABLE:
            logger.warning("LIME not available. Skipping initialization.")
            return
        
        try:
            # Create a prediction function for LIME
            def predict_fn(texts):
                # Preprocess texts
                X = self.model.preprocess_text(texts)
                
                # Get predictions
                return self.model.model.predict(X)
            
            # Create LIME explainer
            self.lime_explainer = lime.lime_text.LimeTextExplainer(
                class_names=list(self.model.category_mapping.keys())
            )
            
            logger.info("Initialized LIME explainer")
        
        except Exception as e:
            logger.error(f"Error initializing LIME explainer: {e}")
    
    def _initialize_shap(self):
        """
        Initialize SHAP explainer.
        """
        if not SHAP_AVAILABLE:
            logger.warning("SHAP not available. Skipping initialization.")
            return
        
        try:
            # Create a prediction function for SHAP
            def predict_fn(texts):
                # Preprocess texts
                X = self.model.preprocess_text(texts)
                
                # Get predictions
                return self.model.model.predict(X)
            
            # Create SHAP explainer
            self.shap_explainer = shap.Explainer(predict_fn, masker=shap.maskers.Text())
            
            logger.info("Initialized SHAP explainer")
        
        except Exception as e:
            logger.error(f"Error initializing SHAP explainer: {e}")
    
    def explain_with_lime(self, text, category=None):
        """
        Explain a prediction using LIME.
        
        Args:
            text (str): Input text
            category (str, optional): Category to explain
            
        Returns:
            dict: Explanation
        """
        if not LIME_AVAILABLE or self.lime_explainer is None:
            logger.warning("LIME not available. Cannot explain prediction.")
            return None
        
        try:
            # Preprocess text
            X = self.model.preprocess_text([text])
            
            # Get prediction
            prediction = self.model.model.predict(X)[0]
            
            # Get predicted category
            predicted_category_idx = np.argmax(prediction)
            predicted_category = list(self.model.category_mapping.keys())[predicted_category_idx]
            
            # Get category index to explain
            if category is not None and category in self.model.category_mapping:
                category_idx = list(self.model.category_mapping.keys()).index(category)
            else:
                category_idx = predicted_category_idx
                category = predicted_category
            
            # Create a prediction function for LIME
            def predict_fn(texts):
                # Preprocess texts
                X = self.model.preprocess_text(texts)
                
                # Get predictions
                return self.model.model.predict(X)
            
            # Generate explanation
            lime_config = self.config["explainable_ai"]["lime"]
            explanation = self.lime_explainer.explain_instance(
                text,
                predict_fn,
                num_features=lime_config["num_features"],
                num_samples=lime_config["num_samples"],
                labels=[category_idx]
            )
            
            # Extract explanation data
            explanation_data = {
                "text": text,
                "predicted_category": predicted_category,
                "predicted_probability": float(prediction[predicted_category_idx]),
                "explained_category": category,
                "explained_probability": float(prediction[category_idx]),
                "features": []
            }
            
            # Get feature importances
            for feature, importance in explanation.as_list(label=category_idx):
                explanation_data["features"].append({
                    "feature": feature,
                    "importance": float(importance)
                })
            
            # Generate visualization
            self._save_lime_visualization(explanation, category_idx, text)
            
            return explanation_data
        
        except Exception as e:
            logger.error(f"Error explaining with LIME: {e}")
            return None
    
    def explain_with_shap(self, text, category=None):
        """
        Explain a prediction using SHAP.
        
        Args:
            text (str): Input text
            category (str, optional): Category to explain
            
        Returns:
            dict: Explanation
        """
        if not SHAP_AVAILABLE or self.shap_explainer is None:
            logger.warning("SHAP not available. Cannot explain prediction.")
            return None
        
        try:
            # Preprocess text
            X = self.model.preprocess_text([text])
            
            # Get prediction
            prediction = self.model.model.predict(X)[0]
            
            # Get predicted category
            predicted_category_idx = np.argmax(prediction)
            predicted_category = list(self.model.category_mapping.keys())[predicted_category_idx]
            
            # Get category index to explain
            if category is not None and category in self.model.category_mapping:
                category_idx = list(self.model.category_mapping.keys()).index(category)
            else:
                category_idx = predicted_category_idx
                category = predicted_category
            
            # Generate SHAP values
            shap_values = self.shap_explainer([text])
            
            # Extract explanation data
            explanation_data = {
                "text": text,
                "predicted_category": predicted_category,
                "predicted_probability": float(prediction[predicted_category_idx]),
                "explained_category": category,
                "explained_probability": float(prediction[category_idx]),
                "base_value": float(shap_values.base_values[0][category_idx]),
                "features": []
            }
            
            # Get feature importances
            for i, token in enumerate(shap_values.data[0].split()):
                if i < len(shap_values.values[0][category_idx]):
                    importance = float(shap_values.values[0][category_idx][i])
                    explanation_data["features"].append({
                        "feature": token,
                        "importance": importance
                    })
            
            # Sort features by absolute importance
            explanation_data["features"].sort(key=lambda x: abs(x["importance"]), reverse=True)
            
            # Generate visualization
            self._save_shap_visualization(shap_values, category_idx, text)
            
            return explanation_data
        
        except Exception as e:
            logger.error(f"Error explaining with SHAP: {e}")
            return None
    
    def explain_with_attention(self, text):
        """
        Explain a prediction using attention weights.
        
        Args:
            text (str): Input text
            
        Returns:
            dict: Explanation
        """
        try:
            # Check if model has attention layer
            attention_layer = None
            attention_config = self.config["explainable_ai"]["attention"]
            
            for layer in self.model.model.layers:
                if attention_config["layer_name"] in layer.name:
                    attention_layer = layer
                    break
            
            if attention_layer is None:
                logger.warning("Attention layer not found in model. Cannot explain with attention.")
                return None
            
            # Create a model that outputs attention weights
            attention_model = tf.keras.Model(
                inputs=self.model.model.inputs,
                outputs=[self.model.model.output, attention_layer.output]
            )
            
            # Preprocess text
            X = self.model.preprocess_text([text])
            
            # Get prediction and attention weights
            prediction, attention_weights = attention_model.predict(X)
            prediction = prediction[0]
            attention_weights = attention_weights[0]
            
            # Get predicted category
            predicted_category_idx = np.argmax(prediction)
            predicted_category = list(self.model.category_mapping.keys())[predicted_category_idx]
            
            # Get tokens
            tokens = self.model.tokenizer.sequences_to_texts([self.model.tokenizer.texts_to_sequences([text])[0]])[0].split()
            
            # Extract explanation data
            explanation_data = {
                "text": text,
                "predicted_category": predicted_category,
                "predicted_probability": float(prediction[predicted_category_idx]),
                "tokens": []
            }
            
            # Add attention weights for each token
            for i, token in enumerate(tokens):
                if i < len(attention_weights):
                    explanation_data["tokens"].append({
                        "token": token,
                        "attention": float(attention_weights[i])
                    })
            
            # Generate visualization
            self._save_attention_visualization(tokens, attention_weights, text)
            
            return explanation_data
        
        except Exception as e:
            logger.error(f"Error explaining with attention: {e}")
            return None
    
    def generate_counterfactual(self, text, target_category):
        """
        Generate counterfactual examples.
        
        Args:
            text (str): Input text
            target_category (str): Target category
            
        Returns:
            dict: Counterfactual examples
        """
        try:
            # Check if target category exists
            if target_category not in self.model.category_mapping:
                logger.warning(f"Target category '{target_category}' not found in model.")
                return None
            
            # Preprocess text
            X = self.model.preprocess_text([text])
            
            # Get prediction
            prediction = self.model.model.predict(X)[0]
            
            # Get predicted category
            predicted_category_idx = np.argmax(prediction)
            predicted_category = list(self.model.category_mapping.keys())[predicted_category_idx]
            
            # Get target category index
            target_category_idx = list(self.model.category_mapping.keys()).index(target_category)
            
            # If already predicted as target category, no need for counterfactual
            if predicted_category == target_category:
                logger.info(f"Text already predicted as '{target_category}'. No counterfactual needed.")
                return {
                    "text": text,
                    "predicted_category": predicted_category,
                    "predicted_probability": float(prediction[predicted_category_idx]),
                    "target_category": target_category,
                    "target_probability": float(prediction[target_category_idx]),
                    "counterfactuals": []
                }
            
            # Get tokens
            tokens = text.split()
            
            # Generate counterfactuals by modifying tokens
            counterfactuals = []
            counterfactual_config = self.config["explainable_ai"]["counterfactual"]
            
            # Try removing tokens
            for i in range(len(tokens)):
                modified_tokens = tokens.copy()
                removed_token = modified_tokens.pop(i)
                modified_text = " ".join(modified_tokens)
                
                # Preprocess modified text
                X_modified = self.model.preprocess_text([modified_text])
                
                # Get prediction
                modified_prediction = self.model.model.predict(X_modified)[0]
                
                # Check if prediction changed to target category
                modified_category_idx = np.argmax(modified_prediction)
                modified_category = list(self.model.category_mapping.keys())[modified_category_idx]
                
                if modified_category == target_category:
                    counterfactuals.append({
                        "text": modified_text,
                        "modification": f"Removed token: '{removed_token}'",
                        "probability": float(modified_prediction[target_category_idx])
                    })
            
            # Try replacing tokens with synonyms (simplified version)
            synonyms = {
                "good": ["great", "excellent", "wonderful", "fantastic"],
                "bad": ["poor", "terrible", "awful", "horrible"],
                "like": ["enjoy", "appreciate", "love", "adore"],
                "dislike": ["hate", "detest", "loathe", "despise"],
                "happy": ["glad", "pleased", "delighted", "joyful"],
                "sad": ["unhappy", "upset", "depressed", "miserable"]
            }
            
            for i, token in enumerate(tokens):
                if token.lower() in synonyms:
                    for synonym in synonyms[token.lower()]:
                        modified_tokens = tokens.copy()
                        modified_tokens[i] = synonym
                        modified_text = " ".join(modified_tokens)
                        
                        # Preprocess modified text
                        X_modified = self.model.preprocess_text([modified_text])
                        
                        # Get prediction
                        modified_prediction = self.model.model.predict(X_modified)[0]
                        
                        # Check if prediction changed to target category
                        modified_category_idx = np.argmax(modified_prediction)
                        modified_category = list(self.model.category_mapping.keys())[modified_category_idx]
                        
                        if modified_category == target_category:
                            counterfactuals.append({
                                "text": modified_text,
                                "modification": f"Replaced '{token}' with '{synonym}'",
                                "probability": float(modified_prediction[target_category_idx])
                            })
            
            # Sort counterfactuals by probability
            counterfactuals.sort(key=lambda x: x["probability"], reverse=True)
            
            # Limit number of counterfactuals
            counterfactuals = counterfactuals[:counterfactual_config["num_examples"]]
            
            return {
                "text": text,
                "predicted_category": predicted_category,
                "predicted_probability": float(prediction[predicted_category_idx]),
                "target_category": target_category,
                "target_probability": float(prediction[target_category_idx]),
                "counterfactuals": counterfactuals
            }
        
        except Exception as e:
            logger.error(f"Error generating counterfactual: {e}")
            return None
    
    def _save_lime_visualization(self, explanation, category_idx, text):
        """
        Save LIME visualization.
        
        Args:
            explanation: LIME explanation
            category_idx (int): Category index
            text (str): Input text
            
        Returns:
            str: Path to saved visualization
        """
        try:
            # Create figure
            visualization_config = self.config["explainable_ai"]["visualization"]
            plt.figure(figsize=tuple(visualization_config["figsize"]))
            
            # Generate visualization
            explanation.as_pyplot_figure(label=category_idx)
            
            # Add title
            category = list(self.model.category_mapping.keys())[category_idx]
            plt.title(f"LIME Explanation for '{category}'")
            
            # Save figure
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"lime_explanation_{timestamp}.{visualization_config['save_format']}"
            filepath = os.path.join(self.output_dir, filename)
            
            plt.savefig(filepath, dpi=visualization_config["dpi"], bbox_inches="tight")
            plt.close()
            
            logger.info(f"Saved LIME visualization to {filepath}")
            return filepath
        
        except Exception as e:
            logger.error(f"Error saving LIME visualization: {e}")
            return None
    
    def _save_shap_visualization(self, shap_values, category_idx, text):
        """
        Save SHAP visualization.
        
        Args:
            shap_values: SHAP values
            category_idx (int): Category index
            text (str): Input text
            
        Returns:
            str: Path to saved visualization
        """
        try:
            # Create figure
            visualization_config = self.config["explainable_ai"]["visualization"]
            plt.figure(figsize=tuple(visualization_config["figsize"]))
            
            # Generate visualization
            shap.plots.text(shap_values[:, :, category_idx], show=False)
            
            # Add title
            category = list(self.model.category_mapping.keys())[category_idx]
            plt.title(f"SHAP Explanation for '{category}'")
            
            # Save figure
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"shap_explanation_{timestamp}.{visualization_config['save_format']}"
            filepath = os.path.join(self.output_dir, filename)
            
            plt.savefig(filepath, dpi=visualization_config["dpi"], bbox_inches="tight")
            plt.close()
            
            logger.info(f"Saved SHAP visualization to {filepath}")
            return filepath
        
        except Exception as e:
            logger.error(f"Error saving SHAP visualization: {e}")
            return None
    
    def _save_attention_visualization(self, tokens, attention_weights, text):
        """
        Save attention visualization.
        
        Args:
            tokens (list): List of tokens
            attention_weights (np.ndarray): Attention weights
            text (str): Input text
            
        Returns:
            str: Path to saved visualization
        """
        try:
            # Create figure
            visualization_config = self.config["explainable_ai"]["visualization"]
            plt.figure(figsize=tuple(visualization_config["figsize"]))
            
            # Generate visualization
            plt.bar(range(len(tokens)), attention_weights[:len(tokens)])
            plt.xticks(range(len(tokens)), tokens, rotation=45, ha="right")
            plt.xlabel("Tokens")
            plt.ylabel("Attention Weight")
            plt.title("Attention Weights")
            plt.tight_layout()
            
            # Save figure
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"attention_visualization_{timestamp}.{visualization_config['save_format']}"
            filepath = os.path.join(self.output_dir, filename)
            
            plt.savefig(filepath, dpi=visualization_config["dpi"], bbox_inches="tight")
            plt.close()
            
            logger.info(f"Saved attention visualization to {filepath}")
            return filepath
        
        except Exception as e:
            logger.error(f"Error saving attention visualization: {e}")
            return None

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Explainable AI for NoahAI")
    parser.add_argument("--config", help="Path to configuration file")
    parser.add_argument("--model-dir", default="data/model", help="Directory for model files")
    parser.add_argument("--output-dir", default="data/explainable_ai", help="Directory for output")
    parser.add_argument("--text", help="Text to explain")
    parser.add_argument("--category", help="Category to explain")
    parser.add_argument("--method", choices=["lime", "shap", "attention", "counterfactual"], default="lime", help="Explanation method")
    parser.add_argument("--target-category", help="Target category for counterfactual")
    args = parser.parse_args()
    
    # Create explainable AI manager
    manager = ExplainableAIManager(
        config_file=args.config,
        model_dir=args.model_dir,
        output_dir=args.output_dir
    )
    
    # Generate explanation if text is provided
    if args.text:
        if args.method == "lime":
            explanation = manager.explain_with_lime(args.text, args.category)
        elif args.method == "shap":
            explanation = manager.explain_with_shap(args.text, args.category)
        elif args.method == "attention":
            explanation = manager.explain_with_attention(args.text)
        elif args.method == "counterfactual":
            if not args.target_category:
                logger.error("Target category required for counterfactual explanation")
                return 1
            explanation = manager.generate_counterfactual(args.text, args.target_category)
        
        # Print explanation
        if explanation:
            print(json.dumps(explanation, indent=2))
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
