"""
Explainable AI Components for NoahAI

This module provides tools for explaining model decisions and generating
human-understandable explanations for AI behavior.
"""

import os
import json
import logging
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional, Union

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

# Try to import TensorFlow
try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    logger.warning("TensorFlow not available. Some explainable AI features will be limited.")

# Try to import visualization libraries
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False
    logger.warning("Visualization libraries not available. Visualization features will be limited.")

class ExplainableAI:
    """
    Explainable AI for NoahAI.
    
    This class provides tools for explaining model decisions and generating
    human-understandable explanations for AI behavior.
    """
    
    def __init__(self, model=None, config=None):
        """
        Initialize the Explainable AI component.
        
        Args:
            model: The model to explain
            config (dict, optional): Configuration for explainable AI
        """
        self.model = model
        self.config = config or {}
        
        # Initialize explanation parameters
        self.explanation_level = self.config.get("explanation_level", "detailed")  # "simple", "detailed", "technical"
        self.use_feature_importance = self.config.get("use_feature_importance", True)
        self.use_counterfactuals = self.config.get("use_counterfactuals", False)
        self.use_attention_visualization = self.config.get("use_attention_visualization", True)
        
        # Initialize explanation history
        self.explanation_history = []
        
        # Create output directory if it doesn't exist
        self.output_dir = self.config.get("output_dir", "data/explanations")
        os.makedirs(self.output_dir, exist_ok=True)
        
        logger.info(f"Explainable AI initialized with explanation level: {self.explanation_level}")
    
    def set_model(self, model):
        """
        Set the model to explain.
        
        Args:
            model: The model to explain
            
        Returns:
            bool: Whether the model was set successfully
        """
        self.model = model
        logger.info("Model set for explanation.")
        return True
    
    def explain_decision(self, input_data, prediction, features=None, feature_names=None):
        """
        Explain a model decision.
        
        Args:
            input_data: Input data that led to the decision
            prediction: Model prediction or decision
            features (optional): Feature values used for the decision
            feature_names (optional): Names of the features
            
        Returns:
            dict: Explanation data
        """
        # Initialize explanation
        explanation = {
            "timestamp": datetime.now().isoformat(),
            "prediction": prediction,
            "confidence": None,
            "explanation_text": "",
            "feature_importance": {},
            "counterfactuals": [],
            "visualization_path": None
        }
        
        try:
            # Get confidence if available
            if hasattr(prediction, "confidence") or (isinstance(prediction, dict) and "confidence" in prediction):
                explanation["confidence"] = prediction.confidence if hasattr(prediction, "confidence") else prediction["confidence"]
            
            # Generate feature importance if enabled and features are provided
            if self.use_feature_importance and features is not None:
                feature_importance = self._calculate_feature_importance(input_data, features, feature_names)
                explanation["feature_importance"] = feature_importance
                
                # Add top features to explanation text
                if feature_importance:
                    top_features = sorted(feature_importance.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
                    explanation["explanation_text"] += "Top factors in this decision:\n"
                    for feature, importance in top_features:
                        direction = "increased" if importance > 0 else "decreased"
                        explanation["explanation_text"] += f"- {feature}: {direction} the likelihood ({importance:.2f})\n"
            
            # Generate counterfactuals if enabled
            if self.use_counterfactuals:
                counterfactuals = self._generate_counterfactuals(input_data, prediction, features, feature_names)
                explanation["counterfactuals"] = counterfactuals
                
                # Add counterfactuals to explanation text
                if counterfactuals:
                    explanation["explanation_text"] += "\nAlternative scenarios:\n"
                    for i, cf in enumerate(counterfactuals[:2]):  # Limit to 2 counterfactuals
                        explanation["explanation_text"] += f"- {cf['description']}\n"
            
            # Generate visualization if enabled
            if self.use_attention_visualization and VISUALIZATION_AVAILABLE:
                visualization_path = self._generate_visualization(input_data, prediction, features, feature_names)
                explanation["visualization_path"] = visualization_path
            
            # Add explanation to history
            self.explanation_history.append(explanation)
            
            return explanation
        except Exception as e:
            logger.error(f"Error generating explanation: {e}")
            explanation["explanation_text"] = "Could not generate explanation due to an error."
            return explanation
    
    def _calculate_feature_importance(self, input_data, features, feature_names=None):
        """
        Calculate feature importance for a decision.
        
        Args:
            input_data: Input data that led to the decision
            features: Feature values used for the decision
            feature_names (optional): Names of the features
            
        Returns:
            dict: Feature importance scores
        """
        if not TENSORFLOW_AVAILABLE or self.model is None:
            return {}
        
        try:
            # Use feature names if provided, otherwise use indices
            if feature_names is None:
                feature_names = [f"feature_{i}" for i in range(len(features))]
            
            # Initialize feature importance
            feature_importance = {}
            
            # Method 1: Permutation importance
            # For each feature, permute its values and measure the change in prediction
            baseline_prediction = self.model.predict(np.array([features]))[0]
            
            for i, feature_name in enumerate(feature_names):
                # Create a copy of the features
                modified_features = features.copy()
                
                # Modify the feature (set to mean or zero)
                original_value = modified_features[i]
                modified_features[i] = 0  # or np.mean(features) if available
                
                # Get new prediction
                new_prediction = self.model.predict(np.array([modified_features]))[0]
                
                # Calculate importance as the difference in predictions
                importance = baseline_prediction - new_prediction
                
                # Store importance
                feature_importance[feature_name] = float(importance)
                
                # Restore original value
                modified_features[i] = original_value
            
            # Normalize importance scores
            max_importance = max(abs(v) for v in feature_importance.values()) if feature_importance else 1
            if max_importance > 0:
                feature_importance = {k: v / max_importance for k, v in feature_importance.items()}
            
            return feature_importance
        except Exception as e:
            logger.error(f"Error calculating feature importance: {e}")
            return {}
    
    def _generate_counterfactuals(self, input_data, prediction, features, feature_names=None):
        """
        Generate counterfactual explanations.
        
        Args:
            input_data: Input data that led to the decision
            prediction: Model prediction or decision
            features: Feature values used for the decision
            feature_names (optional): Names of the features
            
        Returns:
            list: Counterfactual explanations
        """
        if not TENSORFLOW_AVAILABLE or self.model is None:
            return []
        
        try:
            # Use feature names if provided, otherwise use indices
            if feature_names is None:
                feature_names = [f"feature_{i}" for i in range(len(features))]
            
            # Initialize counterfactuals
            counterfactuals = []
            
            # Method: Find minimal changes to features that would change the prediction
            # For simplicity, we'll just modify the top 1-2 important features
            
            # Get feature importance
            feature_importance = self._calculate_feature_importance(input_data, features, feature_names)
            
            # Sort features by importance
            sorted_features = sorted(feature_importance.items(), key=lambda x: abs(x[1]), reverse=True)
            
            # Generate counterfactuals for top features
            for feature_name, importance in sorted_features[:2]:
                # Get feature index
                feature_idx = feature_names.index(feature_name)
                
                # Create a copy of the features
                modified_features = features.copy()
                
                # Modify the feature based on its importance
                if importance > 0:
                    # Decrease the feature value
                    modified_features[feature_idx] = max(0, features[feature_idx] - abs(importance) * features[feature_idx])
                    change_direction = "decreased"
                else:
                    # Increase the feature value
                    modified_features[feature_idx] = features[feature_idx] + abs(importance) * features[feature_idx]
                    change_direction = "increased"
                
                # Get new prediction
                new_prediction = self.model.predict(np.array([modified_features]))[0]
                
                # Create counterfactual
                counterfactual = {
                    "modified_feature": feature_name,
                    "original_value": float(features[feature_idx]),
                    "modified_value": float(modified_features[feature_idx]),
                    "original_prediction": float(prediction) if isinstance(prediction, (int, float)) else prediction,
                    "new_prediction": float(new_prediction) if isinstance(new_prediction, (int, float)) else new_prediction,
                    "description": f"If {feature_name} was {change_direction} from {features[feature_idx]:.2f} to {modified_features[feature_idx]:.2f}, the prediction would change from {prediction} to {new_prediction}"
                }
                
                counterfactuals.append(counterfactual)
            
            return counterfactuals
        except Exception as e:
            logger.error(f"Error generating counterfactuals: {e}")
            return []
    
    def _generate_visualization(self, input_data, prediction, features, feature_names=None):
        """
        Generate visualization for a decision.
        
        Args:
            input_data: Input data that led to the decision
            prediction: Model prediction or decision
            features: Feature values used for the decision
            feature_names (optional): Names of the features
            
        Returns:
            str: Path to the visualization file
        """
        if not VISUALIZATION_AVAILABLE:
            return None
        
        try:
            # Use feature names if provided, otherwise use indices
            if feature_names is None:
                feature_names = [f"feature_{i}" for i in range(len(features))]
            
            # Create a unique filename
            filename = f"explanation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            filepath = os.path.join(self.output_dir, filename)
            
            # Create figure
            plt.figure(figsize=(10, 6))
            
            # Get feature importance
            feature_importance = self._calculate_feature_importance(input_data, features, feature_names)
            
            # Sort features by importance
            sorted_features = sorted(feature_importance.items(), key=lambda x: abs(x[1]), reverse=True)
            
            # Plot feature importance
            if sorted_features:
                # Extract feature names and importance values
                names = [f[0] for f in sorted_features]
                values = [f[1] for f in sorted_features]
                
                # Create bar chart
                colors = ['green' if v > 0 else 'red' for v in values]
                plt.barh(names, values, color=colors)
                plt.xlabel('Feature Importance')
                plt.title('Feature Importance for Prediction')
                plt.grid(axis='x', linestyle='--', alpha=0.6)
                
                # Add prediction information
                plt.figtext(0.5, 0.01, f"Prediction: {prediction}", ha='center', fontsize=12)
                
                # Save figure
                plt.tight_layout()
                plt.savefig(filepath)
                plt.close()
                
                logger.info(f"Saved visualization to {filepath}")
                return filepath
            else:
                logger.warning("No feature importance data available for visualization.")
                return None
        except Exception as e:
            logger.error(f"Error generating visualization: {e}")
            return None
    
    def explain_text_decision(self, text, prediction, attention_weights=None):
        """
        Explain a decision for text input.
        
        Args:
            text (str): Input text
            prediction: Model prediction or decision
            attention_weights (optional): Attention weights for the text
            
        Returns:
            dict: Explanation data
        """
        # Initialize explanation
        explanation = {
            "timestamp": datetime.now().isoformat(),
            "input_text": text,
            "prediction": prediction,
            "confidence": None,
            "explanation_text": "",
            "important_words": [],
            "visualization_path": None
        }
        
        try:
            # Get confidence if available
            if hasattr(prediction, "confidence") or (isinstance(prediction, dict) and "confidence" in prediction):
                explanation["confidence"] = prediction.confidence if hasattr(prediction, "confidence") else prediction["confidence"]
            
            # Identify important words
            important_words = self._identify_important_words(text, prediction, attention_weights)
            explanation["important_words"] = important_words
            
            # Add important words to explanation text
            if important_words:
                explanation["explanation_text"] += "Key words in this decision:\n"
                for word, importance in important_words[:5]:  # Limit to top 5 words
                    explanation["explanation_text"] += f"- {word}: {importance:.2f}\n"
            
            # Generate visualization if enabled
            if self.use_attention_visualization and VISUALIZATION_AVAILABLE and attention_weights is not None:
                visualization_path = self._generate_text_visualization(text, prediction, attention_weights)
                explanation["visualization_path"] = visualization_path
            
            # Add explanation to history
            self.explanation_history.append(explanation)
            
            return explanation
        except Exception as e:
            logger.error(f"Error generating text explanation: {e}")
            explanation["explanation_text"] = "Could not generate explanation due to an error."
            return explanation
    
    def _identify_important_words(self, text, prediction, attention_weights=None):
        """
        Identify important words in text input.
        
        Args:
            text (str): Input text
            prediction: Model prediction or decision
            attention_weights (optional): Attention weights for the text
            
        Returns:
            list: Important words with their importance scores
        """
        try:
            # Split text into words
            words = text.split()
            
            # If attention weights are provided, use them
            if attention_weights is not None and len(attention_weights) == len(words):
                word_importance = [(word, float(weight)) for word, weight in zip(words, attention_weights)]
            else:
                # Simple heuristic: longer words are more important
                word_importance = [(word, len(word) / max(1, len(max(words, key=len)))) for word in words]
                
                # Boost importance of certain keywords
                keywords = ["not", "very", "extremely", "always", "never", "but", "however", "although", "despite"]
                for i, (word, importance) in enumerate(word_importance):
                    if word.lower() in keywords:
                        word_importance[i] = (word, importance * 1.5)
            
            # Sort by importance
            word_importance.sort(key=lambda x: x[1], reverse=True)
            
            return word_importance
        except Exception as e:
            logger.error(f"Error identifying important words: {e}")
            return []
    
    def _generate_text_visualization(self, text, prediction, attention_weights):
        """
        Generate visualization for text input.
        
        Args:
            text (str): Input text
            prediction: Model prediction or decision
            attention_weights: Attention weights for the text
            
        Returns:
            str: Path to the visualization file
        """
        if not VISUALIZATION_AVAILABLE:
            return None
        
        try:
            # Create a unique filename
            filename = f"text_explanation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            filepath = os.path.join(self.output_dir, filename)
            
            # Split text into words
            words = text.split()
            
            # Ensure attention weights match the number of words
            if len(attention_weights) != len(words):
                logger.warning(f"Attention weights length ({len(attention_weights)}) does not match words length ({len(words)})")
                # Truncate or pad attention weights
                if len(attention_weights) > len(words):
                    attention_weights = attention_weights[:len(words)]
                else:
                    attention_weights = list(attention_weights) + [0] * (len(words) - len(attention_weights))
            
            # Create figure
            plt.figure(figsize=(12, 6))
            
            # Plot attention weights
            plt.bar(range(len(words)), attention_weights, color='skyblue')
            plt.xticks(range(len(words)), words, rotation=45, ha='right')
            plt.xlabel('Words')
            plt.ylabel('Attention Weight')
            plt.title('Word Importance in Prediction')
            plt.grid(axis='y', linestyle='--', alpha=0.6)
            
            # Add prediction information
            plt.figtext(0.5, 0.01, f"Prediction: {prediction}", ha='center', fontsize=12)
            
            # Save figure
            plt.tight_layout()
            plt.savefig(filepath)
            plt.close()
            
            logger.info(f"Saved text visualization to {filepath}")
            return filepath
        except Exception as e:
            logger.error(f"Error generating text visualization: {e}")
            return None
    
    def get_explanation_history(self, limit=10):
        """
        Get the history of explanations.
        
        Args:
            limit (int): Maximum number of explanations to return
            
        Returns:
            list: Explanation history
        """
        return self.explanation_history[-limit:]
    
    def clear_explanation_history(self):
        """
        Clear the explanation history.
        
        Returns:
            bool: Whether the history was cleared successfully
        """
        self.explanation_history = []
        logger.info("Cleared explanation history.")
        return True
