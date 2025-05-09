#!/usr/bin/env python3
"""
Tests for the automated labeling system.
"""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch

# Add parent directory to path to import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Mock the NoahAI modules that might not be available during testing
sys.modules['models.deep_learning_model'] = MagicMock()
sys.modules['src.utils.learning_utils'] = MagicMock()

# Now we can import the module to test
from automated_labeling_manager import AutomatedLabelingManager

class TestAutomatedLabeling(unittest.TestCase):
    """Test cases for the automated labeling system."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a mock configuration
        self.config = {
            "model": {
                "model_type": "test_model",
                "max_words": 1000,
                "max_sequence_length": 50,
                "use_reinforcement_learning": False
            },
            "labeling": {
                "confidence_threshold": 0.8,
                "use_ensemble": False,
                "ensemble_size": 1,
                "ensemble_agreement_threshold": 0.7,
                "human_verification_threshold": 0.6,
                "max_examples_per_batch": 100,
                "save_confidence": True,
                "incremental_update": {
                    "enabled": False
                }
            }
        }
        
        # Create a patch for the _load_config method
        self.config_patcher = patch.object(
            AutomatedLabelingManager, 
            '_load_config', 
            return_value=self.config
        )
        self.mock_load_config = self.config_patcher.start()
        
        # Create a patch for the _load_model method
        self.model_patcher = patch.object(
            AutomatedLabelingManager, 
            '_load_model', 
            return_value=MagicMock()
        )
        self.mock_load_model = self.model_patcher.start()
        
        # Create a patch for the _initialize_ensemble method
        self.ensemble_patcher = patch.object(
            AutomatedLabelingManager, 
            '_initialize_ensemble'
        )
        self.mock_initialize_ensemble = self.ensemble_patcher.start()
        
        # Create the manager instance
        self.manager = AutomatedLabelingManager(
            config_file=None,
            model_dir="test_model_dir",
            output_dir="test_output_dir"
        )
    
    def tearDown(self):
        """Tear down test fixtures."""
        self.config_patcher.stop()
        self.model_patcher.stop()
        self.ensemble_patcher.stop()
    
    def test_initialization(self):
        """Test that the manager initializes correctly."""
        self.assertEqual(self.manager.model_dir, "test_model_dir")
        self.assertEqual(self.manager.output_dir, "test_output_dir")
        self.assertEqual(self.manager.config, self.config)
        self.mock_load_model.assert_called_once()
        
        # Check that metrics are initialized
        self.assertIn("total_examples", self.manager.metrics)
        self.assertIn("labeled_examples", self.manager.metrics)
        self.assertIn("high_confidence_examples", self.manager.metrics)
        self.assertIn("low_confidence_examples", self.manager.metrics)
    
    def test_load_unlabeled_data(self):
        """Test loading unlabeled data."""
        # Mock the open function
        mock_data = [
            {"text": "This is a test"},
            {"text": "Another test example"}
        ]
        
        with patch('builtins.open', unittest.mock.mock_open(read_data=str(mock_data))), \
             patch('json.load', return_value=mock_data), \
             patch('os.path.exists', return_value=True):
            
            result = self.manager.load_unlabeled_data("test_file.json")
            
            self.assertEqual(len(result), 2)
            self.assertEqual(result[0]["text"], "This is a test")
            self.assertEqual(result[1]["text"], "Another test example")
    
    def test_predict_with_primary_model(self):
        """Test prediction with the primary model."""
        # Mock the model's preprocess_text and predict methods
        self.manager.primary_model.preprocess_text.return_value = "preprocessed_text"
        self.manager.primary_model.model.predict.return_value = [[0.9, 0.1]]
        self.manager.primary_model.category_mapping = {"category1": 0, "category2": 1}
        
        texts = ["This is a test"]
        predictions, confidences, categories = self.manager.predict_with_primary_model(texts)
        
        self.manager.primary_model.preprocess_text.assert_called_once_with(texts)
        self.manager.primary_model.model.predict.assert_called_once_with("preprocessed_text")
        
        self.assertEqual(predictions, [[0.9, 0.1]])
        self.assertEqual(confidences, [0.9])
        self.assertEqual(categories, ["category1"])
    
    def test_label_data(self):
        """Test labeling data."""
        # Mock the predict_with_primary_model method
        with patch.object(
            self.manager, 
            'predict_with_primary_model', 
            return_value=([[0.9, 0.1], [0.6, 0.4]], [0.9, 0.6], ["category1", "category2"])
        ):
            unlabeled_examples = [
                {"id": 1, "text": "High confidence example"},
                {"id": 2, "text": "Low confidence example"}
            ]
            
            high_confidence, low_confidence = self.manager.label_data(unlabeled_examples)
            
            self.assertEqual(len(high_confidence), 1)
            self.assertEqual(len(low_confidence), 1)
            self.assertEqual(high_confidence[0]["id"], 1)
            self.assertEqual(low_confidence[0]["id"], 2)
            self.assertEqual(high_confidence[0]["category"], "category1")
            self.assertEqual(low_confidence[0]["category"], "category2")
            self.assertEqual(high_confidence[0]["confidence"], 0.9)
            self.assertEqual(low_confidence[0]["confidence"], 0.6)
            self.assertTrue(high_confidence[0]["is_high_confidence"])
            self.assertFalse(low_confidence[0]["is_high_confidence"])

if __name__ == '__main__':
    unittest.main()
