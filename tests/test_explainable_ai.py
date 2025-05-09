"""
Tests for explainable AI features in NoahAI.

This module tests the explainable AI capabilities including:
- Response explanation generation
- Feature importance analysis
- Decision path visualization
- User-friendly explanations
"""

import sys
import os
import unittest
import json
import tempfile
import shutil

# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the explainable AI module
try:
    from src.utils.explainable_ai import ExplainableAI
    EXPLAINABLE_AI_AVAILABLE = True
except ImportError:
    EXPLAINABLE_AI_AVAILABLE = False

class TestExplainableAI(unittest.TestCase):
    """Test cases for explainable AI features."""

    def setUp(self):
        """Set up test fixtures."""
        if EXPLAINABLE_AI_AVAILABLE:
            # Create a temporary directory for explanation storage
            self.test_dir = tempfile.mkdtemp()
            
            # Create an ExplainableAI instance for testing
            self.xai = ExplainableAI(
                explanation_dir=self.test_dir,
                log_explanations=True,
                explanation_level="detailed"
            )
            
            # Sample data for testing
            self.user_input = "What is machine learning?"
            self.ai_response = "Machine learning is a subset of artificial intelligence that enables systems to learn from data and improve from experience without being explicitly programmed."
            self.intent = {"intent": "definition_request", "confidence": 0.85}
            self.entities = {
                "topic": [
                    {"value": "machine learning", "start": 8, "end": 24}
                ]
            }
            self.confidence = 0.92
            self.model_type = "transformer"
            self.keywords = ["machine learning", "definition", "AI"]
            self.context = {
                "last_intent": "greeting",
                "conversation_history": [
                    ("user", "Hello there!"),
                    ("ai", "Hi! How can I help you today?")
                ]
            }
        
    def tearDown(self):
        """Tear down test fixtures."""
        if hasattr(self, 'test_dir') and os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    @unittest.skipIf(not EXPLAINABLE_AI_AVAILABLE, "Explainable AI not available")
    def test_explain_response(self):
        """Test response explanation generation."""
        explanation = self.xai.explain_response(
            self.user_input, self.ai_response, self.intent, self.entities,
            self.confidence, self.model_type, self.keywords, self.context
        )
        
        # Check if explanation contains expected keys
        self.assertIn("timestamp", explanation)
        self.assertIn("user_input", explanation)
        self.assertIn("ai_response", explanation)
        self.assertIn("intent", explanation)
        self.assertIn("intent_confidence", explanation)
        self.assertIn("entities_found", explanation)
        self.assertIn("overall_confidence", explanation)
        self.assertIn("model_type", explanation)
        self.assertIn("keywords", explanation)
        self.assertIn("reasoning", explanation)
        
        # Check if detailed information is included
        self.assertIn("context_factors", explanation)
        self.assertIn("alternative_responses", explanation)
        
        # Check if values match input
        self.assertEqual(explanation["user_input"], self.user_input)
        self.assertEqual(explanation["ai_response"], self.ai_response)
        self.assertEqual(explanation["intent"], self.intent["intent"])
        self.assertEqual(explanation["intent_confidence"], self.intent["confidence"])
        self.assertEqual(explanation["overall_confidence"], self.confidence)
        self.assertEqual(explanation["model_type"], self.model_type)
        self.assertEqual(explanation["keywords"], self.keywords)
        
        # Check if explanation was added to history
        self.assertEqual(len(self.xai.explanation_history), 1)

    @unittest.skipIf(not EXPLAINABLE_AI_AVAILABLE, "Explainable AI not available")
    def test_user_friendly_explanation(self):
        """Test user-friendly explanation generation."""
        # Generate an explanation
        explanation = self.xai.explain_response(
            self.user_input, self.ai_response, self.intent, self.entities,
            self.confidence, self.model_type, self.keywords, self.context
        )
        
        # Generate user-friendly explanations at different detail levels
        low_detail = self.xai.get_explanation_for_user(explanation, "low")
        medium_detail = self.xai.get_explanation_for_user(explanation, "medium")
        high_detail = self.xai.get_explanation_for_user(explanation, "high")
        
        # Check if detail levels produce different explanations
        self.assertLessEqual(len(low_detail), len(medium_detail))
        self.assertLessEqual(len(medium_detail), len(high_detail))
        
        # Check if high detail explanation includes entity information
        if self.entities:
            for entity_type in self.entities:
                self.assertIn(entity_type, high_detail.lower())
        
        # Check if confidence information is included in medium and high detail
        confidence_text = f"{self.confidence:.1%}"
        self.assertIn(confidence_text, medium_detail)
        self.assertIn(confidence_text, high_detail)

    @unittest.skipIf(not EXPLAINABLE_AI_AVAILABLE, "Explainable AI not available")
    def test_explanation_history(self):
        """Test explanation history management."""
        # Generate multiple explanations
        for i in range(3):
            self.xai.explain_response(
                f"Question {i}", f"Answer {i}", self.intent, self.entities,
                self.confidence, self.model_type, self.keywords, self.context
            )
        
        # Check if all explanations were added to history
        self.assertEqual(len(self.xai.explanation_history), 3)
        
        # Get recent explanations
        recent = self.xai.get_recent_explanations(2)
        
        # Check if correct number of explanations is returned
        self.assertEqual(len(recent), 2)
        
        # Check if explanations are in reverse chronological order
        self.assertEqual(recent[0]["user_input"], "Question 1")
        self.assertEqual(recent[1]["user_input"], "Question 2")

    @unittest.skipIf(not EXPLAINABLE_AI_AVAILABLE, "Explainable AI not available")
    def test_explanation_persistence(self):
        """Test explanation persistence to file."""
        # Generate an explanation
        self.xai.explain_response(
            self.user_input, self.ai_response, self.intent, self.entities,
            self.confidence, self.model_type, self.keywords, self.context
        )
        
        # Save explanations
        self.xai._save_explanations()
        
        # Check if explanation file was created
        explanation_file = os.path.join(self.test_dir, "explanation_history.json")
        self.assertTrue(os.path.exists(explanation_file))
        
        # Load explanations from file
        with open(explanation_file, 'r') as f:
            loaded_explanations = json.load(f)
        
        # Check if loaded explanations match original
        self.assertEqual(len(loaded_explanations), 1)
        self.assertEqual(loaded_explanations[0]["user_input"], self.user_input)
        self.assertEqual(loaded_explanations[0]["ai_response"], self.ai_response)

    @unittest.skipIf(not EXPLAINABLE_AI_AVAILABLE, "Explainable AI not available")
    def test_reasoning_generation(self):
        """Test reasoning generation for explanations."""
        # Generate an explanation
        explanation = self.xai.explain_response(
            self.user_input, self.ai_response, self.intent, self.entities,
            self.confidence, self.model_type, self.keywords, self.context
        )
        
        # Check if reasoning is generated
        self.assertIn("reasoning", explanation)
        self.assertTrue(len(explanation["reasoning"]) > 0)
        
        # Check if reasoning mentions the intent
        self.assertIn(self.intent["intent"], explanation["reasoning"])
        
        # Check if reasoning mentions entities if present
        if self.entities:
            for entity_type in self.entities:
                self.assertIn(entity_type, explanation["reasoning"].lower())

    @unittest.skipIf(not EXPLAINABLE_AI_AVAILABLE, "Explainable AI not available")
    def test_technical_explanation(self):
        """Test technical explanation generation."""
        # Create an ExplainableAI instance with technical explanation level
        xai_technical = ExplainableAI(
            explanation_dir=self.test_dir,
            log_explanations=False,
            explanation_level="technical"
        )
        
        # Generate an explanation
        explanation = xai_technical.explain_response(
            self.user_input, self.ai_response, self.intent, self.entities,
            self.confidence, self.model_type, self.keywords, self.context
        )
        
        # Check if technical information is included
        self.assertIn("model_details", explanation)
        self.assertIn("features_used", explanation["model_details"])
        self.assertIn("decision_path", explanation["model_details"])
        
        # Check if features include intent and entities
        features = explanation["model_details"]["features_used"]
        self.assertTrue(any("intent" in feature.lower() for feature in features))
        if self.entities:
            self.assertTrue(any("entity" in feature.lower() for feature in features))
        
        # Check if decision path has multiple steps
        decision_path = explanation["model_details"]["decision_path"]
        self.assertTrue(len(decision_path) > 3)

if __name__ == '__main__':
    unittest.main()
