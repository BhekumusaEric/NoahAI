"""
Tests for reinforcement learning enhancements in NoahAI.

This module tests the enhanced reinforcement learning capabilities including:
- Hierarchical reinforcement learning
- Multi-objective reinforcement learning
- Meta-learning for user preference adaptation
"""

import sys
import os
import unittest
import numpy as np
import tempfile
import shutil

# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the reinforcement learning modules
try:
    from src.utils.hierarchical_rl import HierarchicalRLManager
    HIERARCHICAL_RL_AVAILABLE = True
except ImportError:
    HIERARCHICAL_RL_AVAILABLE = False

try:
    from src.utils.multi_objective_rl import MultiObjectiveRLAgent
    MULTI_OBJECTIVE_RL_AVAILABLE = True
except ImportError:
    MULTI_OBJECTIVE_RL_AVAILABLE = False

try:
    from src.utils.meta_learning import MAMLAgent, UserProfile
    META_LEARNING_AVAILABLE = True
except ImportError:
    META_LEARNING_AVAILABLE = False

# Check if TensorFlow is available
try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

@unittest.skipIf(not TENSORFLOW_AVAILABLE, "TensorFlow not available")
class TestHierarchicalRL(unittest.TestCase):
    """Test cases for hierarchical reinforcement learning."""

    def setUp(self):
        """Set up test fixtures."""
        if HIERARCHICAL_RL_AVAILABLE:
            # Create a temporary directory for model saving/loading
            self.test_dir = tempfile.mkdtemp()
            
            # Create a small HRL manager for testing
            self.hrl = HierarchicalRLManager(
                model_dir=self.test_dir,
                state_size=10,
                action_size=4,
                num_goals=2,
                meta_lr=0.001,
                worker_lr=0.001,
                gamma=0.99,
                use_intrinsic_motivation=True
            )
            
            # Sample state and action for testing
            self.sample_state = np.random.random(10)
            self.sample_next_state = np.random.random(10)
            self.sample_action = 1
            self.sample_reward = 0.5
            self.sample_done = False
        
    def tearDown(self):
        """Tear down test fixtures."""
        if hasattr(self, 'test_dir') and os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    @unittest.skipIf(not HIERARCHICAL_RL_AVAILABLE, "Hierarchical RL not available")
    def test_goal_selection(self):
        """Test goal selection functionality."""
        goal, prob = self.hrl.select_goal(self.sample_state)
        
        # Check if goal is valid
        self.assertIsInstance(goal, (int, np.integer))
        self.assertGreaterEqual(goal, 0)
        self.assertLess(goal, self.hrl.num_goals)
        
        # Check if probability is valid
        self.assertIsInstance(prob, float)
        self.assertGreaterEqual(prob, 0.0)
        self.assertLessEqual(prob, 1.0)

    @unittest.skipIf(not HIERARCHICAL_RL_AVAILABLE, "Hierarchical RL not available")
    def test_action_selection(self):
        """Test action selection functionality."""
        goal, _ = self.hrl.select_goal(self.sample_state)
        action, prob = self.hrl.select_action(self.sample_state, goal)
        
        # Check if action is valid
        self.assertIsInstance(action, (int, np.integer))
        self.assertGreaterEqual(action, 0)
        self.assertLess(action, self.hrl.action_size)
        
        # Check if probability is valid
        self.assertIsInstance(prob, float)
        self.assertGreaterEqual(prob, 0.0)
        self.assertLessEqual(prob, 1.0)

    @unittest.skipIf(not HIERARCHICAL_RL_AVAILABLE, "Hierarchical RL not available")
    def test_intrinsic_motivation(self):
        """Test intrinsic motivation calculation."""
        intrinsic_reward = self.hrl.calculate_intrinsic_reward(
            self.sample_state, self.sample_action, self.sample_next_state
        )
        
        # Check if intrinsic reward is valid
        self.assertIsInstance(intrinsic_reward, float)
        self.assertGreaterEqual(intrinsic_reward, 0.0)
        self.assertLessEqual(intrinsic_reward, 1.0)

    @unittest.skipIf(not HIERARCHICAL_RL_AVAILABLE, "Hierarchical RL not available")
    def test_process_feedback(self):
        """Test feedback processing."""
        result = self.hrl.process_feedback(
            self.sample_state, self.sample_action, self.sample_reward,
            self.sample_next_state, self.sample_done
        )
        
        # Check if result contains expected keys
        self.assertIn("current_goal", result)
        self.assertIn("goal_steps", result)
        self.assertIn("intrinsic_reward", result)
        self.assertIn("total_reward", result)
        
        # Check if values are valid
        self.assertIsInstance(result["current_goal"], (int, np.integer, type(None)))
        self.assertIsInstance(result["goal_steps"], int)
        self.assertIsInstance(result["intrinsic_reward"], float)
        self.assertIsInstance(result["total_reward"], float)

    @unittest.skipIf(not HIERARCHICAL_RL_AVAILABLE, "Hierarchical RL not available")
    def test_save_and_load(self):
        """Test model saving functionality."""
        # Save the model
        save_result = self.hrl.save("test_hrl")
        
        # Check if save was successful
        self.assertTrue(save_result)
        
        # Check if files were created
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, "test_hrl_meta.h5")))
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, "test_hrl_worker_0.h5")))
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, "test_hrl_metadata.json")))

@unittest.skipIf(not TENSORFLOW_AVAILABLE, "TensorFlow not available")
class TestMultiObjectiveRL(unittest.TestCase):
    """Test cases for multi-objective reinforcement learning."""

    def setUp(self):
        """Set up test fixtures."""
        if MULTI_OBJECTIVE_RL_AVAILABLE:
            # Create a temporary directory for model saving/loading
            self.test_dir = tempfile.mkdtemp()
            
            # Create a small MORL agent for testing
            self.morl = MultiObjectiveRLAgent(
                state_size=10,
                action_size=4,
                num_objectives=3,
                model_dir=self.test_dir,
                learning_rate=0.001,
                gamma=0.99,
                preference_learning_rate=0.01
            )
            
            # Sample state and action for testing
            self.sample_state = np.random.random(10)
            self.sample_next_state = np.random.random(10)
            self.sample_action = 1
            self.sample_rewards = [0.5, 0.3, 0.7]  # One reward per objective
            self.sample_done = False
        
    def tearDown(self):
        """Tear down test fixtures."""
        if hasattr(self, 'test_dir') and os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    @unittest.skipIf(not MULTI_OBJECTIVE_RL_AVAILABLE, "Multi-objective RL not available")
    def test_action_selection(self):
        """Test action selection functionality."""
        action = self.morl.get_action(self.sample_state)
        
        # Check if action is valid
        self.assertIsInstance(action, (int, np.integer))
        self.assertGreaterEqual(action, 0)
        self.assertLess(action, self.morl.action_size)

    @unittest.skipIf(not MULTI_OBJECTIVE_RL_AVAILABLE, "Multi-objective RL not available")
    def test_experience_storage(self):
        """Test experience storage functionality."""
        # Store an experience
        self.morl.remember(
            self.sample_state, self.sample_action, self.sample_rewards,
            self.sample_next_state, self.sample_done
        )
        
        # Check if experience was stored
        self.assertEqual(len(self.morl.replay_buffer), 1)
        
        # Check if rewards were recorded
        for i in range(self.morl.num_objectives):
            self.assertEqual(len(self.morl.reward_history[i]), 1)
            self.assertEqual(self.morl.reward_history[i][0], self.sample_rewards[i])

    @unittest.skipIf(not MULTI_OBJECTIVE_RL_AVAILABLE, "Multi-objective RL not available")
    def test_preference_update(self):
        """Test preference weight update functionality."""
        # Initial preference weights
        initial_weights = self.morl.preference_weights.copy()
        
        # Update preferences
        user_feedback = {
            "user_satisfaction": 0.8,
            "information_accuracy": 0.6,
            "response_diversity": 0.4
        }
        
        updated_weights = self.morl.update_preferences(user_feedback)
        
        # Check if weights were updated
        self.assertFalse(np.array_equal(updated_weights, initial_weights))
        
        # Check if weights sum to 1
        self.assertAlmostEqual(np.sum(updated_weights), 1.0, places=5)
        
        # Check if weights are in valid range
        self.assertTrue(np.all(updated_weights >= 0.0))
        self.assertTrue(np.all(updated_weights <= 1.0))

    @unittest.skipIf(not MULTI_OBJECTIVE_RL_AVAILABLE, "Multi-objective RL not available")
    def test_save_functionality(self):
        """Test model saving functionality."""
        # Save the model
        save_result = self.morl.save("test_morl")
        
        # Check if save was successful
        self.assertTrue(save_result)
        
        # Check if files were created
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, "test_morl_metadata.json")))
        
        # Check if objective-specific files were created
        for objective in self.morl.objective_names:
            self.assertTrue(os.path.exists(os.path.join(self.test_dir, f"test_morl_{objective}_q.h5")))

@unittest.skipIf(not TENSORFLOW_AVAILABLE, "TensorFlow not available")
class TestMetaLearning(unittest.TestCase):
    """Test cases for meta-learning."""

    def setUp(self):
        """Set up test fixtures."""
        if META_LEARNING_AVAILABLE:
            # Create a user profile for testing
            self.user_profile = UserProfile("test_user", "Test User")
            
            # Sample preferences for testing
            self.sample_preferences = {
                "response_length": 0.7,
                "formality": 0.3,
                "technical_level": 0.8
            }
            
            # Sample feedback for testing
            self.sample_feedback = {
                "rating": 4,
                "response_length": 0.6,
                "formality": 0.4,
                "topic": "artificial_intelligence"
            }
            
            # Sample interaction for testing
            self.sample_interaction = {
                "message_length": 120,
                "is_question": True,
                "user_input": "What is machine learning?",
                "ai_response": "Machine learning is a subset of artificial intelligence..."
            }
        
    @unittest.skipIf(not META_LEARNING_AVAILABLE, "Meta-learning not available")
    def test_user_profile_preferences(self):
        """Test user profile preference management."""
        # Initial preferences
        initial_prefs = self.user_profile.preferences.copy()
        
        # Update preferences
        self.user_profile.update_preferences(self.sample_preferences)
        
        # Check if preferences were updated
        for key, value in self.sample_preferences.items():
            self.assertNotEqual(self.user_profile.preferences[key], initial_prefs[key])
            # Check if the preference moved in the right direction
            if value > initial_prefs[key]:
                self.assertGreater(self.user_profile.preferences[key], initial_prefs[key])
            elif value < initial_prefs[key]:
                self.assertLess(self.user_profile.preferences[key], initial_prefs[key])

    @unittest.skipIf(not META_LEARNING_AVAILABLE, "Meta-learning not available")
    def test_user_profile_feedback(self):
        """Test user profile feedback processing."""
        # Initial topic interests
        initial_interests = self.user_profile.topic_interests.copy()
        
        # Update from feedback
        self.user_profile.update_from_feedback(self.sample_feedback)
        
        # Check if feedback was recorded
        self.assertEqual(len(self.user_profile.feedback_history), 1)
        
        # Check if topic interest was updated
        topic = self.sample_feedback["topic"]
        self.assertGreater(self.user_profile.topic_interests[topic], initial_interests[topic])

    @unittest.skipIf(not META_LEARNING_AVAILABLE, "Meta-learning not available")
    def test_user_profile_interaction(self):
        """Test user profile interaction processing."""
        # Initial interaction patterns
        initial_patterns = self.user_profile.interaction_patterns.copy()
        
        # Update from interaction
        self.user_profile.update_from_interaction(self.sample_interaction)
        
        # Check if conversation history was updated
        self.assertEqual(len(self.user_profile.conversation_history), 1)
        
        # Check if interaction patterns were updated
        self.assertNotEqual(
            self.user_profile.interaction_patterns["avg_message_length"],
            initial_patterns["avg_message_length"]
        )
        self.assertNotEqual(
            self.user_profile.interaction_patterns["question_frequency"],
            initial_patterns["question_frequency"]
        )

    @unittest.skipIf(not META_LEARNING_AVAILABLE, "Meta-learning not available")
    def test_user_profile_serialization(self):
        """Test user profile serialization."""
        # Update profile with some data
        self.user_profile.update_preferences(self.sample_preferences)
        self.user_profile.update_from_feedback(self.sample_feedback)
        self.user_profile.update_from_interaction(self.sample_interaction)
        
        # Convert to dictionary
        profile_dict = self.user_profile.to_dict()
        
        # Check if dictionary contains expected keys
        self.assertIn("user_id", profile_dict)
        self.assertIn("name", profile_dict)
        self.assertIn("preferences", profile_dict)
        self.assertIn("topic_interests", profile_dict)
        
        # Create a new profile from the dictionary
        new_profile = UserProfile.from_dict(profile_dict)
        
        # Check if the new profile has the same values
        self.assertEqual(new_profile.user_id, self.user_profile.user_id)
        self.assertEqual(new_profile.name, self.user_profile.name)
        for key, value in self.user_profile.preferences.items():
            self.assertEqual(new_profile.preferences[key], value)

if __name__ == '__main__':
    unittest.main()
