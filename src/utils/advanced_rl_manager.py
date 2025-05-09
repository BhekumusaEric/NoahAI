"""
Advanced Reinforcement Learning Manager for NoahAI

This module integrates all the advanced reinforcement learning components:
1. A/B Testing Framework
2. Transformer-Based State Representation
3. Hierarchical Reinforcement Learning
4. Multi-Objective Reinforcement Learning
5. Meta-Learning for User Adaptation
"""

import os
import json
import logging
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/advanced_rl.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("advanced_rl_manager")

# Try to import advanced RL components
try:
    from src.utils.ab_testing import ABTestingFramework
    from src.utils.transformer_state import TransformerStateRepresentation
    from src.utils.hierarchical_rl import HierarchicalRLAgent
    from src.utils.multi_objective_rl import MultiObjectiveRLAgent
    from src.utils.meta_learning import MAMLAgent, UserProfile
    ADVANCED_RL_AVAILABLE = True
except ImportError:
    ADVANCED_RL_AVAILABLE = False
    logger.warning("Advanced reinforcement learning components not available.")

# Try to import TensorFlow
try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    logger.warning("TensorFlow not available. Advanced RL capabilities will be limited.")

class AdvancedRLManager:
    """
    Advanced Reinforcement Learning Manager for NoahAI.
    
    This class integrates all the advanced reinforcement learning components
    and provides a unified interface for using them.
    """
    
    def __init__(self, model_dir="data/rl_models", config=None):
        """
        Initialize the Advanced RL Manager.
        
        Args:
            model_dir (str): Directory to save/load models
            config (dict, optional): Configuration for the RL components
        """
        self.model_dir = model_dir
        self.config = config or {}
        
        # Check if advanced RL components are available
        if not ADVANCED_RL_AVAILABLE:
            logger.error("Advanced reinforcement learning components not available.")
            raise ImportError("Advanced reinforcement learning components not available.")
        
        # Check if TensorFlow is available
        if not TENSORFLOW_AVAILABLE:
            logger.error("TensorFlow is required for advanced reinforcement learning.")
            raise ImportError("TensorFlow is required for advanced reinforcement learning.")
        
        # Create model directory if it doesn't exist
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(os.path.join(self.model_dir, "hierarchical"), exist_ok=True)
        os.makedirs(os.path.join(self.model_dir, "multi_objective"), exist_ok=True)
        os.makedirs(os.path.join(self.model_dir, "meta_learning"), exist_ok=True)
        
        # Initialize components based on configuration
        self.use_transformer_state = self.config.get("use_transformer_state", True)
        self.use_hierarchical_rl = self.config.get("use_hierarchical_rl", False)
        self.use_multi_objective_rl = self.config.get("use_multi_objective_rl", False)
        self.use_meta_learning = self.config.get("use_meta_learning", False)
        self.use_ab_testing = self.config.get("use_ab_testing", False)
        
        # Initialize state representation
        if self.use_transformer_state:
            self.state_representation = TransformerStateRepresentation(
                state_size=self.config.get("state_size", 128),
                model_name=self.config.get("transformer_model", "distilbert-base-uncased"),
                use_attention=self.config.get("use_attention", True),
                max_sequence_length=self.config.get("max_sequence_length", 512),
                cache_dir=os.path.join(self.model_dir, "transformers")
            )
            logger.info("Initialized transformer-based state representation.")
        else:
            from src.utils.rl_manager import StateRepresentation
            self.state_representation = StateRepresentation(
                state_size=self.config.get("state_size", 128)
            )
            logger.info("Initialized standard state representation.")
        
        # Initialize A/B testing framework
        if self.use_ab_testing:
            self.ab_framework = ABTestingFramework(
                experiment_dir=os.path.join(self.model_dir, "experiments"),
                metrics_file="metrics.json"
            )
            logger.info("Initialized A/B testing framework.")
        else:
            self.ab_framework = None
        
        # Initialize RL agents based on configuration
        self.current_agent_type = self.config.get("agent_type", "hierarchical")
        self.agents = {}
        
        # Initialize hierarchical RL agent if enabled
        if self.use_hierarchical_rl:
            self.agents["hierarchical"] = HierarchicalRLAgent(
                state_size=self.config.get("state_size", 128),
                action_size=self.config.get("action_size", 10),
                num_options=self.config.get("num_options", 3),
                model_dir=os.path.join(self.model_dir, "hierarchical"),
                learning_rate=self.config.get("learning_rate", 0.001),
                gamma=self.config.get("gamma", 0.99),
                option_duration=self.config.get("option_duration", 5)
            )
            logger.info("Initialized hierarchical RL agent.")
        
        # Initialize multi-objective RL agent if enabled
        if self.use_multi_objective_rl:
            self.agents["multi_objective"] = MultiObjectiveRLAgent(
                state_size=self.config.get("state_size", 128),
                action_size=self.config.get("action_size", 10),
                model_dir=os.path.join(self.model_dir, "multi_objective"),
                learning_rate=self.config.get("learning_rate", 0.001),
                gamma=self.config.get("gamma", 0.99),
                scalarization_method=self.config.get("scalarization_method", "weighted_sum")
            )
            logger.info("Initialized multi-objective RL agent.")
        
        # Initialize meta-learning agent if enabled
        if self.use_meta_learning:
            self.agents["meta_learning"] = MAMLAgent(
                state_size=self.config.get("state_size", 128),
                action_size=self.config.get("action_size", 10),
                model_dir=os.path.join(self.model_dir, "meta_learning"),
                meta_learning_rate=self.config.get("meta_learning_rate", 0.001),
                adaptation_learning_rate=self.config.get("adaptation_learning_rate", 0.01),
                gamma=self.config.get("gamma", 0.99),
                inner_steps=self.config.get("inner_steps", 5),
                meta_batch_size=self.config.get("meta_batch_size", 4)
            )
            logger.info("Initialized meta-learning agent.")
        
        # Set current agent
        self.current_agent = self.agents.get(self.current_agent_type)
        if self.current_agent is None and self.agents:
            self.current_agent_type = next(iter(self.agents.keys()))
            self.current_agent = self.agents[self.current_agent_type]
        
        # Initialize user ID for meta-learning
        self.current_user_id = "default_user"
        
        # Training metrics
        self.train_step = 0
        self.episode_rewards = []
        self.episode_lengths = []
        self.mean_rewards = []
        
        # Current episode data
        self.current_episode = {
            "states": [],
            "actions": [],
            "rewards": [],
            "next_states": [],
            "dones": [],
            "values": [],
            "log_probs": [],
            "total_reward": 0,
            "length": 0
        }
        
        logger.info(f"Advanced RL Manager initialized with agent type: {self.current_agent_type}")
    
    def set_agent_type(self, agent_type):
        """
        Set the current agent type.
        
        Args:
            agent_type (str): Agent type to use
            
        Returns:
            bool: Whether the agent type was set successfully
        """
        if agent_type not in self.agents:
            logger.error(f"Unknown agent type: {agent_type}")
            return False
        
        self.current_agent_type = agent_type
        self.current_agent = self.agents[agent_type]
        logger.info(f"Set current agent type to: {agent_type}")
        return True
    
    def set_user_id(self, user_id, name=None):
        """
        Set the current user ID for meta-learning.
        
        Args:
            user_id (str): User ID
            name (str, optional): User name
            
        Returns:
            bool: Whether the user ID was set successfully
        """
        self.current_user_id = user_id
        
        # Create user profile if using meta-learning
        if self.use_meta_learning and "meta_learning" in self.agents:
            self.agents["meta_learning"].get_user_profile(user_id, name)
            logger.info(f"Set current user ID to: {user_id}")
            return True
        
        return False
    
    def get_state(self, user_input, ai_response, conversation_history):
        """
        Get the state representation for the current conversation.
        
        Args:
            user_input (str): User input
            ai_response (str): AI response
            conversation_history (list): Conversation history
            
        Returns:
            numpy.ndarray: State representation
        """
        return self.state_representation.get_state(
            user_input, ai_response, conversation_history
        )
    
    def get_action(self, state):
        """
        Get an action from the current agent.
        
        Args:
            state: Current state
            
        Returns:
            int: Selected action
        """
        if self.current_agent is None:
            logger.error("No agent available.")
            return np.random.randint(0, 10)  # Default action size
        
        if self.current_agent_type == "hierarchical":
            action, _ = self.current_agent.get_action(state)
            return action
        elif self.current_agent_type == "multi_objective":
            return self.current_agent.get_action(state)
        elif self.current_agent_type == "meta_learning":
            return self.current_agent.get_action(state, self.current_user_id)
        else:
            logger.error(f"Unknown agent type: {self.current_agent_type}")
            return np.random.randint(0, 10)  # Default action size
    
    def process_feedback(self, user_input, ai_response, rating, conversation_history):
        """
        Process user feedback and update the RL agents.
        
        Args:
            user_input (str): User input
            ai_response (str): AI response
            rating (int): User rating (1-5)
            conversation_history (list): Conversation history
            
        Returns:
            float: Reward value
        """
        # Get current state representation
        current_state = self.get_state(
            user_input, ai_response, conversation_history
        )
        
        reward = None
        
        # Process feedback with the current agent
        if self.current_agent_type == "hierarchical":
            # Get action and option
            action, option_index = self.current_agent.get_action(current_state)
            
            # Calculate reward (normalize rating to [-1, 1])
            reward = (rating - 3) / 2
            
            # Store experience
            self.current_agent.remember(
                current_state, action, reward, current_state, False, option_index
            )
        
        elif self.current_agent_type == "multi_objective":
            # Get action
            action = self.current_agent.get_action(current_state)
            
            # Evaluate objectives
            objective_values = self.current_agent.evaluate_objectives(
                user_input, ai_response, conversation_history
            )
            
            # Calculate scalarized reward
            reward = self.current_agent.scalarize_rewards(objective_values)
            
            # Store experience
            self.current_agent.remember(
                current_state, action, objective_values, current_state, False
            )
        
        elif self.current_agent_type == "meta_learning":
            # Get action
            action = self.current_agent.get_action(current_state, self.current_user_id)
            
            # Get user profile
            profile = self.current_agent.get_user_profile(self.current_user_id)
            
            # Get preference vector
            preference_vector = profile.get_preference_vector()
            
            # Calculate reward (normalize rating to [-1, 1])
            reward = (rating - 3) / 2
            
            # Create experience
            experience = (current_state, action, reward, current_state, False, preference_vector)
            
            # Adapt to user
            self.current_agent.adapt_to_user(self.current_user_id, [experience])
            
            # Update user profile based on feedback
            feedback = {
                "rating": rating,
                "response_length": len(ai_response) / 500,  # Normalize
                "formality": 0.5,  # Default value
                "technical_level": 0.5,  # Default value
                "verbosity": len(ai_response) / 500,  # Normalize
                "humor": 0.5  # Default value
            }
            
            profile.update_from_feedback(feedback)
        
        # Update current episode data
        self.current_episode["states"].append(current_state)
        self.current_episode["rewards"].append(reward if reward is not None else 0)
        self.current_episode["total_reward"] += reward if reward is not None else 0
        self.current_episode["length"] += 1
        
        # If this is the first interaction, we don't have an action yet
        if len(self.current_episode["actions"]) < len(self.current_episode["states"]):
            # Get action for the current state
            if self.current_agent_type == "hierarchical":
                action, _ = self.current_agent.get_action(current_state)
                self.current_episode["actions"].append(action)
                self.current_episode["next_states"].append(current_state)  # Same state for now
                self.current_episode["dones"].append(False)
            elif self.current_agent_type == "multi_objective":
                action = self.current_agent.get_action(current_state)
                self.current_episode["actions"].append(action)
                self.current_episode["next_states"].append(current_state)  # Same state for now
                self.current_episode["dones"].append(False)
            elif self.current_agent_type == "meta_learning":
                action = self.current_agent.get_action(current_state, self.current_user_id)
                self.current_episode["actions"].append(action)
                self.current_episode["next_states"].append(current_state)  # Same state for now
                self.current_episode["dones"].append(False)
        
        # Record experiment data if using A/B testing
        if self.use_ab_testing and self.ab_framework:
            # Record metric for the current agent type
            self.ab_framework.record_metric(
                experiment_id=self.config.get("experiment_id", "default_experiment"),
                variant_name=self.current_agent_type,
                metric_name="reward",
                value=reward if reward is not None else 0
            )
        
        return reward
    
    def train(self):
        """
        Train the current RL agent using collected experiences.
        
        Returns:
            dict: Training results
        """
        if self.current_agent is None:
            logger.error("No agent available for training.")
            return {"success": False, "message": "No agent available for training."}
        
        # Check if we have enough data
        if len(self.current_episode["states"]) < 2:
            logger.info("Not enough data for training.")
            return {"success": False, "message": "Not enough data for training."}
        
        # Mark the last interaction as done
        if self.current_episode["dones"]:
            self.current_episode["dones"][-1] = True
        
        # Train the agent based on the type
        if self.current_agent_type == "hierarchical":
            results = self.current_agent.train()
        elif self.current_agent_type == "multi_objective":
            results = self.current_agent.train()
        elif self.current_agent_type == "meta_learning":
            # Meta-learning is trained during feedback processing
            results = {"success": True, "message": "Meta-learning is trained during feedback processing."}
        else:
            logger.error(f"Unknown agent type: {self.current_agent_type}")
            return {"success": False, "message": f"Unknown agent type: {self.current_agent_type}"}
        
        # Update training metrics
        self.train_step += 1
        self.episode_rewards.append(self.current_episode["total_reward"])
        self.episode_lengths.append(self.current_episode["length"])
        
        # Track mean rewards (for performance evaluation)
        mean_reward = self.current_episode["total_reward"] / max(1, self.current_episode["length"])
        self.mean_rewards.append(mean_reward)
        
        # Log training progress
        logger.info(f"Training step {self.train_step}: total_reward={self.current_episode['total_reward']:.4f}, "
                   f"mean_reward={mean_reward:.4f}, length={self.current_episode['length']}")
        
        # Reset current episode
        self.current_episode = {
            "states": [],
            "actions": [],
            "rewards": [],
            "next_states": [],
            "dones": [],
            "values": [],
            "log_probs": [],
            "total_reward": 0,
            "length": 0
        }
        
        # Add mean reward to result
        if isinstance(results, dict):
            results["mean_reward"] = mean_reward
            results["train_step"] = self.train_step
            results["agent_type"] = self.current_agent_type
        
        return results
    
    def save(self, filename=None):
        """
        Save all RL agents.
        
        Args:
            filename (str, optional): Filename prefix to save the models
            
        Returns:
            bool: Whether the models were saved successfully
        """
        try:
            if filename is None:
                filename = f"advanced_rl_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Save each agent
            for agent_type, agent in self.agents.items():
                agent_filename = f"{filename}_{agent_type}"
                agent.save(agent_filename)
                logger.info(f"Saved {agent_type} agent: {agent_filename}")
            
            # Save manager metadata
            metadata = {
                "current_agent_type": self.current_agent_type,
                "use_transformer_state": self.use_transformer_state,
                "use_hierarchical_rl": self.use_hierarchical_rl,
                "use_multi_objective_rl": self.use_multi_objective_rl,
                "use_meta_learning": self.use_meta_learning,
                "use_ab_testing": self.use_ab_testing,
                "config": self.config,
                "train_step": self.train_step,
                "episode_rewards": self.episode_rewards,
                "episode_lengths": self.episode_lengths,
                "mean_rewards": self.mean_rewards,
                "current_user_id": self.current_user_id,
                "saved_at": datetime.now().isoformat()
            }
            
            metadata_path = os.path.join(self.model_dir, f"{filename}_manager.json")
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Advanced RL manager saved to {metadata_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving advanced RL manager: {e}")
            return False
    
    def load(self, filename):
        """
        Load saved RL agents.
        
        Args:
            filename (str): Filename prefix to load the models
            
        Returns:
            bool: Whether the models were loaded successfully
        """
        try:
            # Load manager metadata
            metadata_path = os.path.join(self.model_dir, f"{filename}_manager.json")
            if not os.path.exists(metadata_path):
                logger.error(f"Metadata file not found: {metadata_path}")
                return False
            
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            # Update manager attributes
            self.current_agent_type = metadata.get("current_agent_type", self.current_agent_type)
            self.use_transformer_state = metadata.get("use_transformer_state", self.use_transformer_state)
            self.use_hierarchical_rl = metadata.get("use_hierarchical_rl", self.use_hierarchical_rl)
            self.use_multi_objective_rl = metadata.get("use_multi_objective_rl", self.use_multi_objective_rl)
            self.use_meta_learning = metadata.get("use_meta_learning", self.use_meta_learning)
            self.use_ab_testing = metadata.get("use_ab_testing", self.use_ab_testing)
            self.config = metadata.get("config", self.config)
            self.train_step = metadata.get("train_step", 0)
            self.episode_rewards = metadata.get("episode_rewards", [])
            self.episode_lengths = metadata.get("episode_lengths", [])
            self.mean_rewards = metadata.get("mean_rewards", [])
            self.current_user_id = metadata.get("current_user_id", "default_user")
            
            # Load each agent
            for agent_type, agent in self.agents.items():
                agent_filename = f"{filename}_{agent_type}"
                success = agent.load(agent_filename)
                if success:
                    logger.info(f"Loaded {agent_type} agent: {agent_filename}")
                else:
                    logger.error(f"Error loading {agent_type} agent: {agent_filename}")
            
            # Set current agent
            self.current_agent = self.agents.get(self.current_agent_type)
            if self.current_agent is None and self.agents:
                self.current_agent_type = next(iter(self.agents.keys()))
                self.current_agent = self.agents[self.current_agent_type]
            
            logger.info(f"Advanced RL manager loaded from {metadata_path}")
            return True
        except Exception as e:
            logger.error(f"Error loading advanced RL manager: {e}")
            return False
    
    def get_explanation(self, state, action, user_input, ai_response):
        """
        Get an explanation for the agent's decision.
        
        Args:
            state: Current state
            action: Selected action
            user_input (str): User input
            ai_response (str): AI response
            
        Returns:
            dict: Explanation data
        """
        explanation = {
            "agent_type": self.current_agent_type,
            "action": action,
            "confidence": None,
            "factors": [],
            "alternatives": []
        }
        
        # Add agent-specific explanations
        if self.current_agent_type == "hierarchical":
            # Get option information
            _, option_index = self.current_agent.get_action(state)
            explanation["option"] = option_index
            explanation["factors"].append(f"Using option {option_index} for this context")
            
            # Add option description
            option_descriptions = [
                "General conversation option",
                "Question answering option",
                "Technical explanation option"
            ]
            if option_index < len(option_descriptions):
                explanation["option_description"] = option_descriptions[option_index]
                explanation["factors"].append(f"Option purpose: {option_descriptions[option_index]}")
        
        elif self.current_agent_type == "multi_objective":
            # Get objective values
            objective_values = self.current_agent.evaluate_objectives(
                user_input, ai_response, []  # Empty conversation history for simplicity
            )
            
            # Add objective values to explanation
            objective_names = ["engagement", "informativeness", "coherence"]
            for i, name in enumerate(objective_names):
                if i < len(objective_values):
                    explanation["factors"].append(f"{name.capitalize()}: {objective_values[i]:.2f}")
            
            # Add scalarized reward
            reward = self.current_agent.scalarize_rewards(objective_values)
            explanation["confidence"] = (reward + 1) / 2  # Normalize to [0, 1]
            explanation["factors"].append(f"Overall reward: {reward:.2f}")
        
        elif self.current_agent_type == "meta_learning":
            # Get user profile
            profile = self.current_agent.get_user_profile(self.current_user_id)
            
            # Add user preferences to explanation
            explanation["factors"].append("User preferences:")
            for pref_name, pref_value in profile.preferences.items():
                explanation["factors"].append(f"- {pref_name}: {pref_value:.2f}")
            
            # Add adaptation rate
            explanation["factors"].append(f"Adaptation rate: {profile.adaptation_rate:.2f}")
            explanation["confidence"] = 0.5  # Default confidence
        
        return explanation
