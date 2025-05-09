"""
Reinforcement Learning Manager for NoahAI

This module provides a central manager for reinforcement learning algorithms
and integrates them with NoahAI.
"""

import os
import json
import numpy as np
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/reinforcement_learning.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("rl_manager")

# Try to import reinforcement learning modules
try:
    from src.utils.reinforcement_learning import DeepQNetwork, AdvantageActorCritic, ExperienceReplay
    from src.utils.ppo_agent import PPOAgent
    RL_MODULES_AVAILABLE = True
except ImportError:
    RL_MODULES_AVAILABLE = False
    logger.warning("Reinforcement learning modules not available.")

# Try to import TensorFlow
try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    logger.warning("TensorFlow not available. Advanced RL capabilities will be limited.")

class RLManager:
    """
    Reinforcement Learning Manager for NoahAI.

    This class provides a central manager for reinforcement learning algorithms
    and integrates them with NoahAI.
    """

    def __init__(self, model_dir="data/rl_models", algorithm="ppo",
                 state_size=128, action_size=10, config=None):
        """
        Initialize the RL Manager.

        Args:
            model_dir (str): Directory to save/load models
            algorithm (str): RL algorithm to use ("dqn", "a2c", "ppo")
            state_size (int): Size of the state space
            action_size (int): Size of the action space
            config (dict, optional): Configuration for the RL algorithm
        """
        self.model_dir = model_dir
        self.algorithm = algorithm.lower()
        self.state_size = state_size
        self.action_size = action_size
        self.config = config or {}

        # Check if RL modules are available
        if not RL_MODULES_AVAILABLE:
            logger.error("Reinforcement learning modules not available.")
            raise ImportError("Reinforcement learning modules not available.")

        # Check if TensorFlow is available
        if not TENSORFLOW_AVAILABLE:
            logger.error("TensorFlow is required for reinforcement learning.")
            raise ImportError("TensorFlow is required for reinforcement learning.")

        # Create model directory if it doesn't exist
        os.makedirs(self.model_dir, exist_ok=True)

        # Initialize the RL agent
        self.agent = self._initialize_agent()

        # Initialize state representation
        self.state_representation = StateRepresentation(state_size)

        # Initialize reward function with weights from config
        reward_weights = config.get("reward_weights", {})
        self.reward_function = RewardFunction(reward_weights)

        # Log reward weights if provided
        if reward_weights:
            logger.info(f"Using custom reward weights: {reward_weights}")

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

    def _initialize_agent(self):
        """
        Initialize the RL agent based on the selected algorithm.

        Returns:
            object: The RL agent
        """
        if self.algorithm == "dqn":
            return DeepQNetwork(
                state_size=self.state_size,
                action_size=self.action_size,
                model_dir=self.model_dir,
                learning_rate=self.config.get("learning_rate", 0.001),
                gamma=self.config.get("gamma", 0.95),
                epsilon=self.config.get("epsilon", 1.0),
                epsilon_decay=self.config.get("epsilon_decay", 0.995),
                epsilon_min=self.config.get("epsilon_min", 0.01),
                batch_size=self.config.get("batch_size", 32),
                update_target_freq=self.config.get("update_target_freq", 100)
            )
        elif self.algorithm == "a2c":
            return AdvantageActorCritic(
                state_size=self.state_size,
                action_size=self.action_size,
                model_dir=self.model_dir,
                actor_lr=self.config.get("actor_lr", 0.001),
                critic_lr=self.config.get("critic_lr", 0.002),
                gamma=self.config.get("gamma", 0.95),
                entropy_beta=self.config.get("entropy_beta", 0.01),
                batch_size=self.config.get("batch_size", 32)
            )
        elif self.algorithm == "ppo":
            return PPOAgent(
                state_size=self.state_size,
                action_size=self.action_size,
                model_dir=self.model_dir,
                actor_lr=self.config.get("actor_lr", 0.0003),
                critic_lr=self.config.get("critic_lr", 0.001),
                gamma=self.config.get("gamma", 0.99),
                clip_ratio=self.config.get("clip_ratio", 0.2),
                target_kl=self.config.get("target_kl", 0.01),
                entropy_coef=self.config.get("entropy_coef", 0.01),
                batch_size=self.config.get("batch_size", 64),
                epochs=self.config.get("epochs", 10),
                lam=self.config.get("lam", 0.95)
            )
        else:
            logger.error(f"Unknown algorithm: {self.algorithm}")
            raise ValueError(f"Unknown algorithm: {self.algorithm}")

    def get_action(self, state):
        """
        Get an action from the RL agent.

        Args:
            state: Current state

        Returns:
            int: Selected action
        """
        if self.algorithm == "dqn":
            return self.agent.act(state)
        elif self.algorithm == "a2c":
            action, _ = self.agent.act(state)
            return action
        elif self.algorithm == "ppo":
            action, log_prob, value = self.agent.act(state)
            return action
        else:
            logger.error(f"Unknown algorithm: {self.algorithm}")
            raise ValueError(f"Unknown algorithm: {self.algorithm}")

    def process_feedback(self, user_input, ai_response, rating, conversation_history):
        """
        Process user feedback and update the RL agent.

        Args:
            user_input (str): User input
            ai_response (str): AI response
            rating (int): User rating (1-5)
            conversation_history (list): Conversation history

        Returns:
            float: Reward value
        """
        # Get current state representation
        current_state = self.state_representation.get_state(
            user_input, ai_response, conversation_history
        )

        # Calculate reward
        reward = self.reward_function.calculate_reward(
            user_input, ai_response, rating, conversation_history
        )

        # Update current episode data
        self.current_episode["states"].append(current_state)
        self.current_episode["rewards"].append(reward)
        self.current_episode["total_reward"] += reward
        self.current_episode["length"] += 1

        # If this is the first interaction, we don't have an action yet
        if len(self.current_episode["actions"]) < len(self.current_episode["states"]):
            # Get action for the current state
            if self.algorithm == "dqn":
                action = self.agent.act(current_state, explore=False)
                self.current_episode["actions"].append(action)
                self.current_episode["next_states"].append(current_state)  # Same state for now
                self.current_episode["dones"].append(False)
            elif self.algorithm == "a2c":
                action, log_prob = self.agent.act(current_state)
                self.current_episode["actions"].append(action)
                self.current_episode["next_states"].append(current_state)  # Same state for now
                self.current_episode["dones"].append(False)
                self.current_episode["log_probs"].append(log_prob)
            elif self.algorithm == "ppo":
                action, log_prob, value = self.agent.act(current_state)
                self.current_episode["actions"].append(action)
                self.current_episode["next_states"].append(current_state)  # Same state for now
                self.current_episode["dones"].append(False)
                self.current_episode["log_probs"].append(log_prob)
                self.current_episode["values"].append(value)

        return reward

    def train(self):
        """
        Train the RL agent using collected experiences.

        Returns:
            dict: Training results
        """
        # Check if we have enough data
        if len(self.current_episode["states"]) < 2:
            logger.info("Not enough data for training.")
            return {"success": False, "message": "Not enough data for training."}

        # Mark the last interaction as done
        if self.current_episode["dones"]:
            self.current_episode["dones"][-1] = True

        # Train the agent based on the algorithm
        if self.algorithm == "dqn":
            # Add experiences to memory
            for i in range(len(self.current_episode["states"]) - 1):
                self.agent.remember(
                    self.current_episode["states"][i],
                    self.current_episode["actions"][i],
                    self.current_episode["rewards"][i],
                    self.current_episode["next_states"][i],
                    self.current_episode["dones"][i]
                )

            # Train the agent
            loss = self.agent.replay()

            result = {
                "success": loss is not None,
                "loss": loss,
                "epsilon": self.agent.epsilon,
                "total_reward": self.current_episode["total_reward"],
                "episode_length": self.current_episode["length"]
            }
        elif self.algorithm == "a2c":
            # Add experiences to memory
            for i in range(len(self.current_episode["states"]) - 1):
                self.agent.remember(
                    self.current_episode["states"][i],
                    self.current_episode["actions"][i],
                    self.current_episode["rewards"][i],
                    self.current_episode["next_states"][i],
                    self.current_episode["dones"][i]
                )

            # Train the agent
            actor_loss, critic_loss = self.agent.train()

            result = {
                "success": actor_loss is not None and critic_loss is not None,
                "actor_loss": actor_loss,
                "critic_loss": critic_loss,
                "total_reward": self.current_episode["total_reward"],
                "episode_length": self.current_episode["length"]
            }
        elif self.algorithm == "ppo":
            # Add experiences to memory
            for i in range(len(self.current_episode["states"]) - 1):
                self.agent.remember(
                    self.current_episode["states"][i],
                    self.current_episode["actions"][i],
                    self.current_episode["rewards"][i],
                    self.current_episode["values"][i],
                    self.current_episode["log_probs"][i],
                    self.current_episode["dones"][i]
                )

            # Train the agent
            actor_loss, critic_loss = self.agent.train()

            result = {
                "success": actor_loss is not None and critic_loss is not None,
                "actor_loss": actor_loss,
                "critic_loss": critic_loss,
                "total_reward": self.current_episode["total_reward"],
                "episode_length": self.current_episode["length"]
            }
        else:
            logger.error(f"Unknown algorithm: {self.algorithm}")
            raise ValueError(f"Unknown algorithm: {self.algorithm}")

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
        if isinstance(result, dict):
            result["mean_reward"] = mean_reward

            # Add exploration metrics if available
            if hasattr(self.agent, 'exploration_strategy'):
                result["exploration_strategy"] = self.agent.exploration_strategy

                if hasattr(self.agent, 'entropy_coef'):
                    result["entropy_coef"] = self.agent.entropy_coef

            # Add training progress metrics
            result["train_step"] = self.train_step
            result["mean_rewards"] = self.mean_rewards[-10:] if len(self.mean_rewards) > 10 else self.mean_rewards

        return result

    def save(self, filename=None):
        """
        Save the RL agent.

        Args:
            filename (str, optional): Filename to save the model

        Returns:
            bool: Whether the model was saved successfully
        """
        try:
            if filename is None:
                filename = f"{self.algorithm}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            # Save the agent
            if self.algorithm == "dqn":
                self.agent.save(filename)
            elif self.algorithm == "a2c":
                self.agent.save(filename)
            elif self.algorithm == "ppo":
                self.agent.save(filename)

            # Save manager metadata
            metadata = {
                "algorithm": self.algorithm,
                "state_size": self.state_size,
                "action_size": self.action_size,
                "config": self.config,
                "train_step": self.train_step,
                "episode_rewards": self.episode_rewards,
                "episode_lengths": self.episode_lengths,
                "mean_rewards": self.mean_rewards,
                "saved_at": datetime.now().isoformat()
            }

            # Add exploration metrics if available
            if hasattr(self.agent, 'exploration_strategy'):
                metadata["exploration_strategy"] = self.agent.exploration_strategy

                if hasattr(self.agent, 'entropy_coef'):
                    metadata["entropy_coef"] = self.agent.entropy_coef

                if hasattr(self.agent, 'mean_entropy_history') and self.agent.mean_entropy_history:
                    metadata["mean_entropy_history"] = self.agent.mean_entropy_history

            metadata_path = os.path.join(self.model_dir, f"{filename}_manager.json")
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)

            logger.info(f"RL manager saved to {metadata_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving RL manager: {e}")
            return False

    def load(self, filename):
        """
        Load a saved RL agent.

        Args:
            filename (str): Filename to load the model

        Returns:
            bool: Whether the model was loaded successfully
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
            self.algorithm = metadata.get("algorithm", self.algorithm)
            self.state_size = metadata.get("state_size", self.state_size)
            self.action_size = metadata.get("action_size", self.action_size)
            self.config = metadata.get("config", self.config)
            self.train_step = metadata.get("train_step", 0)
            self.episode_rewards = metadata.get("episode_rewards", [])
            self.episode_lengths = metadata.get("episode_lengths", [])
            self.mean_rewards = metadata.get("mean_rewards", [])

            # Update agent exploration parameters if available
            if hasattr(self.agent, 'exploration_strategy') and "exploration_strategy" in metadata:
                self.agent.exploration_strategy = metadata["exploration_strategy"]

            if hasattr(self.agent, 'entropy_coef') and "entropy_coef" in metadata:
                self.agent.entropy_coef = metadata["entropy_coef"]

            if hasattr(self.agent, 'mean_entropy_history') and "mean_entropy_history" in metadata:
                self.agent.mean_entropy_history = metadata["mean_entropy_history"]

            # Re-initialize the agent
            self.agent = self._initialize_agent()

            # Load the agent
            if self.algorithm == "dqn":
                model_path = os.path.join(self.model_dir, f"{filename}.h5")
                success = self.agent.load(model_path)
            elif self.algorithm == "a2c":
                actor_path = os.path.join(self.model_dir, f"{filename}_actor.h5")
                critic_path = os.path.join(self.model_dir, f"{filename}_critic.h5")
                success = self.agent.load(actor_path, critic_path, metadata_path)
            elif self.algorithm == "ppo":
                actor_path = os.path.join(self.model_dir, f"{filename}_actor.h5")
                critic_path = os.path.join(self.model_dir, f"{filename}_critic.h5")
                success = self.agent.load(actor_path, critic_path, metadata_path)
            else:
                logger.error(f"Unknown algorithm: {self.algorithm}")
                return False

            if not success:
                logger.error(f"Error loading RL agent: {filename}")
                return False

            logger.info(f"RL manager loaded from {metadata_path}")
            return True
        except Exception as e:
            logger.error(f"Error loading RL manager: {e}")
            return False

class StateRepresentation:
    """
    State representation for reinforcement learning.

    This class converts conversation data into a numerical state
    representation for the RL agent.
    """

    def __init__(self, state_size=128):
        """
        Initialize the state representation.

        Args:
            state_size (int): Size of the state vector
        """
        self.state_size = state_size

        # Try to import TensorFlow for text embedding
        try:
            import tensorflow as tf
            import tensorflow_hub as hub
            self.use_tf_embedding = True

            # Load Universal Sentence Encoder
            try:
                self.embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
                logger.info("Loaded Universal Sentence Encoder for state representation.")
            except Exception as e:
                logger.warning(f"Error loading Universal Sentence Encoder: {e}")
                self.use_tf_embedding = False
        except ImportError:
            self.use_tf_embedding = False
            logger.warning("TensorFlow Hub not available. Using simple state representation.")

    def get_state(self, user_input, ai_response, conversation_history):
        """
        Convert conversation data into a state representation.

        Args:
            user_input (str): User input
            ai_response (str): AI response
            conversation_history (list): Conversation history

        Returns:
            numpy.ndarray: State representation
        """
        if self.use_tf_embedding:
            return self._get_state_with_embedding(user_input, ai_response, conversation_history)
        else:
            return self._get_simple_state(user_input, ai_response, conversation_history)

    def _get_state_with_embedding(self, user_input, ai_response, conversation_history):
        """
        Get state representation using TensorFlow embedding.

        Args:
            user_input (str): User input
            ai_response (str): AI response
            conversation_history (list): Conversation history

        Returns:
            numpy.ndarray: State representation
        """
        # Combine recent conversation history
        recent_history = conversation_history[-5:] if len(conversation_history) > 5 else conversation_history

        # Create a combined text for embedding
        combined_text = user_input + " [SEP] " + ai_response

        # Add recent history
        for turn in recent_history:
            role, text = turn
            combined_text += f" [SEP] {role}: {text}"

        # Get embedding
        embedding = self.embed([combined_text])[0].numpy()

        # Resize to state_size
        if len(embedding) > self.state_size:
            embedding = embedding[:self.state_size]
        elif len(embedding) < self.state_size:
            padding = np.zeros(self.state_size - len(embedding))
            embedding = np.concatenate([embedding, padding])

        return embedding

    def _get_simple_state(self, user_input, ai_response, conversation_history):
        """
        Get a simple state representation without TensorFlow.

        Args:
            user_input (str): User input
            ai_response (str): AI response
            conversation_history (list): Conversation history

        Returns:
            numpy.ndarray: State representation
        """
        # Initialize state vector
        state = np.zeros(self.state_size)

        # Simple features
        features = [
            len(user_input),                                # Length of user input
            len(ai_response),                               # Length of AI response
            len(conversation_history),                      # Conversation length
            user_input.count('?') > 0,                      # User asked a question
            user_input.count('!') > 0,                      # User used exclamation
            ai_response.count('?') > 0,                     # AI asked a question
            ai_response.count('!') > 0,                     # AI used exclamation
            len(user_input.split()) / 20,                   # Normalized word count (user)
            len(ai_response.split()) / 50,                  # Normalized word count (AI)
            sum(1 for c in user_input if c.isupper()) / len(user_input) if len(user_input) > 0 else 0,  # Uppercase ratio (user)
            sum(1 for c in ai_response if c.isupper()) / len(ai_response) if len(ai_response) > 0 else 0,  # Uppercase ratio (AI)
        ]

        # Add features to state vector
        for i, feature in enumerate(features):
            if i < self.state_size:
                state[i] = feature

        # Add character frequency features
        char_offset = len(features)
        for i, c in enumerate("abcdefghijklmnopqrstuvwxyz"):
            if char_offset + i < self.state_size:
                state[char_offset + i] = user_input.lower().count(c) / len(user_input) if len(user_input) > 0 else 0

        return state

class RewardFunction:
    """
    Enhanced reward function for reinforcement learning.

    This class calculates sophisticated rewards based on user feedback,
    conversation quality, and context.
    """

    def __init__(self, reward_weights=None, config=None):
        """
        Initialize the reward function.

        Args:
            reward_weights (dict, optional): Weights for different reward components
            config (dict, optional): Configuration for the reward function
        """
        self.config = config or {}
        reward_weights = reward_weights or {}

        # Default weights for different reward components
        self.rating_weight = reward_weights.get("rating_weight", 1.0)
        self.length_weight = reward_weights.get("length_weight", 0.2)
        self.sentiment_weight = reward_weights.get("sentiment_weight", 0.3)
        self.engagement_weight = reward_weights.get("engagement_weight", 0.5)
        self.coherence_weight = reward_weights.get("coherence_weight", 0.4)
        self.informativeness_weight = reward_weights.get("informativeness_weight", 0.4)
        self.helpfulness_weight = reward_weights.get("helpfulness_weight", 0.6)
        self.consistency_weight = reward_weights.get("consistency_weight", 0.3)

        # Try to import NLP utilities for sentiment analysis
        try:
            from src.utils.nlp_utils import NLPUtils
            self.nlp_utils = NLPUtils()
            self.use_nlp = True
            logger.info("Using NLP utilities for reward calculation.")
        except ImportError:
            self.use_nlp = False
            logger.warning("NLP utilities not available. Using simple reward calculation.")

    def calculate_reward(self, user_input, ai_response, rating, conversation_history):
        """
        Calculate reward based on user feedback and conversation context.

        Args:
            user_input (str): User input
            ai_response (str): AI response
            rating (int): User rating (1-5)
            conversation_history (list): Conversation history

        Returns:
            float: Reward value
        """
        # Base reward from rating (normalize to [-1, 1] range)
        rating_reward = (rating - 3) / 2

        # Length reward (penalize very short or very long responses)
        response_length = len(ai_response)
        if response_length < 10:
            length_reward = -0.5  # Too short
        elif response_length > 500:
            length_reward = -0.3  # Too long
        elif response_length > 200:
            length_reward = 0.1   # Good length
        else:
            length_reward = 0.2   # Ideal length

        # Sentiment reward
        sentiment_reward = 0
        if self.use_nlp:
            # Analyze sentiment of user input and response
            user_sentiment = self.nlp_utils.sentiment_analysis(user_input)
            ai_sentiment = self.nlp_utils.sentiment_analysis(ai_response)

            # Reward for matching sentiment or improving negative sentiment
            if user_sentiment["label"] == "negative" and ai_sentiment["label"] == "positive":
                sentiment_reward = 0.3  # Improved negative sentiment
            elif user_sentiment["label"] == ai_sentiment["label"]:
                sentiment_reward = 0.2  # Matched sentiment
            else:
                sentiment_reward = 0.0  # Neutral

        # Engagement reward
        engagement_reward = 0
        if len(conversation_history) > 1:
            # Reward for maintaining conversation
            engagement_reward = 0.1 * min(len(conversation_history) / 10, 1.0)

            # Additional reward for asking questions (encouraging engagement)
            if "?" in ai_response:
                engagement_reward += 0.2

            # Reward for longer conversations (user is engaged)
            if len(conversation_history) > 5:
                engagement_reward += 0.3

        # Coherence reward - how well the response relates to the user input
        coherence_reward = 0
        if self.use_nlp:
            # Calculate semantic similarity between user input and AI response
            try:
                coherence_score = self.nlp_utils.calculate_similarity(user_input, ai_response)
                # Transform to reward: we want similarity but not too high (which might indicate repetition)
                if coherence_score < 0.2:
                    coherence_reward = -0.2  # Too unrelated
                elif coherence_score > 0.9:
                    coherence_reward = 0.0   # Too similar (might be repetitive)
                else:
                    coherence_reward = 0.3   # Good coherence
            except:
                # Fallback if similarity calculation fails
                coherence_reward = 0.0

        # Informativeness reward - based on response complexity and information content
        informativeness_reward = 0

        # Simple heuristic: count unique words as a proxy for information content
        unique_words = len(set(ai_response.lower().split()))
        total_words = len(ai_response.split())

        if total_words > 0:
            # Lexical diversity (unique words / total words)
            lexical_diversity = unique_words / total_words

            if lexical_diversity < 0.4:
                informativeness_reward = -0.1  # Low diversity, repetitive
            elif lexical_diversity > 0.8:
                informativeness_reward = 0.2   # High diversity, informative
            else:
                informativeness_reward = 0.1   # Moderate diversity

            # Reward for including specific information markers
            info_markers = ["for example", "such as", "specifically", "in particular",
                           "according to", "research shows", "studies indicate"]
            for marker in info_markers:
                if marker in ai_response.lower():
                    informativeness_reward += 0.05

            # Cap the informativeness reward
            informativeness_reward = min(informativeness_reward, 0.5)

        # Helpfulness reward - based on response structure and content
        helpfulness_reward = 0

        # Check for helpful response structures
        if any(marker in ai_response.lower() for marker in ["here's how", "you can", "try this", "steps to"]):
            helpfulness_reward += 0.2

        # Check if response addresses user questions
        if "?" in user_input and any(marker in ai_response.lower() for marker in ["yes", "no", "maybe", "possibly"]):
            helpfulness_reward += 0.1

        # Consistency reward - consistency with previous responses
        consistency_reward = 0
        if len(conversation_history) > 2:
            # Get previous AI responses
            ai_responses = [msg[1] for msg in conversation_history if msg[0] == "ai"]
            if ai_responses and len(ai_responses) >= 2:
                # Check for contradictions (very simple heuristic)
                prev_response = ai_responses[-1]
                if self.use_nlp:
                    try:
                        consistency_score = self.nlp_utils.calculate_similarity(prev_response, ai_response)
                        # We want some consistency but not too much
                        if consistency_score < 0.1:
                            consistency_reward = -0.2  # Potentially contradictory
                        elif consistency_score > 0.8:
                            consistency_reward = -0.1  # Too repetitive
                        else:
                            consistency_reward = 0.2   # Good consistency
                    except:
                        consistency_reward = 0.0

        # Combine rewards with weights
        total_reward = (
            self.rating_weight * rating_reward +
            self.length_weight * length_reward +
            self.sentiment_weight * sentiment_reward +
            self.engagement_weight * engagement_reward +
            self.coherence_weight * coherence_reward +
            self.informativeness_weight * informativeness_reward +
            self.helpfulness_weight * helpfulness_reward +
            self.consistency_weight * consistency_reward
        )

        # Log detailed reward components for analysis
        logger.debug(f"Reward components: rating={rating_reward:.2f}, length={length_reward:.2f}, " +
                    f"sentiment={sentiment_reward:.2f}, engagement={engagement_reward:.2f}, " +
                    f"coherence={coherence_reward:.2f}, informativeness={informativeness_reward:.2f}, " +
                    f"helpfulness={helpfulness_reward:.2f}, consistency={consistency_reward:.2f}, " +
                    f"total={total_reward:.2f}")

        return total_reward
