"""
Meta-Learning for User Adaptation in NoahAI

This module implements meta-learning techniques to quickly adapt to
different user preferences and conversation styles.
"""

import os
import json
import numpy as np
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional, Union
from collections import defaultdict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/meta_learning.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("meta_learning")

# Try to import TensorFlow
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, Model, load_model, clone_model
    from tensorflow.keras.layers import Dense, Dropout, LSTM, Input, Concatenate
    from tensorflow.keras.optimizers import Adam
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    logger.warning("TensorFlow not available. Meta-learning will not be available.")

class UserProfile:
    """
    User profile for personalized reinforcement learning.
    
    This class stores user preferences and conversation patterns
    to enable personalized responses.
    """
    
    def __init__(self, user_id, name=None):
        """
        Initialize a user profile.
        
        Args:
            user_id (str): Unique identifier for the user
            name (str, optional): User's name
        """
        self.user_id = user_id
        self.name = name or user_id
        
        # User preferences
        self.preferences = {
            "response_length": 0.5,  # 0.0 = short, 1.0 = long
            "formality": 0.5,        # 0.0 = casual, 1.0 = formal
            "technical_level": 0.5,  # 0.0 = simple, 1.0 = technical
            "verbosity": 0.5,        # 0.0 = concise, 1.0 = detailed
            "humor": 0.5             # 0.0 = serious, 1.0 = humorous
        }
        
        # Conversation history
        self.conversation_history = []
        
        # Feedback history
        self.feedback_history = []
        
        # Topic interests (topic -> interest level)
        self.topic_interests = defaultdict(float)
        
        # Interaction patterns
        self.interaction_patterns = {
            "avg_message_length": 0,
            "question_frequency": 0,
            "response_time": 0,
            "session_duration": 0,
            "session_frequency": 0
        }
        
        # Meta-learning parameters
        self.adaptation_rate = 0.1  # How quickly to adapt to user feedback
        self.exploration_rate = 0.2  # Exploration rate for trying new approaches
    
    def update_preferences(self, preferences):
        """
        Update user preferences.
        
        Args:
            preferences (dict): New preference values
        """
        for key, value in preferences.items():
            if key in self.preferences:
                # Smooth update
                self.preferences[key] = (1 - self.adaptation_rate) * self.preferences[key] + self.adaptation_rate * value
    
    def update_from_feedback(self, feedback):
        """
        Update profile based on user feedback.
        
        Args:
            feedback (dict): User feedback
        """
        # Add to feedback history
        self.feedback_history.append(feedback)
        
        # Extract rating
        rating = feedback.get("rating", 3)
        normalized_rating = (rating - 3) / 2  # Convert 1-5 to -1 to 1
        
        # Update preferences based on feedback
        if "response_length" in feedback:
            self.preferences["response_length"] += self.adaptation_rate * normalized_rating * (feedback["response_length"] - self.preferences["response_length"])
            
        if "formality" in feedback:
            self.preferences["formality"] += self.adaptation_rate * normalized_rating * (feedback["formality"] - self.preferences["formality"])
            
        if "technical_level" in feedback:
            self.preferences["technical_level"] += self.adaptation_rate * normalized_rating * (feedback["technical_level"] - self.preferences["technical_level"])
            
        if "verbosity" in feedback:
            self.preferences["verbosity"] += self.adaptation_rate * normalized_rating * (feedback["verbosity"] - self.preferences["verbosity"])
            
        if "humor" in feedback:
            self.preferences["humor"] += self.adaptation_rate * normalized_rating * (feedback["humor"] - self.preferences["humor"])
        
        # Update topic interests
        if "topic" in feedback:
            topic = feedback["topic"]
            self.topic_interests[topic] += normalized_rating * self.adaptation_rate
            
            # Normalize topic interests
            if self.topic_interests[topic] > 1.0:
                self.topic_interests[topic] = 1.0
            elif self.topic_interests[topic] < -1.0:
                self.topic_interests[topic] = -1.0
        
        # Adjust adaptation rate based on feedback consistency
        if len(self.feedback_history) > 5:
            recent_ratings = [f.get("rating", 3) for f in self.feedback_history[-5:]]
            rating_variance = np.var(recent_ratings)
            
            # If ratings are consistent, reduce adaptation rate
            if rating_variance < 0.5:
                self.adaptation_rate = max(0.05, self.adaptation_rate * 0.9)
            else:
                # If ratings are inconsistent, increase adaptation rate
                self.adaptation_rate = min(0.3, self.adaptation_rate * 1.1)
    
    def update_from_interaction(self, interaction):
        """
        Update profile based on user interaction.
        
        Args:
            interaction (dict): User interaction data
        """
        # Update interaction patterns
        if "message_length" in interaction:
            # Exponential moving average
            alpha = 0.1
            self.interaction_patterns["avg_message_length"] = (1 - alpha) * self.interaction_patterns["avg_message_length"] + alpha * interaction["message_length"]
        
        if "is_question" in interaction:
            # Update question frequency
            alpha = 0.1
            self.interaction_patterns["question_frequency"] = (1 - alpha) * self.interaction_patterns["question_frequency"] + alpha * (1 if interaction["is_question"] else 0)
        
        # Add to conversation history
        if "user_input" in interaction and "ai_response" in interaction:
            self.conversation_history.append((interaction["user_input"], interaction["ai_response"]))
            
            # Keep only the last 50 interactions
            if len(self.conversation_history) > 50:
                self.conversation_history = self.conversation_history[-50:]
    
    def get_preference_vector(self):
        """
        Get a vector representation of user preferences.
        
        Returns:
            numpy.ndarray: Preference vector
        """
        # Convert preferences to a vector
        pref_vector = np.array([
            self.preferences["response_length"],
            self.preferences["formality"],
            self.preferences["technical_level"],
            self.preferences["verbosity"],
            self.preferences["humor"]
        ])
        
        return pref_vector
    
    def to_dict(self):
        """
        Convert user profile to a dictionary.
        
        Returns:
            dict: User profile as a dictionary
        """
        return {
            "user_id": self.user_id,
            "name": self.name,
            "preferences": self.preferences,
            "topic_interests": dict(self.topic_interests),
            "interaction_patterns": self.interaction_patterns,
            "adaptation_rate": self.adaptation_rate,
            "exploration_rate": self.exploration_rate
        }
    
    @classmethod
    def from_dict(cls, data):
        """
        Create a user profile from a dictionary.
        
        Args:
            data (dict): User profile data
            
        Returns:
            UserProfile: User profile object
        """
        profile = cls(data["user_id"], data.get("name"))
        profile.preferences = data.get("preferences", profile.preferences)
        profile.topic_interests = defaultdict(float, data.get("topic_interests", {}))
        profile.interaction_patterns = data.get("interaction_patterns", profile.interaction_patterns)
        profile.adaptation_rate = data.get("adaptation_rate", profile.adaptation_rate)
        profile.exploration_rate = data.get("exploration_rate", profile.exploration_rate)
        
        return profile

class MAMLAgent:
    """
    Model-Agnostic Meta-Learning (MAML) Agent for NoahAI.
    
    This agent implements MAML for fast adaptation to different users.
    """
    
    def __init__(self, state_size=128, action_size=10, model_dir="data/rl_models/meta_learning",
                 meta_learning_rate=0.001, adaptation_learning_rate=0.01, gamma=0.99,
                 inner_steps=5, meta_batch_size=4):
        """
        Initialize the MAML agent.
        
        Args:
            state_size (int): Size of the state space
            action_size (int): Size of the action space
            model_dir (str): Directory to save/load models
            meta_learning_rate (float): Learning rate for meta-update
            adaptation_learning_rate (float): Learning rate for task adaptation
            gamma (float): Discount factor for future rewards
            inner_steps (int): Number of gradient steps for task adaptation
            meta_batch_size (int): Number of tasks in meta-batch
        """
        self.state_size = state_size
        self.action_size = action_size
        self.model_dir = model_dir
        self.meta_learning_rate = meta_learning_rate
        self.adaptation_learning_rate = adaptation_learning_rate
        self.gamma = gamma
        self.inner_steps = inner_steps
        self.meta_batch_size = meta_batch_size
        
        # Create model directory if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)
        
        # Initialize meta-model
        self.meta_model = self._build_model()
        
        # User-specific adapted models
        self.user_models = {}
        
        # User profiles
        self.user_profiles = {}
        
        # Experience buffer for meta-learning
        self.meta_buffer = []
        
        # Training metrics
        self.meta_loss_history = []
        self.adaptation_loss_history = defaultdict(list)
    
    def _build_model(self):
        """
        Build the policy network.
        
        Returns:
            tf.keras.Model: Policy network
        """
        if not TENSORFLOW_AVAILABLE:
            logger.error("TensorFlow is required for meta-learning.")
            return None
        
        # Input layers
        state_input = Input(shape=(self.state_size,))
        preference_input = Input(shape=(5,))  # User preference vector
        
        # Combine state and preferences
        combined = Concatenate()([state_input, preference_input])
        
        # Hidden layers
        x = Dense(64, activation='relu')(combined)
        x = Dense(64, activation='relu')(x)
        
        # Output layer (action probabilities)
        outputs = Dense(self.action_size, activation='softmax')(x)
        
        # Create model
        model = Model(inputs=[state_input, preference_input], outputs=outputs)
        model.compile(optimizer=Adam(learning_rate=self.meta_learning_rate))
        
        return model
    
    def get_user_model(self, user_id):
        """
        Get or create a user-specific model.
        
        Args:
            user_id (str): User ID
            
        Returns:
            tf.keras.Model: User-specific model
        """
        if user_id not in self.user_models:
            # Clone the meta-model for this user
            if TENSORFLOW_AVAILABLE and self.meta_model is not None:
                self.user_models[user_id] = clone_model(self.meta_model)
                self.user_models[user_id].set_weights(self.meta_model.get_weights())
                self.user_models[user_id].compile(optimizer=Adam(learning_rate=self.adaptation_learning_rate))
            else:
                logger.error("TensorFlow is required for meta-learning.")
                return None
        
        return self.user_models[user_id]
    
    def get_user_profile(self, user_id, name=None):
        """
        Get or create a user profile.
        
        Args:
            user_id (str): User ID
            name (str, optional): User name
            
        Returns:
            UserProfile: User profile
        """
        if user_id not in self.user_profiles:
            self.user_profiles[user_id] = UserProfile(user_id, name)
        
        return self.user_profiles[user_id]
    
    def get_action(self, state, user_id):
        """
        Get an action based on the current state and user profile.
        
        Args:
            state: Current state
            user_id (str): User ID
            
        Returns:
            int: Selected action
        """
        if not TENSORFLOW_AVAILABLE:
            return np.random.randint(0, self.action_size)
        
        # Get user profile
        profile = self.get_user_profile(user_id)
        
        # Get preference vector
        preference_vector = profile.get_preference_vector()
        
        # Get user-specific model
        user_model = self.get_user_model(user_id)
        
        # Reshape inputs
        state = np.reshape(state, [1, self.state_size])
        preference_vector = np.reshape(preference_vector, [1, 5])
        
        # Get action probabilities
        action_probs = user_model.predict([state, preference_vector], verbose=0)[0]
        
        # Exploration: with probability exploration_rate, choose a random action
        if np.random.random() < profile.exploration_rate:
            action = np.random.choice(self.action_size)
        else:
            # Sample action from probabilities
            action = np.random.choice(self.action_size, p=action_probs)
        
        return action
    
    def adapt_to_user(self, user_id, experiences):
        """
        Adapt the model to a specific user.
        
        Args:
            user_id (str): User ID
            experiences (list): List of (state, action, reward, next_state, done, preference_vector) tuples
            
        Returns:
            float: Adaptation loss
        """
        if not TENSORFLOW_AVAILABLE or len(experiences) == 0:
            return None
        
        # Get user model
        user_model = self.get_user_model(user_id)
        
        # Unpack experiences
        states, actions, rewards, next_states, dones, preference_vectors = zip(*experiences)
        
        # Convert to numpy arrays
        states = np.vstack(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        preference_vectors = np.vstack(preference_vectors)
        
        # Create one-hot encoded actions
        actions_one_hot = np.zeros((len(actions), self.action_size))
        for i, action in enumerate(actions):
            actions_one_hot[i, action] = 1
        
        # Perform inner loop update
        for _ in range(self.inner_steps):
            with tf.GradientTape() as tape:
                # Get action probabilities
                action_probs = user_model([states, preference_vectors], training=True)
                
                # Calculate log probabilities of actions
                log_probs = tf.reduce_sum(
                    tf.math.log(action_probs + 1e-10) * actions_one_hot,
                    axis=1
                )
                
                # Calculate loss (negative because we want to maximize)
                loss = -tf.reduce_mean(log_probs * rewards)
            
            # Calculate gradients and apply updates
            gradients = tape.gradient(loss, user_model.trainable_variables)
            user_model.optimizer.apply_gradients(zip(gradients, user_model.trainable_variables))
        
        # Store adaptation loss
        self.adaptation_loss_history[user_id].append(loss.numpy())
        
        # Add to meta-buffer for meta-learning
        self.meta_buffer.append((user_id, experiences))
        
        # If meta-buffer is full, perform meta-update
        if len(self.meta_buffer) >= self.meta_batch_size:
            self._meta_update()
        
        return loss.numpy()
    
    def _meta_update(self):
        """
        Perform meta-update using MAML.
        
        Returns:
            float: Meta-loss
        """
        if not TENSORFLOW_AVAILABLE or len(self.meta_buffer) < self.meta_batch_size:
            return None
        
        # Sample tasks from meta-buffer
        tasks = self.meta_buffer[:self.meta_batch_size]
        self.meta_buffer = self.meta_buffer[self.meta_batch_size:]
        
        meta_loss = 0.0
        
        # Perform meta-update
        with tf.GradientTape() as meta_tape:
            # Iterate over tasks
            for user_id, experiences in tasks:
                # Clone meta-model for this task
                task_model = clone_model(self.meta_model)
                task_model.set_weights(self.meta_model.get_weights())
                task_model.compile(optimizer=Adam(learning_rate=self.adaptation_learning_rate))
                
                # Unpack experiences
                states, actions, rewards, next_states, dones, preference_vectors = zip(*experiences)
                
                # Convert to numpy arrays
                states = np.vstack(states)
                actions = np.array(actions)
                rewards = np.array(rewards)
                preference_vectors = np.vstack(preference_vectors)
                
                # Create one-hot encoded actions
                actions_one_hot = np.zeros((len(actions), self.action_size))
                for i, action in enumerate(actions):
                    actions_one_hot[i, action] = 1
                
                # Split experiences into support (for adaptation) and query (for meta-update)
                split_idx = len(experiences) // 2
                support_states = states[:split_idx]
                support_actions = actions[:split_idx]
                support_rewards = rewards[:split_idx]
                support_actions_one_hot = actions_one_hot[:split_idx]
                support_preference_vectors = preference_vectors[:split_idx]
                
                query_states = states[split_idx:]
                query_actions = actions[split_idx:]
                query_rewards = rewards[split_idx:]
                query_actions_one_hot = actions_one_hot[split_idx:]
                query_preference_vectors = preference_vectors[split_idx:]
                
                # Perform inner loop update on support set
                for _ in range(self.inner_steps):
                    with tf.GradientTape() as tape:
                        # Get action probabilities
                        action_probs = task_model([support_states, support_preference_vectors], training=True)
                        
                        # Calculate log probabilities of actions
                        log_probs = tf.reduce_sum(
                            tf.math.log(action_probs + 1e-10) * support_actions_one_hot,
                            axis=1
                        )
                        
                        # Calculate loss (negative because we want to maximize)
                        inner_loss = -tf.reduce_mean(log_probs * support_rewards)
                    
                    # Calculate gradients and apply updates
                    gradients = tape.gradient(inner_loss, task_model.trainable_variables)
                    task_model.optimizer.apply_gradients(zip(gradients, task_model.trainable_variables))
                
                # Evaluate on query set
                action_probs = task_model([query_states, query_preference_vectors], training=True)
                
                # Calculate log probabilities of actions
                log_probs = tf.reduce_sum(
                    tf.math.log(action_probs + 1e-10) * query_actions_one_hot,
                    axis=1
                )
                
                # Calculate loss (negative because we want to maximize)
                query_loss = -tf.reduce_mean(log_probs * query_rewards)
                
                # Add to meta-loss
                meta_loss += query_loss
        
        # Calculate average meta-loss
        meta_loss /= len(tasks)
        
        # Calculate gradients and apply meta-update
        meta_gradients = meta_tape.gradient(meta_loss, self.meta_model.trainable_variables)
        self.meta_model.optimizer.apply_gradients(zip(meta_gradients, self.meta_model.trainable_variables))
        
        # Store meta-loss
        self.meta_loss_history.append(meta_loss.numpy())
        
        # Update user models with new meta-model weights
        for user_id in self.user_models:
            # Get current weights
            current_weights = self.user_models[user_id].get_weights()
            meta_weights = self.meta_model.get_weights()
            
            # Interpolate between current and meta weights
            new_weights = []
            for cw, mw in zip(current_weights, meta_weights):
                new_weights.append(0.9 * cw + 0.1 * mw)  # 90% current, 10% meta
            
            # Set new weights
            self.user_models[user_id].set_weights(new_weights)
        
        return meta_loss.numpy()
    
    def save(self, filename_prefix=None):
        """
        Save the meta-learning agent.
        
        Args:
            filename_prefix (str, optional): Prefix for the filenames
        """
        if not TENSORFLOW_AVAILABLE:
            logger.error("TensorFlow is required to save models.")
            return False
        
        if filename_prefix is None:
            filename_prefix = f"meta_learning_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Save meta-model
        meta_model_path = os.path.join(self.model_dir, f"{filename_prefix}_meta_model.h5")
        self.meta_model.save(meta_model_path)
        
        # Save user models
        for user_id, model in self.user_models.items():
            user_model_path = os.path.join(self.model_dir, f"{filename_prefix}_user_{user_id}.h5")
            model.save(user_model_path)
        
        # Save user profiles
        profiles_data = {user_id: profile.to_dict() for user_id, profile in self.user_profiles.items()}
        profiles_path = os.path.join(self.model_dir, f"{filename_prefix}_profiles.json")
        with open(profiles_path, 'w') as f:
            json.dump(profiles_data, f, indent=2)
        
        # Save metadata
        metadata = {
            "state_size": self.state_size,
            "action_size": self.action_size,
            "meta_learning_rate": self.meta_learning_rate,
            "adaptation_learning_rate": self.adaptation_learning_rate,
            "gamma": self.gamma,
            "inner_steps": self.inner_steps,
            "meta_batch_size": self.meta_batch_size,
            "meta_loss_history": self.meta_loss_history,
            "adaptation_loss_history": {user_id: losses for user_id, losses in self.adaptation_loss_history.items()},
            "user_ids": list(self.user_profiles.keys()),
            "saved_at": datetime.now().isoformat()
        }
        
        metadata_path = os.path.join(self.model_dir, f"{filename_prefix}_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Meta-learning agent saved to {self.model_dir}/{filename_prefix}_*")
        return True
    
    def load(self, filename_prefix):
        """
        Load a saved meta-learning agent.
        
        Args:
            filename_prefix (str): Prefix for the filenames
            
        Returns:
            bool: Whether the agent was loaded successfully
        """
        if not TENSORFLOW_AVAILABLE:
            logger.error("TensorFlow is required to load models.")
            return False
        
        try:
            # Load meta-model
            meta_model_path = os.path.join(self.model_dir, f"{filename_prefix}_meta_model.h5")
            self.meta_model = load_model(meta_model_path)
            
            # Load metadata
            metadata_path = os.path.join(self.model_dir, f"{filename_prefix}_metadata.json")
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            self.state_size = metadata.get("state_size", self.state_size)
            self.action_size = metadata.get("action_size", self.action_size)
            self.meta_learning_rate = metadata.get("meta_learning_rate", self.meta_learning_rate)
            self.adaptation_learning_rate = metadata.get("adaptation_learning_rate", self.adaptation_learning_rate)
            self.gamma = metadata.get("gamma", self.gamma)
            self.inner_steps = metadata.get("inner_steps", self.inner_steps)
            self.meta_batch_size = metadata.get("meta_batch_size", self.meta_batch_size)
            self.meta_loss_history = metadata.get("meta_loss_history", [])
            
            # Load adaptation loss history
            adaptation_loss_history = metadata.get("adaptation_loss_history", {})
            for user_id, losses in adaptation_loss_history.items():
                self.adaptation_loss_history[user_id] = losses
            
            # Load user profiles
            profiles_path = os.path.join(self.model_dir, f"{filename_prefix}_profiles.json")
            with open(profiles_path, 'r') as f:
                profiles_data = json.load(f)
            
            for user_id, profile_data in profiles_data.items():
                self.user_profiles[user_id] = UserProfile.from_dict(profile_data)
            
            # Load user models
            for user_id in metadata.get("user_ids", []):
                user_model_path = os.path.join(self.model_dir, f"{filename_prefix}_user_{user_id}.h5")
                if os.path.exists(user_model_path):
                    self.user_models[user_id] = load_model(user_model_path)
            
            logger.info(f"Meta-learning agent loaded from {self.model_dir}/{filename_prefix}_*")
            return True
        except Exception as e:
            logger.error(f"Error loading meta-learning agent: {e}")
            return False
