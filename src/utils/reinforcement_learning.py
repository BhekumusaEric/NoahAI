"""
Advanced Reinforcement Learning for NoahAI

This module provides advanced reinforcement learning algorithms and utilities
to improve NoahAI's learning capabilities.
"""

import os
import json
import numpy as np
import random
import logging
from collections import deque
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
logger = logging.getLogger("reinforcement_learning")

# Try to import TensorFlow
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, Model, load_model, clone_model
    from tensorflow.keras.layers import Dense, Dropout, LSTM, Input, Concatenate
    from tensorflow.keras.optimizers import Adam
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    logger.warning("TensorFlow not available. Advanced RL capabilities will be limited.")

class ExperienceReplay:
    """
    Experience Replay buffer for reinforcement learning.

    Stores and manages past experiences (state, action, reward, next_state)
    for more efficient learning.
    """

    def __init__(self, max_size=10000, alpha=0.6, beta=0.4, beta_increment=0.001):
        """
        Initialize the Experience Replay buffer.

        Args:
            max_size (int): Maximum size of the buffer
            alpha (float): Priority exponent (0 = uniform sampling, higher = more prioritization)
            beta (float): Importance sampling exponent (0 = no correction, 1 = full correction)
            beta_increment (float): Increment for beta parameter per sampling
        """
        self.buffer = deque(maxlen=max_size)
        self.priorities = deque(maxlen=max_size)
        self.max_size = max_size
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.epsilon = 1e-6  # Small constant to avoid zero priority

    def add(self, state, action, reward, next_state, done, error=None):
        """
        Add an experience to the buffer.

        Args:
            state: Current state
            action: Action taken
            reward (float): Reward received
            next_state: Next state
            done (bool): Whether the episode is done
            error (float, optional): TD error for prioritization
        """
        experience = (state, action, reward, next_state, done)
        self.buffer.append(experience)

        # Set priority based on TD error or default to max priority
        if error is not None:
            priority = (abs(error) + self.epsilon) ** self.alpha
        else:
            # If no error provided, set to max priority to ensure sampling
            priority = (max(self.priorities) if self.priorities else 1.0)

        self.priorities.append(priority)

    def sample(self, batch_size):
        """
        Sample a batch of experiences from the buffer.

        Args:
            batch_size (int): Number of experiences to sample

        Returns:
            tuple: Batch of (states, actions, rewards, next_states, dones, weights, indices)
        """
        # If buffer is not large enough, return None
        if len(self.buffer) < batch_size:
            return None

        # Calculate sampling probabilities
        priorities = np.array(self.priorities)
        probs = priorities / np.sum(priorities)

        # Sample indices based on priorities
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)

        # Calculate importance sampling weights
        weights = (len(self.buffer) * probs[indices]) ** (-self.beta)
        weights /= np.max(weights)  # Normalize weights

        # Increment beta for next sampling
        self.beta = min(1.0, self.beta + self.beta_increment)

        # Get experiences
        batch = [self.buffer[i] for i in indices]
        states, actions, rewards, next_states, dones = zip(*batch)

        return states, actions, rewards, next_states, dones, weights, indices

    def update_priorities(self, indices, errors):
        """
        Update priorities for experiences.

        Args:
            indices (list): Indices of experiences to update
            errors (list): TD errors for each experience
        """
        for i, error in zip(indices, errors):
            if i < len(self.priorities):
                self.priorities[i] = (abs(error) + self.epsilon) ** self.alpha

    def __len__(self):
        """
        Get the current size of the buffer.

        Returns:
            int: Current buffer size
        """
        return len(self.buffer)

class DeepQNetwork:
    """
    Deep Q-Network (DQN) implementation for reinforcement learning.

    Uses a neural network to approximate the Q-function for better
    action selection in complex environments.
    """

    def __init__(self, state_size, action_size, model_dir="data/rl_models",
                 learning_rate=0.001, gamma=0.95, epsilon=1.0,
                 epsilon_decay=0.995, epsilon_min=0.01,
                 batch_size=32, update_target_freq=100):
        """
        Initialize the DQN agent.

        Args:
            state_size (int): Size of the state space
            action_size (int): Size of the action space
            model_dir (str): Directory to save/load models
            learning_rate (float): Learning rate for the optimizer
            gamma (float): Discount factor for future rewards
            epsilon (float): Exploration rate
            epsilon_decay (float): Decay rate for epsilon
            epsilon_min (float): Minimum value for epsilon
            batch_size (int): Batch size for training
            update_target_freq (int): Frequency to update target network
        """
        if not TENSORFLOW_AVAILABLE:
            logger.error("TensorFlow is required for DQN.")
            raise ImportError("TensorFlow is required for DQN.")

        self.state_size = state_size
        self.action_size = action_size
        self.model_dir = model_dir
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.update_target_freq = update_target_freq

        # Create model directory if it doesn't exist
        os.makedirs(self.model_dir, exist_ok=True)

        # Create experience replay buffer
        self.memory = ExperienceReplay()

        # Build models
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()

        # Training metrics
        self.train_step = 0
        self.loss_history = []

    def _build_model(self):
        """
        Build a neural network model for DQN.

        Returns:
            tf.keras.Model: The DQN model
        """
        model = Sequential()
        model.add(Dense(64, input_dim=self.state_size, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def update_target_model(self):
        """
        Update the target model with weights from the main model.
        """
        self.target_model.set_weights(self.model.get_weights())

    def act(self, state, explore=True):
        """
        Choose an action based on the current state.

        Args:
            state: Current state
            explore (bool): Whether to use exploration

        Returns:
            int: Selected action
        """
        if explore and np.random.rand() <= self.epsilon:
            # Exploration: choose a random action
            return random.randrange(self.action_size)

        # Exploitation: choose the best action based on the model
        state = np.reshape(state, [1, self.state_size])
        q_values = self.model.predict(state, verbose=0)
        return np.argmax(q_values[0])

    def remember(self, state, action, reward, next_state, done):
        """
        Store experience in memory.

        Args:
            state: Current state
            action: Action taken
            reward (float): Reward received
            next_state: Next state
            done (bool): Whether the episode is done
        """
        # Calculate TD error for prioritized replay
        state = np.reshape(state, [1, self.state_size])
        next_state = np.reshape(next_state, [1, self.state_size])

        target = reward
        if not done:
            target += self.gamma * np.amax(self.target_model.predict(next_state, verbose=0)[0])

        current_q = self.model.predict(state, verbose=0)[0]
        td_error = target - current_q[action]

        self.memory.add(state, action, reward, next_state, done, td_error)

    def replay(self):
        """
        Train the model using experiences from memory.

        Returns:
            float: Loss value
        """
        # Sample batch from memory
        batch = self.memory.sample(self.batch_size)
        if batch is None:
            return None

        states, actions, rewards, next_states, dones, weights, indices = batch

        # Convert to numpy arrays
        states = np.vstack(states)
        next_states = np.vstack(next_states)

        # Get current Q values
        current_q = self.model.predict(states, verbose=0)

        # Get next Q values from target model
        next_q = self.target_model.predict(next_states, verbose=0)

        # Update Q values for the actions taken
        targets = current_q.copy()
        td_errors = []

        for i, (action, reward, done) in enumerate(zip(actions, rewards, dones)):
            target = reward
            if not done:
                target += self.gamma * np.amax(next_q[i])

            # Calculate TD error
            td_error = target - targets[i, action]
            td_errors.append(td_error)

            # Update target for the action
            targets[i, action] = target

        # Train the model
        loss = self.model.train_on_batch(states, targets, sample_weight=weights)

        # Update priorities in memory
        self.memory.update_priorities(indices, td_errors)

        # Update target model periodically
        self.train_step += 1
        if self.train_step % self.update_target_freq == 0:
            self.update_target_model()

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        # Store loss
        self.loss_history.append(loss)

        return loss

    def save(self, filename=None):
        """
        Save the model.

        Args:
            filename (str, optional): Filename to save the model
        """
        if filename is None:
            filename = f"dqn_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.h5"

        filepath = os.path.join(self.model_dir, filename)
        self.model.save(filepath)

        # Save metadata
        metadata = {
            "state_size": self.state_size,
            "action_size": self.action_size,
            "learning_rate": self.learning_rate,
            "gamma": self.gamma,
            "epsilon": self.epsilon,
            "epsilon_decay": self.epsilon_decay,
            "epsilon_min": self.epsilon_min,
            "train_step": self.train_step,
            "saved_at": datetime.now().isoformat()
        }

        metadata_path = os.path.join(self.model_dir, f"{filename}.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Model saved to {filepath}")

    def load(self, filepath):
        """
        Load a saved model.

        Args:
            filepath (str): Path to the saved model
        """
        if not os.path.exists(filepath):
            logger.error(f"Model file not found: {filepath}")
            return False

        try:
            self.model = load_model(filepath)
            self.target_model = clone_model(self.model)
            self.target_model.set_weights(self.model.get_weights())

            # Load metadata if available
            metadata_path = f"{filepath}.json"
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)

                self.state_size = metadata.get("state_size", self.state_size)
                self.action_size = metadata.get("action_size", self.action_size)
                self.learning_rate = metadata.get("learning_rate", self.learning_rate)
                self.gamma = metadata.get("gamma", self.gamma)
                self.epsilon = metadata.get("epsilon", self.epsilon)
                self.epsilon_decay = metadata.get("epsilon_decay", self.epsilon_decay)
                self.epsilon_min = metadata.get("epsilon_min", self.epsilon_min)
                self.train_step = metadata.get("train_step", 0)

            logger.info(f"Model loaded from {filepath}")
            return True
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False

class AdvantageActorCritic:
    """
    Advantage Actor-Critic (A2C) implementation for reinforcement learning.

    Uses separate networks for policy (actor) and value function (critic)
    for improved policy learning.
    """

    def __init__(self, state_size, action_size, model_dir="data/rl_models",
                 actor_lr=0.001, critic_lr=0.002, gamma=0.95,
                 entropy_beta=0.01, batch_size=32):
        """
        Initialize the A2C agent.

        Args:
            state_size (int): Size of the state space
            action_size (int): Size of the action space
            model_dir (str): Directory to save/load models
            actor_lr (float): Learning rate for the actor network
            critic_lr (float): Learning rate for the critic network
            gamma (float): Discount factor for future rewards
            entropy_beta (float): Coefficient for entropy regularization
            batch_size (int): Batch size for training
        """
        if not TENSORFLOW_AVAILABLE:
            logger.error("TensorFlow is required for A2C.")
            raise ImportError("TensorFlow is required for A2C.")

        self.state_size = state_size
        self.action_size = action_size
        self.model_dir = model_dir
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.gamma = gamma
        self.entropy_beta = entropy_beta
        self.batch_size = batch_size

        # Create model directory if it doesn't exist
        os.makedirs(self.model_dir, exist_ok=True)

        # Build models
        self.actor, self.critic = self._build_models()

        # Training metrics
        self.train_step = 0
        self.actor_loss_history = []
        self.critic_loss_history = []

        # Experience buffer
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []

    def _build_models(self):
        """
        Build actor and critic networks.

        Returns:
            tuple: (actor_model, critic_model)
        """
        # Actor network (policy)
        actor = Sequential()
        actor.add(Dense(64, input_dim=self.state_size, activation='relu'))
        actor.add(Dense(64, activation='relu'))
        actor.add(Dense(self.action_size, activation='softmax'))
        actor.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=self.actor_lr))

        # Critic network (value function)
        critic = Sequential()
        critic.add(Dense(64, input_dim=self.state_size, activation='relu'))
        critic.add(Dense(64, activation='relu'))
        critic.add(Dense(1, activation='linear'))
        critic.compile(loss='mse', optimizer=Adam(learning_rate=self.critic_lr))

        return actor, critic

    def act(self, state):
        """
        Choose an action based on the current state.

        Args:
            state: Current state

        Returns:
            int: Selected action
            float: Action probability
        """
        state = np.reshape(state, [1, self.state_size])
        action_probs = self.actor.predict(state, verbose=0)[0]

        # Sample action from the probability distribution
        action = np.random.choice(self.action_size, p=action_probs)

        return action, action_probs[action]

    def remember(self, state, action, reward, next_state, done):
        """
        Store experience for batch training.

        Args:
            state: Current state
            action: Action taken
            reward (float): Reward received
            next_state: Next state
            done (bool): Whether the episode is done
        """
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(done)

    def _discount_rewards(self, rewards, dones):
        """
        Calculate discounted rewards.

        Args:
            rewards (list): List of rewards
            dones (list): List of done flags

        Returns:
            numpy.ndarray: Discounted rewards
        """
        discounted_rewards = np.zeros_like(rewards, dtype=np.float32)
        running_reward = 0

        # Calculate discounted rewards in reverse order
        for i in reversed(range(len(rewards))):
            if dones[i]:
                running_reward = 0
            running_reward = rewards[i] + self.gamma * running_reward
            discounted_rewards[i] = running_reward

        # Normalize rewards for stability
        if len(discounted_rewards) > 1:
            discounted_rewards = (discounted_rewards - np.mean(discounted_rewards)) / (np.std(discounted_rewards) + 1e-8)

        return discounted_rewards

    def train(self):
        """
        Train the actor and critic networks.

        Returns:
            tuple: (actor_loss, critic_loss)
        """
        if len(self.states) < self.batch_size:
            return None, None

        # Convert to numpy arrays
        states = np.vstack(self.states)
        actions = np.array(self.actions)
        rewards = np.array(self.rewards)
        next_states = np.vstack(self.next_states)
        dones = np.array(self.dones)

        # Calculate discounted rewards
        discounted_rewards = self._discount_rewards(rewards, dones)

        # Get value predictions
        values = self.critic.predict(states, verbose=0).flatten()
        next_values = self.critic.predict(next_states, verbose=0).flatten()

        # Calculate advantages
        advantages = np.zeros_like(rewards, dtype=np.float32)
        for i in range(len(rewards)):
            if dones[i]:
                advantages[i] = rewards[i] - values[i]
            else:
                advantages[i] = rewards[i] + self.gamma * next_values[i] - values[i]

        # Train critic network
        critic_loss = self.critic.train_on_batch(states, discounted_rewards)

        # Train actor network
        actor_target = np.zeros((len(actions), self.action_size))
        for i, action in enumerate(actions):
            actor_target[i, action] = advantages[i]

        # Add entropy regularization
        action_probs = self.actor.predict(states, verbose=0)
        entropy = -np.sum(action_probs * np.log(action_probs + 1e-8), axis=1)
        actor_target += self.entropy_beta * entropy[:, np.newaxis]

        actor_loss = self.actor.train_on_batch(states, actor_target)

        # Store losses
        self.actor_loss_history.append(actor_loss)
        self.critic_loss_history.append(critic_loss)

        # Increment train step
        self.train_step += 1

        # Clear experience buffer
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []

        return actor_loss, critic_loss

    def save(self, filename_prefix=None):
        """
        Save the actor and critic models.

        Args:
            filename_prefix (str, optional): Prefix for the filenames
        """
        if filename_prefix is None:
            filename_prefix = f"a2c_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        actor_filepath = os.path.join(self.model_dir, f"{filename_prefix}_actor.h5")
        critic_filepath = os.path.join(self.model_dir, f"{filename_prefix}_critic.h5")

        self.actor.save(actor_filepath)
        self.critic.save(critic_filepath)

        # Save metadata
        metadata = {
            "state_size": self.state_size,
            "action_size": self.action_size,
            "actor_lr": self.actor_lr,
            "critic_lr": self.critic_lr,
            "gamma": self.gamma,
            "entropy_beta": self.entropy_beta,
            "train_step": self.train_step,
            "saved_at": datetime.now().isoformat()
        }

        metadata_path = os.path.join(self.model_dir, f"{filename_prefix}_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"A2C models saved to {self.model_dir}/{filename_prefix}_*.h5")

    def load(self, actor_filepath, critic_filepath, metadata_path=None):
        """
        Load saved actor and critic models.

        Args:
            actor_filepath (str): Path to the saved actor model
            critic_filepath (str): Path to the saved critic model
            metadata_path (str, optional): Path to the metadata file

        Returns:
            bool: Whether the models were loaded successfully
        """
        if not os.path.exists(actor_filepath) or not os.path.exists(critic_filepath):
            logger.error(f"Model files not found: {actor_filepath} or {critic_filepath}")
            return False

        try:
            self.actor = load_model(actor_filepath)
            self.critic = load_model(critic_filepath)

            # Load metadata if available
            if metadata_path and os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)

                self.state_size = metadata.get("state_size", self.state_size)
                self.action_size = metadata.get("action_size", self.action_size)
                self.actor_lr = metadata.get("actor_lr", self.actor_lr)
                self.critic_lr = metadata.get("critic_lr", self.critic_lr)
                self.gamma = metadata.get("gamma", self.gamma)
                self.entropy_beta = metadata.get("entropy_beta", self.entropy_beta)
                self.train_step = metadata.get("train_step", 0)

            logger.info(f"A2C models loaded from {actor_filepath} and {critic_filepath}")
            return True
        except Exception as e:
            logger.error(f"Error loading A2C models: {e}")
            return False