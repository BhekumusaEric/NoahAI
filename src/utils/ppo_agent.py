"""
Proximal Policy Optimization (PPO) Agent for NoahAI

This module implements the PPO algorithm for more stable policy updates
in reinforcement learning.
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
logger = logging.getLogger("ppo_agent")

# Try to import TensorFlow
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, Model, load_model
    from tensorflow.keras.layers import Dense, Dropout, LSTM, Input, Concatenate, Lambda
    from tensorflow.keras.optimizers import Adam
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    logger.warning("TensorFlow not available. PPO agent will not be available.")

class PPOAgent:
    """
    Proximal Policy Optimization (PPO) agent for reinforcement learning.

    PPO is a policy gradient method that uses a clipped surrogate objective
    to ensure stable policy updates.
    """

    def __init__(self, state_size, action_size, model_dir="data/rl_models",
                 actor_lr=0.0003, critic_lr=0.001, gamma=0.99,
                 clip_ratio=0.2, target_kl=0.01, entropy_coef=0.01,
                 batch_size=64, epochs=10, lam=0.95,
                 exploration_strategy="adaptive_entropy"):
        """
        Initialize the PPO agent with enhanced exploration strategies.

        Args:
            state_size (int): Size of the state space
            action_size (int): Size of the action space
            model_dir (str): Directory to save/load models
            actor_lr (float): Learning rate for the actor network
            critic_lr (float): Learning rate for the critic network
            gamma (float): Discount factor for future rewards
            clip_ratio (float): PPO clipping parameter
            target_kl (float): Target KL divergence threshold
            entropy_coef (float): Entropy coefficient for exploration
            batch_size (int): Batch size for training
            epochs (int): Number of epochs to train on each batch
            lam (float): GAE-Lambda parameter
            exploration_strategy (str): Strategy for exploration:
                - "fixed_entropy": Fixed entropy coefficient
                - "adaptive_entropy": Adjust entropy coefficient based on performance
                - "curiosity": Use curiosity-driven exploration
                - "noisy_networks": Add noise to network parameters
                - "parameter_noise": Add noise to action parameters
        """
        if not TENSORFLOW_AVAILABLE:
            logger.error("TensorFlow is required for PPO.")
            raise ImportError("TensorFlow is required for PPO.")

        self.state_size = state_size
        self.action_size = action_size
        self.model_dir = model_dir
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.gamma = gamma
        self.clip_ratio = clip_ratio
        self.target_kl = target_kl
        self.entropy_coef = entropy_coef
        self.batch_size = batch_size
        self.epochs = epochs
        self.lam = lam
        self.exploration_strategy = exploration_strategy

        # Create model directory if it doesn't exist
        os.makedirs(self.model_dir, exist_ok=True)

        # Build models
        self.actor, self.critic = self._build_models()

        # Initialize curiosity model if using curiosity-driven exploration
        self.curiosity_model = None
        if self.exploration_strategy == "curiosity":
            self.curiosity_model = self._build_curiosity_model()
            self.curiosity_scale = 0.01  # Scale for curiosity rewards
            self.curiosity_loss_history = []

        # Training metrics
        self.train_step = 0
        self.actor_loss_history = []
        self.critic_loss_history = []
        self.mean_entropy_history = []
        self.mean_reward_history = []

        # Adaptive entropy parameters
        self.initial_entropy_coef = entropy_coef
        self.min_entropy_coef = 0.001
        self.max_entropy_coef = 0.05
        self.entropy_decay_rate = 0.99
        self.entropy_increase_rate = 1.05
        self.reward_threshold = 0.5  # Threshold for good performance

        # Parameter noise for exploration
        self.use_parameter_noise = (exploration_strategy == "parameter_noise")
        self.parameter_noise_scale = 0.1
        self.parameter_noise_decay = 0.9995

        # Noisy networks parameters
        self.use_noisy_networks = (exploration_strategy == "noisy_networks")
        self.noise_scale = 0.1

        # Experience buffer
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []

        logger.info(f"Initialized PPO agent with exploration strategy: {exploration_strategy}")

    def _build_curiosity_model(self):
        """
        Build a curiosity-driven exploration model.

        This model predicts the next state given the current state and action,
        and uses prediction error as an intrinsic reward signal.

        Returns:
            tf.keras.Model: The curiosity model
        """
        # State input
        state_input = Input(shape=(self.state_size,))

        # Action input (one-hot encoded)
        action_input = Input(shape=(self.action_size,))

        # Combine state and action
        combined = Concatenate()([state_input, action_input])

        # Forward model (predicts next state)
        x = Dense(64, activation='relu')(combined)
        x = Dense(64, activation='relu')(x)
        next_state_pred = Dense(self.state_size, activation='linear')(x)

        # Create model
        model = Model(inputs=[state_input, action_input], outputs=next_state_pred)
        model.compile(loss='mse', optimizer=Adam(learning_rate=0.001))

        return model

    def _build_models(self):
        """
        Build actor and critic networks.

        Returns:
            tuple: (actor_model, critic_model)
        """
        # Actor network (policy)
        actor_input = Input(shape=(self.state_size,))
        x = Dense(64, activation='relu')(actor_input)
        x = Dense(64, activation='relu')(x)

        # Output mean and log standard deviation for continuous actions
        # or probabilities for discrete actions
        if isinstance(self.action_size, tuple):  # Continuous action space
            action_mean = Dense(self.action_size[0], activation='tanh')(x)
            log_std = Dense(self.action_size[0], activation='linear')(x)
            actor_output = Concatenate()([action_mean, log_std])
        else:  # Discrete action space
            actor_output = Dense(self.action_size, activation='softmax')(x)

        actor = Model(inputs=actor_input, outputs=actor_output)
        actor.compile(optimizer=Adam(learning_rate=self.actor_lr))

        # Critic network (value function)
        critic_input = Input(shape=(self.state_size,))
        x = Dense(64, activation='relu')(critic_input)
        x = Dense(64, activation='relu')(x)
        critic_output = Dense(1, activation='linear')(x)

        critic = Model(inputs=critic_input, outputs=critic_output)
        critic.compile(loss='mse', optimizer=Adam(learning_rate=self.critic_lr))

        return actor, critic

    def act(self, state, explore=True):
        """
        Choose an action based on the current state with enhanced exploration.

        Args:
            state: Current state
            explore (bool): Whether to use exploration strategies

        Returns:
            tuple: (action, log_prob, value)
        """
        state = np.reshape(state, [1, self.state_size])

        # Apply parameter noise if enabled and exploring
        if explore and self.use_parameter_noise:
            # Create a temporary copy of the actor with noise added to weights
            noisy_actor = clone_model(self.actor)
            noisy_actor.set_weights(self.actor.get_weights())

            # Add noise to weights
            weights = noisy_actor.get_weights()
            for i in range(len(weights)):
                noise = np.random.normal(0, self.parameter_noise_scale, weights[i].shape)
                weights[i] += noise

            noisy_actor.set_weights(weights)

            # Get action probabilities from noisy actor
            action_probs = noisy_actor.predict(state, verbose=0)[0]
        else:
            # Get action probabilities from actor
            action_probs = self.actor.predict(state, verbose=0)[0]

        # Apply noisy networks approach if enabled and exploring
        if explore and self.use_noisy_networks:
            # Add noise to action probabilities
            noise = np.random.normal(0, self.noise_scale, action_probs.shape)
            action_probs = np.abs(action_probs + noise)
            # Renormalize to ensure valid probability distribution
            action_probs = action_probs / np.sum(action_probs)

        # Sample action from the probability distribution
        action = np.random.choice(self.action_size, p=action_probs)

        # Calculate log probability of the action
        log_prob = np.log(action_probs[action] + 1e-10)

        # Get value from critic
        value = self.critic.predict(state, verbose=0)[0, 0]

        # Calculate curiosity reward if using curiosity-driven exploration
        curiosity_reward = 0
        if explore and self.exploration_strategy == "curiosity" and self.curiosity_model and len(self.states) > 0:
            # Get the previous state
            prev_state = self.states[-1]

            # Create one-hot encoded action
            action_one_hot = np.zeros(self.action_size)
            action_one_hot[action] = 1

            # Predict next state
            predicted_next_state = self.curiosity_model.predict([np.reshape(prev_state, [1, self.state_size]),
                                                               np.reshape(action_one_hot, [1, self.action_size])],
                                                              verbose=0)[0]

            # Calculate prediction error (curiosity reward)
            curiosity_reward = np.mean(np.square(predicted_next_state - state[0])) * self.curiosity_scale

            # Log the curiosity reward
            logger.debug(f"Curiosity reward: {curiosity_reward:.4f}")

        return action, log_prob, value, curiosity_reward if self.exploration_strategy == "curiosity" else None

    def remember(self, state, action, reward, value, log_prob, done):
        """
        Store experience for batch training.

        Args:
            state: Current state
            action: Action taken
            reward (float): Reward received
            value (float): Value prediction
            log_prob (float): Log probability of the action
            done (bool): Whether the episode is done
        """
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)

    def _compute_advantages(self, rewards, values, dones):
        """
        Compute advantages using Generalized Advantage Estimation (GAE).

        Args:
            rewards (list): List of rewards
            values (list): List of value predictions
            dones (list): List of done flags

        Returns:
            tuple: (advantages, returns)
        """
        advantages = np.zeros_like(rewards, dtype=np.float32)
        returns = np.zeros_like(rewards, dtype=np.float32)
        gae = 0

        # Calculate advantages in reverse order
        for i in reversed(range(len(rewards) - 1)):
            if dones[i]:
                gae = 0

            # Calculate TD error
            delta = rewards[i] + self.gamma * values[i + 1] * (1 - dones[i]) - values[i]

            # Calculate GAE
            gae = delta + self.gamma * self.lam * (1 - dones[i]) * gae
            advantages[i] = gae

        # Calculate returns
        returns = advantages + np.array(values)

        # Normalize advantages for stability
        if len(advantages) > 1:
            advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)

        return advantages, returns

    def train(self):
        """
        Train the actor and critic networks using PPO with enhanced exploration.

        Returns:
            tuple: (actor_loss, critic_loss)
        """
        if len(self.states) < self.batch_size:
            return None, None

        # Convert to numpy arrays
        states = np.vstack(self.states)
        actions = np.array(self.actions)
        rewards = np.array(self.rewards)
        values = np.array(self.values)
        old_log_probs = np.array(self.log_probs)
        dones = np.array(self.dones)

        # Compute advantages and returns
        advantages, returns = self._compute_advantages(rewards, values, dones)

        # Create one-hot encoded actions for discrete action space
        actions_one_hot = np.zeros((len(actions), self.action_size))
        for i, action in enumerate(actions):
            actions_one_hot[i, action] = 1

        # Train for multiple epochs
        actor_losses = []
        critic_losses = []
        entropy_values = []
        curiosity_losses = []

        # Train curiosity model if using curiosity-driven exploration
        if self.exploration_strategy == "curiosity" and self.curiosity_model and len(states) > 1:
            # Prepare data for curiosity model
            prev_states = states[:-1]
            next_states = states[1:]
            curiosity_actions = actions[:-1]

            # Create one-hot encoded actions
            curiosity_actions_one_hot = np.zeros((len(curiosity_actions), self.action_size))
            for i, action in enumerate(curiosity_actions):
                curiosity_actions_one_hot[i, action] = 1

            # Train curiosity model to predict next states
            curiosity_loss = self.curiosity_model.train_on_batch(
                [prev_states, curiosity_actions_one_hot],
                next_states
            )
            curiosity_losses.append(curiosity_loss)
            logger.debug(f"Curiosity model loss: {curiosity_loss:.4f}")

        for _ in range(self.epochs):
            # Generate random indices for mini-batches
            indices = np.random.permutation(len(states))

            # Train in mini-batches
            for start_idx in range(0, len(states), self.batch_size):
                end_idx = min(start_idx + self.batch_size, len(states))
                batch_indices = indices[start_idx:end_idx]

                # Get batch data
                batch_states = states[batch_indices]
                batch_actions_one_hot = actions_one_hot[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]

                # Train critic
                critic_loss = self.critic.train_on_batch(batch_states, batch_returns)
                critic_losses.append(critic_loss)

                # Train actor using custom loss function
                with tf.GradientTape() as tape:
                    # Get current action probabilities
                    action_probs = self.actor(batch_states, training=True)

                    # Calculate current log probabilities
                    current_log_probs = tf.reduce_sum(
                        tf.math.log(action_probs + 1e-10) * batch_actions_one_hot,
                        axis=1
                    )

                    # Calculate ratio of new and old probabilities
                    ratio = tf.exp(current_log_probs - batch_old_log_probs)

                    # Calculate surrogate losses
                    surrogate1 = ratio * batch_advantages
                    surrogate2 = tf.clip_by_value(
                        ratio, 1 - self.clip_ratio, 1 + self.clip_ratio
                    ) * batch_advantages

                    # Calculate entropy for exploration
                    entropy = -tf.reduce_sum(action_probs * tf.math.log(action_probs + 1e-10), axis=1)
                    mean_entropy = tf.reduce_mean(entropy)
                    entropy_values.append(mean_entropy.numpy())

                    # Calculate actor loss (negative because we want to maximize)
                    actor_loss = -tf.reduce_mean(
                        tf.minimum(surrogate1, surrogate2) + self.entropy_coef * entropy
                    )

                # Calculate gradients and apply updates
                actor_gradients = tape.gradient(actor_loss, self.actor.trainable_variables)
                self.actor.optimizer.apply_gradients(zip(actor_gradients, self.actor.trainable_variables))

                actor_losses.append(actor_loss.numpy())

                # Calculate KL divergence for early stopping
                action_probs_new = self.actor.predict(batch_states, verbose=0)
                kl = tf.reduce_mean(
                    tf.reduce_sum(
                        action_probs * tf.math.log(action_probs / (action_probs_new + 1e-10) + 1e-10),
                        axis=1
                    )
                )

                # Early stopping based on KL divergence
                if kl > 1.5 * self.target_kl:
                    break

        # Store losses and metrics
        self.actor_loss_history.extend(actor_losses)
        self.critic_loss_history.extend(critic_losses)

        # Store mean entropy
        mean_entropy_value = np.mean(entropy_values) if entropy_values else 0
        self.mean_entropy_history.append(mean_entropy_value)

        # Store mean reward
        mean_reward = np.mean(rewards)
        self.mean_reward_history.append(mean_reward)

        # Update exploration parameters based on strategy
        if self.exploration_strategy == "adaptive_entropy":
            # Adjust entropy coefficient based on performance
            if mean_reward > self.reward_threshold:
                # Good performance, reduce exploration
                self.entropy_coef = max(self.min_entropy_coef,
                                       self.entropy_coef * self.entropy_decay_rate)
            else:
                # Poor performance, increase exploration
                self.entropy_coef = min(self.max_entropy_coef,
                                       self.entropy_coef * self.entropy_increase_rate)

            logger.debug(f"Adjusted entropy coefficient to {self.entropy_coef:.6f}")

        # Decay parameter noise if using that strategy
        if self.use_parameter_noise:
            self.parameter_noise_scale *= self.parameter_noise_decay
            logger.debug(f"Parameter noise scale decayed to {self.parameter_noise_scale:.6f}")

        # Increment train step
        self.train_step += 1

        # Clear experience buffer
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []

        # Return training metrics
        result = {
            "actor_loss": np.mean(actor_losses),
            "critic_loss": np.mean(critic_losses),
            "mean_entropy": mean_entropy_value,
            "mean_reward": mean_reward,
            "entropy_coef": self.entropy_coef
        }

        # Add curiosity loss if applicable
        if curiosity_losses:
            result["curiosity_loss"] = np.mean(curiosity_losses)

        return result["actor_loss"], result["critic_loss"]

    def save(self, filename_prefix=None):
        """
        Save the actor and critic models.

        Args:
            filename_prefix (str, optional): Prefix for the filenames
        """
        if filename_prefix is None:
            filename_prefix = f"ppo_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

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
            "clip_ratio": self.clip_ratio,
            "target_kl": self.target_kl,
            "entropy_coef": self.entropy_coef,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "lam": self.lam,
            "train_step": self.train_step,
            "saved_at": datetime.now().isoformat()
        }

        metadata_path = os.path.join(self.model_dir, f"{filename_prefix}_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"PPO models saved to {self.model_dir}/{filename_prefix}_*.h5")

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
                self.clip_ratio = metadata.get("clip_ratio", self.clip_ratio)
                self.target_kl = metadata.get("target_kl", self.target_kl)
                self.entropy_coef = metadata.get("entropy_coef", self.entropy_coef)
                self.batch_size = metadata.get("batch_size", self.batch_size)
                self.epochs = metadata.get("epochs", self.epochs)
                self.lam = metadata.get("lam", self.lam)
                self.train_step = metadata.get("train_step", 0)

            logger.info(f"PPO models loaded from {actor_filepath} and {critic_filepath}")
            return True
        except Exception as e:
            logger.error(f"Error loading PPO models: {e}")
            return False
