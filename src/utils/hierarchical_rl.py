"""
Hierarchical Reinforcement Learning for NoahAI

This module implements hierarchical reinforcement learning with temporal abstraction
for handling complex tasks in conversation.
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
        logging.FileHandler("logs/hierarchical_rl.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("hierarchical_rl")

# Try to import TensorFlow
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, Model, load_model
    from tensorflow.keras.layers import Dense, Dropout, LSTM, Input, Concatenate
    from tensorflow.keras.optimizers import Adam
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    logger.warning("TensorFlow not available. Hierarchical RL will not be available.")

class Option:
    """
    Option class for the options framework in hierarchical RL.
    
    An option is a temporally extended action that consists of:
    - An initiation set (when the option can be started)
    - A policy (what actions to take during the option)
    - A termination condition (when the option ends)
    """
    
    def __init__(self, name, policy_network=None, termination_network=None, 
                 state_size=128, action_size=10, learning_rate=0.001):
        """
        Initialize an option.
        
        Args:
            name (str): Name of the option
            policy_network (tf.keras.Model, optional): Policy network for the option
            termination_network (tf.keras.Model, optional): Termination network for the option
            state_size (int): Size of the state space
            action_size (int): Size of the action space
            learning_rate (float): Learning rate for the networks
        """
        self.name = name
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        
        # Initialize networks
        if policy_network is not None:
            self.policy_network = policy_network
        else:
            self.policy_network = self._build_policy_network()
            
        if termination_network is not None:
            self.termination_network = termination_network
        else:
            self.termination_network = self._build_termination_network()
        
        # Experience buffer
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []
        
        # Training metrics
        self.policy_loss_history = []
        self.termination_loss_history = []
    
    def _build_policy_network(self):
        """
        Build the policy network for the option.
        
        Returns:
            tf.keras.Model: Policy network
        """
        if not TENSORFLOW_AVAILABLE:
            logger.error("TensorFlow is required for hierarchical RL.")
            return None
        
        # Input layer
        inputs = Input(shape=(self.state_size,))
        
        # Hidden layers
        x = Dense(64, activation='relu')(inputs)
        x = Dense(64, activation='relu')(x)
        
        # Output layer (action probabilities)
        outputs = Dense(self.action_size, activation='softmax')(x)
        
        # Create model
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=Adam(learning_rate=self.learning_rate))
        
        return model
    
    def _build_termination_network(self):
        """
        Build the termination network for the option.
        
        Returns:
            tf.keras.Model: Termination network
        """
        if not TENSORFLOW_AVAILABLE:
            logger.error("TensorFlow is required for hierarchical RL.")
            return None
        
        # Input layer
        inputs = Input(shape=(self.state_size,))
        
        # Hidden layers
        x = Dense(64, activation='relu')(inputs)
        x = Dense(32, activation='relu')(x)
        
        # Output layer (termination probability)
        outputs = Dense(1, activation='sigmoid')(x)
        
        # Create model
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=Adam(learning_rate=self.learning_rate),
                     loss='binary_crossentropy')
        
        return model
    
    def get_action(self, state):
        """
        Get an action from the option's policy.
        
        Args:
            state: Current state
            
        Returns:
            int: Selected action
        """
        if not TENSORFLOW_AVAILABLE or self.policy_network is None:
            return np.random.randint(0, self.action_size)
        
        # Reshape state
        state = np.reshape(state, [1, self.state_size])
        
        # Get action probabilities
        action_probs = self.policy_network.predict(state, verbose=0)[0]
        
        # Sample action
        action = np.random.choice(self.action_size, p=action_probs)
        
        return action
    
    def should_terminate(self, state):
        """
        Determine if the option should terminate.
        
        Args:
            state: Current state
            
        Returns:
            bool: Whether the option should terminate
        """
        if not TENSORFLOW_AVAILABLE or self.termination_network is None:
            return np.random.random() < 0.1  # 10% chance to terminate
        
        # Reshape state
        state = np.reshape(state, [1, self.state_size])
        
        # Get termination probability
        termination_prob = self.termination_network.predict(state, verbose=0)[0, 0]
        
        # Sample termination
        return np.random.random() < termination_prob
    
    def train(self, batch_size=32):
        """
        Train the option's networks.
        
        Args:
            batch_size (int): Batch size for training
            
        Returns:
            tuple: (policy_loss, termination_loss)
        """
        if not TENSORFLOW_AVAILABLE or len(self.states) < batch_size:
            return None, None
        
        # Convert to numpy arrays
        states = np.vstack(self.states)
        actions = np.array(self.actions)
        rewards = np.array(self.rewards)
        next_states = np.vstack(self.next_states)
        dones = np.array(self.dones)
        
        # Create one-hot encoded actions
        actions_one_hot = np.zeros((len(actions), self.action_size))
        for i, action in enumerate(actions):
            actions_one_hot[i, action] = 1
        
        # Train policy network
        with tf.GradientTape() as tape:
            # Get action probabilities
            action_probs = self.policy_network(states, training=True)
            
            # Calculate log probabilities of actions
            log_probs = tf.reduce_sum(
                tf.math.log(action_probs + 1e-10) * actions_one_hot,
                axis=1
            )
            
            # Calculate policy loss (negative because we want to maximize)
            policy_loss = -tf.reduce_mean(log_probs * rewards)
        
        # Calculate gradients and apply updates
        policy_gradients = tape.gradient(policy_loss, self.policy_network.trainable_variables)
        self.policy_network.optimizer.apply_gradients(zip(policy_gradients, self.policy_network.trainable_variables))
        
        # Train termination network
        # Use rewards as a proxy for termination signal (high reward = don't terminate)
        termination_targets = (1.0 - rewards).clip(0, 1)  # Invert and clip rewards
        termination_loss = self.termination_network.train_on_batch(states, termination_targets)
        
        # Store losses
        self.policy_loss_history.append(policy_loss.numpy())
        self.termination_loss_history.append(termination_loss)
        
        # Clear experience buffer
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []
        
        return policy_loss.numpy(), termination_loss

class HierarchicalRLAgent:
    """
    Hierarchical Reinforcement Learning Agent for NoahAI.
    
    This agent uses the options framework to handle complex tasks
    with temporal abstraction.
    """
    
    def __init__(self, state_size=128, action_size=10, num_options=3,
                 model_dir="data/rl_models/hierarchical", learning_rate=0.001,
                 gamma=0.99, option_duration=5):
        """
        Initialize the hierarchical RL agent.
        
        Args:
            state_size (int): Size of the state space
            action_size (int): Size of the action space
            num_options (int): Number of options to create
            model_dir (str): Directory to save/load models
            learning_rate (float): Learning rate for the networks
            gamma (float): Discount factor for future rewards
            option_duration (int): Average duration of options in steps
        """
        self.state_size = state_size
        self.action_size = action_size
        self.num_options = num_options
        self.model_dir = model_dir
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.option_duration = option_duration
        
        # Create model directory if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)
        
        # Initialize options
        self.options = []
        for i in range(num_options):
            option = Option(
                name=f"option_{i}",
                state_size=state_size,
                action_size=action_size,
                learning_rate=learning_rate
            )
            self.options.append(option)
        
        # Initialize meta-controller (selects options)
        self.meta_controller = self._build_meta_controller()
        
        # Current option
        self.current_option = None
        self.current_option_steps = 0
        
        # Experience buffer for meta-controller
        self.meta_states = []
        self.meta_options = []
        self.meta_rewards = []
        self.meta_next_states = []
        self.meta_dones = []
        
        # Training metrics
        self.meta_loss_history = []
        self.option_rewards = [[] for _ in range(num_options)]
    
    def _build_meta_controller(self):
        """
        Build the meta-controller network.
        
        Returns:
            tf.keras.Model: Meta-controller network
        """
        if not TENSORFLOW_AVAILABLE:
            logger.error("TensorFlow is required for hierarchical RL.")
            return None
        
        # Input layer
        inputs = Input(shape=(self.state_size,))
        
        # Hidden layers
        x = Dense(64, activation='relu')(inputs)
        x = Dense(64, activation='relu')(x)
        
        # Output layer (option probabilities)
        outputs = Dense(self.num_options, activation='softmax')(x)
        
        # Create model
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=Adam(learning_rate=self.learning_rate))
        
        return model
    
    def select_option(self, state):
        """
        Select an option using the meta-controller.
        
        Args:
            state: Current state
            
        Returns:
            int: Selected option index
        """
        if not TENSORFLOW_AVAILABLE or self.meta_controller is None:
            return np.random.randint(0, self.num_options)
        
        # Reshape state
        state = np.reshape(state, [1, self.state_size])
        
        # Get option probabilities
        option_probs = self.meta_controller.predict(state, verbose=0)[0]
        
        # Sample option
        option_index = np.random.choice(self.num_options, p=option_probs)
        
        return option_index
    
    def get_action(self, state):
        """
        Get an action based on the current state.
        
        Args:
            state: Current state
            
        Returns:
            tuple: (action, option_index)
        """
        # If no current option or option should terminate, select a new option
        if (self.current_option is None or 
            self.current_option.should_terminate(state) or
            self.current_option_steps >= self.option_duration):
            
            option_index = self.select_option(state)
            self.current_option = self.options[option_index]
            self.current_option_steps = 0
        else:
            option_index = self.options.index(self.current_option)
            self.current_option_steps += 1
        
        # Get action from current option
        action = self.current_option.get_action(state)
        
        return action, option_index
    
    def remember(self, state, action, reward, next_state, done, option_index):
        """
        Store experience in memory.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether the episode is done
            option_index: Index of the option used
        """
        # Store experience for the option
        option = self.options[option_index]
        option.states.append(state)
        option.actions.append(action)
        option.rewards.append(reward)
        option.next_states.append(next_state)
        option.dones.append(done)
        
        # Store experience for meta-controller
        # Only store when option terminates or episode ends
        if option.should_terminate(next_state) or done or self.current_option_steps >= self.option_duration:
            self.meta_states.append(state)
            self.meta_options.append(option_index)
            self.meta_rewards.append(reward)  # Use the last reward for simplicity
            self.meta_next_states.append(next_state)
            self.meta_dones.append(done)
            
            # Record option reward
            self.option_rewards[option_index].append(reward)
    
    def train(self, batch_size=32):
        """
        Train the hierarchical RL agent.
        
        Args:
            batch_size (int): Batch size for training
            
        Returns:
            dict: Training results
        """
        results = {}
        
        # Train options
        option_results = []
        for i, option in enumerate(self.options):
            policy_loss, termination_loss = option.train(batch_size)
            option_results.append({
                "option": i,
                "policy_loss": policy_loss,
                "termination_loss": termination_loss
            })
        
        results["options"] = option_results
        
        # Train meta-controller
        if TENSORFLOW_AVAILABLE and len(self.meta_states) >= batch_size:
            # Convert to numpy arrays
            states = np.vstack(self.meta_states)
            options = np.array(self.meta_options)
            rewards = np.array(self.meta_rewards)
            next_states = np.vstack(self.meta_next_states)
            dones = np.array(self.meta_dones)
            
            # Create one-hot encoded options
            options_one_hot = np.zeros((len(options), self.num_options))
            for i, option_idx in enumerate(options):
                options_one_hot[i, option_idx] = 1
            
            # Train meta-controller
            with tf.GradientTape() as tape:
                # Get option probabilities
                option_probs = self.meta_controller(states, training=True)
                
                # Calculate log probabilities of options
                log_probs = tf.reduce_sum(
                    tf.math.log(option_probs + 1e-10) * options_one_hot,
                    axis=1
                )
                
                # Calculate meta-controller loss (negative because we want to maximize)
                meta_loss = -tf.reduce_mean(log_probs * rewards)
            
            # Calculate gradients and apply updates
            meta_gradients = tape.gradient(meta_loss, self.meta_controller.trainable_variables)
            self.meta_controller.optimizer.apply_gradients(zip(meta_gradients, self.meta_controller.trainable_variables))
            
            # Store loss
            self.meta_loss_history.append(meta_loss.numpy())
            
            # Clear experience buffer
            self.meta_states = []
            self.meta_options = []
            self.meta_rewards = []
            self.meta_next_states = []
            self.meta_dones = []
            
            results["meta_loss"] = meta_loss.numpy()
        
        return results
    
    def save(self, filename_prefix=None):
        """
        Save the hierarchical RL agent.
        
        Args:
            filename_prefix (str, optional): Prefix for the filenames
        """
        if not TENSORFLOW_AVAILABLE:
            logger.error("TensorFlow is required to save models.")
            return False
        
        if filename_prefix is None:
            filename_prefix = f"hierarchical_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Save meta-controller
        meta_filepath = os.path.join(self.model_dir, f"{filename_prefix}_meta.h5")
        self.meta_controller.save(meta_filepath)
        
        # Save options
        for i, option in enumerate(self.options):
            option_filepath = os.path.join(self.model_dir, f"{filename_prefix}_option_{i}.h5")
            option.policy_network.save(option_filepath)
            
            termination_filepath = os.path.join(self.model_dir, f"{filename_prefix}_termination_{i}.h5")
            option.termination_network.save(termination_filepath)
        
        # Save metadata
        metadata = {
            "state_size": self.state_size,
            "action_size": self.action_size,
            "num_options": self.num_options,
            "learning_rate": self.learning_rate,
            "gamma": self.gamma,
            "option_duration": self.option_duration,
            "meta_loss_history": self.meta_loss_history,
            "option_rewards": self.option_rewards,
            "saved_at": datetime.now().isoformat()
        }
        
        metadata_path = os.path.join(self.model_dir, f"{filename_prefix}_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Hierarchical RL agent saved to {self.model_dir}/{filename_prefix}_*")
        return True
    
    def load(self, filename_prefix):
        """
        Load a saved hierarchical RL agent.
        
        Args:
            filename_prefix (str): Prefix for the filenames
            
        Returns:
            bool: Whether the agent was loaded successfully
        """
        if not TENSORFLOW_AVAILABLE:
            logger.error("TensorFlow is required to load models.")
            return False
        
        try:
            # Load meta-controller
            meta_filepath = os.path.join(self.model_dir, f"{filename_prefix}_meta.h5")
            self.meta_controller = load_model(meta_filepath)
            
            # Load options
            for i in range(self.num_options):
                option_filepath = os.path.join(self.model_dir, f"{filename_prefix}_option_{i}.h5")
                self.options[i].policy_network = load_model(option_filepath)
                
                termination_filepath = os.path.join(self.model_dir, f"{filename_prefix}_termination_{i}.h5")
                self.options[i].termination_network = load_model(termination_filepath)
            
            # Load metadata
            metadata_path = os.path.join(self.model_dir, f"{filename_prefix}_metadata.json")
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            self.state_size = metadata.get("state_size", self.state_size)
            self.action_size = metadata.get("action_size", self.action_size)
            self.num_options = metadata.get("num_options", self.num_options)
            self.learning_rate = metadata.get("learning_rate", self.learning_rate)
            self.gamma = metadata.get("gamma", self.gamma)
            self.option_duration = metadata.get("option_duration", self.option_duration)
            self.meta_loss_history = metadata.get("meta_loss_history", [])
            self.option_rewards = metadata.get("option_rewards", [[] for _ in range(self.num_options)])
            
            logger.info(f"Hierarchical RL agent loaded from {self.model_dir}/{filename_prefix}_*")
            return True
        except Exception as e:
            logger.error(f"Error loading hierarchical RL agent: {e}")
            return False
