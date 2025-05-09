"""
Multi-Objective Reinforcement Learning for NoahAI

This module implements multi-objective reinforcement learning to balance
different aspects of conversation quality.
"""

import os
import json
import numpy as np
import logging
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/multi_objective_rl.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("multi_objective_rl")

# Try to import TensorFlow
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, Model, load_model
    from tensorflow.keras.layers import Dense, Dropout, LSTM, Input, Concatenate
    from tensorflow.keras.optimizers import Adam
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    logger.warning("TensorFlow not available. Multi-objective RL will not be available.")

class ObjectiveFunction:
    """
    Objective function for multi-objective reinforcement learning.
    
    This class represents a single objective function that evaluates
    a specific aspect of conversation quality.
    """
    
    def __init__(self, name, weight=1.0, min_value=-1.0, max_value=1.0):
        """
        Initialize an objective function.
        
        Args:
            name (str): Name of the objective
            weight (float): Weight of the objective in the scalarization
            min_value (float): Minimum possible value of the objective
            max_value (float): Maximum possible value of the objective
        """
        self.name = name
        self.weight = weight
        self.min_value = min_value
        self.max_value = max_value
        self.values = []
    
    def evaluate(self, user_input, ai_response, conversation_history):
        """
        Evaluate the objective function.
        
        Args:
            user_input (str): User input
            ai_response (str): AI response
            conversation_history (list): Conversation history
            
        Returns:
            float: Objective value
        """
        # This is a base class, subclasses should implement this method
        raise NotImplementedError("Subclasses must implement evaluate method")
    
    def normalize(self, value):
        """
        Normalize the objective value to [0, 1].
        
        Args:
            value (float): Raw objective value
            
        Returns:
            float: Normalized objective value
        """
        return (value - self.min_value) / (self.max_value - self.min_value)
    
    def record_value(self, value):
        """
        Record an objective value.
        
        Args:
            value (float): Objective value to record
        """
        self.values.append(value)

class EngagementObjective(ObjectiveFunction):
    """Objective function for user engagement."""
    
    def evaluate(self, user_input, ai_response, conversation_history):
        """
        Evaluate user engagement.
        
        Args:
            user_input (str): User input
            ai_response (str): AI response
            conversation_history (list): Conversation history
            
        Returns:
            float: Engagement score
        """
        score = 0.0
        
        # Conversation length contributes to engagement
        score += min(0.5, len(conversation_history) / 20)
        
        # User input length indicates engagement
        score += min(0.3, len(user_input) / 100)
        
        # Questions in AI response encourage engagement
        if "?" in ai_response:
            score += 0.2
        
        # User responding to AI questions indicates engagement
        if conversation_history and "?" in conversation_history[-1][1] and "?" not in user_input:
            score += 0.3
        
        # Normalize to [-1, 1]
        score = min(1.0, max(-1.0, score * 2 - 1))
        
        return score

class InformativenessObjective(ObjectiveFunction):
    """Objective function for informativeness."""
    
    def evaluate(self, user_input, ai_response, conversation_history):
        """
        Evaluate informativeness.
        
        Args:
            user_input (str): User input
            ai_response (str): AI response
            conversation_history (list): Conversation history
            
        Returns:
            float: Informativeness score
        """
        score = 0.0
        
        # Response length correlates with informativeness
        score += min(0.4, len(ai_response) / 200)
        
        # Lexical diversity (unique words / total words)
        words = ai_response.lower().split()
        if words:
            unique_words = len(set(words))
            lexical_diversity = unique_words / len(words)
            score += min(0.4, lexical_diversity)
        
        # Presence of information markers
        info_markers = ["for example", "such as", "specifically", "in particular",
                       "according to", "research shows", "studies indicate"]
        for marker in info_markers:
            if marker in ai_response.lower():
                score += 0.1
                break
        
        # Responding to questions with detailed answers
        if "?" in user_input and len(ai_response) > 100:
            score += 0.3
        
        # Normalize to [-1, 1]
        score = min(1.0, max(-1.0, score * 2 - 1))
        
        return score

class CoherenceObjective(ObjectiveFunction):
    """Objective function for coherence."""
    
    def evaluate(self, user_input, ai_response, conversation_history):
        """
        Evaluate coherence.
        
        Args:
            user_input (str): User input
            ai_response (str): AI response
            conversation_history (list): Conversation history
            
        Returns:
            float: Coherence score
        """
        score = 0.0
        
        # Simple word overlap between user input and AI response
        user_words = set(user_input.lower().split())
        ai_words = set(ai_response.lower().split())
        
        if user_words and ai_words:
            overlap = len(user_words.intersection(ai_words))
            overlap_ratio = overlap / len(user_words)
            
            # Some overlap is good, too much might be repetitive
            if overlap_ratio < 0.1:
                score -= 0.3  # Too little overlap
            elif overlap_ratio > 0.7:
                score -= 0.2  # Too much overlap (might be repetitive)
            else:
                score += 0.3  # Good overlap
        
        # Responding to questions with answers
        if "?" in user_input:
            if any(marker in ai_response.lower() for marker in ["yes", "no", "maybe", "possibly"]):
                score += 0.2
        
        # Maintaining context from previous turns
        if len(conversation_history) >= 2:
            prev_user_input = conversation_history[-2][1] if conversation_history[-2][0] == "user" else ""
            prev_user_words = set(prev_user_input.lower().split())
            
            if prev_user_words and ai_words:
                prev_overlap = len(prev_user_words.intersection(ai_words))
                if prev_overlap > 0:
                    score += 0.2
        
        # Default score if no specific patterns are found
        if score == 0.0:
            score = 0.1
        
        # Normalize to [-1, 1]
        score = min(1.0, max(-1.0, score))
        
        return score

class MultiObjectiveRLAgent:
    """
    Multi-Objective Reinforcement Learning Agent for NoahAI.
    
    This agent balances multiple objectives to optimize conversation quality.
    """
    
    def __init__(self, state_size=128, action_size=10, model_dir="data/rl_models/multi_objective",
                 learning_rate=0.001, gamma=0.99, scalarization_method="weighted_sum"):
        """
        Initialize the multi-objective RL agent.
        
        Args:
            state_size (int): Size of the state space
            action_size (int): Size of the action space
            model_dir (str): Directory to save/load models
            learning_rate (float): Learning rate for the networks
            gamma (float): Discount factor for future rewards
            scalarization_method (str): Method to scalarize multiple objectives
                Options: "weighted_sum", "chebyshev", "linear"
        """
        self.state_size = state_size
        self.action_size = action_size
        self.model_dir = model_dir
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.scalarization_method = scalarization_method
        
        # Create model directory if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)
        
        # Initialize objectives
        self.objectives = [
            EngagementObjective("engagement", weight=0.4),
            InformativenessObjective("informativeness", weight=0.3),
            CoherenceObjective("coherence", weight=0.3)
        ]
        
        # Initialize policy network
        self.policy_network = self._build_policy_network()
        
        # Initialize value networks (one per objective)
        self.value_networks = []
        for objective in self.objectives:
            value_network = self._build_value_network()
            self.value_networks.append(value_network)
        
        # Experience buffer
        self.states = []
        self.actions = []
        self.rewards = []  # List of reward vectors
        self.next_states = []
        self.dones = []
        
        # Training metrics
        self.policy_loss_history = []
        self.value_loss_history = []
        self.objective_values = {obj.name: [] for obj in self.objectives}
        self.pareto_front = []
    
    def _build_policy_network(self):
        """
        Build the policy network.
        
        Returns:
            tf.keras.Model: Policy network
        """
        if not TENSORFLOW_AVAILABLE:
            logger.error("TensorFlow is required for multi-objective RL.")
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
    
    def _build_value_network(self):
        """
        Build a value network for an objective.
        
        Returns:
            tf.keras.Model: Value network
        """
        if not TENSORFLOW_AVAILABLE:
            logger.error("TensorFlow is required for multi-objective RL.")
            return None
        
        # Input layer
        inputs = Input(shape=(self.state_size,))
        
        # Hidden layers
        x = Dense(64, activation='relu')(inputs)
        x = Dense(32, activation='relu')(x)
        
        # Output layer (value)
        outputs = Dense(1, activation='linear')(x)
        
        # Create model
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=Adam(learning_rate=self.learning_rate),
                     loss='mse')
        
        return model
    
    def get_action(self, state):
        """
        Get an action based on the current state.
        
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
    
    def evaluate_objectives(self, user_input, ai_response, conversation_history):
        """
        Evaluate all objectives.
        
        Args:
            user_input (str): User input
            ai_response (str): AI response
            conversation_history (list): Conversation history
            
        Returns:
            list: List of objective values
        """
        objective_values = []
        
        for objective in self.objectives:
            value = objective.evaluate(user_input, ai_response, conversation_history)
            objective.record_value(value)
            objective_values.append(value)
            
            # Store for visualization
            self.objective_values[objective.name].append(value)
        
        return objective_values
    
    def scalarize_rewards(self, reward_vector):
        """
        Scalarize a vector of rewards into a single scalar reward.
        
        Args:
            reward_vector (list): Vector of rewards for each objective
            
        Returns:
            float: Scalarized reward
        """
        if self.scalarization_method == "weighted_sum":
            # Weighted sum scalarization
            return sum(w * r for w, r in zip([obj.weight for obj in self.objectives], reward_vector))
        
        elif self.scalarization_method == "chebyshev":
            # Chebyshev scalarization (minimize the maximum weighted deviation from a reference point)
            # Reference point is the maximum possible value for each objective
            reference_point = [obj.max_value for obj in self.objectives]
            weights = [obj.weight for obj in self.objectives]
            
            # Calculate weighted deviations
            deviations = [w * (ref - r) for w, ref, r in zip(weights, reference_point, reward_vector)]
            
            # Return negative of maximum deviation (we want to maximize reward)
            return -max(deviations)
        
        elif self.scalarization_method == "linear":
            # Linear scalarization with normalization
            normalized_rewards = [obj.normalize(r) for obj, r in zip(self.objectives, reward_vector)]
            weights = [obj.weight for obj in self.objectives]
            
            return sum(w * r for w, r in zip(weights, normalized_rewards))
        
        else:
            # Default to weighted sum
            return sum(w * r for w, r in zip([obj.weight for obj in self.objectives], reward_vector))
    
    def remember(self, state, action, reward_vector, next_state, done):
        """
        Store experience in memory.
        
        Args:
            state: Current state
            action: Action taken
            reward_vector (list): Vector of rewards for each objective
            next_state: Next state
            done: Whether the episode is done
        """
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward_vector)
        self.next_states.append(next_state)
        self.dones.append(done)
        
        # Update Pareto front
        self._update_pareto_front(reward_vector)
    
    def _update_pareto_front(self, reward_vector):
        """
        Update the Pareto front with a new reward vector.
        
        Args:
            reward_vector (list): Vector of rewards for each objective
        """
        # Check if the new point is dominated by any point in the Pareto front
        dominated = False
        for point in self.pareto_front:
            if all(p >= r for p, r in zip(point, reward_vector)) and any(p > r for p, r in zip(point, reward_vector)):
                dominated = True
                break
        
        if not dominated:
            # Remove points from Pareto front that are dominated by the new point
            self.pareto_front = [point for point in self.pareto_front if not (
                all(r >= p for r, p in zip(reward_vector, point)) and 
                any(r > p for r, p in zip(reward_vector, point))
            )]
            
            # Add the new point to the Pareto front
            self.pareto_front.append(reward_vector)
    
    def train(self, batch_size=32):
        """
        Train the multi-objective RL agent.
        
        Args:
            batch_size (int): Batch size for training
            
        Returns:
            dict: Training results
        """
        if not TENSORFLOW_AVAILABLE or len(self.states) < batch_size:
            return None
        
        # Convert to numpy arrays
        states = np.vstack(self.states)
        actions = np.array(self.actions)
        rewards = np.array(self.rewards)  # Shape: (n_samples, n_objectives)
        next_states = np.vstack(self.next_states)
        dones = np.array(self.dones)
        
        # Create one-hot encoded actions
        actions_one_hot = np.zeros((len(actions), self.action_size))
        for i, action in enumerate(actions):
            actions_one_hot[i, action] = 1
        
        # Train value networks (one per objective)
        value_losses = []
        for i, value_network in enumerate(self.value_networks):
            # Get objective rewards
            objective_rewards = rewards[:, i]
            
            # Calculate returns (simple Monte Carlo)
            returns = np.zeros_like(objective_rewards)
            discounted_sum = 0
            for t in reversed(range(len(objective_rewards))):
                discounted_sum = objective_rewards[t] + self.gamma * discounted_sum * (1 - dones[t])
                returns[t] = discounted_sum
            
            # Train value network
            value_loss = value_network.train_on_batch(states, returns)
            value_losses.append(value_loss)
        
        # Train policy network
        with tf.GradientTape() as tape:
            # Get action probabilities
            action_probs = self.policy_network(states, training=True)
            
            # Calculate log probabilities of actions
            log_probs = tf.reduce_sum(
                tf.math.log(action_probs + 1e-10) * actions_one_hot,
                axis=1
            )
            
            # Scalarize rewards for policy update
            scalarized_rewards = np.array([self.scalarize_rewards(r) for r in rewards])
            
            # Calculate policy loss (negative because we want to maximize)
            policy_loss = -tf.reduce_mean(log_probs * scalarized_rewards)
        
        # Calculate gradients and apply updates
        policy_gradients = tape.gradient(policy_loss, self.policy_network.trainable_variables)
        self.policy_network.optimizer.apply_gradients(zip(policy_gradients, self.policy_network.trainable_variables))
        
        # Store losses
        self.policy_loss_history.append(policy_loss.numpy())
        self.value_loss_history.append(np.mean(value_losses))
        
        # Clear experience buffer
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []
        
        return {
            "policy_loss": policy_loss.numpy(),
            "value_losses": value_losses,
            "pareto_front_size": len(self.pareto_front)
        }
    
    def save(self, filename_prefix=None):
        """
        Save the multi-objective RL agent.
        
        Args:
            filename_prefix (str, optional): Prefix for the filenames
        """
        if not TENSORFLOW_AVAILABLE:
            logger.error("TensorFlow is required to save models.")
            return False
        
        if filename_prefix is None:
            filename_prefix = f"multi_objective_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Save policy network
        policy_filepath = os.path.join(self.model_dir, f"{filename_prefix}_policy.h5")
        self.policy_network.save(policy_filepath)
        
        # Save value networks
        for i, value_network in enumerate(self.value_networks):
            value_filepath = os.path.join(self.model_dir, f"{filename_prefix}_value_{i}.h5")
            value_network.save(value_filepath)
        
        # Save metadata
        metadata = {
            "state_size": self.state_size,
            "action_size": self.action_size,
            "learning_rate": self.learning_rate,
            "gamma": self.gamma,
            "scalarization_method": self.scalarization_method,
            "objectives": [
                {
                    "name": obj.name,
                    "weight": obj.weight,
                    "min_value": obj.min_value,
                    "max_value": obj.max_value
                }
                for obj in self.objectives
            ],
            "policy_loss_history": self.policy_loss_history,
            "value_loss_history": self.value_loss_history,
            "objective_values": self.objective_values,
            "pareto_front": self.pareto_front,
            "saved_at": datetime.now().isoformat()
        }
        
        metadata_path = os.path.join(self.model_dir, f"{filename_prefix}_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Multi-objective RL agent saved to {self.model_dir}/{filename_prefix}_*")
        return True
    
    def visualize_pareto_front(self, save_path=None):
        """
        Visualize the Pareto front.
        
        Args:
            save_path (str, optional): Path to save the visualization
        """
        if len(self.objectives) < 2 or len(self.pareto_front) == 0:
            logger.warning("Cannot visualize Pareto front with less than 2 objectives or empty front.")
            return
        
        if len(self.objectives) == 2:
            # 2D visualization
            plt.figure(figsize=(10, 6))
            
            # Plot all points
            all_rewards = np.array(self.rewards)
            plt.scatter(all_rewards[:, 0], all_rewards[:, 1], alpha=0.5, label="All points")
            
            # Plot Pareto front
            pareto_front = np.array(self.pareto_front)
            plt.scatter(pareto_front[:, 0], pareto_front[:, 1], color='red', label="Pareto front")
            
            # Connect Pareto front points
            pareto_front = pareto_front[pareto_front[:, 0].argsort()]
            plt.plot(pareto_front[:, 0], pareto_front[:, 1], 'r--')
            
            plt.title("Pareto Front")
            plt.xlabel(self.objectives[0].name)
            plt.ylabel(self.objectives[1].name)
            plt.legend()
            plt.grid(True, alpha=0.3)
            
        elif len(self.objectives) == 3:
            # 3D visualization
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            
            # Plot all points
            all_rewards = np.array(self.rewards)
            ax.scatter(all_rewards[:, 0], all_rewards[:, 1], all_rewards[:, 2], alpha=0.5, label="All points")
            
            # Plot Pareto front
            pareto_front = np.array(self.pareto_front)
            ax.scatter(pareto_front[:, 0], pareto_front[:, 1], pareto_front[:, 2], color='red', label="Pareto front")
            
            ax.set_title("Pareto Front")
            ax.set_xlabel(self.objectives[0].name)
            ax.set_ylabel(self.objectives[1].name)
            ax.set_zlabel(self.objectives[2].name)
            ax.legend()
            
        else:
            # For more than 3 objectives, use parallel coordinates
            plt.figure(figsize=(12, 6))
            
            # Get objective names
            objective_names = [obj.name for obj in self.objectives]
            
            # Plot all points (sample if too many)
            all_rewards = np.array(self.rewards)
            if len(all_rewards) > 100:
                indices = np.random.choice(len(all_rewards), 100, replace=False)
                all_rewards = all_rewards[indices]
            
            # Plot each point
            for i, point in enumerate(all_rewards):
                plt.plot(range(len(objective_names)), point, color='blue', alpha=0.1)
            
            # Plot Pareto front
            for i, point in enumerate(self.pareto_front):
                plt.plot(range(len(objective_names)), point, color='red', alpha=0.7)
            
            plt.title("Parallel Coordinates Plot of Pareto Front")
            plt.xticks(range(len(objective_names)), objective_names)
            plt.grid(True, alpha=0.3)
            
            # Add legend
            plt.plot([], [], color='blue', alpha=0.5, label="All points")
            plt.plot([], [], color='red', alpha=0.7, label="Pareto front")
            plt.legend()
        
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Saved Pareto front visualization to {save_path}")
        else:
            plt.show()
