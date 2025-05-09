"""
Learning utilities for NoahAI

This module provides utilities for collecting feedback and training the AI model.
"""

import os
import json
import time
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/learning_utils.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("learning_utils")

# Try to import reinforcement learning modules
try:
    from src.utils.rl_manager import RLManager
    RL_MANAGER_AVAILABLE = True
    logger.info("Reinforcement learning manager available.")
except ImportError:
    RL_MANAGER_AVAILABLE = False
    logger.warning("Reinforcement learning manager not available.")

class LearningUtils:
    """
    Utility class for learning and improving the AI over time.
    """

    def __init__(self, model, data_dir="data", use_reinforcement_learning=False,
                 rl_algorithm="ppo", rl_config=None, enable_online_learning=True):
        """
        Initialize the learning utilities.

        Args:
            model: The model to train
            data_dir (str): Directory to store learning data
            use_reinforcement_learning (bool): Whether to use reinforcement learning
            rl_algorithm (str): Reinforcement learning algorithm to use
            rl_config (dict, optional): Configuration for the RL algorithm
            enable_online_learning (bool): Whether to enable online learning during conversations
        """
        self.model = model
        self.data_dir = data_dir
        self.feedback_file = os.path.join(data_dir, "feedback.json")
        self.training_log_file = os.path.join(data_dir, "training_log.json")
        self.use_reinforcement_learning = use_reinforcement_learning
        self.rl_algorithm = rl_algorithm
        self.rl_config = rl_config or {}
        self.enable_online_learning = enable_online_learning

        # Online learning parameters
        self.online_learning_threshold = self.rl_config.get("online_learning_threshold", 3)  # Min feedback items before online update
        self.online_learning_frequency = self.rl_config.get("online_learning_frequency", 5)  # Update every N feedback items
        self.online_learning_batch_size = self.rl_config.get("online_learning_batch_size", 2)  # Batch size for online updates
        self.online_learning_epochs = self.rl_config.get("online_learning_epochs", 1)  # Epochs for online updates
        self.recent_feedback_count = 0  # Counter for recent feedback since last online update
        self.online_updates_performed = 0  # Counter for online updates performed

        # Create data directory if it doesn't exist
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(os.path.join(data_dir, "rl_models"), exist_ok=True)
        os.makedirs("logs", exist_ok=True)

        # Initialize feedback data
        self.feedback_data = self._load_feedback_data()

        # Initialize training log
        self.training_log = self._load_training_log()

        # Initialize reinforcement learning manager if available and enabled
        self.rl_manager = None
        if self.use_reinforcement_learning and RL_MANAGER_AVAILABLE:
            try:
                # Extract reward weights from config if available
                reward_weights = self.rl_config.get("reward_weights", {})

                # Set state size based on model complexity
                if hasattr(model, 'model_type'):
                    if model.model_type == "simple_lstm":
                        state_size = 64
                    elif model.model_type == "advanced_lstm":
                        state_size = 128
                    elif model.model_type == "cnn_lstm":
                        state_size = 192
                    elif model.model_type == "transfer_learning":
                        state_size = 256
                    else:
                        state_size = 128  # Default
                else:
                    state_size = 128  # Default

                # Get action size (number of response categories) from model if available
                if hasattr(model, 'responses') and isinstance(model.responses, dict):
                    action_size = len(model.responses.keys())
                else:
                    action_size = 10  # Default

                self.rl_manager = RLManager(
                    model_dir=os.path.join(data_dir, "rl_models"),
                    algorithm=rl_algorithm,
                    state_size=state_size,
                    action_size=action_size,
                    config={**self.rl_config, "reward_weights": reward_weights}
                )
                logger.info(f"Initialized reinforcement learning manager with algorithm: {rl_algorithm}, "
                           f"state_size: {state_size}, action_size: {action_size}")
            except Exception as e:
                logger.error(f"Error initializing reinforcement learning manager: {e}")
                self.rl_manager = None
                self.use_reinforcement_learning = False

    def _load_feedback_data(self):
        """
        Load feedback data from file.

        Returns:
            list: List of feedback entries
        """
        if os.path.exists(self.feedback_file):
            try:
                with open(self.feedback_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading feedback data: {e}")

        return []

    def _load_training_log(self):
        """
        Load training log from file.

        Returns:
            list: List of training sessions
        """
        if os.path.exists(self.training_log_file):
            try:
                with open(self.training_log_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading training log: {e}")

        return []

    def _save_feedback_data(self):
        """
        Save feedback data to file.
        """
        try:
            with open(self.feedback_file, 'w') as f:
                json.dump(self.feedback_data, f, indent=2)
        except Exception as e:
            print(f"Error saving feedback data: {e}")

    def _save_training_log(self):
        """
        Save training log to file.
        """
        try:
            with open(self.training_log_file, 'w') as f:
                json.dump(self.training_log, f, indent=2)
        except Exception as e:
            print(f"Error saving training log: {e}")

    def add_feedback(self, user_input, ai_response, rating, category=None, conversation_history=None):
        """
        Add user feedback for a conversation.

        Args:
            user_input (str): The user's input text
            ai_response (str): The AI's response
            rating (int): User rating (1-5, where 5 is best)
            category (str, optional): The response category
            conversation_history (list, optional): Conversation history

        Returns:
            float or None: Reward value if reinforcement learning is enabled, None otherwise
        """
        # Validate rating
        if not 1 <= rating <= 5:
            raise ValueError("Rating must be between 1 and 5")

        # Create feedback entry
        feedback_entry = {
            "user_input": user_input,
            "ai_response": ai_response,
            "rating": rating,
            "category": category,
            "timestamp": datetime.now().isoformat()
        }

        # Add to feedback data
        self.feedback_data.append(feedback_entry)

        # Save feedback data
        self._save_feedback_data()

        # Also add to model's feedback history if it has that attribute
        if hasattr(self.model, 'add_feedback'):
            self.model.add_feedback(user_input, ai_response, rating)

        reward = None

        # Process feedback with reinforcement learning if enabled
        if self.use_reinforcement_learning and self.rl_manager:
            try:
                # Default conversation history if not provided
                if conversation_history is None:
                    conversation_history = []

                # Process feedback with RL manager
                reward = self.rl_manager.process_feedback(
                    user_input=user_input,
                    ai_response=ai_response,
                    rating=rating,
                    conversation_history=conversation_history
                )

                # Log the reward
                logger.info(f"Reinforcement learning reward: {reward}")

                # Add reward to feedback entry
                feedback_entry["rl_reward"] = reward

                # Update feedback data with reward
                self.feedback_data[-1] = feedback_entry

                # Save updated feedback data
                self._save_feedback_data()

                # Increment recent feedback counter for online learning
                self.recent_feedback_count += 1

                # Check if we should perform online learning
                if (self.enable_online_learning and
                    len(self.feedback_data) >= self.online_learning_threshold and
                    self.recent_feedback_count >= self.online_learning_frequency):

                    # Perform online learning
                    online_result = self._perform_online_learning()

                    if online_result:
                        logger.info(f"Online learning update performed: {online_result}")
                        # Reset counter after successful update
                        self.recent_feedback_count = 0
                        self.online_updates_performed += 1

            except Exception as e:
                logger.error(f"Error processing feedback with reinforcement learning: {e}")

        return reward

    def _perform_online_learning(self):
        """
        Perform online learning update based on recent feedback.

        This enables continuous learning during conversations without
        requiring a full training session.

        Returns:
            dict or None: Results of the online learning update, or None if update failed
        """
        try:
            # Check if we have enough feedback data
            if len(self.feedback_data) < self.online_learning_threshold:
                logger.debug("Not enough feedback data for online learning update")
                return None

            logger.info(f"Performing online learning update (update #{self.online_updates_performed + 1})")

            # Start timer
            start_time = time.time()

            # Get most recent feedback entries
            recent_feedback = self.feedback_data[-self.online_learning_frequency:]

            # Update reinforcement learning model if enabled
            rl_results = None
            if self.use_reinforcement_learning and self.rl_manager:
                try:
                    # Train the RL model with recent experiences
                    rl_results = self.rl_manager.train()
                    logger.info(f"Online RL update results: {rl_results}")
                except Exception as e:
                    logger.error(f"Error in online RL update: {e}")

            # Update deep learning model if available
            dl_results = None
            if hasattr(self.model, 'train'):
                try:
                    # Prepare mini-batch for training
                    texts = []
                    categories = []
                    ratings = []

                    # Process recent feedback
                    for feedback in recent_feedback:
                        user_input = feedback["user_input"]
                        category = feedback["category"]
                        rating = feedback["rating"]

                        # Only use feedback with category information
                        if category:
                            texts.append(user_input)
                            categories.append(category)
                            ratings.append(rating)

                    # Only train if we have categorized feedback
                    if texts and categories:
                        # Train with a single mini-batch update
                        dl_results = self.model.train(
                            epochs=self.online_learning_epochs,
                            batch_size=min(self.online_learning_batch_size, len(texts)),
                            use_early_stopping=False  # No early stopping for online updates
                        )
                        logger.info(f"Online DL update performed with {len(texts)} examples")
                except Exception as e:
                    logger.error(f"Error in online DL update: {e}")

            # Calculate training time
            training_time = time.time() - start_time

            # Create online learning log entry
            online_entry = {
                "timestamp": datetime.now().isoformat(),
                "type": "online_update",
                "update_number": self.online_updates_performed + 1,
                "feedback_count": len(recent_feedback),
                "training_time": training_time,
                "rl_results": rl_results,
                "dl_results": dl_results is not None
            }

            # Add to training log
            self.training_log.append(online_entry)

            # Save training log
            self._save_training_log()

            return online_entry

        except Exception as e:
            logger.error(f"Error performing online learning update: {e}")
            return None

    def train_model(self, epochs=10, batch_size=32, min_feedback_count=10,
                  use_early_stopping=True, use_transfer_learning=False,
                  train_rl=True, save_rl_model=True, evaluate_after_training=True):
        """
        Train the model using collected feedback with enhanced monitoring and evaluation.

        Args:
            epochs (int): Number of training epochs
            batch_size (int): Batch size for training
            min_feedback_count (int): Minimum number of feedback entries required for training
            use_early_stopping (bool): Whether to use early stopping
            use_transfer_learning (bool): Whether to use transfer learning techniques
            train_rl (bool): Whether to train the reinforcement learning model
            save_rl_model (bool): Whether to save the reinforcement learning model
            evaluate_after_training (bool): Whether to evaluate performance after training

        Returns:
            dict: Training results
        """
        # Check if we have enough feedback data
        if len(self.feedback_data) < min_feedback_count:
            logger.warning(f"Not enough feedback data for training. Need at least {min_feedback_count} entries.")
            return None

        # Start training timer
        start_time = time.time()

        # Initialize results
        results = {}

        # Get pre-training performance metrics for comparison
        if evaluate_after_training:
            pre_training_metrics = self.evaluate_performance(detailed=False)
            logger.info(f"Pre-training metrics: {pre_training_metrics}")

        # Train the deep learning model
        if hasattr(self.model, 'train'):
            logger.info("Training deep learning model...")
            training_history = self.model.train(
                epochs=epochs,
                batch_size=batch_size,
                use_early_stopping=use_early_stopping,
                use_transfer_learning=use_transfer_learning
            )
            results["deep_learning"] = training_history

            # Log training results
            if isinstance(training_history, dict):
                if "accuracy" in training_history:
                    final_accuracy = training_history["accuracy"][-1] if isinstance(training_history["accuracy"], list) else training_history["accuracy"]
                    logger.info(f"Training completed with final accuracy: {final_accuracy:.4f}")
                if "loss" in training_history:
                    final_loss = training_history["loss"][-1] if isinstance(training_history["loss"], list) else training_history["loss"]
                    logger.info(f"Training completed with final loss: {final_loss:.4f}")
        else:
            logger.warning("Model does not support training.")

        # Train the reinforcement learning model if enabled
        if self.use_reinforcement_learning and self.rl_manager and train_rl:
            try:
                logger.info(f"Training reinforcement learning model ({self.rl_algorithm})...")
                rl_results = self.rl_manager.train()
                results["reinforcement_learning"] = rl_results

                # Log RL training results
                if isinstance(rl_results, dict):
                    if "actor_loss" in rl_results and "critic_loss" in rl_results:
                        logger.info(f"RL training completed with actor loss: {rl_results['actor_loss']:.4f}, "
                                   f"critic loss: {rl_results['critic_loss']:.4f}")
                    if "mean_reward" in rl_results:
                        logger.info(f"RL training completed with mean reward: {rl_results['mean_reward']:.4f}")
                    if "entropy_coef" in rl_results:
                        logger.info(f"Current entropy coefficient: {rl_results['entropy_coef']:.6f}")

                # Save the RL model if requested
                if save_rl_model and (isinstance(rl_results, dict) and rl_results.get("success", False) or
                                     isinstance(rl_results, tuple) and rl_results[0] is not None):
                    model_filename = f"{self.rl_algorithm}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    self.rl_manager.save(model_filename)
                    logger.info(f"Saved reinforcement learning model: {model_filename}")
            except Exception as e:
                logger.error(f"Error training reinforcement learning model: {e}")
                results["reinforcement_learning"] = {"success": False, "error": str(e)}

        # Calculate training time
        training_time = time.time() - start_time

        # Create training log entry
        training_entry = {
            "timestamp": datetime.now().isoformat(),
            "type": "full",  # Mark as full training (not online update)
            "epochs": epochs,
            "batch_size": batch_size,
            "feedback_count": len(self.feedback_data),
            "training_time": training_time,
            "use_early_stopping": use_early_stopping,
            "use_transfer_learning": use_transfer_learning,
            "use_reinforcement_learning": self.use_reinforcement_learning,
            "rl_algorithm": self.rl_algorithm if self.use_reinforcement_learning else None,
            "results": results
        }

        # Add to training log
        self.training_log.append(training_entry)

        # Save training log
        self._save_training_log()

        # Evaluate performance after training if requested
        if evaluate_after_training:
            post_training_metrics = self.evaluate_performance(detailed=True)

            # Calculate improvements
            improvements = {}
            for key in post_training_metrics:
                if key in pre_training_metrics and isinstance(post_training_metrics[key], (int, float)) and key != "evaluation_time":
                    improvements[f"{key}_change"] = post_training_metrics[key] - pre_training_metrics[key]

            # Log performance improvements
            if improvements:
                logger.info(f"Training improvements: {improvements}")

            # Add evaluation to training entry
            training_entry["performance_evaluation"] = post_training_metrics
            training_entry["performance_improvements"] = improvements

            # Update training log
            self.training_log[-1] = training_entry
            self._save_training_log()

        return training_entry

    def get_feedback_stats(self):
        """
        Get statistics about the collected feedback.

        Returns:
            dict: Feedback statistics
        """
        if not self.feedback_data:
            return {
                "count": 0,
                "average_rating": 0,
                "rating_distribution": {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
            }

        # Calculate average rating
        ratings = [entry["rating"] for entry in self.feedback_data]
        avg_rating = sum(ratings) / len(ratings)

        # Calculate rating distribution
        rating_dist = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
        for rating in ratings:
            rating_dist[rating] += 1

        # Calculate category distribution if available
        category_dist = {}
        for entry in self.feedback_data:
            if "category" in entry and entry["category"]:
                cat = entry["category"]
                category_dist[cat] = category_dist.get(cat, 0) + 1

        return {
            "count": len(self.feedback_data),
            "average_rating": avg_rating,
            "rating_distribution": rating_dist,
            "category_distribution": category_dist
        }

    def get_training_stats(self):
        """
        Get statistics about the training sessions.

        Returns:
            dict: Training statistics
        """
        if not self.training_log:
            return {
                "count": 0,
                "last_training": None,
                "total_training_time": 0
            }

        # Calculate total training time
        total_time = sum(entry.get("training_time", 0) for entry in self.training_log)

        # Get last training timestamp
        last_training = self.training_log[-1]["timestamp"]

        # Count different types of training sessions
        full_training_count = sum(1 for entry in self.training_log if entry.get("type", "full") == "full")
        online_updates_count = sum(1 for entry in self.training_log if entry.get("type") == "online_update")

        # Calculate improvement metrics if we have enough training sessions
        improvement_metrics = self._calculate_improvement_metrics() if len(self.training_log) >= 2 else {}

        stats = {
            "count": len(self.training_log),
            "full_training_count": full_training_count,
            "online_updates_count": online_updates_count,
            "last_training": last_training,
            "total_training_time": total_time,
            "average_time_per_session": total_time / len(self.training_log)
        }

        # Add improvement metrics if available
        if improvement_metrics:
            stats.update(improvement_metrics)

        return stats

    def _calculate_improvement_metrics(self):
        """
        Calculate metrics to measure improvement over time.

        Returns:
            dict: Improvement metrics
        """
        # Initialize metrics
        metrics = {}

        # Filter full training sessions (not online updates)
        full_training_sessions = [entry for entry in self.training_log
                                 if entry.get("type", "full") == "full"]

        if len(full_training_sessions) >= 2:
            # Get first and last full training sessions
            first_session = full_training_sessions[0]
            last_session = full_training_sessions[-1]

            # Extract metrics if available
            if "results" in first_session and "results" in last_session:
                first_results = first_session["results"]
                last_results = last_session["results"]

                # Deep learning metrics
                if "deep_learning" in first_results and "deep_learning" in last_results:
                    first_dl = first_results["deep_learning"]
                    last_dl = last_results["deep_learning"]

                    # Calculate accuracy improvement if available
                    if isinstance(first_dl, dict) and isinstance(last_dl, dict):
                        if "accuracy" in first_dl and "accuracy" in last_dl:
                            first_acc = first_dl.get("accuracy", [0])[-1]
                            last_acc = last_dl.get("accuracy", [0])[-1]
                            metrics["accuracy_improvement"] = last_acc - first_acc
                            metrics["accuracy_improvement_percent"] = (
                                (last_acc - first_acc) / max(0.001, first_acc) * 100
                            )

                        # Calculate loss improvement
                        if "loss" in first_dl and "loss" in last_dl:
                            first_loss = first_dl.get("loss", [1])[-1]
                            last_loss = last_dl.get("loss", [1])[-1]
                            metrics["loss_improvement"] = first_loss - last_loss
                            metrics["loss_improvement_percent"] = (
                                (first_loss - last_loss) / max(0.001, first_loss) * 100
                            )

                # Reinforcement learning metrics
                if "reinforcement_learning" in first_results and "reinforcement_learning" in last_results:
                    first_rl = first_results["reinforcement_learning"]
                    last_rl = last_results["reinforcement_learning"]

                    if isinstance(first_rl, dict) and isinstance(last_rl, dict):
                        # Calculate reward improvement
                        if "mean_reward" in first_rl and "mean_reward" in last_rl:
                            first_reward = first_rl.get("mean_reward", 0)
                            last_reward = last_rl.get("mean_reward", 0)
                            metrics["reward_improvement"] = last_reward - first_reward
                            metrics["reward_improvement_percent"] = (
                                (last_reward - first_reward) / max(0.001, abs(first_reward)) * 100
                            ) if first_reward != 0 else 0

        # Calculate feedback quality improvement
        early_feedback = self.feedback_data[:min(10, len(self.feedback_data))]
        recent_feedback = self.feedback_data[-min(10, len(self.feedback_data)):]

        if early_feedback and recent_feedback:
            early_ratings = [entry.get("rating", 0) for entry in early_feedback]
            recent_ratings = [entry.get("rating", 0) for entry in recent_feedback]

            early_avg = sum(early_ratings) / len(early_ratings)
            recent_avg = sum(recent_ratings) / len(recent_ratings)

            metrics["rating_improvement"] = recent_avg - early_avg
            metrics["rating_improvement_percent"] = (
                (recent_avg - early_avg) / max(0.001, early_avg) * 100
            )

        # Calculate online learning effectiveness if we have online updates
        online_updates = [entry for entry in self.training_log
                         if entry.get("type") == "online_update"]

        if online_updates:
            # Calculate average reward before and after online updates
            pre_online_rewards = []
            post_online_rewards = []

            for update in online_updates:
                update_time = datetime.fromisoformat(update["timestamp"])

                # Get feedback entries before and after this update
                pre_update_feedback = [
                    entry for entry in self.feedback_data
                    if "timestamp" in entry and
                    datetime.fromisoformat(entry["timestamp"]) < update_time and
                    "rl_reward" in entry
                ][-5:]  # Last 5 before update

                post_update_feedback = [
                    entry for entry in self.feedback_data
                    if "timestamp" in entry and
                    datetime.fromisoformat(entry["timestamp"]) > update_time and
                    "rl_reward" in entry
                ][:5]  # First 5 after update

                if pre_update_feedback:
                    pre_online_rewards.append(
                        sum(entry["rl_reward"] for entry in pre_update_feedback) / len(pre_update_feedback)
                    )

                if post_update_feedback:
                    post_online_rewards.append(
                        sum(entry["rl_reward"] for entry in post_update_feedback) / len(post_update_feedback)
                    )

            if pre_online_rewards and post_online_rewards:
                avg_pre_reward = sum(pre_online_rewards) / len(pre_online_rewards)
                avg_post_reward = sum(post_online_rewards) / len(post_online_rewards)

                metrics["online_learning_reward_improvement"] = avg_post_reward - avg_pre_reward
                metrics["online_learning_effectiveness"] = (
                    (avg_post_reward - avg_pre_reward) / max(0.001, abs(avg_pre_reward)) * 100
                ) if avg_pre_reward != 0 else 0

        return metrics

    def evaluate_performance(self, detailed=False):
        """
        Evaluate the model's performance and improvement over time.

        Args:
            detailed (bool): Whether to include detailed metrics

        Returns:
            dict: Performance evaluation metrics
        """
        # Get basic training and feedback stats
        training_stats = self.get_training_stats()
        feedback_stats = self.get_feedback_stats()

        # Calculate performance metrics
        performance = {
            "training_count": training_stats.get("count", 0),
            "online_updates_count": training_stats.get("online_updates_count", 0),
            "feedback_count": feedback_stats.get("count", 0),
            "average_rating": feedback_stats.get("average_rating", 0),
        }

        # Add improvement metrics if available
        for key, value in training_stats.items():
            if "improvement" in key:
                performance[key] = value

        # Calculate learning efficiency
        if performance["training_count"] > 0 and performance["feedback_count"] > 0:
            performance["learning_efficiency"] = performance["average_rating"] / max(1, performance["training_count"])

            # Add rating improvement per training session if available
            if "rating_improvement" in performance:
                performance["improvement_per_session"] = (
                    performance["rating_improvement"] / max(1, performance["training_count"])
                )

        # Add detailed metrics if requested
        if detailed and self.use_reinforcement_learning and self.rl_manager:
            # Get RL agent metrics if available
            if hasattr(self.rl_manager, 'agent'):
                agent = self.rl_manager.agent

                if hasattr(agent, 'mean_reward_history') and agent.mean_reward_history:
                    performance["rl_mean_reward"] = agent.mean_reward_history[-1]
                    performance["rl_reward_trend"] = (
                        "improving" if len(agent.mean_reward_history) > 1 and
                        agent.mean_reward_history[-1] > agent.mean_reward_history[0] else "stable"
                    )

                if hasattr(agent, 'mean_entropy_history') and agent.mean_entropy_history:
                    performance["rl_mean_entropy"] = agent.mean_entropy_history[-1]

                if hasattr(agent, 'exploration_strategy'):
                    performance["exploration_strategy"] = agent.exploration_strategy

        # Add timestamp
        performance["evaluation_time"] = datetime.now().isoformat()

        return performance
