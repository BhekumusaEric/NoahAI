"""
Continual Learning Manager for NoahAI

This module provides functionality for continual learning, allowing the model
to learn new knowledge without forgetting previously learned information.
"""

import os
import json
import logging
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional, Union
from collections import deque

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/continual_learning.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("continual_learning")

# Try to import TensorFlow
try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    logger.warning("TensorFlow not available. Continual learning will be limited.")

class ContinualLearningManager:
    """
    Continual Learning Manager for NoahAI.
    
    This class provides functionality for continual learning, allowing the model
    to learn new knowledge without forgetting previously learned information.
    """
    
    def __init__(self, model=None, model_dir="data/continual_learning", config=None):
        """
        Initialize the Continual Learning Manager.
        
        Args:
            model: The model to train
            model_dir (str): Directory to save/load models and data
            config (dict, optional): Configuration for continual learning
        """
        self.model = model
        self.model_dir = model_dir
        self.config = config or {}
        
        # Check if TensorFlow is available
        if not TENSORFLOW_AVAILABLE:
            logger.error("TensorFlow is required for continual learning.")
            raise ImportError("TensorFlow is required for continual learning.")
        
        # Create model directory if it doesn't exist
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(os.path.join(self.model_dir, "snapshots"), exist_ok=True)
        os.makedirs(os.path.join(self.model_dir, "replay_buffer"), exist_ok=True)
        
        # Initialize continual learning parameters
        self.use_replay_memory = self.config.get("use_replay_memory", True)
        self.use_elastic_weight_consolidation = self.config.get("use_elastic_weight_consolidation", True)
        self.use_knowledge_distillation = self.config.get("use_knowledge_distillation", True)
        self.use_model_snapshots = self.config.get("use_model_snapshots", True)
        
        # Initialize replay memory
        self.replay_buffer_size = self.config.get("replay_buffer_size", 1000)
        self.replay_buffer = deque(maxlen=self.replay_buffer_size)
        
        # Initialize EWC parameters
        self.ewc_lambda = self.config.get("ewc_lambda", 0.1)  # Importance of old tasks
        self.fisher_information = {}  # Fisher information matrix for EWC
        
        # Initialize knowledge distillation parameters
        self.distillation_temp = self.config.get("distillation_temp", 2.0)
        self.distillation_alpha = self.config.get("distillation_alpha", 0.5)
        
        # Initialize model snapshots
        self.snapshot_frequency = self.config.get("snapshot_frequency", 10)  # Save snapshot every N updates
        self.max_snapshots = self.config.get("max_snapshots", 5)
        self.snapshots = []
        
        # Initialize training metrics
        self.train_step = 0
        self.task_history = []
        self.forgetting_metrics = []
        
        logger.info("Continual Learning Manager initialized.")
    
    def set_model(self, model):
        """
        Set the model for continual learning.
        
        Args:
            model: The model to train
            
        Returns:
            bool: Whether the model was set successfully
        """
        self.model = model
        logger.info("Model set for continual learning.")
        return True
    
    def add_to_replay_buffer(self, data):
        """
        Add data to the replay buffer.
        
        Args:
            data: Data to add to the replay buffer
            
        Returns:
            bool: Whether the data was added successfully
        """
        try:
            # Add data to replay buffer
            self.replay_buffer.append(data)
            
            # Save replay buffer if it's getting large
            if len(self.replay_buffer) % 100 == 0:
                self._save_replay_buffer()
            
            return True
        except Exception as e:
            logger.error(f"Error adding data to replay buffer: {e}")
            return False
    
    def _save_replay_buffer(self):
        """
        Save the replay buffer to disk.
        
        Returns:
            bool: Whether the replay buffer was saved successfully
        """
        try:
            # Create a filename based on the current time
            filename = f"replay_buffer_{datetime.now().strftime('%Y%m%d_%H%M%S')}.npz"
            filepath = os.path.join(self.model_dir, "replay_buffer", filename)
            
            # Convert replay buffer to numpy arrays
            buffer_data = list(self.replay_buffer)
            
            # Extract inputs and targets
            inputs = np.array([data[0] for data in buffer_data])
            targets = np.array([data[1] for data in buffer_data])
            
            # Save to disk
            np.savez_compressed(filepath, inputs=inputs, targets=targets)
            
            logger.info(f"Saved replay buffer to {filepath}")
            return True
        except Exception as e:
            logger.error(f"Error saving replay buffer: {e}")
            return False
    
    def _load_replay_buffer(self, filepath):
        """
        Load the replay buffer from disk.
        
        Args:
            filepath (str): Path to the replay buffer file
            
        Returns:
            bool: Whether the replay buffer was loaded successfully
        """
        try:
            # Load from disk
            data = np.load(filepath)
            
            # Extract inputs and targets
            inputs = data["inputs"]
            targets = data["targets"]
            
            # Clear current replay buffer
            self.replay_buffer.clear()
            
            # Add data to replay buffer
            for i in range(len(inputs)):
                self.replay_buffer.append((inputs[i], targets[i]))
            
            logger.info(f"Loaded replay buffer from {filepath} with {len(self.replay_buffer)} items")
            return True
        except Exception as e:
            logger.error(f"Error loading replay buffer: {e}")
            return False
    
    def create_model_snapshot(self):
        """
        Create a snapshot of the current model.
        
        Returns:
            bool: Whether the snapshot was created successfully
        """
        if not self.use_model_snapshots or self.model is None:
            return False
        
        try:
            # Create a filename based on the current time
            snapshot_name = f"snapshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            snapshot_dir = os.path.join(self.model_dir, "snapshots", snapshot_name)
            os.makedirs(snapshot_dir, exist_ok=True)
            
            # Save model weights
            weights_path = os.path.join(snapshot_dir, "weights.h5")
            self.model.save_weights(weights_path)
            
            # Save snapshot metadata
            metadata = {
                "snapshot_name": snapshot_name,
                "train_step": self.train_step,
                "created_at": datetime.now().isoformat()
            }
            
            metadata_path = os.path.join(snapshot_dir, "metadata.json")
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            # Add to snapshots list
            self.snapshots.append(snapshot_name)
            
            # Keep only the most recent snapshots
            if len(self.snapshots) > self.max_snapshots:
                oldest_snapshot = self.snapshots.pop(0)
                oldest_snapshot_dir = os.path.join(self.model_dir, "snapshots", oldest_snapshot)
                
                # Delete oldest snapshot (in a production environment, you might want to archive instead)
                import shutil
                if os.path.exists(oldest_snapshot_dir):
                    shutil.rmtree(oldest_snapshot_dir)
            
            logger.info(f"Created model snapshot: {snapshot_name}")
            return True
        except Exception as e:
            logger.error(f"Error creating model snapshot: {e}")
            return False
    
    def load_model_snapshot(self, snapshot_name):
        """
        Load a model snapshot.
        
        Args:
            snapshot_name (str): Name of the snapshot to load
            
        Returns:
            bool: Whether the snapshot was loaded successfully
        """
        if not self.use_model_snapshots or self.model is None:
            return False
        
        try:
            # Get snapshot directory
            snapshot_dir = os.path.join(self.model_dir, "snapshots", snapshot_name)
            
            if not os.path.exists(snapshot_dir):
                logger.error(f"Snapshot directory not found: {snapshot_dir}")
                return False
            
            # Load model weights
            weights_path = os.path.join(snapshot_dir, "weights.h5")
            self.model.load_weights(weights_path)
            
            # Load snapshot metadata
            metadata_path = os.path.join(snapshot_dir, "metadata.json")
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                
                self.train_step = metadata.get("train_step", 0)
            
            logger.info(f"Loaded model snapshot: {snapshot_name}")
            return True
        except Exception as e:
            logger.error(f"Error loading model snapshot: {e}")
            return False
    
    def _calculate_fisher_information(self, dataset):
        """
        Calculate Fisher information matrix for EWC.
        
        Args:
            dataset: Dataset to calculate Fisher information
            
        Returns:
            dict: Fisher information matrix
        """
        if not TENSORFLOW_AVAILABLE or self.model is None:
            return {}
        
        try:
            # Initialize Fisher information matrix
            fisher_information = {}
            for var in self.model.trainable_variables:
                fisher_information[var.name] = tf.zeros_like(var)
            
            # Calculate Fisher information
            for batch in dataset:
                inputs, _ = batch
                
                with tf.GradientTape() as tape:
                    # Forward pass
                    outputs = self.model(inputs, training=True)
                    
                    # Calculate log likelihood
                    log_likelihood = tf.reduce_mean(tf.math.log(outputs + 1e-10))
                
                # Calculate gradients
                gradients = tape.gradient(log_likelihood, self.model.trainable_variables)
                
                # Update Fisher information
                for i, var in enumerate(self.model.trainable_variables):
                    if gradients[i] is not None:
                        fisher_information[var.name] += tf.square(gradients[i])
            
            # Normalize Fisher information
            for var_name in fisher_information:
                fisher_information[var_name] /= len(dataset)
            
            return fisher_information
        except Exception as e:
            logger.error(f"Error calculating Fisher information: {e}")
            return {}
    
    def _ewc_loss(self, current_weights):
        """
        Calculate EWC loss to prevent catastrophic forgetting.
        
        Args:
            current_weights: Current model weights
            
        Returns:
            tf.Tensor: EWC loss
        """
        if not self.fisher_information:
            return tf.constant(0.0)
        
        try:
            # Initialize EWC loss
            ewc_loss = tf.constant(0.0)
            
            # Calculate EWC loss
            for i, var in enumerate(self.model.trainable_variables):
                if var.name in self.fisher_information and var.name in current_weights:
                    # Calculate squared difference between current and old weights
                    weight_diff = tf.square(var - current_weights[var.name])
                    
                    # Weight by Fisher information
                    ewc_loss += tf.reduce_sum(self.fisher_information[var.name] * weight_diff)
            
            # Apply EWC lambda
            ewc_loss *= self.ewc_lambda / 2.0
            
            return ewc_loss
        except Exception as e:
            logger.error(f"Error calculating EWC loss: {e}")
            return tf.constant(0.0)
    
    def _knowledge_distillation_loss(self, teacher_model, student_model, inputs, temperature=2.0):
        """
        Calculate knowledge distillation loss.
        
        Args:
            teacher_model: Teacher model (old model)
            student_model: Student model (new model)
            inputs: Input data
            temperature (float): Temperature for softening the distributions
            
        Returns:
            tf.Tensor: Knowledge distillation loss
        """
        try:
            # Get teacher predictions
            teacher_logits = teacher_model(inputs, training=False)
            
            # Get student predictions
            student_logits = student_model(inputs, training=True)
            
            # Apply temperature scaling
            teacher_probs = tf.nn.softmax(teacher_logits / temperature)
            student_probs = tf.nn.softmax(student_logits / temperature)
            
            # Calculate KL divergence
            kl_divergence = tf.reduce_mean(
                tf.reduce_sum(
                    teacher_probs * tf.math.log(teacher_probs / (student_probs + 1e-10) + 1e-10),
                    axis=1
                )
            )
            
            # Scale by temperature squared
            distillation_loss = kl_divergence * (temperature ** 2)
            
            return distillation_loss
        except Exception as e:
            logger.error(f"Error calculating knowledge distillation loss: {e}")
            return tf.constant(0.0)
    
    def train_with_continual_learning(self, dataset, new_task_name=None, epochs=1, batch_size=32):
        """
        Train the model with continual learning techniques.
        
        Args:
            dataset: Dataset to train on
            new_task_name (str, optional): Name of the new task
            epochs (int): Number of training epochs
            batch_size (int): Batch size for training
            
        Returns:
            dict: Training results
        """
        if not TENSORFLOW_AVAILABLE or self.model is None:
            logger.error("TensorFlow or model not available for continual learning.")
            return {"success": False, "message": "TensorFlow or model not available for continual learning."}
        
        try:
            # Start training timer
            start_time = datetime.now()
            
            # Create a snapshot of the current model before training
            if self.use_model_snapshots and self.train_step % self.snapshot_frequency == 0:
                self.create_model_snapshot()
            
            # Save current weights for EWC
            if self.use_elastic_weight_consolidation:
                current_weights = {}
                for var in self.model.trainable_variables:
                    current_weights[var.name] = tf.identity(var)
            
            # Create a copy of the model for knowledge distillation
            if self.use_knowledge_distillation:
                teacher_model = tf.keras.models.clone_model(self.model)
                teacher_model.set_weights(self.model.get_weights())
            
            # Prepare replay buffer data if available
            replay_data = None
            if self.use_replay_memory and self.replay_buffer:
                # Convert replay buffer to TensorFlow dataset
                replay_inputs = np.array([data[0] for data in self.replay_buffer])
                replay_targets = np.array([data[1] for data in self.replay_buffer])
                
                # Create TensorFlow dataset
                replay_data = tf.data.Dataset.from_tensor_slices((replay_inputs, replay_targets))
                replay_data = replay_data.batch(batch_size)
            
            # Initialize training metrics
            training_metrics = {
                "loss": [],
                "accuracy": [],
                "ewc_loss": [],
                "distillation_loss": [],
                "replay_loss": []
            }
            
            # Train for the specified number of epochs
            for epoch in range(epochs):
                epoch_metrics = {
                    "loss": [],
                    "accuracy": [],
                    "ewc_loss": [],
                    "distillation_loss": [],
                    "replay_loss": []
                }
                
                # Train on new data
                for batch in dataset:
                    inputs, targets = batch
                    
                    with tf.GradientTape() as tape:
                        # Forward pass
                        predictions = self.model(inputs, training=True)
                        
                        # Calculate task loss
                        task_loss = tf.keras.losses.sparse_categorical_crossentropy(
                            targets, predictions
                        )
                        task_loss = tf.reduce_mean(task_loss)
                        
                        # Initialize total loss
                        total_loss = task_loss
                        
                        # Add EWC loss if enabled
                        ewc_loss = tf.constant(0.0)
                        if self.use_elastic_weight_consolidation and self.fisher_information:
                            ewc_loss = self._ewc_loss(current_weights)
                            total_loss += ewc_loss
                        
                        # Add knowledge distillation loss if enabled
                        distillation_loss = tf.constant(0.0)
                        if self.use_knowledge_distillation:
                            distillation_loss = self._knowledge_distillation_loss(
                                teacher_model, self.model, inputs, self.distillation_temp
                            )
                            total_loss += self.distillation_alpha * distillation_loss
                    
                    # Calculate gradients
                    gradients = tape.gradient(total_loss, self.model.trainable_variables)
                    
                    # Apply gradients
                    self.model.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
                    
                    # Calculate accuracy
                    accuracy = tf.keras.metrics.sparse_categorical_accuracy(targets, predictions)
                    accuracy = tf.reduce_mean(accuracy)
                    
                    # Update epoch metrics
                    epoch_metrics["loss"].append(float(task_loss))
                    epoch_metrics["accuracy"].append(float(accuracy))
                    epoch_metrics["ewc_loss"].append(float(ewc_loss))
                    epoch_metrics["distillation_loss"].append(float(distillation_loss))
                
                # Train on replay buffer if available
                replay_loss = 0.0
                if self.use_replay_memory and replay_data is not None:
                    replay_losses = []
                    
                    for replay_batch in replay_data:
                        replay_inputs, replay_targets = replay_batch
                        
                        with tf.GradientTape() as tape:
                            # Forward pass
                            replay_predictions = self.model(replay_inputs, training=True)
                            
                            # Calculate replay loss
                            batch_replay_loss = tf.keras.losses.sparse_categorical_crossentropy(
                                replay_targets, replay_predictions
                            )
                            batch_replay_loss = tf.reduce_mean(batch_replay_loss)
                        
                        # Calculate gradients
                        gradients = tape.gradient(batch_replay_loss, self.model.trainable_variables)
                        
                        # Apply gradients
                        self.model.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
                        
                        replay_losses.append(float(batch_replay_loss))
                    
                    # Calculate average replay loss
                    replay_loss = np.mean(replay_losses) if replay_losses else 0.0
                    epoch_metrics["replay_loss"].append(replay_loss)
                
                # Calculate epoch average metrics
                for key in epoch_metrics:
                    if epoch_metrics[key]:
                        training_metrics[key].append(np.mean(epoch_metrics[key]))
                
                # Log epoch progress
                logger.info(f"Epoch {epoch + 1}/{epochs}: "
                           f"loss={training_metrics['loss'][-1]:.4f}, "
                           f"accuracy={training_metrics['accuracy'][-1]:.4f}, "
                           f"ewc_loss={training_metrics['ewc_loss'][-1]:.4f}, "
                           f"distillation_loss={training_metrics['distillation_loss'][-1]:.4f}, "
                           f"replay_loss={replay_loss:.4f}")
            
            # Calculate Fisher information for the new task
            if self.use_elastic_weight_consolidation:
                self.fisher_information = self._calculate_fisher_information(dataset)
            
            # Add data to replay buffer
            if self.use_replay_memory:
                for batch in dataset:
                    inputs, targets = batch
                    
                    # Add a subset of the batch to the replay buffer
                    for i in range(min(len(inputs), 10)):  # Limit to 10 samples per batch
                        self.add_to_replay_buffer((inputs[i].numpy(), targets[i].numpy()))
            
            # Update train step
            self.train_step += 1
            
            # Create a snapshot of the model after training
            if self.use_model_snapshots:
                self.create_model_snapshot()
            
            # Add task to history
            if new_task_name:
                self.task_history.append({
                    "task_name": new_task_name,
                    "train_step": self.train_step,
                    "trained_at": datetime.now().isoformat()
                })
            
            # Calculate training time
            training_time = (datetime.now() - start_time).total_seconds()
            
            # Return training results
            return {
                "success": True,
                "train_step": self.train_step,
                "epochs": epochs,
                "training_time": training_time,
                "final_loss": training_metrics["loss"][-1] if training_metrics["loss"] else None,
                "final_accuracy": training_metrics["accuracy"][-1] if training_metrics["accuracy"] else None,
                "ewc_loss": training_metrics["ewc_loss"][-1] if training_metrics["ewc_loss"] else None,
                "distillation_loss": training_metrics["distillation_loss"][-1] if training_metrics["distillation_loss"] else None,
                "replay_loss": training_metrics["replay_loss"][-1] if training_metrics["replay_loss"] else None,
                "replay_buffer_size": len(self.replay_buffer),
                "snapshots_count": len(self.snapshots)
            }
        except Exception as e:
            logger.error(f"Error in continual learning training: {e}")
            return {"success": False, "message": f"Error in continual learning training: {e}"}
    
    def evaluate_forgetting(self, old_task_dataset, old_task_name):
        """
        Evaluate how much the model has forgotten about an old task.
        
        Args:
            old_task_dataset: Dataset for the old task
            old_task_name (str): Name of the old task
            
        Returns:
            dict: Forgetting metrics
        """
        if not TENSORFLOW_AVAILABLE or self.model is None:
            logger.error("TensorFlow or model not available for evaluation.")
            return {"success": False, "message": "TensorFlow or model not available for evaluation."}
        
        try:
            # Evaluate on old task
            loss = []
            accuracy = []
            
            for batch in old_task_dataset:
                inputs, targets = batch
                
                # Forward pass
                predictions = self.model(inputs, training=False)
                
                # Calculate loss
                batch_loss = tf.keras.losses.sparse_categorical_crossentropy(
                    targets, predictions
                )
                batch_loss = tf.reduce_mean(batch_loss)
                
                # Calculate accuracy
                batch_accuracy = tf.keras.metrics.sparse_categorical_accuracy(targets, predictions)
                batch_accuracy = tf.reduce_mean(batch_accuracy)
                
                loss.append(float(batch_loss))
                accuracy.append(float(batch_accuracy))
            
            # Calculate average metrics
            avg_loss = np.mean(loss) if loss else 0.0
            avg_accuracy = np.mean(accuracy) if accuracy else 0.0
            
            # Find original performance on this task if available
            original_accuracy = None
            for task in self.task_history:
                if task["task_name"] == old_task_name:
                    original_accuracy = task.get("accuracy")
                    break
            
            # Calculate forgetting
            forgetting = None
            if original_accuracy is not None:
                forgetting = original_accuracy - avg_accuracy
            
            # Add to forgetting metrics
            forgetting_metric = {
                "task_name": old_task_name,
                "evaluated_at": datetime.now().isoformat(),
                "loss": avg_loss,
                "accuracy": avg_accuracy,
                "original_accuracy": original_accuracy,
                "forgetting": forgetting
            }
            
            self.forgetting_metrics.append(forgetting_metric)
            
            logger.info(f"Evaluated forgetting for task '{old_task_name}': "
                       f"accuracy={avg_accuracy:.4f}, "
                       f"forgetting={forgetting:.4f} if forgetting is not None else 'N/A'")
            
            return forgetting_metric
        except Exception as e:
            logger.error(f"Error evaluating forgetting: {e}")
            return {"success": False, "message": f"Error evaluating forgetting: {e}"}
    
    def get_status(self):
        """
        Get the status of the continual learning manager.
        
        Returns:
            dict: Status information
        """
        return {
            "train_step": self.train_step,
            "replay_buffer_size": len(self.replay_buffer),
            "snapshots_count": len(self.snapshots),
            "snapshots": self.snapshots,
            "task_history": self.task_history,
            "use_replay_memory": self.use_replay_memory,
            "use_elastic_weight_consolidation": self.use_elastic_weight_consolidation,
            "use_knowledge_distillation": self.use_knowledge_distillation,
            "use_model_snapshots": self.use_model_snapshots
        }
