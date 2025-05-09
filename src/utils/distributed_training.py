"""
Distributed Training Manager for NoahAI

This module provides functionality for distributed training across multiple machines,
enabling faster learning and more efficient use of computational resources.
"""

import os
import json
import time
import socket
import logging
import threading
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/distributed_training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("distributed_training")

# Try to import TensorFlow
try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    logger.warning("TensorFlow not available. Distributed training will be limited.")

# Try to import multiprocessing
try:
    import multiprocessing as mp
    MULTIPROCESSING_AVAILABLE = True
except ImportError:
    MULTIPROCESSING_AVAILABLE = False
    logger.warning("Multiprocessing not available. Distributed training will be limited.")

class DistributedTrainingManager:
    """
    Distributed Training Manager for NoahAI.
    
    This class provides functionality for distributed training across multiple machines,
    enabling faster learning and more efficient use of computational resources.
    """
    
    def __init__(self, model_dir="data/distributed", config=None):
        """
        Initialize the Distributed Training Manager.
        
        Args:
            model_dir (str): Directory to save/load models and checkpoints
            config (dict, optional): Configuration for distributed training
        """
        self.model_dir = model_dir
        self.config = config or {}
        
        # Check if TensorFlow is available
        if not TENSORFLOW_AVAILABLE:
            logger.error("TensorFlow is required for distributed training.")
            raise ImportError("TensorFlow is required for distributed training.")
        
        # Create model directory if it doesn't exist
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(os.path.join(self.model_dir, "checkpoints"), exist_ok=True)
        
        # Initialize distributed training parameters
        self.role = self.config.get("role", "worker")  # "worker" or "parameter_server"
        self.cluster_config = self.config.get("cluster_config", {
            "workers": ["localhost:2222"],
            "parameter_servers": ["localhost:2223"]
        })
        self.worker_index = self.config.get("worker_index", 0)
        self.ps_index = self.config.get("ps_index", 0)
        
        # Initialize synchronization parameters
        self.sync_frequency = self.config.get("sync_frequency", 10)  # Sync every N steps
        self.sync_timeout = self.config.get("sync_timeout", 60)  # Timeout in seconds
        self.use_async_updates = self.config.get("use_async_updates", False)
        
        # Initialize training parameters
        self.batch_size = self.config.get("batch_size", 32)
        self.learning_rate = self.config.get("learning_rate", 0.001)
        self.epochs = self.config.get("epochs", 10)
        
        # Initialize distributed TensorFlow if available
        self.tf_server = None
        self.cluster = None
        self.server_thread = None
        self.is_running = False
        
        # Initialize model and optimizer
        self.model = None
        self.optimizer = None
        
        # Initialize metrics
        self.train_step = 0
        self.global_step = tf.Variable(0, trainable=False, dtype=tf.int64)
        self.training_metrics = {
            "loss_history": [],
            "accuracy_history": [],
            "training_time": 0,
            "sync_time": 0,
            "worker_metrics": {}
        }
        
        # Initialize locks and events
        self.sync_lock = threading.Lock()
        self.stop_event = threading.Event()
        
        logger.info(f"Distributed Training Manager initialized with role: {self.role}")
    
    def initialize_cluster(self):
        """
        Initialize the TensorFlow distributed cluster.
        
        Returns:
            bool: Whether the cluster was initialized successfully
        """
        try:
            # Create TensorFlow cluster
            self.cluster = tf.train.ClusterSpec(self.cluster_config)
            
            # Create TensorFlow server
            if self.role == "worker":
                job_name = "worker"
                task_index = self.worker_index
            else:
                job_name = "parameter_servers"
                task_index = self.ps_index
            
            self.tf_server = tf.distribute.Server(
                self.cluster,
                job_name=job_name,
                task_index=task_index
            )
            
            logger.info(f"Initialized TensorFlow server with role: {self.role}, task index: {task_index}")
            return True
        except Exception as e:
            logger.error(f"Error initializing TensorFlow cluster: {e}")
            return False
    
    def start_server(self):
        """
        Start the TensorFlow server in a separate thread.
        
        Returns:
            bool: Whether the server was started successfully
        """
        if self.tf_server is None:
            success = self.initialize_cluster()
            if not success:
                return False
        
        # Start server in a separate thread
        def server_thread_func():
            logger.info(f"Starting TensorFlow server with role: {self.role}")
            self.is_running = True
            self.tf_server.join()
            self.is_running = False
            logger.info(f"TensorFlow server stopped with role: {self.role}")
        
        self.server_thread = threading.Thread(target=server_thread_func)
        self.server_thread.daemon = True
        self.server_thread.start()
        
        # Wait for server to start
        time.sleep(2)
        
        if self.is_running:
            logger.info(f"TensorFlow server started with role: {self.role}")
            return True
        else:
            logger.error(f"Error starting TensorFlow server with role: {self.role}")
            return False
    
    def stop_server(self):
        """
        Stop the TensorFlow server.
        
        Returns:
            bool: Whether the server was stopped successfully
        """
        if not self.is_running:
            logger.warning("TensorFlow server is not running.")
            return True
        
        # Signal server to stop
        self.stop_event.set()
        
        # Wait for server to stop
        if self.server_thread is not None:
            self.server_thread.join(timeout=5)
        
        self.is_running = False
        logger.info(f"TensorFlow server stopped with role: {self.role}")
        return True
    
    def initialize_model(self, model):
        """
        Initialize the model for distributed training.
        
        Args:
            model: The model to train
            
        Returns:
            bool: Whether the model was initialized successfully
        """
        try:
            self.model = model
            
            # Create optimizer with distributed strategy
            if self.role == "worker":
                with tf.device(f"/job:parameter_servers/task:0"):
                    self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
            
            logger.info(f"Initialized model for distributed training with role: {self.role}")
            return True
        except Exception as e:
            logger.error(f"Error initializing model for distributed training: {e}")
            return False
    
    def _get_device_assignment(self):
        """
        Get the device assignment for the current role.
        
        Returns:
            str: Device assignment
        """
        if self.role == "worker":
            return f"/job:worker/task:{self.worker_index}"
        else:
            return f"/job:parameter_servers/task:{self.ps_index}"
    
    def _sync_parameters(self):
        """
        Synchronize model parameters across workers.
        
        Returns:
            bool: Whether the parameters were synchronized successfully
        """
        if self.role != "worker":
            logger.warning("Parameter synchronization is only performed by workers.")
            return False
        
        try:
            with self.sync_lock:
                sync_start_time = time.time()
                
                # Get global variables
                with tf.device("/job:parameter_servers/task:0"):
                    global_vars = tf.compat.v1.global_variables()
                
                # Sync operation
                sync_op = tf.compat.v1.train.SyncReplicasOptimizer(
                    self.optimizer,
                    replicas_to_aggregate=len(self.cluster_config["workers"]),
                    total_num_replicas=len(self.cluster_config["workers"])
                )
                
                # Execute sync operation
                with tf.compat.v1.Session(self.tf_server.target) as sess:
                    sess.run(sync_op)
                
                sync_time = time.time() - sync_start_time
                self.training_metrics["sync_time"] += sync_time
                
                logger.info(f"Synchronized parameters in {sync_time:.2f} seconds")
                return True
        except Exception as e:
            logger.error(f"Error synchronizing parameters: {e}")
            return False
    
    def train_step_distributed(self, inputs, targets):
        """
        Perform a single distributed training step.
        
        Args:
            inputs: Input data
            targets: Target data
            
        Returns:
            dict: Training metrics
        """
        if self.role != "worker":
            logger.warning("Training is only performed by workers.")
            return None
        
        if self.model is None:
            logger.error("Model not initialized for distributed training.")
            return None
        
        try:
            # Get device assignment
            device = self._get_device_assignment()
            
            # Perform training step
            with tf.device(device):
                with tf.GradientTape() as tape:
                    # Forward pass
                    predictions = self.model(inputs, training=True)
                    
                    # Calculate loss
                    loss = tf.keras.losses.sparse_categorical_crossentropy(
                        targets, predictions
                    )
                    loss = tf.reduce_mean(loss)
                
                # Calculate gradients
                gradients = tape.gradient(loss, self.model.trainable_variables)
                
                # Apply gradients
                with tf.device("/job:parameter_servers/task:0"):
                    self.optimizer.apply_gradients(
                        zip(gradients, self.model.trainable_variables)
                    )
                    self.global_step.assign_add(1)
            
            # Calculate accuracy
            accuracy = tf.keras.metrics.sparse_categorical_accuracy(targets, predictions)
            accuracy = tf.reduce_mean(accuracy)
            
            # Update metrics
            self.train_step += 1
            self.training_metrics["loss_history"].append(float(loss))
            self.training_metrics["accuracy_history"].append(float(accuracy))
            
            # Synchronize parameters if needed
            if self.train_step % self.sync_frequency == 0 and not self.use_async_updates:
                self._sync_parameters()
            
            # Log progress
            if self.train_step % 10 == 0:
                logger.info(f"Worker {self.worker_index} - Step {self.train_step}: "
                           f"loss={float(loss):.4f}, accuracy={float(accuracy):.4f}")
            
            return {
                "loss": float(loss),
                "accuracy": float(accuracy),
                "step": self.train_step,
                "global_step": int(self.global_step)
            }
        except Exception as e:
            logger.error(f"Error in distributed training step: {e}")
            return None
    
    def train_distributed(self, dataset, epochs=None, steps_per_epoch=None):
        """
        Train the model in a distributed manner.
        
        Args:
            dataset: Training dataset
            epochs (int, optional): Number of epochs to train
            steps_per_epoch (int, optional): Number of steps per epoch
            
        Returns:
            dict: Training results
        """
        if self.role != "worker":
            logger.warning("Training is only performed by workers.")
            return None
        
        if self.model is None:
            logger.error("Model not initialized for distributed training.")
            return None
        
        # Use configured values if not provided
        epochs = epochs or self.epochs
        
        try:
            # Start training timer
            start_time = time.time()
            
            # Initialize training metrics
            self.training_metrics["loss_history"] = []
            self.training_metrics["accuracy_history"] = []
            self.training_metrics["sync_time"] = 0
            
            # Train for the specified number of epochs
            for epoch in range(epochs):
                epoch_loss = []
                epoch_accuracy = []
                
                # Iterate over the dataset
                for step, (inputs, targets) in enumerate(dataset):
                    # Perform training step
                    step_metrics = self.train_step_distributed(inputs, targets)
                    
                    if step_metrics is not None:
                        epoch_loss.append(step_metrics["loss"])
                        epoch_accuracy.append(step_metrics["accuracy"])
                    
                    # Stop if we've reached the specified number of steps
                    if steps_per_epoch is not None and step >= steps_per_epoch - 1:
                        break
                
                # Calculate epoch metrics
                avg_loss = np.mean(epoch_loss) if epoch_loss else 0
                avg_accuracy = np.mean(epoch_accuracy) if epoch_accuracy else 0
                
                # Log epoch progress
                logger.info(f"Worker {self.worker_index} - Epoch {epoch + 1}/{epochs}: "
                           f"loss={avg_loss:.4f}, accuracy={avg_accuracy:.4f}")
                
                # Synchronize parameters at the end of each epoch
                if not self.use_async_updates:
                    self._sync_parameters()
            
            # Calculate training time
            training_time = time.time() - start_time
            self.training_metrics["training_time"] = training_time
            
            # Log final metrics
            logger.info(f"Worker {self.worker_index} - Training completed in {training_time:.2f} seconds")
            logger.info(f"Worker {self.worker_index} - Final metrics: "
                       f"loss={self.training_metrics['loss_history'][-1]:.4f}, "
                       f"accuracy={self.training_metrics['accuracy_history'][-1]:.4f}")
            
            # Save worker metrics
            self.training_metrics["worker_metrics"][f"worker_{self.worker_index}"] = {
                "loss": self.training_metrics["loss_history"][-1],
                "accuracy": self.training_metrics["accuracy_history"][-1],
                "training_time": training_time,
                "sync_time": self.training_metrics["sync_time"]
            }
            
            return self.training_metrics
        except Exception as e:
            logger.error(f"Error in distributed training: {e}")
            return None
    
    def save_checkpoint(self, checkpoint_name=None):
        """
        Save a checkpoint of the model.
        
        Args:
            checkpoint_name (str, optional): Name of the checkpoint
            
        Returns:
            bool: Whether the checkpoint was saved successfully
        """
        if self.model is None:
            logger.error("Model not initialized for distributed training.")
            return False
        
        try:
            if checkpoint_name is None:
                checkpoint_name = f"checkpoint_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Create checkpoint directory
            checkpoint_dir = os.path.join(self.model_dir, "checkpoints", checkpoint_name)
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            # Save model weights
            weights_path = os.path.join(checkpoint_dir, "weights.h5")
            self.model.save_weights(weights_path)
            
            # Save optimizer state
            optimizer_path = os.path.join(checkpoint_dir, "optimizer.npy")
            np.save(optimizer_path, self.optimizer.get_weights())
            
            # Save training metrics
            metrics_path = os.path.join(checkpoint_dir, "metrics.json")
            with open(metrics_path, 'w') as f:
                json.dump(self.training_metrics, f, indent=2)
            
            # Save checkpoint metadata
            metadata = {
                "checkpoint_name": checkpoint_name,
                "role": self.role,
                "worker_index": self.worker_index,
                "train_step": self.train_step,
                "global_step": int(self.global_step),
                "saved_at": datetime.now().isoformat()
            }
            
            metadata_path = os.path.join(checkpoint_dir, "metadata.json")
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Saved checkpoint: {checkpoint_name}")
            return True
        except Exception as e:
            logger.error(f"Error saving checkpoint: {e}")
            return False
    
    def load_checkpoint(self, checkpoint_name):
        """
        Load a checkpoint of the model.
        
        Args:
            checkpoint_name (str): Name of the checkpoint
            
        Returns:
            bool: Whether the checkpoint was loaded successfully
        """
        if self.model is None:
            logger.error("Model not initialized for distributed training.")
            return False
        
        try:
            # Get checkpoint directory
            checkpoint_dir = os.path.join(self.model_dir, "checkpoints", checkpoint_name)
            
            if not os.path.exists(checkpoint_dir):
                logger.error(f"Checkpoint directory not found: {checkpoint_dir}")
                return False
            
            # Load model weights
            weights_path = os.path.join(checkpoint_dir, "weights.h5")
            self.model.load_weights(weights_path)
            
            # Load optimizer state
            optimizer_path = os.path.join(checkpoint_dir, "optimizer.npy")
            if os.path.exists(optimizer_path):
                optimizer_weights = np.load(optimizer_path, allow_pickle=True)
                self.optimizer.set_weights(optimizer_weights)
            
            # Load training metrics
            metrics_path = os.path.join(checkpoint_dir, "metrics.json")
            if os.path.exists(metrics_path):
                with open(metrics_path, 'r') as f:
                    self.training_metrics = json.load(f)
            
            # Load checkpoint metadata
            metadata_path = os.path.join(checkpoint_dir, "metadata.json")
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                
                self.train_step = metadata.get("train_step", 0)
                self.global_step.assign(metadata.get("global_step", 0))
            
            logger.info(f"Loaded checkpoint: {checkpoint_name}")
            return True
        except Exception as e:
            logger.error(f"Error loading checkpoint: {e}")
            return False
    
    def get_status(self):
        """
        Get the status of the distributed training.
        
        Returns:
            dict: Status information
        """
        return {
            "role": self.role,
            "worker_index": self.worker_index if self.role == "worker" else None,
            "ps_index": self.ps_index if self.role == "parameter_server" else None,
            "is_running": self.is_running,
            "train_step": self.train_step,
            "global_step": int(self.global_step),
            "training_metrics": self.training_metrics,
            "cluster_config": self.cluster_config
        }
