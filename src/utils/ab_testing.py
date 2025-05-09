"""
A/B Testing Framework for NoahAI Reinforcement Learning

This module provides tools for comparing different reinforcement learning
algorithms and hyperparameters through systematic A/B testing.
"""

import os
import json
import time
import logging
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional, Union
from collections import defaultdict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/ab_testing.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("ab_testing")

class ABTestingFramework:
    """
    A/B Testing Framework for comparing different RL algorithms and hyperparameters.
    
    This class provides tools for running controlled experiments to compare
    the performance of different reinforcement learning configurations.
    """
    
    def __init__(self, experiment_dir="data/experiments", metrics_file="metrics.json"):
        """
        Initialize the A/B Testing Framework.
        
        Args:
            experiment_dir (str): Directory to store experiment data
            metrics_file (str): File to store metrics data
        """
        self.experiment_dir = experiment_dir
        self.metrics_file = os.path.join(experiment_dir, metrics_file)
        self.current_experiment = None
        self.experiments = {}
        self.metrics = defaultdict(list)
        
        # Create experiment directory if it doesn't exist
        os.makedirs(experiment_dir, exist_ok=True)
        
        # Load existing metrics if available
        self._load_metrics()
    
    def _load_metrics(self):
        """Load metrics from file if it exists."""
        if os.path.exists(self.metrics_file):
            try:
                with open(self.metrics_file, 'r') as f:
                    data = json.load(f)
                    self.experiments = data.get("experiments", {})
                    
                    # Convert defaultdict values
                    for exp_id, metrics in data.get("metrics", {}).items():
                        for metric_name, values in metrics.items():
                            self.metrics[exp_id].append({
                                "name": metric_name,
                                "values": values
                            })
                            
                logger.info(f"Loaded metrics for {len(self.experiments)} experiments")
            except Exception as e:
                logger.error(f"Error loading metrics: {e}")
    
    def _save_metrics(self):
        """Save metrics to file."""
        try:
            # Convert metrics to serializable format
            serializable_metrics = {}
            for exp_id, metrics_list in self.metrics.items():
                serializable_metrics[exp_id] = {}
                for metric in metrics_list:
                    serializable_metrics[exp_id][metric["name"]] = metric["values"]
            
            data = {
                "experiments": self.experiments,
                "metrics": serializable_metrics,
                "last_updated": datetime.now().isoformat()
            }
            
            with open(self.metrics_file, 'w') as f:
                json.dump(data, f, indent=2)
                
            logger.info(f"Saved metrics to {self.metrics_file}")
        except Exception as e:
            logger.error(f"Error saving metrics: {e}")
    
    def create_experiment(self, name, description=None, variants=None):
        """
        Create a new A/B testing experiment.
        
        Args:
            name (str): Name of the experiment
            description (str, optional): Description of the experiment
            variants (list, optional): List of variant configurations to test
            
        Returns:
            str: Experiment ID
        """
        experiment_id = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.experiments[experiment_id] = {
            "id": experiment_id,
            "name": name,
            "description": description,
            "created_at": datetime.now().isoformat(),
            "status": "created",
            "variants": variants or [],
            "results": None
        }
        
        self.current_experiment = experiment_id
        self._save_metrics()
        
        logger.info(f"Created experiment: {name} (ID: {experiment_id})")
        return experiment_id
    
    def add_variant(self, experiment_id, variant_name, config):
        """
        Add a variant to an experiment.
        
        Args:
            experiment_id (str): ID of the experiment
            variant_name (str): Name of the variant
            config (dict): Configuration for the variant
            
        Returns:
            bool: Whether the variant was added successfully
        """
        if experiment_id not in self.experiments:
            logger.error(f"Experiment not found: {experiment_id}")
            return False
        
        variant = {
            "name": variant_name,
            "config": config,
            "status": "created"
        }
        
        self.experiments[experiment_id]["variants"].append(variant)
        self._save_metrics()
        
        logger.info(f"Added variant '{variant_name}' to experiment {experiment_id}")
        return True
    
    def start_experiment(self, experiment_id=None):
        """
        Start an experiment.
        
        Args:
            experiment_id (str, optional): ID of the experiment to start.
                If None, uses the current experiment.
                
        Returns:
            bool: Whether the experiment was started successfully
        """
        experiment_id = experiment_id or self.current_experiment
        
        if experiment_id not in self.experiments:
            logger.error(f"Experiment not found: {experiment_id}")
            return False
        
        self.experiments[experiment_id]["status"] = "running"
        self.experiments[experiment_id]["started_at"] = datetime.now().isoformat()
        self._save_metrics()
        
        logger.info(f"Started experiment: {experiment_id}")
        return True
    
    def record_metric(self, experiment_id, variant_name, metric_name, value, step=None):
        """
        Record a metric for a variant in an experiment.
        
        Args:
            experiment_id (str): ID of the experiment
            variant_name (str): Name of the variant
            metric_name (str): Name of the metric
            value (float): Value of the metric
            step (int, optional): Step number for the metric
            
        Returns:
            bool: Whether the metric was recorded successfully
        """
        if experiment_id not in self.experiments:
            logger.error(f"Experiment not found: {experiment_id}")
            return False
        
        # Create a unique key for the variant
        variant_key = f"{experiment_id}_{variant_name}_{metric_name}"
        
        # Find or create the metric entry
        metric_entry = None
        for entry in self.metrics[experiment_id]:
            if entry["name"] == f"{variant_name}_{metric_name}":
                metric_entry = entry
                break
        
        if metric_entry is None:
            metric_entry = {
                "name": f"{variant_name}_{metric_name}",
                "values": []
            }
            self.metrics[experiment_id].append(metric_entry)
        
        # Add the value with step information
        if step is not None:
            metric_entry["values"].append({"step": step, "value": value})
        else:
            metric_entry["values"].append({"step": len(metric_entry["values"]), "value": value})
        
        self._save_metrics()
        return True
    
    def complete_experiment(self, experiment_id=None, results=None):
        """
        Mark an experiment as complete and record results.
        
        Args:
            experiment_id (str, optional): ID of the experiment to complete.
                If None, uses the current experiment.
            results (dict, optional): Results of the experiment
                
        Returns:
            bool: Whether the experiment was completed successfully
        """
        experiment_id = experiment_id or self.current_experiment
        
        if experiment_id not in self.experiments:
            logger.error(f"Experiment not found: {experiment_id}")
            return False
        
        self.experiments[experiment_id]["status"] = "completed"
        self.experiments[experiment_id]["completed_at"] = datetime.now().isoformat()
        
        if results:
            self.experiments[experiment_id]["results"] = results
        
        self._save_metrics()
        
        logger.info(f"Completed experiment: {experiment_id}")
        return True
    
    def get_experiment_results(self, experiment_id=None):
        """
        Get the results of an experiment.
        
        Args:
            experiment_id (str, optional): ID of the experiment.
                If None, uses the current experiment.
                
        Returns:
            dict: Results of the experiment
        """
        experiment_id = experiment_id or self.current_experiment
        
        if experiment_id not in self.experiments:
            logger.error(f"Experiment not found: {experiment_id}")
            return None
        
        # Compile results from metrics
        results = {
            "experiment": self.experiments[experiment_id],
            "metrics": {}
        }
        
        for metric_entry in self.metrics[experiment_id]:
            # Extract variant name and metric name
            parts = metric_entry["name"].split("_", 1)
            if len(parts) == 2:
                variant_name, metric_name = parts
                
                if variant_name not in results["metrics"]:
                    results["metrics"][variant_name] = {}
                
                # Extract values only (not steps)
                values = [entry["value"] for entry in metric_entry["values"]]
                results["metrics"][variant_name][metric_name] = values
        
        return results
    
    def visualize_experiment(self, experiment_id=None, metric_name=None, save_path=None):
        """
        Visualize the results of an experiment.
        
        Args:
            experiment_id (str, optional): ID of the experiment.
                If None, uses the current experiment.
            metric_name (str, optional): Name of the metric to visualize.
                If None, visualizes all metrics.
            save_path (str, optional): Path to save the visualization.
                If None, displays the visualization.
                
        Returns:
            bool: Whether the visualization was created successfully
        """
        experiment_id = experiment_id or self.current_experiment
        
        if experiment_id not in self.experiments:
            logger.error(f"Experiment not found: {experiment_id}")
            return False
        
        # Get experiment results
        results = self.get_experiment_results(experiment_id)
        if not results:
            return False
        
        # Create visualization
        plt.figure(figsize=(12, 8))
        
        # If metric_name is specified, only visualize that metric
        if metric_name:
            plt.title(f"{results['experiment']['name']} - {metric_name}")
            plt.xlabel("Step")
            plt.ylabel(metric_name)
            
            for variant_name, metrics in results["metrics"].items():
                if metric_name in metrics:
                    values = metrics[metric_name]
                    plt.plot(range(len(values)), values, label=variant_name)
            
            plt.legend()
        else:
            # Visualize all metrics
            metrics_set = set()
            for variant_metrics in results["metrics"].values():
                metrics_set.update(variant_metrics.keys())
            
            # Create subplots for each metric
            fig, axes = plt.subplots(len(metrics_set), 1, figsize=(12, 6 * len(metrics_set)))
            fig.suptitle(results['experiment']['name'], fontsize=16)
            
            for i, metric in enumerate(sorted(metrics_set)):
                ax = axes[i] if len(metrics_set) > 1 else axes
                ax.set_title(metric)
                ax.set_xlabel("Step")
                ax.set_ylabel(metric)
                
                for variant_name, metrics in results["metrics"].items():
                    if metric in metrics:
                        values = metrics[metric]
                        ax.plot(range(len(values)), values, label=variant_name)
                
                ax.legend()
            
            plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        # Save or display the visualization
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Saved visualization to {save_path}")
        else:
            plt.show()
        
        return True
