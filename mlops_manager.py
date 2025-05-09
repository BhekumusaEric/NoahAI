#!/usr/bin/env python3
"""
MLOps Manager for NoahAI

This script implements MLOps (Machine Learning Operations) for NoahAI.
It integrates with popular MLOps tools to track experiments, monitor models,
and manage the ML lifecycle.

Features:
- Experiment tracking with MLflow
- Model versioning and registry
- Performance monitoring
- Deployment management
- A/B testing integration
- Automated reporting
"""

import os
import sys
import json
import time
import argparse
import logging
import numpy as np
import tensorflow as tf
from datetime import datetime
from typing import List, Dict, Tuple, Any, Optional, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/mlops.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("mlops")

# Try to import NoahAI modules
try:
    from models.deep_learning_model import DeepLearningModel
    from src.utils.learning_utils import LearningUtils
    NOAHAI_AVAILABLE = True
except ImportError:
    NOAHAI_AVAILABLE = False
    logger.error("NoahAI modules not available. Please run this script from the NoahAI directory.")
    sys.exit(1)

# Try to import MLOps libraries
try:
    import mlflow
    import mlflow.tensorflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    logger.warning("MLflow not available. Install with: pip install mlflow")

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    logger.warning("Weights & Biases not available. Install with: pip install wandb")

try:
    import tensorboard
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    logger.warning("TensorBoard not available. Install with: pip install tensorboard")

class MLOpsManager:
    """
    Manager for MLOps integration with NoahAI.
    """
    
    def __init__(self, config_file=None, model_dir="data/model", output_dir="data/mlops"):
        """
        Initialize the MLOps manager.
        
        Args:
            config_file (str): Path to configuration file
            model_dir (str): Directory for model files
            output_dir (str): Directory for output
        """
        self.model_dir = model_dir
        self.output_dir = output_dir
        
        # Create directories
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)
        
        # Load configuration
        self.config = self._load_config(config_file)
        
        # Initialize model
        self.model = self._load_model()
        
        # Initialize MLOps tools
        self.mlflow_client = None
        self.wandb_run = None
        self.tensorboard_writer = None
        
        self._initialize_mlops_tools()
    
    def _load_config(self, config_file):
        """
        Load configuration from file or use defaults.
        
        Args:
            config_file (str): Path to configuration file
            
        Returns:
            dict: Configuration dictionary
        """
        default_config = {
            "model": {
                "model_type": "advanced_lstm",
                "max_words": 10000,
                "max_sequence_length": 100,
                "use_reinforcement_learning": True
            },
            "mlops": {
                "mlflow": {
                    "enabled": True,
                    "tracking_uri": "http://localhost:5000",
                    "experiment_name": "noahai",
                    "register_model": True
                },
                "wandb": {
                    "enabled": False,
                    "project": "noahai",
                    "entity": None,
                    "tags": ["deep-learning", "nlp", "tensorflow"]
                },
                "tensorboard": {
                    "enabled": True,
                    "log_dir": "logs/tensorboard"
                },
                "monitoring": {
                    "enabled": True,
                    "metrics": ["accuracy", "loss", "precision", "recall", "f1"],
                    "frequency": 100  # Log metrics every N steps
                },
                "deployment": {
                    "enabled": False,
                    "target": "local",  # local, docker, kubernetes
                    "auto_deploy": False,
                    "deployment_threshold": 0.8  # Minimum accuracy for auto-deployment
                },
                "reporting": {
                    "enabled": True,
                    "format": "markdown",  # markdown, html, pdf
                    "include_plots": True,
                    "auto_generate": True
                }
            }
        }
        
        if config_file and os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    user_config = json.load(f)
                
                # Merge user config with defaults
                for section, settings in user_config.items():
                    if section in default_config:
                        if isinstance(settings, dict) and isinstance(default_config[section], dict):
                            default_config[section].update(settings)
                        else:
                            default_config[section] = settings
                    else:
                        default_config[section] = settings
                
                logger.info(f"Loaded configuration from {config_file}")
            except Exception as e:
                logger.error(f"Error loading configuration: {e}")
        
        return default_config
    
    def _load_model(self):
        """
        Load or create a model.
        
        Returns:
            DeepLearningModel: Model instance
        """
        model_config = self.config["model"]
        
        # Check if model exists
        model_file = os.path.join(self.model_dir, f"model_{model_config['model_type']}.h5")
        
        if os.path.exists(model_file):
            logger.info(f"Loading existing model from {model_file}")
            
            try:
                model = DeepLearningModel(
                    model_dir=self.model_dir,
                    responses_file="data/responses.json",
                    max_words=model_config["max_words"],
                    max_sequence_length=model_config["max_sequence_length"],
                    use_advanced_nlp=True,
                    model_type=model_config["model_type"],
                    use_reinforcement_learning=model_config["use_reinforcement_learning"]
                )
                
                # Load the model
                model.load_model()
                
                return model
            
            except Exception as e:
                logger.error(f"Error loading model: {e}")
                logger.info("Creating new model instead")
        
        # Create new model
        logger.info("Creating new model")
        
        model = DeepLearningModel(
            model_dir=self.model_dir,
            responses_file="data/responses.json",
            max_words=model_config["max_words"],
            max_sequence_length=model_config["max_sequence_length"],
            use_advanced_nlp=True,
            model_type=model_config["model_type"],
            use_reinforcement_learning=model_config["use_reinforcement_learning"]
        )
        
        return model
    
    def _initialize_mlops_tools(self):
        """
        Initialize MLOps tools.
        """
        # Initialize MLflow
        if MLFLOW_AVAILABLE and self.config["mlops"]["mlflow"]["enabled"]:
            try:
                mlflow_config = self.config["mlops"]["mlflow"]
                
                # Set tracking URI
                if mlflow_config["tracking_uri"]:
                    mlflow.set_tracking_uri(mlflow_config["tracking_uri"])
                
                # Set experiment
                mlflow.set_experiment(mlflow_config["experiment_name"])
                
                # Create client
                self.mlflow_client = mlflow.tracking.MlflowClient()
                
                logger.info(f"Initialized MLflow with experiment: {mlflow_config['experiment_name']}")
            
            except Exception as e:
                logger.error(f"Error initializing MLflow: {e}")
        
        # Initialize Weights & Biases
        if WANDB_AVAILABLE and self.config["mlops"]["wandb"]["enabled"]:
            try:
                wandb_config = self.config["mlops"]["wandb"]
                
                # Initialize wandb
                self.wandb_run = wandb.init(
                    project=wandb_config["project"],
                    entity=wandb_config["entity"],
                    tags=wandb_config["tags"],
                    config=self.config
                )
                
                logger.info(f"Initialized Weights & Biases with project: {wandb_config['project']}")
            
            except Exception as e:
                logger.error(f"Error initializing Weights & Biases: {e}")
        
        # Initialize TensorBoard
        if TENSORBOARD_AVAILABLE and self.config["mlops"]["tensorboard"]["enabled"]:
            try:
                tensorboard_config = self.config["mlops"]["tensorboard"]
                
                # Create log directory
                os.makedirs(tensorboard_config["log_dir"], exist_ok=True)
                
                # Create writer
                self.tensorboard_writer = tf.summary.create_file_writer(
                    os.path.join(tensorboard_config["log_dir"], datetime.now().strftime("%Y%m%d-%H%M%S"))
                )
                
                logger.info(f"Initialized TensorBoard with log directory: {tensorboard_config['log_dir']}")
            
            except Exception as e:
                logger.error(f"Error initializing TensorBoard: {e}")
    
    def start_run(self, run_name=None, tags=None):
        """
        Start a new run in MLOps tools.
        
        Args:
            run_name (str, optional): Name for the run
            tags (dict, optional): Tags for the run
            
        Returns:
            str: Run ID
        """
        run_id = None
        
        # Start MLflow run
        if MLFLOW_AVAILABLE and self.config["mlops"]["mlflow"]["enabled"]:
            try:
                mlflow.start_run(run_name=run_name)
                run_id = mlflow.active_run().info.run_id
                
                # Log tags
                if tags:
                    mlflow.set_tags(tags)
                
                # Log model parameters
                mlflow.log_params({
                    "model_type": self.config["model"]["model_type"],
                    "max_words": self.config["model"]["max_words"],
                    "max_sequence_length": self.config["model"]["max_sequence_length"],
                    "use_reinforcement_learning": self.config["model"]["use_reinforcement_learning"]
                })
                
                logger.info(f"Started MLflow run: {run_id}")
            
            except Exception as e:
                logger.error(f"Error starting MLflow run: {e}")
        
        # Update Weights & Biases run
        if WANDB_AVAILABLE and self.config["mlops"]["wandb"]["enabled"] and self.wandb_run:
            try:
                if run_name:
                    wandb.run.name = run_name
                
                if tags:
                    wandb.run.tags = wandb.run.tags + list(tags.values())
                
                logger.info(f"Updated Weights & Biases run: {wandb.run.id}")
            
            except Exception as e:
                logger.error(f"Error updating Weights & Biases run: {e}")
        
        return run_id
    
    def end_run(self):
        """
        End the current run in MLOps tools.
        """
        # End MLflow run
        if MLFLOW_AVAILABLE and self.config["mlops"]["mlflow"]["enabled"]:
            try:
                mlflow.end_run()
                logger.info("Ended MLflow run")
            
            except Exception as e:
                logger.error(f"Error ending MLflow run: {e}")
        
        # End Weights & Biases run
        if WANDB_AVAILABLE and self.config["mlops"]["wandb"]["enabled"] and self.wandb_run:
            try:
                wandb.finish()
                logger.info("Ended Weights & Biases run")
            
            except Exception as e:
                logger.error(f"Error ending Weights & Biases run: {e}")
    
    def log_metrics(self, metrics, step=None):
        """
        Log metrics to MLOps tools.
        
        Args:
            metrics (dict): Metrics to log
            step (int, optional): Step number
        """
        # Log to MLflow
        if MLFLOW_AVAILABLE and self.config["mlops"]["mlflow"]["enabled"]:
            try:
                mlflow.log_metrics(metrics, step=step)
                logger.info(f"Logged metrics to MLflow: {metrics}")
            
            except Exception as e:
                logger.error(f"Error logging metrics to MLflow: {e}")
        
        # Log to Weights & Biases
        if WANDB_AVAILABLE and self.config["mlops"]["wandb"]["enabled"] and self.wandb_run:
            try:
                wandb.log(metrics, step=step)
                logger.info(f"Logged metrics to Weights & Biases: {metrics}")
            
            except Exception as e:
                logger.error(f"Error logging metrics to Weights & Biases: {e}")
        
        # Log to TensorBoard
        if TENSORBOARD_AVAILABLE and self.config["mlops"]["tensorboard"]["enabled"] and self.tensorboard_writer:
            try:
                with self.tensorboard_writer.as_default():
                    for name, value in metrics.items():
                        tf.summary.scalar(name, value, step=step)
                
                logger.info(f"Logged metrics to TensorBoard: {metrics}")
            
            except Exception as e:
                logger.error(f"Error logging metrics to TensorBoard: {e}")
    
    def log_model(self, model=None, artifact_path="model"):
        """
        Log model to MLOps tools.
        
        Args:
            model (tf.keras.Model, optional): Model to log
            artifact_path (str): Path for the artifact
            
        Returns:
            str: Path to the logged model
        """
        model = model or self.model.model
        
        # Log to MLflow
        if MLFLOW_AVAILABLE and self.config["mlops"]["mlflow"]["enabled"]:
            try:
                mlflow.tensorflow.log_model(model, artifact_path)
                
                # Register model if enabled
                if self.config["mlops"]["mlflow"]["register_model"]:
                    mlflow.tensorflow.log_model(
                        model,
                        artifact_path,
                        registered_model_name=f"noahai_{self.config['model']['model_type']}"
                    )
                
                logger.info(f"Logged model to MLflow: {artifact_path}")
                
                # Get path to the logged model
                client = mlflow.tracking.MlflowClient()
                run_id = mlflow.active_run().info.run_id
                model_uri = f"runs:/{run_id}/{artifact_path}"
                
                return model_uri
            
            except Exception as e:
                logger.error(f"Error logging model to MLflow: {e}")
        
        # Log to Weights & Biases
        if WANDB_AVAILABLE and self.config["mlops"]["wandb"]["enabled"] and self.wandb_run:
            try:
                # Save model to temporary file
                temp_model_path = os.path.join(self.output_dir, "temp_model.h5")
                model.save(temp_model_path)
                
                # Log model as artifact
                artifact = wandb.Artifact(
                    name=f"model_{self.config['model']['model_type']}",
                    type="model",
                    description=f"NoahAI {self.config['model']['model_type']} model"
                )
                artifact.add_file(temp_model_path)
                wandb.log_artifact(artifact)
                
                # Clean up
                os.remove(temp_model_path)
                
                logger.info(f"Logged model to Weights & Biases")
            
            except Exception as e:
                logger.error(f"Error logging model to Weights & Biases: {e}")
        
        return None
    
    def log_dataset(self, dataset_path, dataset_name="training_data"):
        """
        Log dataset to MLOps tools.
        
        Args:
            dataset_path (str): Path to dataset file
            dataset_name (str): Name for the dataset
        """
        if not os.path.exists(dataset_path):
            logger.error(f"Dataset file not found: {dataset_path}")
            return
        
        # Log to MLflow
        if MLFLOW_AVAILABLE and self.config["mlops"]["mlflow"]["enabled"]:
            try:
                mlflow.log_artifact(dataset_path, artifact_path="datasets")
                logger.info(f"Logged dataset to MLflow: {dataset_path}")
            
            except Exception as e:
                logger.error(f"Error logging dataset to MLflow: {e}")
        
        # Log to Weights & Biases
        if WANDB_AVAILABLE and self.config["mlops"]["wandb"]["enabled"] and self.wandb_run:
            try:
                artifact = wandb.Artifact(
                    name=dataset_name,
                    type="dataset",
                    description=f"NoahAI dataset: {dataset_name}"
                )
                artifact.add_file(dataset_path)
                wandb.log_artifact(artifact)
                
                logger.info(f"Logged dataset to Weights & Biases: {dataset_path}")
            
            except Exception as e:
                logger.error(f"Error logging dataset to Weights & Biases: {e}")
    
    def log_figure(self, figure, figure_name):
        """
        Log a matplotlib figure to MLOps tools.
        
        Args:
            figure: Matplotlib figure
            figure_name (str): Name for the figure
        """
        # Save figure to temporary file
        temp_figure_path = os.path.join(self.output_dir, f"{figure_name}.png")
        figure.savefig(temp_figure_path)
        
        # Log to MLflow
        if MLFLOW_AVAILABLE and self.config["mlops"]["mlflow"]["enabled"]:
            try:
                mlflow.log_artifact(temp_figure_path, artifact_path="figures")
                logger.info(f"Logged figure to MLflow: {figure_name}")
            
            except Exception as e:
                logger.error(f"Error logging figure to MLflow: {e}")
        
        # Log to Weights & Biases
        if WANDB_AVAILABLE and self.config["mlops"]["wandb"]["enabled"] and self.wandb_run:
            try:
                wandb.log({figure_name: wandb.Image(temp_figure_path)})
                logger.info(f"Logged figure to Weights & Biases: {figure_name}")
            
            except Exception as e:
                logger.error(f"Error logging figure to Weights & Biases: {e}")
        
        # Clean up
        os.remove(temp_figure_path)
    
    def generate_report(self, metrics=None, figures=None, output_format=None):
        """
        Generate a report of the model training and evaluation.
        
        Args:
            metrics (dict, optional): Metrics to include in the report
            figures (list, optional): Figures to include in the report
            output_format (str, optional): Output format for the report
            
        Returns:
            str: Path to the generated report
        """
        reporting_config = self.config["mlops"]["reporting"]
        
        if not reporting_config["enabled"]:
            logger.info("Reporting is disabled in configuration")
            return None
        
        output_format = output_format or reporting_config["format"]
        
        try:
            # Create report directory
            report_dir = os.path.join(self.output_dir, "reports")
            os.makedirs(report_dir, exist_ok=True)
            
            # Generate report filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_filename = f"noahai_report_{timestamp}.{output_format}"
            report_path = os.path.join(report_dir, report_filename)
            
            # Generate report content
            if output_format == "markdown":
                with open(report_path, 'w') as f:
                    f.write("# NoahAI Model Report\n\n")
                    f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                    
                    # Model information
                    f.write("## Model Information\n\n")
                    f.write(f"- **Model Type:** {self.config['model']['model_type']}\n")
                    f.write(f"- **Max Words:** {self.config['model']['max_words']}\n")
                    f.write(f"- **Max Sequence Length:** {self.config['model']['max_sequence_length']}\n")
                    f.write(f"- **Reinforcement Learning:** {'Enabled' if self.config['model']['use_reinforcement_learning'] else 'Disabled'}\n\n")
                    
                    # Metrics
                    if metrics:
                        f.write("## Metrics\n\n")
                        f.write("| Metric | Value |\n")
                        f.write("|--------|-------|\n")
                        
                        for name, value in metrics.items():
                            f.write(f"| {name} | {value:.4f} |\n")
                        
                        f.write("\n")
                    
                    # Figures
                    if figures and reporting_config["include_plots"]:
                        f.write("## Figures\n\n")
                        
                        for i, figure_path in enumerate(figures):
                            f.write(f"### Figure {i+1}\n\n")
                            f.write(f"![Figure {i+1}]({figure_path})\n\n")
            
            elif output_format == "html":
                with open(report_path, 'w') as f:
                    f.write("<html><head><title>NoahAI Model Report</title></head><body>\n")
                    f.write(f"<h1>NoahAI Model Report</h1>\n")
                    f.write(f"<p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>\n")
                    
                    # Model information
                    f.write("<h2>Model Information</h2>\n")
                    f.write("<ul>\n")
                    f.write(f"<li><b>Model Type:</b> {self.config['model']['model_type']}</li>\n")
                    f.write(f"<li><b>Max Words:</b> {self.config['model']['max_words']}</li>\n")
                    f.write(f"<li><b>Max Sequence Length:</b> {self.config['model']['max_sequence_length']}</li>\n")
                    f.write(f"<li><b>Reinforcement Learning:</b> {'Enabled' if self.config['model']['use_reinforcement_learning'] else 'Disabled'}</li>\n")
                    f.write("</ul>\n")
                    
                    # Metrics
                    if metrics:
                        f.write("<h2>Metrics</h2>\n")
                        f.write("<table border='1'>\n")
                        f.write("<tr><th>Metric</th><th>Value</th></tr>\n")
                        
                        for name, value in metrics.items():
                            f.write(f"<tr><td>{name}</td><td>{value:.4f}</td></tr>\n")
                        
                        f.write("</table>\n")
                    
                    # Figures
                    if figures and reporting_config["include_plots"]:
                        f.write("<h2>Figures</h2>\n")
                        
                        for i, figure_path in enumerate(figures):
                            f.write(f"<h3>Figure {i+1}</h3>\n")
                            f.write(f"<img src='{figure_path}' alt='Figure {i+1}'>\n")
                    
                    f.write("</body></html>\n")
            
            else:
                logger.warning(f"Unsupported report format: {output_format}")
                return None
            
            logger.info(f"Generated report: {report_path}")
            return report_path
        
        except Exception as e:
            logger.error(f"Error generating report: {e}")
            return None
    
    def deploy_model(self, model=None, deployment_target=None):
        """
        Deploy the model to the specified target.
        
        Args:
            model (tf.keras.Model, optional): Model to deploy
            deployment_target (str, optional): Deployment target
            
        Returns:
            bool: Whether deployment was successful
        """
        deployment_config = self.config["mlops"]["deployment"]
        
        if not deployment_config["enabled"]:
            logger.info("Deployment is disabled in configuration")
            return False
        
        model = model or self.model.model
        deployment_target = deployment_target or deployment_config["target"]
        
        try:
            if deployment_target == "local":
                # Save model to deployment directory
                deployment_dir = os.path.join(self.output_dir, "deployment")
                os.makedirs(deployment_dir, exist_ok=True)
                
                model_path = os.path.join(deployment_dir, f"model_{self.config['model']['model_type']}.h5")
                model.save(model_path)
                
                # Save tokenizer
                tokenizer_path = os.path.join(deployment_dir, "tokenizer.pkl")
                self.model._save_tokenizer(tokenizer_path)
                
                # Save category mapping
                mapping_path = os.path.join(deployment_dir, "category_mapping.json")
                with open(mapping_path, 'w') as f:
                    json.dump(self.model.category_mapping, f, indent=2)
                
                logger.info(f"Deployed model to local directory: {deployment_dir}")
                return True
            
            elif deployment_target == "docker":
                logger.warning("Docker deployment not implemented yet")
                return False
            
            elif deployment_target == "kubernetes":
                logger.warning("Kubernetes deployment not implemented yet")
                return False
            
            else:
                logger.warning(f"Unknown deployment target: {deployment_target}")
                return False
        
        except Exception as e:
            logger.error(f"Error deploying model: {e}")
            return False

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="MLOps for NoahAI")
    parser.add_argument("--config", help="Path to configuration file")
    parser.add_argument("--model-dir", default="data/model", help="Directory for model files")
    parser.add_argument("--output-dir", default="data/mlops", help="Directory for output")
    parser.add_argument("--start-run", action="store_true", help="Start a new run")
    parser.add_argument("--end-run", action="store_true", help="End the current run")
    parser.add_argument("--log-model", action="store_true", help="Log the model")
    parser.add_argument("--dataset", help="Path to dataset file to log")
    parser.add_argument("--generate-report", action="store_true", help="Generate a report")
    parser.add_argument("--deploy", action="store_true", help="Deploy the model")
    args = parser.parse_args()
    
    # Create MLOps manager
    manager = MLOpsManager(
        config_file=args.config,
        model_dir=args.model_dir,
        output_dir=args.output_dir
    )
    
    # Start run if requested
    if args.start_run:
        manager.start_run(run_name=f"noahai_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    
    # Log model if requested
    if args.log_model:
        manager.log_model()
    
    # Log dataset if provided
    if args.dataset:
        manager.log_dataset(args.dataset)
    
    # Generate report if requested
    if args.generate_report:
        manager.generate_report()
    
    # Deploy model if requested
    if args.deploy:
        manager.deploy_model()
    
    # End run if requested
    if args.end_run:
        manager.end_run()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
