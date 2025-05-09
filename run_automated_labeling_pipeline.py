#!/usr/bin/env python3
"""
Automated Labeling Pipeline for NoahAI

This script runs the complete automated labeling pipeline for NoahAI:
1. Generate or collect unlabeled data
2. Label data using the trained model
3. Separate high and low confidence examples
4. Update the model with high confidence examples
5. Send low confidence examples for human verification
6. Integrate with active learning for uncertain examples
7. Perform quality assurance
8. Track metrics with MLOps tools
"""

import os
import sys
import json
import time
import argparse
import logging
import subprocess
from datetime import datetime
from typing import List, Dict, Tuple, Any, Optional, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/automated_labeling_pipeline.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("automated_labeling_pipeline")

class AutomatedLabelingPipeline:
    """
    Pipeline for automated labeling of NoahAI data.
    """
    
    def __init__(self, config_file="automated_labeling_config.json", output_dir="data/automated_labeling_pipeline"):
        """
        Initialize the pipeline.
        
        Args:
            config_file (str): Path to configuration file
            output_dir (str): Directory for output
        """
        self.config_file = config_file
        self.output_dir = output_dir
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Load configuration
        self.config = self._load_config(config_file)
        
        # Initialize pipeline metrics
        self.pipeline_metrics = {
            "start_time": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
            "end_time": None,
            "duration": 0,
            "steps_completed": [],
            "steps_failed": [],
            "data_generation": {},
            "automated_labeling": {},
            "model_update": {},
            "human_verification": {},
            "active_learning": {},
            "quality_assurance": {},
            "mlops": {}
        }
    
    def _load_config(self, config_file):
        """
        Load configuration from file or use defaults.
        
        Args:
            config_file (str): Path to configuration file
            
        Returns:
            dict: Configuration dictionary
        """
        default_config = {
            "pipeline": {
                "generate_data": True,
                "automated_labeling": True,
                "model_update": True,
                "human_verification": True,
                "active_learning": True,
                "quality_assurance": True,
                "mlops": True
            },
            "data_generation": {
                "num_examples": 1000,
                "output_file": "data/unlabeled_data.json"
            },
            "automated_labeling": {
                "confidence_threshold": 0.9,
                "use_ensemble": True,
                "max_examples_per_batch": 1000
            },
            "model_update": {
                "enabled": True,
                "update_frequency": 500,
                "min_examples_for_update": 100
            },
            "human_verification": {
                "enabled": True,
                "verification_file": "data/human_verification.json"
            },
            "active_learning": {
                "enabled": True,
                "strategy": "uncertainty",
                "batch_size": 100
            },
            "quality_assurance": {
                "enabled": True,
                "validation_split": 0.1,
                "min_accuracy_threshold": 0.8
            },
            "mlops": {
                "enabled": True,
                "tracking_tool": "mlflow",
                "experiment_name": "noahai_automated_labeling"
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
    
    def run_pipeline(self):
        """
        Run the complete automated labeling pipeline.
        
        Returns:
            dict: Pipeline metrics
        """
        pipeline_config = self.config.get("pipeline", {})
        
        # Step 1: Generate or collect unlabeled data
        if pipeline_config.get("generate_data", True):
            self._run_data_generation()
        
        # Step 2: Label data using the trained model
        if pipeline_config.get("automated_labeling", True):
            self._run_automated_labeling()
        
        # Step 3: Update the model with high confidence examples
        if pipeline_config.get("model_update", True):
            self._run_model_update()
        
        # Step 4: Send low confidence examples for human verification
        if pipeline_config.get("human_verification", True):
            self._run_human_verification()
        
        # Step 5: Integrate with active learning for uncertain examples
        if pipeline_config.get("active_learning", True):
            self._run_active_learning()
        
        # Step 6: Perform quality assurance
        if pipeline_config.get("quality_assurance", True):
            self._run_quality_assurance()
        
        # Step 7: Track metrics with MLOps tools
        if pipeline_config.get("mlops", True):
            self._run_mlops()
        
        # Finalize pipeline metrics
        self.pipeline_metrics["end_time"] = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
        start_time = datetime.strptime(self.pipeline_metrics["start_time"], "%Y-%m-%dT%H:%M:%S")
        end_time = datetime.strptime(self.pipeline_metrics["end_time"], "%Y-%m-%dT%H:%M:%S")
        self.pipeline_metrics["duration"] = (end_time - start_time).total_seconds()
        
        # Save pipeline metrics
        self._save_pipeline_metrics()
        
        return self.pipeline_metrics
    
    def _run_data_generation(self):
        """
        Run the data generation step.
        """
        logger.info("Starting data generation step")
        start_time = time.time()
        
        try:
            # Get data generation config
            data_config = self.config.get("data_generation", {})
            num_examples = data_config.get("num_examples", 1000)
            output_file = data_config.get("output_file", "data/unlabeled_data.json")
            
            # Build command
            cmd = [
                "python", "generate_unlabeled_data.py",
                "--num-examples", str(num_examples),
                "--output", output_file
            ]
            
            # Run data generation
            logger.info(f"Running command: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info("Data generation completed successfully")
                
                # Try to parse output to get number of examples generated
                try:
                    if os.path.exists(output_file):
                        with open(output_file, 'r') as f:
                            data = json.load(f)
                        self.pipeline_metrics["data_generation"]["examples_generated"] = len(data)
                except Exception as e:
                    logger.warning(f"Error parsing generated data: {e}")
                
                self.pipeline_metrics["steps_completed"].append("data_generation")
            else:
                logger.error(f"Data generation failed with code {result.returncode}")
                logger.error(f"Error: {result.stderr}")
                self.pipeline_metrics["steps_failed"].append("data_generation")
        
        except Exception as e:
            logger.error(f"Error in data generation step: {e}")
            self.pipeline_metrics["steps_failed"].append("data_generation")
        
        # Update metrics
        self.pipeline_metrics["data_generation"]["duration"] = time.time() - start_time
        logger.info(f"Data generation step completed in {self.pipeline_metrics['data_generation']['duration']:.2f} seconds")
    
    def _run_automated_labeling(self):
        """
        Run the automated labeling step.
        """
        logger.info("Starting automated labeling step")
        start_time = time.time()
        
        try:
            # Get automated labeling config
            labeling_config = self.config.get("automated_labeling", {})
            data_config = self.config.get("data_generation", {})
            
            # Build command
            cmd = [
                "python", "automated_labeling_manager.py",
                "--config", self.config_file,
                "--output-dir", os.path.join(self.output_dir, "automated_labeling"),
                "--data", data_config.get("output_file", "data/unlabeled_data.json")
            ]
            
            # Run automated labeling
            logger.info(f"Running command: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info("Automated labeling completed successfully")
                
                # Try to load labeling metrics
                try:
                    metrics_file = os.path.join(self.output_dir, "automated_labeling", "labeling_metrics.json")
                    if os.path.exists(metrics_file):
                        with open(metrics_file, 'r') as f:
                            labeling_metrics = json.load(f)
                        
                        # Extract key metrics
                        self.pipeline_metrics["automated_labeling"]["total_examples"] = labeling_metrics.get("total_examples", 0)
                        self.pipeline_metrics["automated_labeling"]["labeled_examples"] = labeling_metrics.get("labeled_examples", 0)
                        self.pipeline_metrics["automated_labeling"]["high_confidence_examples"] = labeling_metrics.get("high_confidence_examples", 0)
                        self.pipeline_metrics["automated_labeling"]["low_confidence_examples"] = labeling_metrics.get("low_confidence_examples", 0)
                        self.pipeline_metrics["automated_labeling"]["estimated_accuracy"] = labeling_metrics.get("estimated_accuracy", 0)
                except Exception as e:
                    logger.warning(f"Error loading labeling metrics: {e}")
                
                self.pipeline_metrics["steps_completed"].append("automated_labeling")
            else:
                logger.error(f"Automated labeling failed with code {result.returncode}")
                logger.error(f"Error: {result.stderr}")
                self.pipeline_metrics["steps_failed"].append("automated_labeling")
        
        except Exception as e:
            logger.error(f"Error in automated labeling step: {e}")
            self.pipeline_metrics["steps_failed"].append("automated_labeling")
        
        # Update metrics
        self.pipeline_metrics["automated_labeling"]["duration"] = time.time() - start_time
        logger.info(f"Automated labeling step completed in {self.pipeline_metrics['automated_labeling']['duration']:.2f} seconds")
    
    def _run_model_update(self):
        """
        Run the model update step.
        """
        logger.info("Starting model update step")
        start_time = time.time()
        
        try:
            # Check if model update is enabled
            if not self.config.get("model_update", {}).get("enabled", True):
                logger.info("Model update is disabled in configuration")
                return
            
            # Find the latest high confidence file
            automated_labeling_dir = os.path.join(self.output_dir, "automated_labeling")
            high_confidence_files = [f for f in os.listdir(automated_labeling_dir) if f.startswith("high_confidence_")]
            
            if not high_confidence_files:
                logger.warning("No high confidence files found for model update")
                return
            
            # Sort by timestamp (newest first)
            high_confidence_files.sort(reverse=True)
            latest_file = os.path.join(automated_labeling_dir, high_confidence_files[0])
            
            # Check if file exists and has enough examples
            if not os.path.exists(latest_file):
                logger.warning(f"High confidence file not found: {latest_file}")
                return
            
            # Load high confidence examples
            with open(latest_file, 'r') as f:
                high_confidence_examples = json.load(f)
            
            min_examples = self.config.get("model_update", {}).get("min_examples_for_update", 100)
            if len(high_confidence_examples) < min_examples:
                logger.warning(f"Not enough high confidence examples for model update. Have {len(high_confidence_examples)}, need {min_examples}")
                return
            
            # Model update is handled automatically by the automated_labeling_manager.py script
            # We just need to record the metrics
            self.pipeline_metrics["model_update"]["examples_used"] = len(high_confidence_examples)
            self.pipeline_metrics["model_update"]["high_confidence_file"] = latest_file
            
            self.pipeline_metrics["steps_completed"].append("model_update")
        
        except Exception as e:
            logger.error(f"Error in model update step: {e}")
            self.pipeline_metrics["steps_failed"].append("model_update")
        
        # Update metrics
        self.pipeline_metrics["model_update"]["duration"] = time.time() - start_time
        logger.info(f"Model update step completed in {self.pipeline_metrics['model_update']['duration']:.2f} seconds")
    
    def _run_human_verification(self):
        """
        Run the human verification step.
        """
        logger.info("Starting human verification step")
        start_time = time.time()
        
        try:
            # Check if human verification is enabled
            if not self.config.get("human_verification", {}).get("enabled", True):
                logger.info("Human verification is disabled in configuration")
                return
            
            # Find the latest low confidence file
            automated_labeling_dir = os.path.join(self.output_dir, "automated_labeling")
            low_confidence_files = [f for f in os.listdir(automated_labeling_dir) if f.startswith("low_confidence_")]
            
            if not low_confidence_files:
                logger.warning("No low confidence files found for human verification")
                return
            
            # Sort by timestamp (newest first)
            low_confidence_files.sort(reverse=True)
            latest_file = os.path.join(automated_labeling_dir, low_confidence_files[0])
            
            # Check if file exists
            if not os.path.exists(latest_file):
                logger.warning(f"Low confidence file not found: {latest_file}")
                return
            
            # Load low confidence examples
            with open(latest_file, 'r') as f:
                low_confidence_examples = json.load(f)
            
            # Create human verification file
            verification_file = self.config.get("human_verification", {}).get("verification_file", "data/human_verification.json")
            os.makedirs(os.path.dirname(verification_file), exist_ok=True)
            
            # Save examples for human verification
            with open(verification_file, 'w') as f:
                json.dump(low_confidence_examples, f, indent=2)
            
            logger.info(f"Saved {len(low_confidence_examples)} examples for human verification to {verification_file}")
            
            # Update metrics
            self.pipeline_metrics["human_verification"]["examples_for_verification"] = len(low_confidence_examples)
            self.pipeline_metrics["human_verification"]["verification_file"] = verification_file
            
            self.pipeline_metrics["steps_completed"].append("human_verification")
        
        except Exception as e:
            logger.error(f"Error in human verification step: {e}")
            self.pipeline_metrics["steps_failed"].append("human_verification")
        
        # Update metrics
        self.pipeline_metrics["human_verification"]["duration"] = time.time() - start_time
        logger.info(f"Human verification step completed in {self.pipeline_metrics['human_verification']['duration']:.2f} seconds")
    
    def _run_active_learning(self):
        """
        Run the active learning step.
        """
        logger.info("Starting active learning step")
        start_time = time.time()
        
        try:
            # Check if active learning is enabled
            if not self.config.get("active_learning", {}).get("enabled", True):
                logger.info("Active learning is disabled in configuration")
                return
            
            # Active learning integration is handled by the automated_labeling_manager.py script
            # We just need to check if the active learning file was created
            active_learning_dir = os.path.join(self.output_dir, "automated_labeling", "active_learning")
            
            if not os.path.exists(active_learning_dir):
                logger.warning("Active learning directory not found")
                return
            
            active_learning_files = [f for f in os.listdir(active_learning_dir) if f.startswith("active_learning_")]
            
            if not active_learning_files:
                logger.warning("No active learning files found")
                return
            
            # Sort by timestamp (newest first)
            active_learning_files.sort(reverse=True)
            latest_file = os.path.join(active_learning_dir, active_learning_files[0])
            
            # Check if file exists
            if not os.path.exists(latest_file):
                logger.warning(f"Active learning file not found: {latest_file}")
                return
            
            # Load active learning examples
            with open(latest_file, 'r') as f:
                active_learning_examples = json.load(f)
            
            # Update metrics
            self.pipeline_metrics["active_learning"]["examples_for_active_learning"] = len(active_learning_examples)
            self.pipeline_metrics["active_learning"]["active_learning_file"] = latest_file
            
            self.pipeline_metrics["steps_completed"].append("active_learning")
        
        except Exception as e:
            logger.error(f"Error in active learning step: {e}")
            self.pipeline_metrics["steps_failed"].append("active_learning")
        
        # Update metrics
        self.pipeline_metrics["active_learning"]["duration"] = time.time() - start_time
        logger.info(f"Active learning step completed in {self.pipeline_metrics['active_learning']['duration']:.2f} seconds")
    
    def _run_quality_assurance(self):
        """
        Run the quality assurance step.
        """
        logger.info("Starting quality assurance step")
        start_time = time.time()
        
        try:
            # Check if quality assurance is enabled
            if not self.config.get("quality_assurance", {}).get("enabled", True):
                logger.info("Quality assurance is disabled in configuration")
                return
            
            # Quality assurance is handled by the automated_labeling_manager.py script
            # We just need to check if the QA metrics file was created
            qa_metrics_file = os.path.join(self.output_dir, "automated_labeling", "quality_assurance_metrics.json")
            
            if not os.path.exists(qa_metrics_file):
                logger.warning("Quality assurance metrics file not found")
                return
            
            # Load QA metrics
            with open(qa_metrics_file, 'r') as f:
                qa_metrics = json.load(f)
            
            # Update metrics
            self.pipeline_metrics["quality_assurance"]["total_examples"] = qa_metrics.get("total_examples", 0)
            self.pipeline_metrics["quality_assurance"]["high_confidence_examples"] = qa_metrics.get("high_confidence_examples", 0)
            self.pipeline_metrics["quality_assurance"]["low_confidence_examples"] = qa_metrics.get("low_confidence_examples", 0)
            self.pipeline_metrics["quality_assurance"]["high_confidence_ratio"] = qa_metrics.get("high_confidence_ratio", 0)
            self.pipeline_metrics["quality_assurance"]["average_confidence"] = qa_metrics.get("average_confidence", 0)
            
            self.pipeline_metrics["steps_completed"].append("quality_assurance")
        
        except Exception as e:
            logger.error(f"Error in quality assurance step: {e}")
            self.pipeline_metrics["steps_failed"].append("quality_assurance")
        
        # Update metrics
        self.pipeline_metrics["quality_assurance"]["duration"] = time.time() - start_time
        logger.info(f"Quality assurance step completed in {self.pipeline_metrics['quality_assurance']['duration']:.2f} seconds")
    
    def _run_mlops(self):
        """
        Run the MLOps step.
        """
        logger.info("Starting MLOps step")
        start_time = time.time()
        
        try:
            # Check if MLOps is enabled
            if not self.config.get("mlops", {}).get("enabled", True):
                logger.info("MLOps is disabled in configuration")
                return
            
            # Build command
            cmd = [
                "python", "mlops_manager.py",
                "--config", self.config_file,
                "--output-dir", os.path.join(self.output_dir, "mlops"),
                "--start-run",
                "--log-model",
                "--generate-report",
                "--end-run"
            ]
            
            # Run MLOps
            logger.info(f"Running command: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info("MLOps completed successfully")
                self.pipeline_metrics["steps_completed"].append("mlops")
            else:
                logger.error(f"MLOps failed with code {result.returncode}")
                logger.error(f"Error: {result.stderr}")
                self.pipeline_metrics["steps_failed"].append("mlops")
        
        except Exception as e:
            logger.error(f"Error in MLOps step: {e}")
            self.pipeline_metrics["steps_failed"].append("mlops")
        
        # Update metrics
        self.pipeline_metrics["mlops"]["duration"] = time.time() - start_time
        logger.info(f"MLOps step completed in {self.pipeline_metrics['mlops']['duration']:.2f} seconds")
    
    def _save_pipeline_metrics(self):
        """
        Save pipeline metrics to file.
        """
        metrics_file = os.path.join(self.output_dir, "pipeline_metrics.json")
        
        try:
            with open(metrics_file, 'w') as f:
                json.dump(self.pipeline_metrics, f, indent=2)
            
            logger.info(f"Saved pipeline metrics to {metrics_file}")
        except Exception as e:
            logger.error(f"Error saving pipeline metrics: {e}")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Automated labeling pipeline for NoahAI")
    parser.add_argument("--config", default="automated_labeling_config.json", help="Path to configuration file")
    parser.add_argument("--output-dir", default="data/automated_labeling_pipeline", help="Directory for output")
    parser.add_argument("--step", choices=["data", "labeling", "update", "verification", "active", "qa", "mlops", "all"], default="all", help="Step to run")
    args = parser.parse_args()
    
    # Create pipeline
    pipeline = AutomatedLabelingPipeline(
        config_file=args.config,
        output_dir=args.output_dir
    )
    
    # Run specific step or all steps
    if args.step == "all":
        pipeline.run_pipeline()
    elif args.step == "data":
        pipeline._run_data_generation()
    elif args.step == "labeling":
        pipeline._run_automated_labeling()
    elif args.step == "update":
        pipeline._run_model_update()
    elif args.step == "verification":
        pipeline._run_human_verification()
    elif args.step == "active":
        pipeline._run_active_learning()
    elif args.step == "qa":
        pipeline._run_quality_assurance()
    elif args.step == "mlops":
        pipeline._run_mlops()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
