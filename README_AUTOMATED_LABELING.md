# Automated Data Labeling for NoahAI

This system implements automated data labeling for NoahAI, allowing the model to automatically label new data with high confidence, reducing the need for manual labeling and accelerating the training pipeline.

## Features

- **Confidence-based Labeling**: Automatically labels data based on model confidence
- **Ensemble Consensus**: Uses multiple models to reach a consensus on labels
- **Human-in-the-Loop Verification**: Routes low-confidence examples for human verification
- **Active Learning Integration**: Prioritizes uncertain examples for manual labeling
- **Incremental Model Updates**: Continuously improves the model with newly labeled data
- **Quality Assurance**: Monitors labeling quality and model performance
- **MLOps Integration**: Tracks metrics and experiments

## Components

1. **Automated Labeling Manager** (`automated_labeling_manager.py`): Core component that handles the automated labeling process
2. **Human Verification Interface** (`human_verification_interface.py`): Simple CLI for human verification of low-confidence examples
3. **Unlabeled Data Generator** (`generate_unlabeled_data.py`): Generates synthetic unlabeled data for testing
4. **Automated Labeling Pipeline** (`run_automated_labeling_pipeline.py`): Orchestrates the complete labeling pipeline
5. **Configuration** (`automated_labeling_config.json`): Configuration file for the labeling system

## How It Works

1. **Data Collection**: Collect or generate unlabeled data
2. **Automated Labeling**: Use the trained model to predict labels for the data
3. **Confidence Filtering**:
   - High-confidence examples are automatically labeled
   - Low-confidence examples are routed for human verification
4. **Model Update**: The model is incrementally updated with high-confidence examples
5. **Human Verification**: Low-confidence examples are verified by humans
6. **Active Learning**: The most informative examples are prioritized for manual labeling
7. **Quality Assurance**: The labeling quality is monitored and evaluated

## Usage

### 1. Run the Complete Automated Labeling Pipeline

```bash
python run_automated_labeling_pipeline.py --config automated_labeling_config.json
```

This will execute all steps in sequence:
1. Generate or collect unlabeled data
2. Label data using the trained model
3. Separate high and low confidence examples
4. Update the model with high confidence examples
5. Send low confidence examples for human verification
6. Integrate with active learning for uncertain examples
7. Perform quality assurance
8. Track metrics with MLOps tools

### 2. Run Individual Components

You can also run specific components of the pipeline:

```bash
# Generate unlabeled data
python generate_unlabeled_data.py --num-examples 1000 --output data/unlabeled_data.json

# Run automated labeling
python automated_labeling_manager.py --config automated_labeling_config.json --data data/unlabeled_data.json

# Perform human verification
python human_verification_interface.py --input data/human_verification.json
```

### 3. Integrate with NoahAI Training Pipeline

The automated labeling system is integrated with the main NoahAI training pipeline:

```bash
# Run only the automated labeling step
python run_massive_training.py --step labeling

# Run the complete pipeline including automated labeling
python run_massive_training.py
```

## Configuration

You can customize the automated labeling system by editing the `automated_labeling_config.json` file:

```json
{
  "model": {
    "model_type": "advanced_lstm",
    "max_words": 10000,
    "max_sequence_length": 100,
    "use_reinforcement_learning": true
  },
  "labeling": {
    "confidence_threshold": 0.9,
    "use_ensemble": true,
    "ensemble_size": 3,
    "ensemble_agreement_threshold": 0.7,
    "human_verification_threshold": 0.7,
    "max_examples_per_batch": 1000,
    "save_confidence": true,
    "incremental_update": {
      "enabled": true,
      "update_frequency": 500,
      "min_examples_for_update": 100,
      "epochs": 3,
      "batch_size": 32
    }
  },
  "quality_assurance": {
    "enabled": true,
    "validation_split": 0.1,
    "min_accuracy_threshold": 0.8,
    "confusion_matrix": true,
    "cross_validation": false
  },
  "active_learning_integration": {
    "enabled": true,
    "strategy": "uncertainty",
    "batch_size": 100
  },
  "output": {
    "format": "json",
    "include_metadata": true,
    "separate_files_by_confidence": true
  }
}
```

## Key Parameters

- **confidence_threshold**: Minimum confidence for automatic labeling (0.0-1.0)
- **use_ensemble**: Whether to use an ensemble of models for consensus
- **ensemble_size**: Number of models in the ensemble
- **ensemble_agreement_threshold**: Minimum agreement ratio among ensemble models
- **human_verification_threshold**: Confidence below which human verification is required
- **incremental_update.enabled**: Whether to update the model incrementally with new labeled data
- **active_learning_integration.strategy**: Strategy for selecting examples for active learning

## Output Files

The system generates several output files:

- **high_confidence_[timestamp].json**: Automatically labeled examples with high confidence
- **low_confidence_[timestamp].json**: Examples with low confidence for human verification
- **verified_examples_[timestamp].json**: Examples verified by humans
- **corrected_examples_[timestamp].json**: Examples corrected by humans
- **labeling_metrics.json**: Metrics about the labeling process
- **quality_assurance_metrics.json**: Metrics about the labeling quality

## Best Practices

1. **Start with a High Confidence Threshold**: Begin with a high confidence threshold (e.g., 0.9) and gradually lower it as the model improves
2. **Use Ensemble Labeling**: Enable ensemble labeling for more reliable predictions
3. **Regularly Verify Low-Confidence Examples**: Regularly review and verify low-confidence examples to improve the model
4. **Monitor Labeling Quality**: Keep an eye on the quality assurance metrics to ensure high-quality labels
5. **Integrate with Active Learning**: Use active learning to prioritize the most informative examples for manual labeling
6. **Incrementally Update the Model**: Enable incremental model updates to continuously improve the model with new labeled data

## Advanced Usage

### Custom Ensemble Models

You can create custom ensemble models by placing them in subdirectories of the model directory:

```
data/model/ensemble_1/
data/model/ensemble_2/
data/model/ensemble_3/
```

### Custom Labeling Strategies

You can implement custom labeling strategies by modifying the `automated_labeling_manager.py` script:

1. Add a new method for your custom strategy
2. Update the `label_data` method to use your custom strategy
3. Update the configuration file to include your custom strategy

### Integration with External Systems

The automated labeling system can be integrated with external systems:

1. **Data Sources**: Connect to external data sources by modifying the data loading code
2. **Human Verification**: Integrate with external human verification systems by modifying the human verification interface
3. **MLOps Tools**: Connect to external MLOps tools by modifying the MLOps integration code
