#!/usr/bin/env python3
"""
Script to train the NoahAI model directly.
"""

import os
import sys
import json
import collections
from models.deep_learning_model import DeepLearningModel

def main():
    """
    Main function to train the model.
    """
    print("Starting model training...")

    # Create model directory if it doesn't exist
    os.makedirs("data/model", exist_ok=True)

    # Load feedback data
    feedback_path = "data/feedback.json"
    if os.path.exists(feedback_path):
        with open(feedback_path, 'r') as f:
            feedback_data = json.load(f)
        print(f"Loaded {len(feedback_data)} feedback entries from {feedback_path}")
    else:
        print(f"Feedback file {feedback_path} not found.")
        return 1

    # Analyze feedback data categories
    categories = collections.Counter()
    for entry in feedback_data:
        category = entry.get("category")
        if category:
            categories[category] += 1

    print("\nCategory distribution:")
    for category, count in categories.most_common():
        print(f"  - {category}: {count} entries")

    # Initialize the model
    model = DeepLearningModel(
        model_dir="data/model",
        responses_file="data/responses.json",
        model_type="advanced_lstm",
        use_reinforcement_learning=True
    )

    # Add feedback data to the model, grouped by category
    # This helps the model learn patterns within each category
    for category in categories:
        category_entries = [entry for entry in feedback_data if entry.get("category") == category]
        print(f"\nAdding {len(category_entries)} entries for category '{category}'")

        for entry in category_entries:
            # Store category in metadata if possible, otherwise just add the feedback
            try:
                model.add_feedback(
                    user_input=entry.get("user_input", ""),
                    ai_response=entry.get("ai_response", ""),
                    rating=entry.get("rating", 3),
                    metadata={"category": category}  # Try to store category in metadata
                )
            except TypeError:
                # If metadata parameter is not supported, fall back to basic add_feedback
                model.add_feedback(
                    user_input=entry.get("user_input", ""),
                    ai_response=entry.get("ai_response", ""),
                    rating=entry.get("rating", 3)
                )

    # Add any uncategorized entries
    uncategorized = [entry for entry in feedback_data if entry.get("category") is None]
    if uncategorized:
        print(f"\nAdding {len(uncategorized)} uncategorized entries")
        for entry in uncategorized:
            model.add_feedback(
                user_input=entry.get("user_input", ""),
                ai_response=entry.get("ai_response", ""),
                rating=entry.get("rating", 3)
            )

    print(f"\nTotal: Added {len(feedback_data)} feedback entries to the model")

    # Train the model
    print("\nTraining model...")
    try:
        # Try with eager execution enabled
        import tensorflow as tf
        tf.config.run_functions_eagerly(True)
        print("Enabled eager execution for TensorFlow")

        history = model.train(
            epochs=15,  # Increased epochs for better learning
            batch_size=16,  # Smaller batch size for better generalization
            use_early_stopping=True,
            use_transfer_learning=True  # Enable transfer learning
        )
    except Exception as e:
        print(f"Error during training: {e}")
        print("Trying with different parameters...")

        # Try with different parameters
        history = model.train(
            epochs=10,  # Reduced epochs
            batch_size=32,  # Larger batch size
            use_early_stopping=True,
            use_transfer_learning=False  # Disable transfer learning
        )

    if history:
        print("Training completed successfully!")
        print(f"Training history: {history}")

        # Save training metrics
        metrics_path = "data/training_metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(history, f, indent=2)
        print(f"Saved training metrics to {metrics_path}")
    else:
        print("Training failed or no feedback data available.")

    return 0

if __name__ == "__main__":
    sys.exit(main())
