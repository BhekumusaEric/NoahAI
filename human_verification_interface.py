#!/usr/bin/env python3
"""
Human Verification Interface for NoahAI

This script provides a simple command-line interface for human verification
of low-confidence examples from the automated labeling system.
"""

import os
import sys
import json
import argparse
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/human_verification.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("human_verification")

class HumanVerificationInterface:
    """
    Interface for human verification of labeled examples.
    """
    
    def __init__(self, input_file, output_dir="data/human_verified"):
        """
        Initialize the interface.
        
        Args:
            input_file (str): Path to file with examples for verification
            output_dir (str): Directory for verified examples
        """
        self.input_file = input_file
        self.output_dir = output_dir
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Load examples
        self.examples = self._load_examples()
        
        # Initialize metrics
        self.metrics = {
            "total_examples": len(self.examples),
            "verified_examples": 0,
            "corrected_examples": 0,
            "skipped_examples": 0,
            "start_time": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
            "end_time": None,
            "duration": 0
        }
    
    def _load_examples(self):
        """
        Load examples from file.
        
        Returns:
            list: List of examples
        """
        if not os.path.exists(self.input_file):
            logger.error(f"Input file not found: {self.input_file}")
            return []
        
        try:
            with open(self.input_file, 'r') as f:
                examples = json.load(f)
            
            logger.info(f"Loaded {len(examples)} examples from {self.input_file}")
            return examples
        
        except Exception as e:
            logger.error(f"Error loading examples: {e}")
            return []
    
    def run_verification(self):
        """
        Run the verification process.
        
        Returns:
            dict: Verification metrics
        """
        if not self.examples:
            logger.error("No examples to verify")
            return self.metrics
        
        verified_examples = []
        
        print("\n=== NoahAI Human Verification Interface ===")
        print(f"Total examples to verify: {len(self.examples)}")
        print("Instructions:")
        print("  - Review each example and its predicted category")
        print("  - Accept the category (press Enter)")
        print("  - Correct the category (type the correct category)")
        print("  - Skip the example (type 'skip')")
        print("  - Quit verification (type 'quit')")
        print("=" * 50)
        
        # Get available categories
        categories = set()
        for example in self.examples:
            if "category" in example:
                categories.add(example["category"])
        
        categories = sorted(list(categories))
        
        for i, example in enumerate(self.examples):
            print(f"\nExample {i+1}/{len(self.examples)}:")
            print(f"Text: {example['text']}")
            print(f"Predicted category: {example.get('category', 'unknown')}")
            print(f"Confidence: {example.get('confidence', 0):.2f}")
            
            if categories:
                print(f"Available categories: {', '.join(categories)}")
            
            response = input("Correct category (Enter to accept, 'skip' to skip, 'quit' to quit): ").strip()
            
            if response.lower() == "quit":
                print("Verification process terminated by user")
                break
            
            if response.lower() == "skip":
                self.metrics["skipped_examples"] += 1
                continue
            
            # Create verified example
            verified_example = example.copy()
            
            if response:
                # User provided a correction
                verified_example["category"] = response
                verified_example["corrected"] = True
                verified_example["original_category"] = example.get("category")
                self.metrics["corrected_examples"] += 1
            else:
                # User accepted the prediction
                verified_example["corrected"] = False
            
            verified_example["verified"] = True
            verified_example["verification_time"] = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
            
            verified_examples.append(verified_example)
            self.metrics["verified_examples"] += 1
            
            # Add new category to the list if it's not already there
            if response and response not in categories:
                categories.append(response)
                categories.sort()
        
        # Save verified examples
        if verified_examples:
            self._save_verified_examples(verified_examples)
        
        # Update metrics
        self.metrics["end_time"] = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
        start_time = datetime.strptime(self.metrics["start_time"], "%Y-%m-%dT%H:%M:%S")
        end_time = datetime.strptime(self.metrics["end_time"], "%Y-%m-%dT%H:%M:%S")
        self.metrics["duration"] = (end_time - start_time).total_seconds()
        
        # Save metrics
        self._save_metrics()
        
        return self.metrics
    
    def _save_verified_examples(self, verified_examples):
        """
        Save verified examples to file.
        
        Args:
            verified_examples (list): List of verified examples
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(self.output_dir, f"verified_examples_{timestamp}.json")
        
        try:
            with open(output_file, 'w') as f:
                json.dump(verified_examples, f, indent=2)
            
            logger.info(f"Saved {len(verified_examples)} verified examples to {output_file}")
            
            # Also save corrected examples separately
            corrected_examples = [ex for ex in verified_examples if ex.get("corrected", False)]
            
            if corrected_examples:
                corrected_file = os.path.join(self.output_dir, f"corrected_examples_{timestamp}.json")
                
                with open(corrected_file, 'w') as f:
                    json.dump(corrected_examples, f, indent=2)
                
                logger.info(f"Saved {len(corrected_examples)} corrected examples to {corrected_file}")
        
        except Exception as e:
            logger.error(f"Error saving verified examples: {e}")
    
    def _save_metrics(self):
        """
        Save metrics to file.
        """
        metrics_file = os.path.join(self.output_dir, "verification_metrics.json")
        
        try:
            with open(metrics_file, 'w') as f:
                json.dump(self.metrics, f, indent=2)
            
            logger.info(f"Saved metrics to {metrics_file}")
        except Exception as e:
            logger.error(f"Error saving metrics: {e}")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Human verification interface for NoahAI")
    parser.add_argument("--input", required=True, help="Path to file with examples for verification")
    parser.add_argument("--output-dir", default="data/human_verified", help="Directory for verified examples")
    args = parser.parse_args()
    
    # Create interface
    interface = HumanVerificationInterface(
        input_file=args.input,
        output_dir=args.output_dir
    )
    
    # Run verification
    metrics = interface.run_verification()
    
    # Print summary
    print("\n=== Verification Summary ===")
    print(f"Total examples: {metrics['total_examples']}")
    print(f"Verified examples: {metrics['verified_examples']}")
    print(f"Corrected examples: {metrics['corrected_examples']}")
    print(f"Skipped examples: {metrics['skipped_examples']}")
    print(f"Duration: {metrics['duration']:.2f} seconds")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
