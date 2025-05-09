#!/usr/bin/env python3
"""
Massive Data Generator for NoahAI

This script generates large volumes of training data for NoahAI's deep learning model.
It supports:
- Synthetic data generation
- Data augmentation
- Parallel data processing
- Batch generation for distributed training
"""

import os
import sys
import json
import random
import argparse
import logging
import numpy as np
import multiprocessing as mp
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Any, Optional, Union
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/data_generator.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("data_generator")

# Import templates from existing generator if available
try:
    from generate_training_data import TEMPLATES
    TEMPLATES_AVAILABLE = True
except ImportError:
    TEMPLATES_AVAILABLE = False
    logger.warning("Could not import templates from generate_training_data.py. Using default templates.")
    # Define basic templates
    TEMPLATES = {
        "greeting": {
            "user_inputs": [
                "Hello", "Hi", "Hey there", "Good morning", "Good afternoon",
                "Good evening", "Greetings", "Hi NoahAI", "Hello there"
            ],
            "ai_responses": [
                "Hello, {user_name}! How can I help you today?",
                "Hi there, {user_name}! What can I do for you?",
                "Greetings, {user_name}! How may I assist you?"
            ]
        },
        "farewell": {
            "user_inputs": [
                "Goodbye", "Bye", "See you later", "Take care", "Farewell",
                "I'm leaving now", "Until next time", "Bye bye", "Have a good day"
            ],
            "ai_responses": [
                "Goodbye, {user_name}! Have a great day!",
                "See you later, {user_name}!",
                "Take care, {user_name}! Come back soon!"
            ]
        }
    }

class MassiveDataGenerator:
    """
    Generator for creating massive training datasets for NoahAI.
    """
    
    def __init__(self, templates=None, output_dir="data", max_rating=5):
        """
        Initialize the data generator.
        
        Args:
            templates (dict): Templates for generating data
            output_dir (str): Directory to save generated data
            max_rating (int): Maximum rating value
        """
        self.templates = templates or TEMPLATES
        self.output_dir = output_dir
        self.max_rating = max_rating
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize statistics
        self.stats = {
            "total_generated": 0,
            "by_category": {},
            "start_time": datetime.now(),
            "end_time": None
        }
    
    def generate_variations(self, text, num_variations=5):
        """
        Generate variations of a text by adding/removing words, changing word order, etc.
        
        Args:
            text (str): Original text
            num_variations (int): Number of variations to generate
            
        Returns:
            list: List of text variations
        """
        variations = [text]
        
        # Split text into words
        words = text.split()
        
        # Skip if too few words
        if len(words) < 3:
            return variations
        
        # Generate variations
        for _ in range(num_variations):
            variation_type = random.choice(["swap", "remove", "add", "replace"])
            
            if variation_type == "swap" and len(words) >= 2:
                # Swap two adjacent words
                idx = random.randint(0, len(words) - 2)
                new_words = words.copy()
                new_words[idx], new_words[idx + 1] = new_words[idx + 1], new_words[idx]
                variations.append(" ".join(new_words))
            
            elif variation_type == "remove" and len(words) >= 3:
                # Remove a non-essential word
                idx = random.randint(0, len(words) - 1)
                new_words = words.copy()
                new_words.pop(idx)
                variations.append(" ".join(new_words))
            
            elif variation_type == "add":
                # Add a filler word
                filler_words = ["please", "maybe", "actually", "just", "really", "basically", "simply"]
                filler = random.choice(filler_words)
                idx = random.randint(0, len(words))
                new_words = words.copy()
                new_words.insert(idx, filler)
                variations.append(" ".join(new_words))
            
            elif variation_type == "replace":
                # Replace a word with a synonym
                synonyms = {
                    "hello": ["hi", "hey", "greetings"],
                    "goodbye": ["bye", "farewell", "see you"],
                    "help": ["assist", "aid", "support"],
                    "good": ["great", "nice", "excellent"],
                    "bad": ["poor", "terrible", "awful"],
                    "happy": ["glad", "pleased", "delighted"],
                    "sad": ["unhappy", "upset", "disappointed"]
                }
                
                for i, word in enumerate(words):
                    if word.lower() in synonyms:
                        new_words = words.copy()
                        new_words[i] = random.choice(synonyms[word.lower()])
                        variations.append(" ".join(new_words))
                        break
        
        # Remove duplicates
        variations = list(set(variations))
        
        return variations
    
    def generate_batch(self, category, batch_size=100, include_variations=True):
        """
        Generate a batch of training data for a specific category.
        
        Args:
            category (str): Category to generate data for
            batch_size (int): Number of entries to generate
            include_variations (bool): Whether to include text variations
            
        Returns:
            list: List of generated data entries
        """
        if category not in self.templates:
            logger.warning(f"Category '{category}' not found in templates")
            return []
        
        templates = self.templates[category]
        user_inputs = templates["user_inputs"]
        ai_responses = templates["ai_responses"]
        
        # Generate expanded inputs with variations if requested
        if include_variations:
            expanded_inputs = []
            for input_text in user_inputs:
                variations = self.generate_variations(input_text)
                expanded_inputs.extend(variations)
            user_inputs = expanded_inputs
        
        # Generate batch
        batch = []
        timestamp = datetime.now()
        
        for _ in range(batch_size):
            # Go back in time a bit for each entry
            timestamp -= timedelta(minutes=random.randint(5, 30))
            
            # Select random user input and AI response
            user_input = random.choice(user_inputs)
            ai_response = random.choice(ai_responses)
            
            # Assign a rating (weighted towards higher ratings)
            weights = [0.05, 0.1, 0.15, 0.3, 0.4]  # 5% 1-star, 40% 5-star
            rating = random.choices(range(1, self.max_rating + 1), weights=weights)[0]
            
            # Create entry
            entry = {
                "user_input": user_input,
                "ai_response": ai_response.replace("{user_name}", "User"),
                "rating": rating,
                "category": category,
                "timestamp": timestamp.strftime("%Y-%m-%dT%H:%M:%S.%f"),
                "is_synthetic": True
            }
            
            # Add reinforcement learning reward for higher ratings
            if rating >= 4:
                entry["rl_reward"] = rating / 4  # Scale to ~1.0-1.25 for 4-5 ratings
            
            batch.append(entry)
        
        # Update statistics
        self.stats["total_generated"] += len(batch)
        if category not in self.stats["by_category"]:
            self.stats["by_category"][category] = 0
        self.stats["by_category"][category] += len(batch)
        
        return batch
    
    def generate_massive_dataset(self, entries_per_category=1000, num_processes=None, chunk_size=100):
        """
        Generate a massive dataset using parallel processing.
        
        Args:
            entries_per_category (int): Number of entries per category
            num_processes (int): Number of processes to use (default: CPU count)
            chunk_size (int): Size of each processing chunk
            
        Returns:
            list: Complete dataset
        """
        # Determine number of processes
        if num_processes is None:
            num_processes = mp.cpu_count()
        
        logger.info(f"Generating massive dataset with {entries_per_category} entries per category")
        logger.info(f"Using {num_processes} processes with chunk size {chunk_size}")
        
        # Prepare tasks
        tasks = []
        for category in self.templates.keys():
            # Split into chunks
            num_chunks = (entries_per_category + chunk_size - 1) // chunk_size
            for _ in range(num_chunks):
                actual_chunk_size = min(chunk_size, entries_per_category - len(tasks) * chunk_size)
                if actual_chunk_size > 0:
                    tasks.append((category, actual_chunk_size))
        
        # Create process pool
        with mp.Pool(processes=num_processes) as pool:
            # Map tasks to processes with progress bar
            results = list(tqdm(
                pool.starmap(self._generate_chunk, tasks),
                total=len(tasks),
                desc="Generating data chunks"
            ))
        
        # Combine results
        dataset = []
        for result in results:
            dataset.extend(result)
        
        # Update statistics
        self.stats["end_time"] = datetime.now()
        self.stats["duration"] = (self.stats["end_time"] - self.stats["start_time"]).total_seconds()
        self.stats["entries_per_second"] = self.stats["total_generated"] / self.stats["duration"]
        
        logger.info(f"Generated {len(dataset)} total entries")
        logger.info(f"Generation took {self.stats['duration']:.2f} seconds")
        logger.info(f"Generation rate: {self.stats['entries_per_second']:.2f} entries/second")
        
        return dataset
    
    def _generate_chunk(self, category, chunk_size):
        """
        Generate a chunk of data (used by multiprocessing).
        
        Args:
            category (str): Category to generate data for
            chunk_size (int): Number of entries to generate
            
        Returns:
            list: Generated data chunk
        """
        return self.generate_batch(category, batch_size=chunk_size)
    
    def save_dataset(self, dataset, filename="massive_training_data.json"):
        """
        Save the generated dataset to a file.
        
        Args:
            dataset (list): Dataset to save
            filename (str): Output filename
            
        Returns:
            str: Path to the saved file
        """
        output_path = os.path.join(self.output_dir, filename)
        
        with open(output_path, 'w') as f:
            json.dump(dataset, f, indent=2)
        
        logger.info(f"Saved dataset with {len(dataset)} entries to {output_path}")
        
        # Save statistics
        stats_path = os.path.join(self.output_dir, "generation_stats.json")
        with open(stats_path, 'w') as f:
            json.dump(self.stats, f, indent=2)
        
        return output_path

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Generate massive training data for NoahAI")
    parser.add_argument("--entries", type=int, default=1000, help="Number of entries per category")
    parser.add_argument("--processes", type=int, default=None, help="Number of processes to use")
    parser.add_argument("--chunk-size", type=int, default=100, help="Size of each processing chunk")
    parser.add_argument("--output", default="massive_training_data.json", help="Output filename")
    parser.add_argument("--output-dir", default="data", help="Output directory")
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create generator
    generator = MassiveDataGenerator(output_dir=args.output_dir)
    
    # Generate dataset
    dataset = generator.generate_massive_dataset(
        entries_per_category=args.entries,
        num_processes=args.processes,
        chunk_size=args.chunk_size
    )
    
    # Save dataset
    generator.save_dataset(dataset, filename=args.output)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
