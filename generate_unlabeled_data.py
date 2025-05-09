#!/usr/bin/env python3
"""
Generate Unlabeled Data for NoahAI

This script generates unlabeled data for testing the automated labeling system.
It creates realistic examples that can be used to evaluate the labeling performance.
"""

import os
import sys
import json
import random
import argparse
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/generate_unlabeled.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("generate_unlabeled")

# Sample templates for generating data
TEMPLATES = {
    "greeting": [
        "Hello there",
        "Hi, how are you?",
        "Good morning",
        "Hey, what's up?",
        "Greetings",
        "Hello, nice to meet you",
        "Hi everyone",
        "Hey there",
        "Good afternoon",
        "Good evening"
    ],
    "farewell": [
        "Goodbye",
        "See you later",
        "Bye for now",
        "Take care",
        "Until next time",
        "Farewell",
        "Have a nice day",
        "Bye bye",
        "See you soon",
        "Later"
    ],
    "thanks": [
        "Thank you",
        "Thanks a lot",
        "I appreciate it",
        "Thanks for your help",
        "Thank you so much",
        "Many thanks",
        "I'm grateful",
        "Thanks a million",
        "I owe you one",
        "Thank you kindly"
    ],
    "question": [
        "What time is it?",
        "How does this work?",
        "Can you help me?",
        "Where is the nearest store?",
        "When does the event start?",
        "Who is in charge here?",
        "Why is this happening?",
        "How much does it cost?",
        "What's the weather like today?",
        "How do I get to the station?"
    ],
    "complaint": [
        "This isn't working properly",
        "I'm not satisfied with the service",
        "The quality is poor",
        "This is taking too long",
        "I've been waiting for hours",
        "The product is defective",
        "I want to speak to a manager",
        "This is unacceptable",
        "I'm very disappointed",
        "This is not what I ordered"
    ]
}

def generate_variations(text, num_variations=3):
    """
    Generate variations of a text.
    
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

def generate_unlabeled_data(num_examples=1000, output_file="data/unlabeled_data.json"):
    """
    Generate unlabeled data.
    
    Args:
        num_examples (int): Number of examples to generate
        output_file (str): Output file path
        
    Returns:
        list: Generated examples
    """
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Generate examples
    examples = []
    
    for _ in range(num_examples):
        # Select a random category
        category = random.choice(list(TEMPLATES.keys()))
        
        # Select a random template
        template = random.choice(TEMPLATES[category])
        
        # Generate variations
        variations = generate_variations(template)
        
        # Select a random variation
        text = random.choice(variations)
        
        # Create example
        example = {
            "text": text,
            "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
        }
        
        examples.append(example)
    
    # Save examples to file
    with open(output_file, 'w') as f:
        json.dump(examples, f, indent=2)
    
    logger.info(f"Generated {len(examples)} unlabeled examples and saved to {output_file}")
    
    return examples

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Generate unlabeled data for NoahAI")
    parser.add_argument("--num-examples", type=int, default=1000, help="Number of examples to generate")
    parser.add_argument("--output", default="data/unlabeled_data.json", help="Output file path")
    args = parser.parse_args()
    
    # Generate unlabeled data
    generate_unlabeled_data(args.num_examples, args.output)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
