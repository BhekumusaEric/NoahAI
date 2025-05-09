#!/usr/bin/env python3
"""
Setup script for NoahAI

This script installs required packages and downloads necessary models.
"""

import os
import sys
import subprocess
import argparse

def install_requirements():
    """
    Install required packages from requirements.txt
    """
    print("Installing required packages...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    print("Required packages installed successfully.")

def download_nltk_data():
    """
    Download required NLTK data
    """
    print("Downloading NLTK data...")
    import nltk
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('maxent_ne_chunker')
    nltk.download('words')
    print("NLTK data downloaded successfully.")

def download_spacy_models():
    """
    Download required spaCy models
    """
    print("Downloading spaCy models...")
    subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
    subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_md"])
    print("spaCy models downloaded successfully.")

def download_transformers_models():
    """
    Download required transformers models
    """
    print("Downloading transformers models...")
    from transformers import AutoTokenizer, AutoModel
    
    # BERT for intent recognition
    print("Downloading BERT model...")
    AutoTokenizer.from_pretrained("bert-base-uncased")
    AutoModel.from_pretrained("bert-base-uncased")
    
    # DistilBERT for sentiment analysis
    print("Downloading DistilBERT model...")
    AutoTokenizer.from_pretrained("distilbert-base-uncased")
    AutoModel.from_pretrained("distilbert-base-uncased")
    
    print("Transformers models downloaded successfully.")

def create_directories():
    """
    Create necessary directories
    """
    print("Creating directories...")
    os.makedirs("data/model", exist_ok=True)
    os.makedirs("data/cache", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    os.makedirs("web/static", exist_ok=True)
    os.makedirs("web/templates", exist_ok=True)
    print("Directories created successfully.")

def setup_environment():
    """
    Set up environment variables
    """
    print("Setting up environment...")
    
    # Create .env file if it doesn't exist
    if not os.path.exists(".env"):
        with open(".env", "w") as f:
            f.write("# NoahAI Environment Variables\n")
            f.write("OPENWEATHER_API_KEY=your_api_key_here\n")
            f.write("NEWS_API_KEY=your_api_key_here\n")
            f.write("GOOGLE_CLIENT_ID=your_client_id_here\n")
            f.write("GOOGLE_CLIENT_SECRET=your_client_secret_here\n")
        print("Created .env file. Please update it with your API keys.")
    else:
        print(".env file already exists.")

def main():
    """
    Main function
    """
    parser = argparse.ArgumentParser(description="Setup script for NoahAI")
    parser.add_argument("--skip-downloads", action="store_true", help="Skip downloading models")
    args = parser.parse_args()
    
    try:
        install_requirements()
        create_directories()
        setup_environment()
        
        if not args.skip_downloads:
            download_nltk_data()
            download_spacy_models()
            download_transformers_models()
        
        print("\nSetup completed successfully!")
        print("You can now run NoahAI with: python noah_ai.py")
    
    except Exception as e:
        print(f"Error during setup: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
