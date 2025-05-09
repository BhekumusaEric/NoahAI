"""
Simple Model for NoahAI
"""

import random
import json
import os
from src.utils.nlp_utils import NLPUtils

class SimpleResponseModel:
    """
    A simple model that generates responses based on pattern matching and templates.
    """
    
    def __init__(self, responses_file=None):
        """
        Initialize the model.
        
        Args:
            responses_file (str): Path to a JSON file with response templates
        """
        self.responses = self._load_responses(responses_file)
        self.nlp_utils = NLPUtils()
        self.conversation_history = []
        
    def _load_responses(self, responses_file):
        """
        Load response templates from a JSON file or use defaults.
        
        Args:
            responses_file (str): Path to a JSON file with response templates
            
        Returns:
            dict: A dictionary of response templates
        """
        try:
            if responses_file and os.path.exists(responses_file):
                with open(responses_file, "r") as f:
                    return json.load(f)
        except Exception as e:
            print(f"Error loading responses: {e}")
            
        # Default responses if file doesn't exist or has an error
        return {
            "greeting": ["Hello!", "Hi there!", "Greetings!"],
            "farewell": ["Goodbye!", "See you later!", "Until next time!"],
            "unknown": ["I'm not sure I understand.", "I don't know how to respond to that."]
        }
    
    def generate_response(self, user_input, user_name=None):
        """
        Generate a response based on the user input.
        
        Args:
            user_input (str): The user's input text
            user_name (str): The user's name for personalization
            
        Returns:
            str: The generated response
        """
        # Add to conversation history
        self.conversation_history.append(("user", user_input))
        
        # Generate a response using NLP utils
        response = self.nlp_utils.generate_response(user_input, self.responses)
        
        # Personalize the response if we have a user name
        if user_name and "{user_name}" in response:
            response = response.format(user_name=user_name)
            
        # Add to conversation history
        self.conversation_history.append(("ai", response))
        
        return response
    
    def get_conversation_history(self):
        """
        Get the conversation history.
        
        Returns:
            list: A list of (speaker, text) tuples
        """
        return self.conversation_history
    
    def clear_conversation_history(self):
        """
        Clear the conversation history.
        """
        self.conversation_history = []
