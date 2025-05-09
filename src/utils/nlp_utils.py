"""
Natural Language Processing utilities for NoahAI

This module provides advanced NLP capabilities including:
- Tokenization and text preprocessing
- Keyword extraction
- Sentiment analysis
- Intent recognition
- Entity extraction
- Context management
- Topic modeling
- Semantic similarity
- Text embedding
- Conversation coherence analysis
"""

import re
import string
import random
import os
import json
import logging
import numpy as np
from collections import Counter
from datetime import datetime
from typing import List, Dict, Tuple, Any, Optional, Union, Set

# Import advanced NLP libraries
try:
    import nltk
    from nltk.tokenize import word_tokenize, sent_tokenize
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    from nltk.util import ngrams
    from nltk.metrics.distance import edit_distance
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    print("NLTK not available. Using basic tokenization.")

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    print("spaCy not available. Using basic NLP processing.")

try:
    import gensim
    from gensim import corpora
    from gensim.models import LdaModel, Word2Vec
    GENSIM_AVAILABLE = True
except ImportError:
    GENSIM_AVAILABLE = False
    print("Gensim not available. Using basic topic modeling.")

try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Transformers not available. Using basic NLP processing.")

# Import our advanced NLP components
try:
    from src.utils.advanced_nlp import SentimentAnalyzer, EntityRecognizer, IntentClassifier
    ADVANCED_NLP_AVAILABLE = True
except ImportError:
    ADVANCED_NLP_AVAILABLE = False
    print("Advanced NLP components not available. Using basic NLP processing.")

# Import multilingual support
try:
    from src.utils.multilingual import MultilingualProcessor
    MULTILINGUAL_AVAILABLE = True
except ImportError:
    MULTILINGUAL_AVAILABLE = False
    print("Multilingual support not available. Using English-only processing.")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/nlp.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("nlp_utils")

class NLPUtils:
    """
    Utility class for advanced natural language processing tasks.

    This class provides methods for:
    - Text preprocessing and tokenization
    - Keyword extraction
    - Sentiment analysis
    - Intent recognition
    - Entity extraction
    - Context management
    """

    def __init__(self, use_advanced_nlp=True, models_dir="data/models", use_multilingual=True):
        """
        Initialize the NLP utilities.

        Args:
            use_advanced_nlp (bool): Whether to use advanced NLP capabilities
            models_dir (str): Directory to store/load NLP models
            use_multilingual (bool): Whether to use multilingual capabilities
        """
        self.use_advanced_nlp = use_advanced_nlp
        self.models_dir = models_dir
        self.use_multilingual = use_multilingual

        # Create models directory if it doesn't exist
        os.makedirs(models_dir, exist_ok=True)

        # Initialize NLP components
        self._initialize_nlp_components()

        # Load intent and entity configurations
        self.intents = self._load_intents()
        self.entities = self._load_entities()

        # Initialize advanced NLP components
        self.sentiment_analyzer = None
        self.entity_recognizer = None
        self.intent_classifier = None

        if ADVANCED_NLP_AVAILABLE and self.use_advanced_nlp:
            try:
                self.sentiment_analyzer = SentimentAnalyzer(model_dir=os.path.join(models_dir, "sentiment"))
                self.entity_recognizer = EntityRecognizer(model_dir=os.path.join(models_dir, "ner"))
                self.intent_classifier = IntentClassifier(model_dir=os.path.join(models_dir, "intent"))
                logger.info("Initialized advanced NLP components.")
            except Exception as e:
                logger.error(f"Error initializing advanced NLP components: {e}")

        # Initialize multilingual support
        self.multilingual = None
        if MULTILINGUAL_AVAILABLE and self.use_multilingual:
            try:
                self.multilingual = MultilingualProcessor(models_dir=os.path.join(models_dir, "multilingual"))
                logger.info("Initialized multilingual support.")
            except Exception as e:
                logger.error(f"Error initializing multilingual support: {e}")

        # Initialize context manager
        self.context = {}

    def _initialize_nlp_components(self):
        """
        Initialize NLP components based on available libraries.
        """
        # Initialize NLTK components
        if NLTK_AVAILABLE and self.use_advanced_nlp:
            self.lemmatizer = WordNetLemmatizer()
            self.stop_words = set(stopwords.words('english'))
            logger.info("NLTK components initialized.")
        else:
            self.lemmatizer = None
            self.stop_words = set()

        # Initialize spaCy model
        if SPACY_AVAILABLE and self.use_advanced_nlp:
            try:
                self.nlp = spacy.load("en_core_web_md")
                logger.info("spaCy model loaded.")
            except OSError:
                logger.warning("Could not load spaCy model. Run 'python -m spacy download en_core_web_md' first.")
                self.nlp = None
        else:
            self.nlp = None

        # Initialize transformers models
        if TRANSFORMERS_AVAILABLE and self.use_advanced_nlp:
            try:
                # Sentiment analysis pipeline
                self.sentiment_pipeline = pipeline("sentiment-analysis")

                # Intent classification model
                self.intent_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
                self.intent_model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")

                logger.info("Transformers models loaded.")
            except Exception as e:
                logger.error(f"Error loading transformers models: {e}")
                self.sentiment_pipeline = None
                self.intent_tokenizer = None
                self.intent_model = None
        else:
            self.sentiment_pipeline = None
            self.intent_tokenizer = None
            self.intent_model = None

    def _load_intents(self):
        """
        Load intent configurations from file.

        Returns:
            dict: Intent configurations
        """
        intent_file = os.path.join(self.models_dir, "intents.json")

        if os.path.exists(intent_file):
            try:
                with open(intent_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading intents: {e}")

        # Default intents
        return {
            "greeting": {
                "patterns": ["hello", "hi", "hey", "good morning", "good afternoon", "good evening"],
                "responses": ["Hello!", "Hi there!", "Hey!"]
            },
            "farewell": {
                "patterns": ["goodbye", "bye", "see you", "see you later", "farewell"],
                "responses": ["Goodbye!", "See you later!", "Farewell!"]
            },
            "thanks": {
                "patterns": ["thank you", "thanks", "appreciate it", "thank you so much"],
                "responses": ["You're welcome!", "Happy to help!", "My pleasure!"]
            },
            "help": {
                "patterns": ["help", "help me", "can you help", "need assistance", "support"],
                "responses": ["How can I help you?", "What do you need help with?", "I'm here to assist you."]
            }
        }

    def _load_entities(self):
        """
        Load entity configurations from file.

        Returns:
            dict: Entity configurations
        """
        entity_file = os.path.join(self.models_dir, "entities.json")

        if os.path.exists(entity_file):
            try:
                with open(entity_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading entities: {e}")

        # Default entities
        return {
            "date": {
                "patterns": [
                    r"\d{1,2}/\d{1,2}/\d{2,4}",
                    r"\d{1,2}-\d{1,2}-\d{2,4}",
                    r"(january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2}(st|nd|rd|th)?,?\s+\d{2,4}"
                ]
            },
            "time": {
                "patterns": [
                    r"\d{1,2}:\d{2}\s*(am|pm)?",
                    r"\d{1,2}\s*(am|pm)"
                ]
            },
            "email": {
                "patterns": [
                    r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
                ]
            },
            "phone": {
                "patterns": [
                    r"\+?\d{1,3}[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}"
                ]
            }
        }

    def tokenize(self, text):
        """
        Tokenize text into words using advanced NLP libraries if available.

        Args:
            text (str): The input text to tokenize

        Returns:
            list: A list of tokens (words)
        """
        # Check for empty text
        if not text:
            return []

        try:
            # Use NLTK for advanced tokenization (now that we have all NLTK data)
            if NLTK_AVAILABLE and self.use_advanced_nlp:
                # Tokenize the text into words
                tokens = word_tokenize(text)

                # Convert to lowercase
                tokens = [token.lower() for token in tokens]

                # Remove punctuation but keep important tokens
                tokens = [token for token in tokens if token not in string.punctuation or token in ['?', '!', '.', ',']]

                # Remove stopwords if available
                if hasattr(self, 'stop_words') and self.stop_words:
                    tokens = [token for token in tokens if token not in self.stop_words]

                # Lemmatize tokens if lemmatizer is available
                if self.lemmatizer:
                    # Get part-of-speech tags for better lemmatization
                    if 'averaged_perceptron_tagger' in nltk.data.path:
                        try:
                            pos_tags = nltk.pos_tag(tokens)
                            lemmatized_tokens = []

                            for token, pos in pos_tags:
                                # Convert POS tag to WordNet format
                                if pos.startswith('J'):
                                    wordnet_pos = 'a'  # adjective
                                elif pos.startswith('V'):
                                    wordnet_pos = 'v'  # verb
                                elif pos.startswith('N'):
                                    wordnet_pos = 'n'  # noun
                                elif pos.startswith('R'):
                                    wordnet_pos = 'r'  # adverb
                                else:
                                    wordnet_pos = 'n'  # default to noun

                                # Lemmatize with POS tag
                                lemmatized_tokens.append(self.lemmatizer.lemmatize(token, pos=wordnet_pos))

                            tokens = lemmatized_tokens
                        except Exception as e:
                            logger.warning(f"Error in POS tagging: {e}. Falling back to basic lemmatization.")
                            tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
                    else:
                        # Basic lemmatization without POS tags
                        tokens = [self.lemmatizer.lemmatize(token) for token in tokens]

                return tokens

            # Use spaCy for tokenization if available
            if SPACY_AVAILABLE and self.nlp and self.use_advanced_nlp:
                try:
                    doc = self.nlp(text)
                    # Get lemmatized tokens with POS filtering
                    tokens = [token.lemma_.lower() for token in doc
                             if not token.is_punct and not token.is_space and not token.is_stop]
                    return tokens
                except Exception as e:
                    logger.warning(f"Error in spaCy tokenization: {e}. Falling back to basic tokenization.")

            # Fallback to basic tokenization
            text = text.lower()
            text = text.translate(str.maketrans('', '', string.punctuation))
            tokens = text.split()

            return tokens

        except Exception as e:
            logger.warning(f"Error in tokenization: {e}. Falling back to basic tokenization.")
            # Simple fallback tokenization
            return text.lower().split()

    def extract_keywords(self, text, top_n=5):
        """
        Extract the most important keywords from text using advanced NLP techniques.

        Args:
            text (str): The input text
            top_n (int): Number of top keywords to return

        Returns:
            list: A list of the top keywords with their importance scores
        """
        if not text:
            return []

        # Use NLTK for advanced keyword extraction
        if NLTK_AVAILABLE and self.use_advanced_nlp:
            try:
                # Tokenize and get POS tags
                tokens = word_tokenize(text.lower())
                pos_tags = nltk.pos_tag(tokens)

                # Extract nouns, proper nouns, and adjectives as potential keywords
                keywords = []
                for token, pos in pos_tags:
                    # Filter by part of speech and length
                    if (pos.startswith('NN') or pos.startswith('JJ')) and len(token) > 2:
                        # Skip stopwords
                        if token not in self.stop_words and token not in string.punctuation:
                            keywords.append(token)

                # Use collocations (phrases that occur together frequently)
                if len(tokens) > 3:
                    try:
                        # Create a text object for collocation finding
                        text_obj = nltk.Text(tokens)
                        # Find bigram collocations
                        bigram_measures = nltk.collocations.BigramAssocMeasures()
                        finder = nltk.collocations.BigramCollocationFinder.from_words(tokens)
                        # Filter out stopwords
                        finder.apply_word_filter(lambda w: w in self.stop_words or len(w) < 3)
                        # Get top collocations by PMI (Pointwise Mutual Information)
                        collocations = finder.nbest(bigram_measures.pmi, 5)
                        # Add collocations to keywords
                        for w1, w2 in collocations:
                            keywords.append(f"{w1} {w2}")
                    except Exception as e:
                        logger.warning(f"Error in collocation finding: {e}")

                # Use frequency distribution to rank keywords
                fdist = nltk.FreqDist(keywords)
                top_keywords = [word for word, _ in fdist.most_common(top_n)]

                return top_keywords
            except Exception as e:
                logger.warning(f"Error in NLTK keyword extraction: {e}. Trying spaCy.")

        # Use spaCy for keyword extraction if available
        if SPACY_AVAILABLE and self.nlp and self.use_advanced_nlp:
            try:
                doc = self.nlp(text)

                # Extract nouns, proper nouns, and adjectives as potential keywords
                keywords = []
                for token in doc:
                    if (token.pos_ in ['NOUN', 'PROPN', 'ADJ'] and
                        not token.is_stop and
                        len(token.text) > 1):
                        keywords.append((token.text.lower(), token.vector_norm))

                # Sort by importance (vector norm) and take top_n
                keywords.sort(key=lambda x: x[1], reverse=True)
                return [keyword for keyword, _ in keywords[:top_n]]
            except Exception as e:
                logger.warning(f"Error in spaCy keyword extraction: {e}. Falling back to basic extraction.")

        # Fallback to basic keyword extraction
        # Tokenize the text
        tokens = self.tokenize(text)

        # Filter out stop words
        filtered_tokens = [token for token in tokens if token not in self.stop_words]

        # Count the occurrences of each token
        token_counts = Counter(filtered_tokens)

        # Get the top N most common tokens
        top_keywords = [keyword for keyword, _ in token_counts.most_common(top_n)]

        return top_keywords

    def sentiment_analysis(self, text):
        """
        Perform sentiment analysis on text using advanced NLP techniques.

        Args:
            text (str): The input text

        Returns:
            dict: Sentiment analysis results with sentiment label and confidence score
        """
        if not text:
            return {"label": "neutral", "score": 0.5}

        # Use NLTK's VADER sentiment analyzer if available
        if NLTK_AVAILABLE and self.use_advanced_nlp:
            try:
                # Import VADER sentiment analyzer
                from nltk.sentiment.vader import SentimentIntensityAnalyzer

                # Initialize the analyzer
                sid = SentimentIntensityAnalyzer()

                # Get sentiment scores
                scores = sid.polarity_scores(text)

                # Determine sentiment label based on compound score
                if scores['compound'] >= 0.05:
                    label = "positive"
                elif scores['compound'] <= -0.05:
                    label = "negative"
                else:
                    label = "neutral"

                # Map the compound score to a 0-1 range
                score = (scores['compound'] + 1) / 2

                return {
                    "label": label,
                    "score": score,
                    "details": {
                        "positive": scores['pos'],
                        "negative": scores['neg'],
                        "neutral": scores['neu'],
                        "compound": scores['compound']
                    }
                }
            except Exception as e:
                logger.warning(f"Error in NLTK VADER sentiment analysis: {e}. Trying other methods.")

        # Use our dedicated sentiment analyzer if available
        if self.sentiment_analyzer and self.use_advanced_nlp:
            try:
                return self.sentiment_analyzer.analyze(text)
            except Exception as e:
                logger.warning(f"Error in advanced sentiment analysis: {e}. Falling back to basic analysis.")

        # Use transformers for sentiment analysis if available
        if TRANSFORMERS_AVAILABLE and self.sentiment_pipeline and self.use_advanced_nlp:
            try:
                result = self.sentiment_pipeline(text)[0]

                # Map the result to our format
                sentiment = {
                    "label": result["label"].lower(),
                    "score": result["score"]
                }

                return sentiment
            except Exception as e:
                logger.warning(f"Error in transformers sentiment analysis: {e}. Falling back to basic analysis.")

        # Use spaCy for sentiment analysis if available
        if SPACY_AVAILABLE and self.nlp and self.use_advanced_nlp:
            try:
                doc = self.nlp(text)

                # Calculate sentiment based on token polarity
                sentiment_score = 0
                for token in doc:
                    # Use the token's sentiment attribute if available
                    if hasattr(token, 'sentiment'):
                        sentiment_score += token.sentiment

                # Normalize the score to [-1, 1] range
                if len(doc) > 0:
                    sentiment_score = sentiment_score / len(doc)

                # Map the score to a label
                if sentiment_score > 0.1:
                    label = "positive"
                elif sentiment_score < -0.1:
                    label = "negative"
                else:
                    label = "neutral"

                return {
                    "label": label,
                    "score": (sentiment_score + 1) / 2  # Convert to [0, 1] range
                }
            except Exception as e:
                logger.warning(f"Error in spaCy sentiment analysis: {e}. Falling back to basic analysis.")

        # Use NLTK's text classification approach if VADER is not available
        if NLTK_AVAILABLE and self.use_advanced_nlp:
            try:
                # Tokenize and get word features
                tokens = word_tokenize(text.lower())

                # Define positive and negative word lists using NLTK's opinion lexicon if available
                try:
                    from nltk.corpus import opinion_lexicon
                    positive_words = set(opinion_lexicon.positive())
                    negative_words = set(opinion_lexicon.negative())
                except:
                    # Fallback to basic word lists
                    positive_words = {'good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic',
                                     'happy', 'glad', 'positive', 'nice', 'love', 'like', 'awesome',
                                     'beautiful', 'best', 'better', 'enjoy', 'thanks', 'thank'}
                    negative_words = {'bad', 'terrible', 'awful', 'horrible', 'poor', 'negative',
                                     'sad', 'unhappy', 'angry', 'upset', 'hate', 'dislike', 'worst',
                                     'worse', 'disappointing', 'disappointed', 'annoying', 'annoyed'}

                # Count positive and negative words
                positive_count = sum(1 for token in tokens if token in positive_words)
                negative_count = sum(1 for token in tokens if token in negative_words)

                # Consider negations (e.g., "not good" should be negative)
                for i in range(len(tokens) - 1):
                    if tokens[i] in {'not', 'no', 'never', "don't", "doesn't", "didn't", "won't", "wouldn't", "couldn't", "shouldn't"}:
                        if tokens[i+1] in positive_words:
                            positive_count -= 1
                            negative_count += 1
                        elif tokens[i+1] in negative_words:
                            negative_count -= 1
                            positive_count += 1

                # Calculate sentiment score
                total_count = positive_count + negative_count
                if total_count > 0:
                    sentiment_score = (positive_count - negative_count) / total_count
                else:
                    sentiment_score = 0

                # Determine sentiment
                if sentiment_score > 0.05:
                    label = "positive"
                elif sentiment_score < -0.05:
                    label = "negative"
                else:
                    label = "neutral"

                return {
                    "label": label,
                    "score": (sentiment_score + 1) / 2  # Convert to [0, 1] range
                }
            except Exception as e:
                logger.warning(f"Error in NLTK text classification: {e}. Falling back to basic analysis.")

        # Fallback to basic sentiment analysis
        # Convert to lowercase
        text = text.lower()

        # Define positive and negative word lists
        positive_words = {'good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic',
                         'happy', 'glad', 'positive', 'nice', 'love', 'like', 'awesome',
                         'beautiful', 'best', 'better', 'enjoy', 'thanks', 'thank'}

        negative_words = {'bad', 'terrible', 'awful', 'horrible', 'poor', 'negative',
                         'sad', 'unhappy', 'angry', 'upset', 'hate', 'dislike', 'worst',
                         'worse', 'disappointing', 'disappointed', 'annoying', 'annoyed'}

        # Tokenize the text
        tokens = self.tokenize(text)

        # Count positive and negative words
        positive_count = sum(1 for token in tokens if token in positive_words)
        negative_count = sum(1 for token in tokens if token in negative_words)
        total_count = len(tokens)

        # Calculate sentiment score
        if total_count > 0:
            sentiment_score = (positive_count - negative_count) / total_count
        else:
            sentiment_score = 0

        # Determine sentiment
        if sentiment_score > 0.05:
            label = "positive"
        elif sentiment_score < -0.05:
            label = "negative"
        else:
            label = "neutral"

        return {
            "label": label,
            "score": (sentiment_score + 1) / 2  # Convert to [0, 1] range
        }

    def generate_response(self, input_text, response_templates, user_name=None, context=None):
        """
        Generate a response based on the input text, templates, and context.

        Args:
            input_text (str): The user's input text
            response_templates (dict): Dictionary of response templates
            user_name (str, optional): The user's name for personalization
            context (dict, optional): Conversation context

        Returns:
            str: A generated response
        """
        if not input_text:
            return random.choice(response_templates.get('unknown', ["I don't understand that yet."]))

        # Update context if provided
        if context:
            self.context.update(context)

        # Recognize intent
        intent_result = self.recognize_intent(input_text)
        intent = intent_result["intent"]
        confidence = intent_result["confidence"]

        # Extract entities
        entities = self.extract_entities(input_text)

        # Log the NLP analysis
        logger.debug(f"Intent: {intent} (confidence: {confidence:.2f})")
        logger.debug(f"Entities: {entities}")

        # Update context with recognized intent and entities
        self.context["last_intent"] = intent
        self.context["last_entities"] = entities

        # Generate response based on intent and entities
        if intent in response_templates and confidence > 0.5:
            response = random.choice(response_templates.get(intent, ["I'm not sure how to respond to that."]))
        else:
            # Fallback to keyword matching for low-confidence intents
            keywords = self.extract_keywords(input_text)

            # Check for specific keywords and return appropriate responses
            if any(word in keywords for word in ['hello', 'hi', 'hey']):
                response = random.choice(response_templates.get('greeting', ['Hello!']))
            elif any(word in keywords for word in ['bye', 'goodbye', 'exit', 'quit']):
                response = random.choice(response_templates.get('farewell', ['Goodbye!']))
            elif any(word in keywords for word in ['thanks', 'thank']):
                response = random.choice(response_templates.get('thanks', ["You're welcome!"]))
            elif any(word in keywords for word in ['weather', 'temperature', 'forecast']):
                response = random.choice(response_templates.get('weather', ["I don't have weather data."]))
            elif any(word in keywords for word in ['joke', 'funny']):
                response = random.choice(response_templates.get('joke', ["Why did the AI cross the road? To get to the other dataset!"]))
            elif any(word in keywords for word in ['time', 'date']):
                response = random.choice(response_templates.get('time', ["I don't have access to the current time."]))
            elif any(word in keywords for word in ['name', 'called']):
                response = random.choice(response_templates.get('name', ["My name is Noah!"]))
            elif any(phrase in input_text.lower() for phrase in ['how are you', 'how do you do', 'how are things']):
                response = random.choice(response_templates.get('how_are_you', ["I'm doing well, thanks!"]))
            else:
                # Default to unknown response if no matches
                response = random.choice(response_templates.get('unknown', ["I don't understand that yet."]))

        # Personalize the response if we have a user name
        if user_name and "{user_name}" in response:
            response = response.format(user_name=user_name)

        # Replace entity placeholders if any
        for entity_type, entity_values in entities.items():
            if entity_values and f"{{{entity_type}}}" in response:
                response = response.replace(f"{{{entity_type}}}", entity_values[0]["value"])

        return response

    def recognize_intent(self, text):
        """
        Recognize the intent of the input text.

        Args:
            text (str): The input text

        Returns:
            dict: Intent recognition result with intent name and confidence score
        """
        if not text:
            return {"intent": "unknown", "confidence": 0.0}

        # Use our dedicated intent classifier if available
        if self.intent_classifier and self.use_advanced_nlp:
            try:
                return self.intent_classifier.classify_intent(text)
            except Exception as e:
                logger.warning(f"Error in advanced intent classification: {e}. Falling back to basic classification.")

        # Use transformers for intent recognition if available
        if TRANSFORMERS_AVAILABLE and self.intent_model and self.intent_tokenizer and self.use_advanced_nlp:
            try:
                # This is a simplified example - in a real implementation, you would fine-tune
                # the model on your specific intents
                inputs = self.intent_tokenizer(text, return_tensors="pt", padding=True, truncation=True)
                outputs = self.intent_model(**inputs)

                # Get the predicted class and confidence
                predicted_class = torch.argmax(outputs.logits, dim=1).item()
                confidence = torch.softmax(outputs.logits, dim=1)[0][predicted_class].item()

                # Map the class to an intent name (simplified)
                intent_names = list(self.intents.keys())
                if predicted_class < len(intent_names):
                    intent = intent_names[predicted_class]
                else:
                    intent = "unknown"

                return {"intent": intent, "confidence": confidence}
            except Exception as e:
                logger.warning(f"Error in transformers intent recognition: {e}. Falling back to pattern matching.")

        # Fallback to pattern matching
        best_intent = "unknown"
        best_confidence = 0.0

        # Normalize the input text
        text_lower = text.lower()

        # Check each intent's patterns
        for intent_name, intent_data in self.intents.items():
            patterns = intent_data.get("patterns", [])

            for pattern in patterns:
                if pattern.lower() in text_lower:
                    # Simple confidence calculation based on pattern length
                    confidence = len(pattern) / len(text_lower)

                    if confidence > best_confidence:
                        best_intent = intent_name
                        best_confidence = confidence

        return {"intent": best_intent, "confidence": best_confidence}

    def extract_entities(self, text):
        """
        Extract entities from the input text.

        Args:
            text (str): The input text

        Returns:
            dict: Dictionary of extracted entities by type
        """
        if not text:
            return {}

        # Use our dedicated entity recognizer if available
        if self.entity_recognizer and self.use_advanced_nlp:
            try:
                return self.entity_recognizer.extract_entities(text)
            except Exception as e:
                logger.warning(f"Error in advanced entity extraction: {e}. Falling back to basic extraction.")

        entities = {}

        # Use spaCy for entity extraction if available
        if SPACY_AVAILABLE and self.nlp and self.use_advanced_nlp:
            try:
                doc = self.nlp(text)

                # Extract named entities
                for ent in doc.ents:
                    entity_type = ent.label_.lower()
                    entity_value = ent.text

                    if entity_type not in entities:
                        entities[entity_type] = []

                    entities[entity_type].append({
                        "value": entity_value,
                        "start": ent.start_char,
                        "end": ent.end_char
                    })

                # Extract noun chunks as potential entities
                for chunk in doc.noun_chunks:
                    # Skip if already captured as a named entity
                    if any(chunk.start_char >= ent.start_char and chunk.end_char <= ent.end_char for ent in doc.ents):
                        continue

                    # Add as a generic entity
                    if "noun_phrase" not in entities:
                        entities["noun_phrase"] = []

                    entities["noun_phrase"].append({
                        "value": chunk.text,
                        "start": chunk.start_char,
                        "end": chunk.end_char
                    })

                return entities
            except Exception as e:
                logger.warning(f"Error in spaCy entity extraction: {e}. Falling back to pattern matching.")

        # Fallback to pattern matching
        for entity_type, entity_data in self.entities.items():
            patterns = entity_data.get("patterns", [])

            for pattern in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)

                for match in matches:
                    if entity_type not in entities:
                        entities[entity_type] = []

                    entities[entity_type].append({
                        "value": match.group(0),
                        "start": match.start(),
                        "end": match.end()
                    })

        return entities

    def update_context(self, key, value):
        """
        Update the conversation context.

        Args:
            key (str): Context key
            value (any): Context value
        """
        self.context[key] = value

    def get_context(self, key=None):
        """
        Get the conversation context.

        Args:
            key (str, optional): Context key to retrieve

        Returns:
            any: Context value if key is provided, otherwise the entire context
        """
        if key:
            return self.context.get(key)
        return self.context

    def clear_context(self):
        """
        Clear the conversation context.
        """
        self.context = {}

    def detect_language(self, text):
        """
        Detect the language of the input text.

        Args:
            text (str): The input text

        Returns:
            dict: Detected language information
        """
        if not text:
            return {"lang_code": "en", "lang_name": "English", "confidence": 1.0}

        # Use multilingual processor if available
        if self.multilingual and self.use_multilingual:
            try:
                return self.multilingual.detect_language(text)
            except Exception as e:
                logger.warning(f"Error in multilingual language detection: {e}")

        # Fallback to English
        return {"lang_code": "en", "lang_name": "English", "confidence": 1.0}

    def translate(self, text, source_lang=None, target_lang="en"):
        """
        Translate text from source language to target language.

        Args:
            text (str): Input text
            source_lang (str, optional): Source language code. If None, language is detected.
            target_lang (str): Target language code

        Returns:
            str: Translated text
        """
        if not text:
            return ""

        # Use multilingual processor if available
        if self.multilingual and self.use_multilingual:
            try:
                return self.multilingual.translate(text, source_lang, target_lang)
            except Exception as e:
                logger.warning(f"Error in multilingual translation: {e}")

        # Fallback message
        return f"[Translation not available]"

    def analyze_topic(self, text, num_topics=1, num_words=5):
        """
        Analyze the topic of the input text.

        Args:
            text (str): The input text
            num_topics (int): Number of topics to extract
            num_words (int): Number of words per topic

        Returns:
            list: List of topics, each containing key words
        """
        if not text or len(text.strip()) < 10:
            return [{"topic_id": 0, "words": [], "confidence": 0.0}]

        # Use Gensim for advanced topic modeling if available
        if GENSIM_AVAILABLE and self.use_advanced_nlp:
            try:
                # Tokenize and preprocess
                tokens = word_tokenize(text.lower())

                # Remove stopwords and short words
                filtered_tokens = [token for token in tokens
                                  if token not in self.stop_words
                                  and len(token) > 2
                                  and token not in string.punctuation]

                # Lemmatize if possible
                if self.lemmatizer:
                    filtered_tokens = [self.lemmatizer.lemmatize(token) for token in filtered_tokens]

                # Create a dictionary
                dictionary = corpora.Dictionary([filtered_tokens])

                # Create a document-term matrix
                corpus = [dictionary.doc2bow(filtered_tokens)]

                # Train LDA model
                lda_model = LdaModel(corpus=corpus,
                                     id2word=dictionary,
                                     num_topics=num_topics,
                                     passes=10,
                                     alpha='auto')

                # Extract topics
                topics = []
                for topic_id in range(num_topics):
                    topic_words = [word for word, _ in lda_model.show_topic(topic_id, num_words)]
                    topics.append({
                        "topic_id": topic_id,
                        "words": topic_words,
                        "confidence": 0.9  # High confidence for LDA model
                    })

                return topics
            except Exception as e:
                logger.warning(f"Error in Gensim topic modeling: {e}. Trying other methods.")

        # Use NLTK for topic modeling if available
        if NLTK_AVAILABLE and self.use_advanced_nlp:

            try:
                # Alternative approach using NLTK's collocations
                tokens = word_tokenize(text.lower())

                # Remove stopwords and short words
                filtered_tokens = [token for token in tokens
                                  if token not in self.stop_words
                                  and len(token) > 2
                                  and token not in string.punctuation]

                # Create NLTK Text object
                text_obj = nltk.Text(filtered_tokens)

                # Find collocations (phrases that occur together frequently)
                finder = nltk.collocations.BigramCollocationFinder.from_words(filtered_tokens)
                bigram_measures = nltk.collocations.BigramAssocMeasures()

                # Get top collocations
                collocations = finder.nbest(bigram_measures.pmi, num_words)
                collocation_phrases = [f"{w1} {w2}" for w1, w2 in collocations]

                # Get frequency distribution of individual words
                fdist = nltk.FreqDist(filtered_tokens)
                top_words = [word for word, _ in fdist.most_common(num_words)]

                # Create topics from collocations and individual words
                topics = []
                if collocations:
                    topics.append({
                        "topic_id": 0,
                        "words": collocation_phrases,
                        "confidence": 0.8  # Good confidence for collocations
                    })

                topics.append({
                    "topic_id": len(topics),
                    "words": top_words,
                    "confidence": 0.7  # Decent confidence for frequency distribution
                })

                # Limit to requested number of topics
                return topics[:num_topics]
            except Exception as e:
                logger.warning(f"Error in NLTK collocation analysis: {e}. Falling back to other methods.")

        # Use spaCy for topic modeling if available
        if SPACY_AVAILABLE and self.nlp and self.use_advanced_nlp:
            try:
                doc = self.nlp(text)

                # Extract important words (nouns, proper nouns, verbs, adjectives)
                important_words = []
                for token in doc:
                    if (token.pos_ in ['NOUN', 'PROPN', 'VERB', 'ADJ'] and
                        not token.is_stop and
                        len(token.text) > 2):
                        important_words.append((token.text.lower(), token.vector_norm))

                # Group words by similarity to form topics
                topics = []
                if important_words:
                    # Sort by importance
                    important_words.sort(key=lambda x: x[1], reverse=True)

                    # Take top words for the main topic
                    top_words = [word for word, _ in important_words[:num_words]]
                    topics.append({
                        "topic_id": 0,
                        "words": top_words,
                        "confidence": 0.8  # Arbitrary confidence
                    })

                return topics
            except Exception as e:
                logger.warning(f"Error in spaCy topic modeling: {e}. Falling back to basic topic modeling.")

        # Fallback to basic topic modeling
        tokens = self.tokenize(text)

        # Remove stopwords
        if NLTK_AVAILABLE:
            tokens = [token for token in tokens if token not in self.stop_words]

        # Count word frequencies
        word_freq = Counter(tokens)

        # Get most common words as the topic
        top_words = [word for word, _ in word_freq.most_common(num_words)]

        topics = [{
            "topic_id": 0,
            "words": top_words,
            "confidence": 0.5  # Lower confidence for basic method
        }]

        return topics

    def calculate_similarity(self, text1, text2):
        """
        Calculate semantic similarity between two texts.

        Args:
            text1 (str): First text
            text2 (str): Second text

        Returns:
            float: Similarity score (0-1)
        """
        if not text1 or not text2:
            return 0.0

        # Use spaCy for semantic similarity if available
        if SPACY_AVAILABLE and self.nlp and self.use_advanced_nlp:
            try:
                doc1 = self.nlp(text1)
                doc2 = self.nlp(text2)

                # Calculate similarity using spaCy's vector representation
                similarity = doc1.similarity(doc2)

                return max(0.0, min(1.0, similarity))  # Ensure in range [0, 1]
            except Exception as e:
                logger.warning(f"Error in spaCy similarity calculation: {e}. Falling back to basic similarity.")

        # Fallback to basic similarity calculation
        tokens1 = set(self.tokenize(text1))
        tokens2 = set(self.tokenize(text2))

        if not tokens1 or not tokens2:
            return 0.0

        # Jaccard similarity: intersection over union
        intersection = tokens1.intersection(tokens2)
        union = tokens1.union(tokens2)

        similarity = len(intersection) / len(union) if union else 0.0

        return similarity

    def analyze_conversation_coherence(self, conversation_history):
        """
        Analyze the coherence of a conversation.

        Args:
            conversation_history (list): List of conversation turns [(speaker, text), ...]

        Returns:
            dict: Coherence analysis results
        """
        if not conversation_history or len(conversation_history) < 2:
            return {
                "coherence_score": 1.0,
                "topic_consistency": 1.0,
                "response_relevance": 1.0,
                "issues": []
            }

        # Extract just the text from conversation history
        texts = [text for _, text in conversation_history]

        # Calculate pairwise similarities between consecutive turns
        similarities = []
        for i in range(1, len(texts)):
            similarity = self.calculate_similarity(texts[i-1], texts[i])
            similarities.append(similarity)

        # Calculate average similarity as a measure of coherence
        avg_similarity = sum(similarities) / len(similarities) if similarities else 1.0

        # Identify potential coherence issues
        issues = []
        for i in range(1, len(similarities)):
            if similarities[i] < 0.1 and similarities[i-1] > 0.3:
                issues.append({
                    "type": "topic_shift",
                    "position": i+1,
                    "description": "Abrupt topic shift detected"
                })
            elif similarities[i] < 0.05:
                issues.append({
                    "type": "irrelevant_response",
                    "position": i+1,
                    "description": "Response appears unrelated to previous message"
                })

        # Analyze topic consistency
        all_topics = []
        for text in texts:
            topics = self.analyze_topic(text, num_topics=1, num_words=3)
            if topics and topics[0]["words"]:
                all_topics.append(set(topics[0]["words"]))

        # Calculate topic overlap across conversation
        topic_consistency = 0.0
        if all_topics:
            # Find words that appear in multiple topic sets
            topic_overlaps = []
            for i in range(1, len(all_topics)):
                overlap = len(all_topics[i].intersection(all_topics[i-1]))
                max_possible = min(len(all_topics[i]), len(all_topics[i-1]))
                topic_overlaps.append(overlap / max_possible if max_possible > 0 else 0)

            topic_consistency = sum(topic_overlaps) / len(topic_overlaps) if topic_overlaps else 0.0

        return {
            "coherence_score": avg_similarity,
            "topic_consistency": topic_consistency,
            "response_relevance": min(1.0, avg_similarity * 1.25),  # Slightly boost relevance score
            "issues": issues
        }

    def get_text_embedding(self, text):
        """
        Get a vector embedding for the input text.

        Args:
            text (str): Input text

        Returns:
            list: Vector embedding
        """
        if not text:
            return [0.0] * 300  # Default empty embedding

        # Use spaCy for text embedding if available
        if SPACY_AVAILABLE and self.nlp and self.use_advanced_nlp:
            try:
                doc = self.nlp(text)

                # Get document vector
                vector = doc.vector

                # Normalize vector
                norm = np.linalg.norm(vector)
                if norm > 0:
                    vector = vector / norm

                return vector.tolist()
            except Exception as e:
                logger.warning(f"Error in spaCy text embedding: {e}. Falling back to basic embedding.")

        # Fallback to basic embedding using word counts
        tokens = self.tokenize(text)

        # Create a simple bag-of-words representation
        # This is a very basic embedding method
        embedding = np.zeros(300)

        for i, token in enumerate(tokens):
            # Use hash of word to determine the index
            idx = hash(token) % 300
            embedding[idx] += 1

        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding.tolist()
