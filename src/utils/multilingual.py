"""
Multilingual Support for NoahAI

This module provides multilingual capabilities for NoahAI:
- Language detection
- Multilingual tokenization
- Multilingual sentiment analysis
- Multilingual entity recognition
- Translation utilities
"""

import os
import json
import logging
import numpy as np
from typing import List, Dict, Tuple, Any, Optional, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/multilingual.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("multilingual")

# Import NLP libraries
try:
    import nltk
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    logger.warning("NLTK not available. Multilingual capabilities will be limited.")

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    logger.warning("spaCy not available. Multilingual capabilities will be limited.")

try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("Transformers not available. Multilingual capabilities will be limited.")

class MultilingualProcessor:
    """
    Processor for multilingual text analysis.
    """
    
    def __init__(self, models_dir="data/models/multilingual"):
        """
        Initialize the multilingual processor.
        
        Args:
            models_dir (str): Directory to store/load multilingual models
        """
        self.models_dir = models_dir
        os.makedirs(models_dir, exist_ok=True)
        
        # Supported languages
        self.supported_languages = {
            "en": "English",
            "fr": "French",
            "es": "Spanish",
            "de": "German"
        }
        
        # Initialize language-specific resources
        self._initialize_resources()
    
    def _initialize_resources(self):
        """Initialize language-specific resources."""
        self.nlp_models = {}
        self.stop_words = {}
        self.tokenizers = {}
        
        # Initialize spaCy models for supported languages
        if SPACY_AVAILABLE:
            try:
                # English
                if spacy.util.is_package("en_core_web_md"):
                    self.nlp_models["en"] = spacy.load("en_core_web_md")
                    logger.info("Loaded English spaCy model")
                
                # French
                if spacy.util.is_package("fr_core_news_md"):
                    self.nlp_models["fr"] = spacy.load("fr_core_news_md")
                    logger.info("Loaded French spaCy model")
                
                # Spanish
                if spacy.util.is_package("es_core_news_md"):
                    try:
                        self.nlp_models["es"] = spacy.load("es_core_news_md")
                        logger.info("Loaded Spanish spaCy model")
                    except:
                        logger.warning("Spanish model not available. Run 'python -m spacy download es_core_news_md'")
                
                # German
                if spacy.util.is_package("de_core_news_md"):
                    try:
                        self.nlp_models["de"] = spacy.load("de_core_news_md")
                        logger.info("Loaded German spaCy model")
                    except:
                        logger.warning("German model not available. Run 'python -m spacy download de_core_news_md'")
            
            except Exception as e:
                logger.error(f"Error loading spaCy models: {e}")
        
        # Initialize NLTK resources for supported languages
        if NLTK_AVAILABLE:
            try:
                # Load stopwords for supported languages
                for lang_code in self.supported_languages:
                    try:
                        self.stop_words[lang_code] = set(stopwords.words(self._map_lang_code_to_nltk(lang_code)))
                        logger.info(f"Loaded stopwords for {self.supported_languages[lang_code]}")
                    except:
                        logger.warning(f"Stopwords not available for {lang_code}")
            except Exception as e:
                logger.error(f"Error loading NLTK resources: {e}")
        
        # Initialize transformers models
        if TRANSFORMERS_AVAILABLE:
            try:
                # Multilingual sentiment analysis
                self.sentiment_pipeline = pipeline(
                    "sentiment-analysis",
                    model="nlptown/bert-base-multilingual-uncased-sentiment"
                )
                logger.info("Loaded multilingual sentiment analysis model")
                
                # Multilingual translation
                self.translation_pipelines = {}
                for lang_code in self.supported_languages:
                    if lang_code != "en":
                        # English to target language
                        try:
                            self.translation_pipelines[f"en-{lang_code}"] = pipeline(
                                "translation",
                                model=f"Helsinki-NLP/opus-mt-en-{lang_code}"
                            )
                            logger.info(f"Loaded English to {self.supported_languages[lang_code]} translation model")
                        except:
                            logger.warning(f"Translation model not available for en-{lang_code}")
                        
                        # Target language to English
                        try:
                            self.translation_pipelines[f"{lang_code}-en"] = pipeline(
                                "translation",
                                model=f"Helsinki-NLP/opus-mt-{lang_code}-en"
                            )
                            logger.info(f"Loaded {self.supported_languages[lang_code]} to English translation model")
                        except:
                            logger.warning(f"Translation model not available for {lang_code}-en")
            
            except Exception as e:
                logger.error(f"Error loading transformers models: {e}")
                self.sentiment_pipeline = None
                self.translation_pipelines = {}
    
    def _map_lang_code_to_nltk(self, lang_code):
        """Map language code to NLTK language name."""
        mapping = {
            "en": "english",
            "fr": "french",
            "es": "spanish",
            "de": "german"
        }
        return mapping.get(lang_code, "english")
    
    def detect_language(self, text):
        """
        Detect the language of the input text.
        
        Args:
            text (str): Input text
            
        Returns:
            dict: Detected language information
        """
        if not text:
            return {"lang_code": "en", "lang_name": "English", "confidence": 1.0}
        
        # Use spaCy for language detection if available
        if SPACY_AVAILABLE and "en" in self.nlp_models:
            try:
                # Count language-specific characters and words
                lang_scores = {}
                
                for lang_code, nlp in self.nlp_models.items():
                    doc = nlp(text)
                    
                    # Calculate score based on token recognition
                    recognized_tokens = sum(1 for token in doc if not token.is_oov)
                    total_tokens = len(doc)
                    
                    if total_tokens > 0:
                        lang_scores[lang_code] = recognized_tokens / total_tokens
                    else:
                        lang_scores[lang_code] = 0
                
                # Get the language with the highest score
                if lang_scores:
                    best_lang = max(lang_scores.items(), key=lambda x: x[1])
                    lang_code = best_lang[0]
                    confidence = best_lang[1]
                    
                    return {
                        "lang_code": lang_code,
                        "lang_name": self.supported_languages.get(lang_code, "Unknown"),
                        "confidence": confidence
                    }
            
            except Exception as e:
                logger.warning(f"Error in spaCy language detection: {e}")
        
        # Fallback to simple character-based detection
        # This is a very basic approach and should be replaced with a proper language detection library
        char_counts = {}
        
        # Character sets that are more common in specific languages
        lang_chars = {
            "en": "abcdefghijklmnopqrstuvwxyz",
            "fr": "éèêëàâäôöùûüÿçœæ",
            "es": "áéíóúüñ¿¡",
            "de": "äöüß"
        }
        
        # Count occurrences of language-specific characters
        text_lower = text.lower()
        for lang_code, chars in lang_chars.items():
            char_counts[lang_code] = sum(text_lower.count(c) for c in chars)
        
        # Get the language with the most character matches
        if sum(char_counts.values()) > 0:
            best_lang = max(char_counts.items(), key=lambda x: x[1])
            lang_code = best_lang[0]
            
            # If very few special characters, default to English
            if best_lang[1] < 2 and lang_code != "en":
                lang_code = "en"
            
            return {
                "lang_code": lang_code,
                "lang_name": self.supported_languages.get(lang_code, "Unknown"),
                "confidence": 0.6  # Low confidence for this simple method
            }
        
        # Default to English
        return {"lang_code": "en", "lang_name": "English", "confidence": 0.5}
    
    def tokenize(self, text, lang_code=None):
        """
        Tokenize text in the specified language.
        
        Args:
            text (str): Input text
            lang_code (str, optional): Language code. If None, language is detected.
            
        Returns:
            list: Tokenized text
        """
        if not text:
            return []
        
        # Detect language if not specified
        if not lang_code:
            lang_result = self.detect_language(text)
            lang_code = lang_result["lang_code"]
        
        # Use spaCy for tokenization if available
        if SPACY_AVAILABLE and lang_code in self.nlp_models:
            try:
                doc = self.nlp_models[lang_code](text)
                
                # Get tokens, filtering out punctuation and spaces
                tokens = [token.text.lower() for token in doc
                         if not token.is_punct and not token.is_space]
                
                return tokens
            except Exception as e:
                logger.warning(f"Error in spaCy tokenization: {e}")
        
        # Use NLTK for tokenization if available
        if NLTK_AVAILABLE:
            try:
                tokens = word_tokenize(text, language=self._map_lang_code_to_nltk(lang_code))
                return [token.lower() for token in tokens if token.isalpha()]
            except Exception as e:
                logger.warning(f"Error in NLTK tokenization: {e}")
        
        # Fallback to basic tokenization
        return text.lower().split()
    
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
        
        # Detect source language if not specified
        if not source_lang:
            lang_result = self.detect_language(text)
            source_lang = lang_result["lang_code"]
        
        # If source and target are the same, return the original text
        if source_lang == target_lang:
            return text
        
        # Use transformers for translation if available
        if TRANSFORMERS_AVAILABLE and hasattr(self, 'translation_pipelines'):
            pipeline_key = f"{source_lang}-{target_lang}"
            
            if pipeline_key in self.translation_pipelines:
                try:
                    result = self.translation_pipelines[pipeline_key](text, max_length=512)
                    return result[0]["translation_text"]
                except Exception as e:
                    logger.warning(f"Error in transformers translation: {e}")
        
        # Fallback message
        return f"[Translation from {self.supported_languages.get(source_lang, source_lang)} to {self.supported_languages.get(target_lang, target_lang)} not available]"
    
    def sentiment_analysis(self, text, lang_code=None):
        """
        Perform sentiment analysis on text in the specified language.
        
        Args:
            text (str): Input text
            lang_code (str, optional): Language code. If None, language is detected.
            
        Returns:
            dict: Sentiment analysis results
        """
        if not text:
            return {"label": "neutral", "score": 0.5}
        
        # Detect language if not specified
        if not lang_code:
            lang_result = self.detect_language(text)
            lang_code = lang_result["lang_code"]
        
        # Use transformers for multilingual sentiment analysis if available
        if TRANSFORMERS_AVAILABLE and hasattr(self, 'sentiment_pipeline'):
            try:
                result = self.sentiment_pipeline(text)[0]
                
                # Map the result to our format
                label = result["label"].lower()
                if "stars" in label:
                    # Handle star-based labels (1-5 stars)
                    stars = int(label.split()[0])
                    if stars >= 4:
                        sentiment_label = "positive"
                    elif stars <= 2:
                        sentiment_label = "negative"
                    else:
                        sentiment_label = "neutral"
                    
                    # Map 1-5 stars to 0-1 score
                    score = (stars - 1) / 4
                else:
                    # Handle standard sentiment labels
                    sentiment_label = label
                    score = result["score"]
                
                return {
                    "label": sentiment_label,
                    "score": score,
                    "language": self.supported_languages.get(lang_code, "Unknown")
                }
            
            except Exception as e:
                logger.warning(f"Error in transformers sentiment analysis: {e}")
        
        # Use spaCy for language-specific sentiment analysis if available
        if SPACY_AVAILABLE and lang_code in self.nlp_models:
            try:
                doc = self.nlp_models[lang_code](text)
                
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
                    "score": (sentiment_score + 1) / 2,  # Convert to [0, 1] range
                    "language": self.supported_languages.get(lang_code, "Unknown")
                }
            
            except Exception as e:
                logger.warning(f"Error in spaCy sentiment analysis: {e}")
        
        # Fallback to basic sentiment analysis
        # Translate to English first if not English
        if lang_code != "en":
            translated_text = self.translate(text, source_lang=lang_code, target_lang="en")
            if translated_text != text:
                # Use English sentiment analysis on translated text
                return self.sentiment_analysis(translated_text, lang_code="en")
        
        # Very basic English sentiment analysis
        positive_words = {'good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic',
                         'happy', 'glad', 'positive', 'nice', 'love', 'like', 'awesome'}
        negative_words = {'bad', 'terrible', 'awful', 'horrible', 'poor', 'negative',
                         'sad', 'unhappy', 'angry', 'upset', 'hate', 'dislike', 'worst'}
        
        tokens = self.tokenize(text, lang_code="en")
        
        positive_count = sum(1 for token in tokens if token in positive_words)
        negative_count = sum(1 for token in tokens if token in negative_words)
        
        if positive_count > negative_count:
            return {"label": "positive", "score": 0.75, "language": "English (fallback)"}
        elif negative_count > positive_count:
            return {"label": "negative", "score": 0.25, "language": "English (fallback)"}
        else:
            return {"label": "neutral", "score": 0.5, "language": "English (fallback)"}
