"""
Advanced NLP capabilities for NoahAI

This module provides specialized NLP models for:
- Dedicated sentiment analysis
- Named entity recognition
- Intent classification
- Question answering
- Text summarization
- Translation
- Voice capabilities
"""

import os
import json
import logging
import numpy as np
from typing import Dict, List, Any, Optional, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/advanced_nlp.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("advanced_nlp")

# Try to import TensorFlow and related libraries
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, Model, load_model
    from tensorflow.keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional, Input
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    logger.warning("TensorFlow not available. Advanced NLP models will not be available.")

# Try to import transformers
try:
    from transformers import (
        pipeline,
        AutoTokenizer,
        AutoModelForSequenceClassification,
        AutoModelForTokenClassification,
        AutoModelForQuestionAnswering,
        AutoModelForSeq2SeqLM
    )
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("Transformers not available. Using basic NLP processing.")

# Try to import speech recognition
try:
    import speech_recognition as sr
    SPEECH_RECOGNITION_AVAILABLE = True
except ImportError:
    SPEECH_RECOGNITION_AVAILABLE = False
    logger.warning("Speech recognition not available.")

# Try to import text-to-speech
try:
    import pyttsx3
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False
    logger.warning("Text-to-speech not available.")

class SentimentAnalyzer:
    """
    Dedicated sentiment analysis model using either TensorFlow or Transformers.
    """

    def __init__(self, model_dir="data/models/sentiment", use_transformers=True):
        """
        Initialize the sentiment analyzer.

        Args:
            model_dir (str): Directory to save/load model files
            use_transformers (bool): Whether to use transformers models (if available)
        """
        self.model_dir = model_dir
        self.use_transformers = use_transformers and TRANSFORMERS_AVAILABLE

        # Create model directory if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)

        # Initialize the model
        self.model = None
        self.tokenizer = None
        self.pipeline = None

        self._initialize_model()

    def _initialize_model(self):
        """
        Initialize or load the sentiment analysis model.
        """
        if self.use_transformers:
            try:
                # Use pre-trained sentiment analysis model from Hugging Face
                self.pipeline = pipeline("sentiment-analysis")
                logger.info("Loaded transformers sentiment analysis model.")
            except Exception as e:
                logger.error(f"Error loading transformers sentiment model: {e}")
                self.use_transformers = False

        if not self.use_transformers and TENSORFLOW_AVAILABLE:
            model_path = os.path.join(self.model_dir, "sentiment_model.h5")
            tokenizer_path = os.path.join(self.model_dir, "sentiment_tokenizer.json")

            # Check if model exists
            if os.path.exists(model_path) and os.path.exists(tokenizer_path):
                try:
                    # Load existing model
                    self.model = load_model(model_path)

                    # Load tokenizer
                    with open(tokenizer_path, 'r') as f:
                        tokenizer_json = json.load(f)
                        self.tokenizer = Tokenizer()
                        self.tokenizer.word_index = tokenizer_json

                    logger.info("Loaded TensorFlow sentiment analysis model.")
                except Exception as e:
                    logger.error(f"Error loading TensorFlow sentiment model: {e}")
                    self._create_model()
            else:
                self._create_model()

    def _create_model(self):
        """
        Create a new sentiment analysis model using TensorFlow.
        """
        if not TENSORFLOW_AVAILABLE:
            logger.error("TensorFlow not available. Cannot create sentiment model.")
            return

        # Create a simple sentiment analysis model
        self.tokenizer = Tokenizer(num_words=10000)

        # Create model
        model = Sequential()
        model.add(Embedding(10000, 128))
        model.add(Bidirectional(LSTM(64)))
        model.add(Dropout(0.2))
        model.add(Dense(32, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(1, activation='sigmoid'))  # Binary classification (positive/negative)

        # Compile model
        model.compile(loss='binary_crossentropy',
                     optimizer='adam',
                     metrics=['accuracy'])

        self.model = model
        logger.info("Created new TensorFlow sentiment analysis model.")

    def analyze(self, text):
        """
        Analyze the sentiment of the given text.

        Args:
            text (str): The text to analyze

        Returns:
            dict: Sentiment analysis result with label and score
        """
        if not text:
            return {"label": "neutral", "score": 0.5}

        if self.use_transformers and self.pipeline:
            try:
                result = self.pipeline(text)[0]

                # Map the result to our format
                sentiment = {
                    "label": result["label"].lower(),
                    "score": result["score"]
                }

                return sentiment
            except Exception as e:
                logger.error(f"Error in transformers sentiment analysis: {e}")

        # Fallback to basic sentiment analysis
        # This is a placeholder - in a real implementation, you would use the TensorFlow model
        # But since we haven't trained it yet, we'll use a simple rule-based approach

        # Define positive and negative word lists
        positive_words = {'good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic',
                         'happy', 'glad', 'positive', 'nice', 'love', 'like', 'awesome',
                         'beautiful', 'best', 'better', 'enjoy', 'thanks', 'thank'}

        negative_words = {'bad', 'terrible', 'awful', 'horrible', 'poor', 'negative',
                         'sad', 'unhappy', 'angry', 'upset', 'hate', 'dislike', 'worst',
                         'worse', 'disappointing', 'disappointed', 'annoying', 'annoyed'}

        # Convert to lowercase and split into words
        words = text.lower().split()

        # Count positive and negative words
        positive_count = sum(1 for word in words if word in positive_words)
        negative_count = sum(1 for word in words if word in negative_words)

        # Calculate sentiment score
        if len(words) > 0:
            sentiment_score = (positive_count - negative_count) / len(words)
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

class EntityRecognizer:
    """
    Named entity recognition using transformers or pattern matching.
    """

    def __init__(self, model_dir="data/models/ner", use_transformers=True):
        """
        Initialize the entity recognizer.

        Args:
            model_dir (str): Directory to save/load model files
            use_transformers (bool): Whether to use transformers models (if available)
        """
        self.model_dir = model_dir
        self.use_transformers = use_transformers and TRANSFORMERS_AVAILABLE

        # Create model directory if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)

        # Initialize the model
        self.pipeline = None

        # Load entity patterns
        self.entity_patterns = self._load_entity_patterns()

        self._initialize_model()

    def _initialize_model(self):
        """
        Initialize or load the entity recognition model.
        """
        if self.use_transformers:
            try:
                # Use pre-trained NER model from Hugging Face
                self.pipeline = pipeline("ner")
                logger.info("Loaded transformers NER model.")
            except Exception as e:
                logger.error(f"Error loading transformers NER model: {e}")
                self.use_transformers = False

    def _load_entity_patterns(self):
        """
        Load entity patterns from file.

        Returns:
            dict: Entity patterns
        """
        patterns_file = os.path.join(self.model_dir, "entity_patterns.json")

        if os.path.exists(patterns_file):
            try:
                with open(patterns_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading entity patterns: {e}")

        # Default entity patterns
        return {
            "date": [
                r"\d{1,2}/\d{1,2}/\d{2,4}",
                r"\d{1,2}-\d{1,2}-\d{2,4}",
                r"(january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2}(st|nd|rd|th)?,?\s+\d{2,4}"
            ],
            "time": [
                r"\d{1,2}:\d{2}\s*(am|pm)?",
                r"\d{1,2}\s*(am|pm)"
            ],
            "email": [
                r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
            ],
            "phone": [
                r"\+?\d{1,3}[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}"
            ],
            "url": [
                r"https?://[^\s]+"
            ],
            "location": [
                r"in\s+([A-Z][a-z]+(\s+[A-Z][a-z]+)*)",
                r"at\s+([A-Z][a-z]+(\s+[A-Z][a-z]+)*)"
            ]
        }

    def extract_entities(self, text):
        """
        Extract entities from the given text.

        Args:
            text (str): The text to analyze

        Returns:
            dict: Dictionary of extracted entities by type
        """
        if not text:
            return {}

        entities = {}

        if self.use_transformers and self.pipeline:
            try:
                # Use transformers NER pipeline
                ner_results = self.pipeline(text)

                # Group entities by type
                for entity in ner_results:
                    entity_type = entity["entity"].lower()
                    entity_value = entity["word"]

                    if entity_type not in entities:
                        entities[entity_type] = []

                    entities[entity_type].append({
                        "value": entity_value,
                        "start": entity["start"],
                        "end": entity["end"]
                    })

                return entities
            except Exception as e:
                logger.error(f"Error in transformers NER: {e}")

        # Fallback to pattern matching
        import re

        for entity_type, patterns in self.entity_patterns.items():
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

class IntentClassifier:
    """
    Intent classification using transformers or pattern matching.
    """

    def __init__(self, model_dir="data/models/intent", use_transformers=True):
        """
        Initialize the intent classifier.

        Args:
            model_dir (str): Directory to save/load model files
            use_transformers (bool): Whether to use transformers models (if available)
        """
        self.model_dir = model_dir
        self.use_transformers = use_transformers and TRANSFORMERS_AVAILABLE

        # Create model directory if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)

        # Initialize the model
        self.model = None
        self.tokenizer = None

        # Load intent patterns
        self.intent_patterns = self._load_intent_patterns()

        self._initialize_model()

    def _initialize_model(self):
        """
        Initialize or load the intent classification model.
        """
        if self.use_transformers:
            try:
                # Use pre-trained text classification model from Hugging Face
                self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
                self.model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")
                logger.info("Loaded transformers intent classification model.")
            except Exception as e:
                logger.error(f"Error loading transformers intent model: {e}")
                self.use_transformers = False

    def _load_intent_patterns(self):
        """
        Load intent patterns from file.

        Returns:
            dict: Intent patterns
        """
        patterns_file = os.path.join(self.model_dir, "intent_patterns.json")

        if os.path.exists(patterns_file):
            try:
                with open(patterns_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading intent patterns: {e}")

        # Default intent patterns
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
            },
            "weather": {
                "patterns": ["weather", "temperature", "forecast", "is it going to rain", "how hot is it"],
                "responses": ["I can check the weather for you."]
            },
            "calendar": {
                "patterns": ["schedule", "calendar", "appointment", "meeting", "event", "remind me"],
                "responses": ["I can help with your calendar."]
            },
            "message": {
                "patterns": ["send a message", "text", "message", "send an email", "email"],
                "responses": ["I can help you send a message."]
            }
        }

    def classify_intent(self, text):
        """
        Classify the intent of the given text.

        Args:
            text (str): The text to classify

        Returns:
            dict: Intent classification result with intent name and confidence score
        """
        if not text:
            return {"intent": "unknown", "confidence": 0.0}

        if self.use_transformers and self.model and self.tokenizer:
            try:
                # This is a simplified example - in a real implementation, you would fine-tune
                # the model on your specific intents
                inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
                outputs = self.model(**inputs)

                # Get the predicted class and confidence
                predicted_class = torch.argmax(outputs.logits, dim=1).item()
                confidence = torch.softmax(outputs.logits, dim=1)[0][predicted_class].item()

                # Map the class to an intent name (simplified)
                intent_names = list(self.intent_patterns.keys())
                if predicted_class < len(intent_names):
                    intent = intent_names[predicted_class]
                else:
                    intent = "unknown"

                return {"intent": intent, "confidence": confidence}
            except Exception as e:
                logger.error(f"Error in transformers intent classification: {e}")

        # Fallback to pattern matching
        best_intent = "unknown"
        best_confidence = 0.0

        # Normalize the input text
        text_lower = text.lower()

        # Check each intent's patterns
        for intent_name, intent_data in self.intent_patterns.items():
            patterns = intent_data.get("patterns", [])

            for pattern in patterns:
                if pattern.lower() in text_lower:
                    # Simple confidence calculation based on pattern length
                    confidence = len(pattern) / len(text_lower)

                    if confidence > best_confidence:
                        best_intent = intent_name
                        best_confidence = confidence

        return {"intent": best_intent, "confidence": best_confidence}

class QuestionAnswerer:
    """
    Question answering using transformers.
    """

    def __init__(self, model_dir="data/models/qa", use_transformers=True):
        """
        Initialize the question answerer.

        Args:
            model_dir (str): Directory to save/load model files
            use_transformers (bool): Whether to use transformers models (if available)
        """
        self.model_dir = model_dir
        self.use_transformers = use_transformers and TRANSFORMERS_AVAILABLE

        # Create model directory if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)

        # Initialize the model
        self.pipeline = None

        self._initialize_model()

    def _initialize_model(self):
        """
        Initialize or load the question answering model.
        """
        if self.use_transformers:
            try:
                # Use pre-trained QA model from Hugging Face
                self.pipeline = pipeline("question-answering")
                logger.info("Loaded transformers question answering model.")
            except Exception as e:
                logger.error(f"Error loading transformers QA model: {e}")
                self.use_transformers = False

    def answer_question(self, question, context):
        """
        Answer a question based on the given context.

        Args:
            question (str): The question to answer
            context (str): The context to extract the answer from

        Returns:
            dict: Answer with text, confidence score, and position
        """
        if not question or not context:
            return {"answer": "", "score": 0.0, "start": 0, "end": 0}

        if self.use_transformers and self.pipeline:
            try:
                # Use transformers QA pipeline
                result = self.pipeline(question=question, context=context)

                return {
                    "answer": result["answer"],
                    "score": result["score"],
                    "start": result["start"],
                    "end": result["end"]
                }
            except Exception as e:
                logger.error(f"Error in transformers QA: {e}")

        # Fallback to simple keyword matching
        question_lower = question.lower()
        context_lower = context.lower()

        # Extract question type
        question_words = ["what", "who", "where", "when", "why", "how"]
        question_type = None

        for word in question_words:
            if question_lower.startswith(word):
                question_type = word
                break

        # Extract keywords from question
        keywords = []
        for word in question_lower.split():
            if word not in question_words and len(word) > 3:
                keywords.append(word)

        # Find sentences in context that contain keywords
        import re
        sentences = re.split(r'[.!?]', context)
        relevant_sentences = []

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            sentence_lower = sentence.lower()
            keyword_count = sum(1 for keyword in keywords if keyword in sentence_lower)

            if keyword_count > 0:
                relevant_sentences.append((sentence, keyword_count))

        # Sort by keyword count
        relevant_sentences.sort(key=lambda x: x[1], reverse=True)

        if relevant_sentences:
            best_sentence = relevant_sentences[0][0]
            return {
                "answer": best_sentence,
                "score": min(relevant_sentences[0][1] / len(keywords), 1.0) if keywords else 0.5,
                "start": context.find(best_sentence),
                "end": context.find(best_sentence) + len(best_sentence)
            }

        return {"answer": "", "score": 0.0, "start": 0, "end": 0}

class TextSummarizer:
    """
    Text summarization using transformers.
    """

    def __init__(self, model_dir="data/models/summarization", use_transformers=True):
        """
        Initialize the text summarizer.

        Args:
            model_dir (str): Directory to save/load model files
            use_transformers (bool): Whether to use transformers models (if available)
        """
        self.model_dir = model_dir
        self.use_transformers = use_transformers and TRANSFORMERS_AVAILABLE

        # Create model directory if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)

        # Initialize the model
        self.pipeline = None

        self._initialize_model()

    def _initialize_model(self):
        """
        Initialize or load the summarization model.
        """
        if self.use_transformers:
            try:
                # Use pre-trained summarization model from Hugging Face
                self.pipeline = pipeline("summarization")
                logger.info("Loaded transformers summarization model.")
            except Exception as e:
                logger.error(f"Error loading transformers summarization model: {e}")
                self.use_transformers = False

    def summarize(self, text, max_length=100, min_length=30):
        """
        Summarize the given text.

        Args:
            text (str): The text to summarize
            max_length (int): Maximum length of the summary
            min_length (int): Minimum length of the summary

        Returns:
            str: Summarized text
        """
        if not text:
            return ""

        if self.use_transformers and self.pipeline:
            try:
                # Use transformers summarization pipeline
                result = self.pipeline(text, max_length=max_length, min_length=min_length, do_sample=False)
                return result[0]["summary_text"]
            except Exception as e:
                logger.error(f"Error in transformers summarization: {e}")

        # Fallback to extractive summarization
        import re
        from collections import Counter

        # Split text into sentences
        sentences = re.split(r'[.!?]', text)
        sentences = [sentence.strip() for sentence in sentences if sentence.strip()]

        if not sentences:
            return ""

        # If text is already short, return it as is
        if len(sentences) <= 3:
            return text

        # Tokenize sentences
        words = [word.lower() for sentence in sentences for word in re.findall(r'\w+', sentence)]

        # Count word frequencies
        word_freq = Counter(words)

        # Calculate sentence scores based on word frequencies
        sentence_scores = []
        for sentence in sentences:
            score = sum(word_freq[word.lower()] for word in re.findall(r'\w+', sentence))
            sentence_scores.append((sentence, score))

        # Sort sentences by score
        sentence_scores.sort(key=lambda x: x[1], reverse=True)

        # Select top sentences
        num_sentences = max(1, min(3, len(sentences) // 3))
        top_sentences = [sentence for sentence, _ in sentence_scores[:num_sentences]]

        # Reorder sentences based on their original position
        ordered_sentences = []
        for sentence in sentences:
            if sentence in top_sentences:
                ordered_sentences.append(sentence)
                if len(ordered_sentences) >= num_sentences:
                    break

        # Join sentences
        summary = ". ".join(ordered_sentences)
        if not summary.endswith((".", "!", "?")):
            summary += "."

        return summary

class Translator:
    """
    Text translation using transformers.
    """

    def __init__(self, model_dir="data/models/translation", use_transformers=True):
        """
        Initialize the translator.

        Args:
            model_dir (str): Directory to save/load model files
            use_transformers (bool): Whether to use transformers models (if available)
        """
        self.model_dir = model_dir
        self.use_transformers = use_transformers and TRANSFORMERS_AVAILABLE

        # Create model directory if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)

        # Initialize the models
        self.pipelines = {}
        self.supported_languages = ["fr", "es", "de", "it", "zh", "ja", "ru", "ar"]

        self._initialize_models()

    def _initialize_models(self):
        """
        Initialize or load the translation models.
        """
        if self.use_transformers:
            try:
                # Load English to French translation model
                self.pipelines["fr"] = pipeline("translation_en_to_fr")
                logger.info("Loaded transformers English to French translation model.")

                # Load English to Spanish translation model
                self.pipelines["es"] = pipeline("translation_en_to_es")
                logger.info("Loaded transformers English to Spanish translation model.")

                # Note: In a real implementation, you would load models for all supported languages
                # For now, we'll just load these two as examples
            except Exception as e:
                logger.error(f"Error loading transformers translation models: {e}")
                self.use_transformers = False

    def translate(self, text, target_language="fr"):
        """
        Translate the given text to the target language.

        Args:
            text (str): The text to translate
            target_language (str): The target language code (e.g., "fr" for French)

        Returns:
            str: Translated text
        """
        if not text:
            return ""

        if target_language not in self.supported_languages:
            return f"Translation to {target_language} is not supported."

        if self.use_transformers and target_language in self.pipelines:
            try:
                # Use transformers translation pipeline
                result = self.pipelines[target_language](text, max_length=400)
                return result[0]["translation_text"]
            except Exception as e:
                logger.error(f"Error in transformers translation: {e}")

        # Fallback to simple translation (just a placeholder)
        return f"[Translation to {target_language} not available]"

class VoiceProcessor:
    """
    Voice processing using speech recognition and text-to-speech.
    """

    def __init__(self):
        """
        Initialize the voice processor.
        """
        self.speech_recognition_available = SPEECH_RECOGNITION_AVAILABLE
        self.tts_available = TTS_AVAILABLE

        # Initialize speech recognition
        self.recognizer = None
        if self.speech_recognition_available:
            try:
                self.recognizer = sr.Recognizer()
                logger.info("Initialized speech recognition.")
            except Exception as e:
                logger.error(f"Error initializing speech recognition: {e}")
                self.speech_recognition_available = False

        # Initialize text-to-speech
        self.tts_engine = None
        if self.tts_available:
            try:
                self.tts_engine = pyttsx3.init()
                logger.info("Initialized text-to-speech.")
            except Exception as e:
                logger.error(f"Error initializing text-to-speech: {e}")
                self.tts_available = False

    def recognize_speech(self, audio_data=None, timeout=5):
        """
        Recognize speech from audio data or microphone.

        Args:
            audio_data (bytes, optional): Audio data to recognize
            timeout (int): Timeout for microphone recording in seconds

        Returns:
            dict: Recognition result with text and confidence
        """
        if not self.speech_recognition_available:
            return {"success": False, "error": "Speech recognition not available"}

        try:
            if audio_data is None:
                # Record from microphone
                with sr.Microphone() as source:
                    self.recognizer.adjust_for_ambient_noise(source)
                    logger.info(f"Listening for {timeout} seconds...")
                    audio_data = self.recognizer.listen(source, timeout=timeout)

            # Recognize speech
            text = self.recognizer.recognize_google(audio_data)
            return {"success": True, "text": text, "confidence": 0.8}  # Google doesn't provide confidence
        except sr.WaitTimeoutError:
            return {"success": False, "error": "Listening timed out"}
        except sr.UnknownValueError:
            return {"success": False, "error": "Could not understand audio"}
        except sr.RequestError as e:
            return {"success": False, "error": f"Recognition service error: {e}"}
        except Exception as e:
            logger.error(f"Error in speech recognition: {e}")
            return {"success": False, "error": str(e)}

    def text_to_speech(self, text, rate=200, volume=1.0, voice=None):
        """
        Convert text to speech.

        Args:
            text (str): Text to convert to speech
            rate (int): Speech rate (words per minute)
            volume (float): Volume (0.0 to 1.0)
            voice (str, optional): Voice ID

        Returns:
            dict: Result with success status
        """
        if not self.tts_available:
            return {"success": False, "error": "Text-to-speech not available"}

        if not text:
            return {"success": False, "error": "No text provided"}

        try:
            # Set properties
            self.tts_engine.setProperty('rate', rate)
            self.tts_engine.setProperty('volume', volume)

            if voice:
                self.tts_engine.setProperty('voice', voice)

            # Say the text
            self.tts_engine.say(text)
            self.tts_engine.runAndWait()

            return {"success": True}
        except Exception as e:
            logger.error(f"Error in text-to-speech: {e}")
            return {"success": False, "error": str(e)}

    def get_available_voices(self):
        """
        Get available voices.

        Returns:
            list: List of available voices
        """
        if not self.tts_available:
            return []

        try:
            voices = self.tts_engine.getProperty('voices')
            return [{"id": voice.id, "name": voice.name, "languages": voice.languages} for voice in voices]
        except Exception as e:
            logger.error(f"Error getting available voices: {e}")
            return []
