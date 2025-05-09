"""
Advanced Text Embedding for NoahAI

This module provides advanced text embedding capabilities using transformer models
for better semantic understanding of text.
"""

import os
import numpy as np
import logging
from typing import List, Dict, Any, Union, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/text_embedding.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("text_embedding")

# Try to import transformer libraries
try:
    import tensorflow as tf
    import tensorflow_hub as hub
    TENSORFLOW_HUB_AVAILABLE = True
except ImportError:
    TENSORFLOW_HUB_AVAILABLE = False
    logger.warning("TensorFlow Hub not available. Using fallback embedding methods.")

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    SentenceTransformer = None  # Define as None to avoid NameError
    logger.warning("Sentence Transformers not available. Using fallback embedding methods.")

class TextEmbedding:
    """
    Advanced text embedding using transformer models.
    """

    def __init__(self, model_name="all-MiniLM-L6-v2", embedding_size=384,
                 cache_dir="data/embeddings", use_tensorflow_hub=False):
        """
        Initialize the text embedding model.

        Args:
            model_name (str): Name of the embedding model to use
            embedding_size (int): Size of the embedding vectors
            cache_dir (str): Directory to cache embeddings
            use_tensorflow_hub (bool): Whether to use TensorFlow Hub models
        """
        self.model_name = model_name
        self.embedding_size = embedding_size
        self.cache_dir = cache_dir
        self.use_tensorflow_hub = use_tensorflow_hub

        # Create cache directory if it doesn't exist
        os.makedirs(self.cache_dir, exist_ok=True)

        # Initialize embedding model
        self.model = None
        self._initialize_model()

    def _initialize_model(self):
        """
        Initialize the embedding model based on available libraries.
        """
        # Try to use Sentence Transformers (preferred)
        if SENTENCE_TRANSFORMERS_AVAILABLE and not self.use_tensorflow_hub:
            try:
                self.model = SentenceTransformer(self.model_name)
                logger.info(f"Initialized Sentence Transformers model: {self.model_name}")
                return
            except Exception as e:
                logger.error(f"Error initializing Sentence Transformers model: {e}")

        # Try to use TensorFlow Hub
        if TENSORFLOW_HUB_AVAILABLE and self.use_tensorflow_hub:
            try:
                # Use Universal Sentence Encoder
                self.model = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
                logger.info("Initialized Universal Sentence Encoder from TensorFlow Hub")
                return
            except Exception as e:
                logger.error(f"Error initializing TensorFlow Hub model: {e}")

        logger.warning("No advanced embedding model available. Using fallback method.")

    def get_embedding(self, text: str) -> np.ndarray:
        """
        Get embedding vector for a text.

        Args:
            text (str): Input text

        Returns:
            numpy.ndarray: Embedding vector
        """
        if not text:
            # Return zero vector for empty text
            return np.zeros(self.embedding_size)

        # Use Sentence Transformers
        if SENTENCE_TRANSFORMERS_AVAILABLE and self.model is not None:
            try:
                return self.model.encode(text)
            except Exception as e:
                logger.error(f"Error encoding with Sentence Transformers: {e}")

        # Use TensorFlow Hub
        elif TENSORFLOW_HUB_AVAILABLE and self.use_tensorflow_hub and self.model is not None:
            try:
                return self.model([text])[0].numpy()
            except Exception as e:
                logger.error(f"Error encoding with TensorFlow Hub: {e}")

        # Fallback to simple embedding
        return self._fallback_embedding(text)

    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Get embedding vectors for multiple texts.

        Args:
            texts (list): List of input texts

        Returns:
            numpy.ndarray: Array of embedding vectors
        """
        if not texts:
            return np.array([])

        # Use Sentence Transformers
        if SENTENCE_TRANSFORMERS_AVAILABLE and self.model is not None:
            try:
                return self.model.encode(texts)
            except Exception as e:
                logger.error(f"Error batch encoding with Sentence Transformers: {e}")

        # Use TensorFlow Hub
        elif TENSORFLOW_HUB_AVAILABLE and self.use_tensorflow_hub and self.model is not None:
            try:
                return self.model(texts).numpy()
            except Exception as e:
                logger.error(f"Error batch encoding with TensorFlow Hub: {e}")

        # Fallback to simple embedding
        return np.array([self._fallback_embedding(text) for text in texts])

    def _fallback_embedding(self, text: str) -> np.ndarray:
        """
        Simple fallback embedding method when advanced models are not available.

        Args:
            text (str): Input text

        Returns:
            numpy.ndarray: Simple embedding vector
        """
        # Create a simple bag-of-words representation
        words = text.lower().split()
        # Use hash of words to create a simple embedding
        embedding = np.zeros(self.embedding_size)

        for i, word in enumerate(words):
            # Use hash of word to determine the index
            idx = hash(word) % self.embedding_size
            embedding[idx] += 1

        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding

    def calculate_similarity(self, text1: str, text2: str) -> float:
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

        # Get embeddings
        embedding1 = self.get_embedding(text1)
        embedding2 = self.get_embedding(text2)

        # Calculate cosine similarity
        similarity = self._cosine_similarity(embedding1, embedding2)

        return similarity

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two vectors.

        Args:
            vec1 (numpy.ndarray): First vector
            vec2 (numpy.ndarray): Second vector

        Returns:
            float: Cosine similarity
        """
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return np.dot(vec1, vec2) / (norm1 * norm2)

    def find_most_similar(self, query: str, candidates: List[str], top_n: int = 1) -> List[Dict[str, Any]]:
        """
        Find the most similar texts from a list of candidates.

        Args:
            query (str): Query text
            candidates (list): List of candidate texts
            top_n (int): Number of top results to return

        Returns:
            list: List of dictionaries with text and similarity score
        """
        if not query or not candidates:
            return []

        # Get query embedding
        query_embedding = self.get_embedding(query)

        # Get candidate embeddings
        candidate_embeddings = self.get_embeddings(candidates)

        # Calculate similarities
        similarities = []
        for i, embedding in enumerate(candidate_embeddings):
            similarity = self._cosine_similarity(query_embedding, embedding)
            similarities.append((candidates[i], similarity))

        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)

        # Return top N results
        return [{"text": text, "similarity": score} for text, score in similarities[:top_n]]
