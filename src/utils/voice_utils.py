"""
Voice Utilities for NoahAI

This module provides speech recognition and text-to-speech capabilities.
"""

import os
import logging
import tempfile
from typing import Optional, Dict, Any, List, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/voice.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("voice_utils")

# Try to import speech recognition
try:
    import speech_recognition as sr
    SPEECH_RECOGNITION_AVAILABLE = True
except ImportError:
    SPEECH_RECOGNITION_AVAILABLE = False
    logger.warning("SpeechRecognition not available. Speech recognition will be disabled.")

# Try to import text-to-speech
try:
    import pyttsx3
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False
    logger.warning("pyttsx3 not available. Text-to-speech will be disabled.")


class VoiceUtils:
    """
    Utility class for voice recognition and synthesis.
    """
    
    def __init__(self, language="en-US"):
        """
        Initialize voice utilities.
        
        Args:
            language (str): Language code for speech recognition
        """
        self.language = language
        self.recognizer = None
        self.engine = None
        
        # Initialize speech recognition if available
        if SPEECH_RECOGNITION_AVAILABLE:
            try:
                self.recognizer = sr.Recognizer()
                logger.info("Speech recognition initialized.")
            except Exception as e:
                logger.error(f"Error initializing speech recognition: {e}")
                self.recognizer = None
        
        # Initialize text-to-speech if available
        if TTS_AVAILABLE:
            try:
                self.engine = pyttsx3.init()
                
                # Get available voices
                voices = self.engine.getProperty('voices')
                
                # Set a default voice (usually the first one)
                if voices:
                    self.engine.setProperty('voice', voices[0].id)
                
                # Set default properties
                self.engine.setProperty('rate', 150)  # Speed
                self.engine.setProperty('volume', 1.0)  # Volume (0.0 to 1.0)
                
                logger.info("Text-to-speech initialized.")
            except Exception as e:
                logger.error(f"Error initializing text-to-speech: {e}")
                self.engine = None
    
    def recognize_speech(self, audio_file=None, timeout=5) -> Dict[str, Any]:
        """
        Recognize speech from microphone or audio file.
        
        Args:
            audio_file (str, optional): Path to audio file. If None, use microphone.
            timeout (int): Timeout for microphone recording in seconds
            
        Returns:
            dict: Recognition result with text and confidence
        """
        if not SPEECH_RECOGNITION_AVAILABLE or not self.recognizer:
            return {"error": "Speech recognition not available"}
        
        try:
            # Use microphone if no audio file is provided
            if not audio_file:
                with sr.Microphone() as source:
                    logger.info("Listening...")
                    
                    # Adjust for ambient noise
                    self.recognizer.adjust_for_ambient_noise(source)
                    
                    # Listen for audio
                    audio = self.recognizer.listen(source, timeout=timeout)
            else:
                # Use audio file
                with sr.AudioFile(audio_file) as source:
                    audio = self.recognizer.record(source)
            
            # Recognize speech
            text = self.recognizer.recognize_google(audio, language=self.language, show_all=True)
            
            # Format the result
            if isinstance(text, dict) and "alternative" in text:
                alternatives = text["alternative"]
                if alternatives:
                    best_result = alternatives[0]
                    return {
                        "text": best_result["transcript"],
                        "confidence": best_result.get("confidence", 1.0),
                        "alternatives": [alt["transcript"] for alt in alternatives[1:]]
                    }
            
            # If no structured result, return the text as a string
            if isinstance(text, str):
                return {
                    "text": text,
                    "confidence": 1.0,
                    "alternatives": []
                }
            
            # Fallback for empty results
            return {
                "text": "",
                "confidence": 0.0,
                "alternatives": []
            }
        
        except sr.UnknownValueError:
            logger.warning("Speech recognition could not understand audio")
            return {"error": "Could not understand audio"}
        
        except sr.RequestError as e:
            logger.error(f"Speech recognition service error: {e}")
            return {"error": f"Recognition service error: {e}"}
        
        except Exception as e:
            logger.error(f"Speech recognition error: {e}")
            return {"error": str(e)}
    
    def text_to_speech(self, text, output_file=None, voice_id=None, rate=None, volume=None) -> Dict[str, Any]:
        """
        Convert text to speech.
        
        Args:
            text (str): Text to convert to speech
            output_file (str, optional): Path to save audio file. If None, play directly.
            voice_id (str, optional): Voice ID to use
            rate (int, optional): Speech rate (words per minute)
            volume (float, optional): Volume (0.0 to 1.0)
            
        Returns:
            dict: Result of the operation
        """
        if not TTS_AVAILABLE or not self.engine:
            return {"error": "Text-to-speech not available"}
        
        try:
            # Set voice if provided
            if voice_id:
                self.engine.setProperty('voice', voice_id)
            
            # Set rate if provided
            if rate:
                self.engine.setProperty('rate', rate)
            
            # Set volume if provided
            if volume is not None:
                self.engine.setProperty('volume', max(0.0, min(1.0, volume)))
            
            # Save to file if output_file is provided
            if output_file:
                self.engine.save_to_file(text, output_file)
                self.engine.runAndWait()
                
                return {
                    "status": "success",
                    "file": output_file
                }
            else:
                # Play directly
                self.engine.say(text)
                self.engine.runAndWait()
                
                return {
                    "status": "success"
                }
        
        except Exception as e:
            logger.error(f"Text-to-speech error: {e}")
            return {"error": str(e)}
    
    def get_available_voices(self) -> List[Dict[str, Any]]:
        """
        Get available voices for text-to-speech.
        
        Returns:
            list: List of available voices
        """
        if not TTS_AVAILABLE or not self.engine:
            return []
        
        try:
            voices = self.engine.getProperty('voices')
            
            return [
                {
                    "id": voice.id,
                    "name": voice.name,
                    "languages": voice.languages,
                    "gender": voice.gender
                }
                for voice in voices
            ]
        
        except Exception as e:
            logger.error(f"Error getting available voices: {e}")
            return []
    
    def set_voice_properties(self, voice_id=None, rate=None, volume=None) -> Dict[str, Any]:
        """
        Set voice properties for text-to-speech.
        
        Args:
            voice_id (str, optional): Voice ID to use
            rate (int, optional): Speech rate (words per minute)
            volume (float, optional): Volume (0.0 to 1.0)
            
        Returns:
            dict: Current voice properties
        """
        if not TTS_AVAILABLE or not self.engine:
            return {"error": "Text-to-speech not available"}
        
        try:
            # Set voice if provided
            if voice_id:
                self.engine.setProperty('voice', voice_id)
            
            # Set rate if provided
            if rate:
                self.engine.setProperty('rate', rate)
            
            # Set volume if provided
            if volume is not None:
                self.engine.setProperty('volume', max(0.0, min(1.0, volume)))
            
            # Get current properties
            current_voice = self.engine.getProperty('voice')
            current_rate = self.engine.getProperty('rate')
            current_volume = self.engine.getProperty('volume')
            
            return {
                "voice": current_voice,
                "rate": current_rate,
                "volume": current_volume
            }
        
        except Exception as e:
            logger.error(f"Error setting voice properties: {e}")
            return {"error": str(e)}
