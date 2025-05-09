"""
Integration Manager for NoahAI

This module provides a central manager for all external integrations.
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/integrations.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("integration_manager")

# Import integrations
try:
    from src.integrations.weather_api import WeatherAPI
    WEATHER_API_AVAILABLE = True
except ImportError:
    WEATHER_API_AVAILABLE = False
    logger.warning("Weather API not available.")

try:
    from src.integrations.calendar_api import CalendarAPI
    CALENDAR_API_AVAILABLE = True
except ImportError:
    CALENDAR_API_AVAILABLE = False
    logger.warning("Calendar API not available.")

try:
    from src.integrations.messaging_api import EmailAPI, SMSAPI
    MESSAGING_API_AVAILABLE = True
except ImportError:
    MESSAGING_API_AVAILABLE = False
    logger.warning("Messaging API not available.")

class IntegrationManager:
    """
    Integration Manager for NoahAI.

    This class provides a central manager for all external integrations.
    """

    def __init__(self, config_file="config/integrations.json"):
        """
        Initialize the Integration Manager.

        Args:
            config_file (str): Path to the integrations configuration file
        """
        self.config_file = config_file
        self.config = self._load_config()

        # Initialize integrations
        self.weather_api = None
        self.calendar_api = None
        self.email_api = None
        self.sms_api = None

        self._initialize_integrations()

    def _load_config(self):
        """
        Load the integrations configuration.

        Returns:
            dict: Configuration data
        """
        # Default configuration
        default_config = {
            "weather": {
                "enabled": True,
                "api_key": os.environ.get("WEATHER_API_KEY", ""),
                "api_provider": "openweathermap"
            },
            "calendar": {
                "enabled": True,
                "credentials_file": os.environ.get("GOOGLE_CALENDAR_CREDENTIALS", ""),
                "token_file": "data/token.json",
                "calendar_id": "primary"
            },
            "email": {
                "enabled": True,
                "smtp_server": os.environ.get("SMTP_SERVER", ""),
                "smtp_port": int(os.environ.get("SMTP_PORT", 587)),
                "username": os.environ.get("SMTP_USERNAME", ""),
                "password": os.environ.get("SMTP_PASSWORD", ""),
                "use_tls": True
            },
            "sms": {
                "enabled": True,
                "account_sid": os.environ.get("TWILIO_ACCOUNT_SID", ""),
                "auth_token": os.environ.get("TWILIO_AUTH_TOKEN", ""),
                "from_number": os.environ.get("TWILIO_PHONE_NUMBER", "")
            }
        }

        # Try to load configuration from file
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    config = json.load(f)

                # Merge with default config
                for section in default_config:
                    if section in config:
                        for key in default_config[section]:
                            if key not in config[section]:
                                config[section][key] = default_config[section][key]
                    else:
                        config[section] = default_config[section]

                return config
            except Exception as e:
                logger.error(f"Error loading integrations config: {e}")

        # Create config directory if it doesn't exist
        os.makedirs(os.path.dirname(self.config_file), exist_ok=True)

        # Save default config
        try:
            with open(self.config_file, 'w') as f:
                json.dump(default_config, f, indent=4)
        except Exception as e:
            logger.error(f"Error saving default integrations config: {e}")

        return default_config

    def _initialize_integrations(self):
        """
        Initialize all enabled integrations.
        """
        # Initialize Weather API
        if WEATHER_API_AVAILABLE and self.config["weather"]["enabled"]:
            try:
                self.weather_api = WeatherAPI(
                    api_key=self.config["weather"]["api_key"],
                    api_provider=self.config["weather"]["api_provider"]
                )
                logger.info("Weather API initialized.")
            except Exception as e:
                logger.error(f"Error initializing Weather API: {e}")

        # Initialize Calendar API
        if CALENDAR_API_AVAILABLE and self.config["calendar"]["enabled"]:
            try:
                self.calendar_api = CalendarAPI(
                    credentials_file=self.config["calendar"]["credentials_file"],
                    token_file=self.config["calendar"]["token_file"],
                    calendar_id=self.config["calendar"]["calendar_id"]
                )
                logger.info("Calendar API initialized.")
            except Exception as e:
                logger.error(f"Error initializing Calendar API: {e}")

        # Initialize Email API
        if MESSAGING_API_AVAILABLE and self.config["email"]["enabled"]:
            try:
                self.email_api = EmailAPI(
                    smtp_server=self.config["email"]["smtp_server"],
                    smtp_port=self.config["email"]["smtp_port"],
                    username=self.config["email"]["username"],
                    password=self.config["email"]["password"],
                    use_tls=self.config["email"]["use_tls"]
                )
                logger.info("Email API initialized.")
            except Exception as e:
                logger.error(f"Error initializing Email API: {e}")

        # Initialize SMS API
        if MESSAGING_API_AVAILABLE and self.config["sms"]["enabled"]:
            try:
                self.sms_api = SMSAPI(
                    account_sid=self.config["sms"]["account_sid"],
                    auth_token=self.config["sms"]["auth_token"],
                    from_number=self.config["sms"]["from_number"]
                )
                logger.info("SMS API initialized.")
            except Exception as e:
                logger.error(f"Error initializing SMS API: {e}")

    def get_weather(self, location):
        """
        Get current weather for a location.

        Args:
            location (str): City name or location

        Returns:
            dict: Weather data or error message
        """
        if not self.weather_api:
            return {"error": "Weather API not available."}

        return self.weather_api.get_current_weather(location)

    def get_forecast(self, location, days=5):
        """
        Get weather forecast for a location.

        Args:
            location (str): City name or location
            days (int): Number of days for the forecast

        Returns:
            dict: Forecast data or error message
        """
        if not self.weather_api:
            return {"error": "Weather API not available."}

        return self.weather_api.get_forecast(location, days)

    def get_events(self, start_date=None, end_date=None, max_results=10):
        """
        Get events from the calendar.

        Args:
            start_date (datetime, optional): Start date for events
            end_date (datetime, optional): End date for events
            max_results (int): Maximum number of events to return

        Returns:
            list: List of events or error message
        """
        if not self.calendar_api:
            return {"error": "Calendar API not available."}

        return self.calendar_api.get_events(start_date, end_date, max_results)

    def create_event(self, summary, start_time, end_time, description=None, location=None):
        """
        Create a new event in the calendar.

        Args:
            summary (str): Event title
            start_time (datetime): Event start time
            end_time (datetime): Event end time
            description (str, optional): Event description
            location (str, optional): Event location

        Returns:
            dict: Created event or error message
        """
        if not self.calendar_api:
            return {"error": "Calendar API not available."}

        return self.calendar_api.create_event(summary, start_time, end_time, description, location)

    def update_event(self, event_id, summary=None, start_time=None, end_time=None, description=None, location=None):
        """
        Update an existing event in the calendar.

        Args:
            event_id (str): ID of the event to update
            summary (str, optional): New event title
            start_time (datetime, optional): New event start time
            end_time (datetime, optional): New event end time
            description (str, optional): New event description
            location (str, optional): New event location

        Returns:
            dict: Updated event or error message
        """
        if not self.calendar_api:
            return {"error": "Calendar API not available."}

        return self.calendar_api.update_event(event_id, summary, start_time, end_time, description, location)

    def delete_event(self, event_id):
        """
        Delete an event from the calendar.

        Args:
            event_id (str): ID of the event to delete

        Returns:
            dict: Success message or error message
        """
        if not self.calendar_api:
            return {"error": "Calendar API not available."}

        return self.calendar_api.delete_event(event_id)

    def send_email(self, to_email, subject, body, from_email=None, html_body=None):
        """
        Send an email.

        Args:
            to_email (str): Recipient email address
            subject (str): Email subject
            body (str): Email body (plain text)
            from_email (str, optional): Sender email address
            html_body (str, optional): HTML version of the email body

        Returns:
            dict: Success message or error message
        """
        if not self.email_api:
            return {"error": "Email API not available."}

        return self.email_api.send_email(to_email, subject, body, from_email, html_body)

    def send_sms(self, to_number, message):
        """
        Send an SMS message.

        Args:
            to_number (str): Recipient phone number
            message (str): SMS message

        Returns:
            dict: Success message or error message
        """
        if not self.sms_api:
            return {"error": "SMS API not available."}

        return self.sms_api.send_sms(to_number, message)

    def get_sms_status(self, sms_sid):
        """
        Get the status of an SMS message.

        Args:
            sms_sid (str): Twilio SMS SID

        Returns:
            dict: SMS status or error message
        """
        if not self.sms_api:
            return {"error": "SMS API not available."}

        return self.sms_api.get_sms_status(sms_sid)

    def handle_intent(self, intent, entities=None, user_input=None):
        """
        Handle an intent with the appropriate integration.

        Args:
            intent (str): The intent to handle
            entities (dict, optional): Entities extracted from user input
            user_input (str, optional): The original user input

        Returns:
            dict: Result of the integration action
        """
        entities = entities or {}

        # Handle weather intent
        if intent == "weather":
            location = None

            # Extract location from entities
            if "location" in entities and entities["location"]:
                location = entities["location"][0]["value"]

            # If no location found, try to extract from user input
            if not location and user_input:
                # Simple extraction - in a real implementation, you would use NLP
                if "in " in user_input:
                    location = user_input.split("in ")[1].split(" ")[0]
                elif "at " in user_input:
                    location = user_input.split("at ")[1].split(" ")[0]
                elif "for " in user_input:
                    location = user_input.split("for ")[1].split(" ")[0]

            # Default to a common location if none found
            location = location or "New York"

            # Get weather data
            return self.get_weather(location)

        # Handle forecast intent
        elif intent == "forecast":
            location = None
            days = 5

            # Extract location from entities
            if "location" in entities and entities["location"]:
                location = entities["location"][0]["value"]

            # Extract days from entities
            if "number" in entities and entities["number"]:
                try:
                    days = int(entities["number"][0]["value"])
                    days = max(1, min(days, 7))  # Limit to 1-7 days
                except (ValueError, TypeError):
                    pass

            # If no location found, try to extract from user input
            if not location and user_input:
                # Simple extraction - in a real implementation, you would use NLP
                if "in " in user_input:
                    location = user_input.split("in ")[1].split(" ")[0]
                elif "at " in user_input:
                    location = user_input.split("at ")[1].split(" ")[0]
                elif "for " in user_input:
                    location = user_input.split("for ")[1].split(" ")[0]

            # Default to a common location if none found
            location = location or "New York"

            # Get forecast data
            return self.get_forecast(location, days)

        # Handle calendar intent
        elif intent == "calendar":
            # Get upcoming events
            return self.get_events()

        # Handle unknown intent
        else:
            return {"error": f"Unknown intent: {intent}"}
