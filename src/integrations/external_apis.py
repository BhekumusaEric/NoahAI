"""
External API Integrations for NoahAI

This module provides integrations with external APIs for:
- Weather data
- News
- Calendar
- Email
"""

import os
import json
import logging
import datetime
import requests
from typing import Dict, List, Any, Optional, Union
from dotenv import load_dotenv

# Google API imports
try:
    from googleapiclient.discovery import build
    from google_auth_oauthlib.flow import InstalledAppFlow
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    GOOGLE_APIS_AVAILABLE = True
except ImportError:
    GOOGLE_APIS_AVAILABLE = False
    print("Google APIs not available. Calendar and Gmail integrations will be disabled.")

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/integrations.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("external_apis")

class WeatherAPI:
    """
    Integration with OpenWeatherMap API for weather data.
    """
    
    def __init__(self, api_key=None):
        """
        Initialize the WeatherAPI.
        
        Args:
            api_key (str, optional): OpenWeatherMap API key. If not provided, will try to get from environment.
        """
        self.api_key = api_key or os.getenv('OPENWEATHER_API_KEY')
        self.base_url = "https://api.openweathermap.org/data/2.5"
        
        if not self.api_key:
            logger.warning("OpenWeatherMap API key not found. Weather functionality will be limited.")
    
    def get_current_weather(self, location: str, units: str = "metric") -> Dict[str, Any]:
        """
        Get current weather for a location.
        
        Args:
            location (str): City name or location
            units (str): Units of measurement (metric, imperial, standard)
            
        Returns:
            dict: Weather data
        """
        if not self.api_key:
            return {"error": "API key not configured"}
        
        try:
            url = f"{self.base_url}/weather"
            params = {
                "q": location,
                "appid": self.api_key,
                "units": units
            }
            
            response = requests.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            # Format the response
            weather = {
                "location": data["name"],
                "country": data["sys"]["country"],
                "temperature": data["main"]["temp"],
                "feels_like": data["main"]["feels_like"],
                "description": data["weather"][0]["description"],
                "humidity": data["main"]["humidity"],
                "wind_speed": data["wind"]["speed"],
                "timestamp": datetime.datetime.now().isoformat()
            }
            
            return weather
        
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching weather data: {e}")
            return {"error": str(e)}
    
    def get_forecast(self, location: str, days: int = 5, units: str = "metric") -> Dict[str, Any]:
        """
        Get weather forecast for a location.
        
        Args:
            location (str): City name or location
            days (int): Number of days for forecast (max 5)
            units (str): Units of measurement (metric, imperial, standard)
            
        Returns:
            dict: Forecast data
        """
        if not self.api_key:
            return {"error": "API key not configured"}
        
        try:
            url = f"{self.base_url}/forecast"
            params = {
                "q": location,
                "appid": self.api_key,
                "units": units,
                "cnt": min(days * 8, 40)  # 8 forecasts per day (3-hour intervals)
            }
            
            response = requests.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            # Format the response
            forecast = {
                "location": data["city"]["name"],
                "country": data["city"]["country"],
                "forecasts": []
            }
            
            # Group forecasts by day
            current_day = None
            day_forecasts = []
            
            for item in data["list"]:
                date = datetime.datetime.fromtimestamp(item["dt"])
                day = date.strftime("%Y-%m-%d")
                
                if current_day != day:
                    if current_day is not None:
                        forecast["forecasts"].append({
                            "date": current_day,
                            "items": day_forecasts
                        })
                    current_day = day
                    day_forecasts = []
                
                day_forecasts.append({
                    "time": date.strftime("%H:%M"),
                    "temperature": item["main"]["temp"],
                    "description": item["weather"][0]["description"],
                    "humidity": item["main"]["humidity"],
                    "wind_speed": item["wind"]["speed"]
                })
            
            # Add the last day
            if current_day is not None and day_forecasts:
                forecast["forecasts"].append({
                    "date": current_day,
                    "items": day_forecasts
                })
            
            return forecast
        
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching forecast data: {e}")
            return {"error": str(e)}


class NewsAPI:
    """
    Integration with NewsAPI for news data.
    """
    
    def __init__(self, api_key=None):
        """
        Initialize the NewsAPI.
        
        Args:
            api_key (str, optional): NewsAPI key. If not provided, will try to get from environment.
        """
        self.api_key = api_key or os.getenv('NEWS_API_KEY')
        self.base_url = "https://newsapi.org/v2"
        
        if not self.api_key:
            logger.warning("NewsAPI key not found. News functionality will be limited.")
    
    def get_top_headlines(self, country: str = "us", category: str = None, query: str = None, page_size: int = 5) -> Dict[str, Any]:
        """
        Get top headlines.
        
        Args:
            country (str): Country code (e.g., us, gb, au)
            category (str, optional): Category (business, entertainment, health, science, sports, technology)
            query (str, optional): Search query
            page_size (int): Number of results to return
            
        Returns:
            dict: News data
        """
        if not self.api_key:
            return {"error": "API key not configured"}
        
        try:
            url = f"{self.base_url}/top-headlines"
            params = {
                "apiKey": self.api_key,
                "country": country,
                "pageSize": min(page_size, 20)
            }
            
            if category:
                params["category"] = category
            
            if query:
                params["q"] = query
            
            response = requests.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            # Format the response
            news = {
                "total_results": data["totalResults"],
                "articles": []
            }
            
            for article in data["articles"]:
                news["articles"].append({
                    "title": article["title"],
                    "source": article["source"]["name"],
                    "description": article["description"],
                    "url": article["url"],
                    "published_at": article["publishedAt"]
                })
            
            return news
        
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching news data: {e}")
            return {"error": str(e)}
    
    def search_news(self, query: str, from_date: str = None, to_date: str = None, sort_by: str = "relevancy", page_size: int = 5) -> Dict[str, Any]:
        """
        Search for news articles.
        
        Args:
            query (str): Search query
            from_date (str, optional): Start date (YYYY-MM-DD)
            to_date (str, optional): End date (YYYY-MM-DD)
            sort_by (str): Sort order (relevancy, popularity, publishedAt)
            page_size (int): Number of results to return
            
        Returns:
            dict: News data
        """
        if not self.api_key:
            return {"error": "API key not configured"}
        
        try:
            url = f"{self.base_url}/everything"
            params = {
                "apiKey": self.api_key,
                "q": query,
                "sortBy": sort_by,
                "pageSize": min(page_size, 20)
            }
            
            if from_date:
                params["from"] = from_date
            
            if to_date:
                params["to"] = to_date
            
            response = requests.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            # Format the response
            news = {
                "total_results": data["totalResults"],
                "articles": []
            }
            
            for article in data["articles"]:
                news["articles"].append({
                    "title": article["title"],
                    "source": article["source"]["name"],
                    "description": article["description"],
                    "url": article["url"],
                    "published_at": article["publishedAt"]
                })
            
            return news
        
        except requests.exceptions.RequestException as e:
            logger.error(f"Error searching news: {e}")
            return {"error": str(e)}


class CalendarAPI:
    """
    Integration with Google Calendar API.
    """
    
    def __init__(self, credentials_file="credentials.json", token_file="token.json"):
        """
        Initialize the CalendarAPI.
        
        Args:
            credentials_file (str): Path to Google API credentials file
            token_file (str): Path to token file
        """
        self.credentials_file = credentials_file
        self.token_file = token_file
        self.service = None
        
        if not GOOGLE_APIS_AVAILABLE:
            logger.warning("Google APIs not available. Calendar functionality will be limited.")
            return
        
        # Initialize the API
        self._initialize_api()
    
    def _initialize_api(self):
        """
        Initialize the Google Calendar API.
        """
        if not GOOGLE_APIS_AVAILABLE:
            return
        
        try:
            creds = None
            
            # Check if token file exists
            if os.path.exists(self.token_file):
                creds = Credentials.from_authorized_user_info(
                    json.load(open(self.token_file)), 
                    ["https://www.googleapis.com/auth/calendar"]
                )
            
            # If credentials don't exist or are invalid, get new ones
            if not creds or not creds.valid:
                if creds and creds.expired and creds.refresh_token:
                    creds.refresh(Request())
                else:
                    flow = InstalledAppFlow.from_client_secrets_file(
                        self.credentials_file, 
                        ["https://www.googleapis.com/auth/calendar"]
                    )
                    creds = flow.run_local_server(port=0)
                
                # Save the credentials
                with open(self.token_file, "w") as token:
                    token.write(creds.to_json())
            
            # Build the service
            self.service = build("calendar", "v3", credentials=creds)
            logger.info("Google Calendar API initialized successfully.")
        
        except Exception as e:
            logger.error(f"Error initializing Google Calendar API: {e}")
            self.service = None
    
    def get_events(self, max_results=10, time_min=None, time_max=None) -> Dict[str, Any]:
        """
        Get upcoming events from the calendar.
        
        Args:
            max_results (int): Maximum number of events to return
            time_min (str, optional): Start time (RFC3339 timestamp)
            time_max (str, optional): End time (RFC3339 timestamp)
            
        Returns:
            dict: Calendar events
        """
        if not GOOGLE_APIS_AVAILABLE or not self.service:
            return {"error": "Google Calendar API not available"}
        
        try:
            # Set default time range if not provided
            if not time_min:
                time_min = datetime.datetime.utcnow().isoformat() + "Z"
            
            # Get events
            events_result = self.service.events().list(
                calendarId="primary",
                timeMin=time_min,
                timeMax=time_max,
                maxResults=max_results,
                singleEvents=True,
                orderBy="startTime"
            ).execute()
            
            events = events_result.get("items", [])
            
            # Format the response
            calendar_events = {
                "events": []
            }
            
            for event in events:
                start = event["start"].get("dateTime", event["start"].get("date"))
                end = event["end"].get("dateTime", event["end"].get("date"))
                
                calendar_events["events"].append({
                    "summary": event["summary"],
                    "start": start,
                    "end": end,
                    "location": event.get("location", ""),
                    "description": event.get("description", ""),
                    "link": event.get("htmlLink", "")
                })
            
            return calendar_events
        
        except Exception as e:
            logger.error(f"Error fetching calendar events: {e}")
            return {"error": str(e)}
    
    def create_event(self, summary, start_time, end_time, description=None, location=None) -> Dict[str, Any]:
        """
        Create a new calendar event.
        
        Args:
            summary (str): Event summary/title
            start_time (str): Start time (RFC3339 timestamp or YYYY-MM-DD)
            end_time (str): End time (RFC3339 timestamp or YYYY-MM-DD)
            description (str, optional): Event description
            location (str, optional): Event location
            
        Returns:
            dict: Created event
        """
        if not GOOGLE_APIS_AVAILABLE or not self.service:
            return {"error": "Google Calendar API not available"}
        
        try:
            # Determine if this is an all-day event
            is_all_day = len(start_time) <= 10  # YYYY-MM-DD format
            
            # Create event body
            event = {
                "summary": summary
            }
            
            if is_all_day:
                event["start"] = {"date": start_time}
                event["end"] = {"date": end_time}
            else:
                event["start"] = {"dateTime": start_time, "timeZone": "UTC"}
                event["end"] = {"dateTime": end_time, "timeZone": "UTC"}
            
            if description:
                event["description"] = description
            
            if location:
                event["location"] = location
            
            # Create the event
            created_event = self.service.events().insert(
                calendarId="primary",
                body=event
            ).execute()
            
            # Format the response
            return {
                "id": created_event["id"],
                "summary": created_event["summary"],
                "start": created_event["start"],
                "end": created_event["end"],
                "link": created_event["htmlLink"]
            }
        
        except Exception as e:
            logger.error(f"Error creating calendar event: {e}")
            return {"error": str(e)}


class EmailAPI:
    """
    Integration with Gmail API.
    """
    
    def __init__(self, credentials_file="credentials.json", token_file="token.json"):
        """
        Initialize the EmailAPI.
        
        Args:
            credentials_file (str): Path to Google API credentials file
            token_file (str): Path to token file
        """
        self.credentials_file = credentials_file
        self.token_file = token_file
        self.service = None
        
        if not GOOGLE_APIS_AVAILABLE:
            logger.warning("Google APIs not available. Email functionality will be limited.")
            return
        
        # Initialize the API
        self._initialize_api()
    
    def _initialize_api(self):
        """
        Initialize the Gmail API.
        """
        if not GOOGLE_APIS_AVAILABLE:
            return
        
        try:
            creds = None
            
            # Check if token file exists
            if os.path.exists(self.token_file):
                creds = Credentials.from_authorized_user_info(
                    json.load(open(self.token_file)), 
                    ["https://www.googleapis.com/auth/gmail.compose"]
                )
            
            # If credentials don't exist or are invalid, get new ones
            if not creds or not creds.valid:
                if creds and creds.expired and creds.refresh_token:
                    creds.refresh(Request())
                else:
                    flow = InstalledAppFlow.from_client_secrets_file(
                        self.credentials_file, 
                        ["https://www.googleapis.com/auth/gmail.compose"]
                    )
                    creds = flow.run_local_server(port=0)
                
                # Save the credentials
                with open(self.token_file, "w") as token:
                    token.write(creds.to_json())
            
            # Build the service
            self.service = build("gmail", "v1", credentials=creds)
            logger.info("Gmail API initialized successfully.")
        
        except Exception as e:
            logger.error(f"Error initializing Gmail API: {e}")
            self.service = None
    
    def send_email(self, to, subject, body, cc=None, bcc=None) -> Dict[str, Any]:
        """
        Send an email.
        
        Args:
            to (str): Recipient email address
            subject (str): Email subject
            body (str): Email body
            cc (str, optional): CC recipients
            bcc (str, optional): BCC recipients
            
        Returns:
            dict: Result of the operation
        """
        if not GOOGLE_APIS_AVAILABLE or not self.service:
            return {"error": "Gmail API not available"}
        
        try:
            import base64
            from email.mime.text import MIMEText
            
            # Create the message
            message = MIMEText(body)
            message["to"] = to
            message["subject"] = subject
            
            if cc:
                message["cc"] = cc
            
            if bcc:
                message["bcc"] = bcc
            
            # Encode the message
            raw_message = base64.urlsafe_b64encode(message.as_bytes()).decode()
            
            # Send the message
            sent_message = self.service.users().messages().send(
                userId="me",
                body={"raw": raw_message}
            ).execute()
            
            return {
                "id": sent_message["id"],
                "status": "sent"
            }
        
        except Exception as e:
            logger.error(f"Error sending email: {e}")
            return {"error": str(e)}
