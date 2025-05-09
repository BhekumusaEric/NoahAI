"""
Weather API integration for NoahAI

This module provides integration with weather APIs to get current weather and forecasts.
"""

import os
import json
import logging
import requests
from datetime import datetime, timedelta
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
logger = logging.getLogger("weather_api")

class WeatherAPI:
    """
    Weather API integration for NoahAI.

    This class provides methods to get current weather and forecasts
    using various weather APIs.
    """

    def __init__(self, api_key=None, api_provider="openweathermap"):
        """
        Initialize the Weather API integration.

        Args:
            api_key (str, optional): API key for the weather service
            api_provider (str): The weather API provider to use
                                ("openweathermap", "weatherapi", etc.)
        """
        self.api_key = api_key or os.environ.get("WEATHER_API_KEY")
        self.api_provider = api_provider

        # Check if API key is available
        if not self.api_key:
            logger.warning("No Weather API key provided. Weather functionality will be limited.")

        # Cache for weather data to avoid excessive API calls
        self.cache = {}
        self.cache_expiry = {}

    def get_current_weather(self, location):
        """
        Get current weather for a location.

        Args:
            location (str): City name or location

        Returns:
            dict: Weather data or error message
        """
        # Check cache first
        cache_key = f"current_{location.lower()}"
        if cache_key in self.cache and datetime.now() < self.cache_expiry.get(cache_key, datetime.now()):
            logger.info(f"Using cached weather data for {location}")
            return self.cache[cache_key]

        # If no API key, return mock data
        if not self.api_key:
            return self._get_mock_weather(location)

        try:
            if self.api_provider == "openweathermap":
                return self._get_openweathermap_current(location)
            elif self.api_provider == "weatherapi":
                return self._get_weatherapi_current(location)
            else:
                logger.error(f"Unsupported weather API provider: {self.api_provider}")
                return {"error": f"Unsupported weather API provider: {self.api_provider}"}
        except Exception as e:
            logger.error(f"Error getting weather data: {e}")
            return {"error": f"Error getting weather data: {str(e)}"}

    def get_forecast(self, location, days=5):
        """
        Get weather forecast for a location.

        Args:
            location (str): City name or location
            days (int): Number of days for the forecast (1-7)

        Returns:
            dict: Forecast data or error message
        """
        # Check cache first
        cache_key = f"forecast_{location.lower()}_{days}"
        if cache_key in self.cache and datetime.now() < self.cache_expiry.get(cache_key, datetime.now()):
            logger.info(f"Using cached forecast data for {location}")
            return self.cache[cache_key]

        # If no API key, return mock data
        if not self.api_key:
            return self._get_mock_forecast(location, days)

        try:
            if self.api_provider == "openweathermap":
                return self._get_openweathermap_forecast(location, days)
            elif self.api_provider == "weatherapi":
                return self._get_weatherapi_forecast(location, days)
            else:
                logger.error(f"Unsupported weather API provider: {self.api_provider}")
                return {"error": f"Unsupported weather API provider: {self.api_provider}"}
        except Exception as e:
            logger.error(f"Error getting forecast data: {e}")
            return {"error": f"Error getting forecast data: {str(e)}"}

    def _get_openweathermap_current(self, location):
        """
        Get current weather from OpenWeatherMap API.

        Args:
            location (str): City name or location

        Returns:
            dict: Weather data
        """
        url = f"https://api.openweathermap.org/data/2.5/weather"
        params = {
            "q": location,
            "appid": self.api_key,
            "units": "metric"
        }

        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()

        # Process the data
        weather = {
            "location": data["name"],
            "country": data["sys"]["country"],
            "temperature": data["main"]["temp"],
            "feels_like": data["main"]["feels_like"],
            "humidity": data["main"]["humidity"],
            "pressure": data["main"]["pressure"],
            "wind_speed": data["wind"]["speed"],
            "wind_direction": data["wind"]["deg"],
            "description": data["weather"][0]["description"],
            "icon": data["weather"][0]["icon"],
            "timestamp": datetime.now().isoformat()
        }

        # Cache the result for 30 minutes
        cache_key = f"current_{location.lower()}"
        self.cache[cache_key] = weather
        self.cache_expiry[cache_key] = datetime.now() + timedelta(minutes=30)

        return weather

    def _get_openweathermap_forecast(self, location, days=5):
        """
        Get weather forecast from OpenWeatherMap API.

        Args:
            location (str): City name or location
            days (int): Number of days for the forecast

        Returns:
            dict: Forecast data
        """
        url = f"https://api.openweathermap.org/data/2.5/forecast"
        params = {
            "q": location,
            "appid": self.api_key,
            "units": "metric"
        }

        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()

        # Process the data
        forecast = {
            "location": data["city"]["name"],
            "country": data["city"]["country"],
            "days": []
        }

        # Group by day
        day_forecasts = {}
        for item in data["list"]:
            dt = datetime.fromtimestamp(item["dt"])
            day = dt.strftime("%Y-%m-%d")

            if day not in day_forecasts:
                day_forecasts[day] = []

            day_forecasts[day].append({
                "time": dt.strftime("%H:%M"),
                "temperature": item["main"]["temp"],
                "feels_like": item["main"]["feels_like"],
                "humidity": item["main"]["humidity"],
                "pressure": item["main"]["pressure"],
                "wind_speed": item["wind"]["speed"],
                "wind_direction": item["wind"]["deg"],
                "description": item["weather"][0]["description"],
                "icon": item["weather"][0]["icon"]
            })

        # Take only the requested number of days
        days_list = sorted(day_forecasts.keys())[:days]
        for day in days_list:
            forecast["days"].append({
                "date": day,
                "forecasts": day_forecasts[day]
            })

        # Cache the result for 1 hour
        cache_key = f"forecast_{location.lower()}_{days}"
        self.cache[cache_key] = forecast
        self.cache_expiry[cache_key] = datetime.now() + timedelta(hours=1)

        return forecast

    def _get_weatherapi_current(self, location):
        """
        Get current weather from WeatherAPI.com.

        Args:
            location (str): City name or location

        Returns:
            dict: Weather data
        """
        url = f"https://api.weatherapi.com/v1/current.json"
        params = {
            "q": location,
            "key": self.api_key
        }

        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()

        # Process the data
        weather = {
            "location": data["location"]["name"],
            "country": data["location"]["country"],
            "temperature": data["current"]["temp_c"],
            "feels_like": data["current"]["feelslike_c"],
            "humidity": data["current"]["humidity"],
            "pressure": data["current"]["pressure_mb"],
            "wind_speed": data["current"]["wind_kph"],
            "wind_direction": data["current"]["wind_degree"],
            "description": data["current"]["condition"]["text"],
            "icon": data["current"]["condition"]["icon"],
            "timestamp": datetime.now().isoformat()
        }

        # Cache the result for 30 minutes
        cache_key = f"current_{location.lower()}"
        self.cache[cache_key] = weather
        self.cache_expiry[cache_key] = datetime.now() + timedelta(minutes=30)

        return weather

    def _get_weatherapi_forecast(self, location, days=5):
        """
        Get weather forecast from WeatherAPI.com.

        Args:
            location (str): City name or location
            days (int): Number of days for the forecast

        Returns:
            dict: Forecast data
        """
        url = f"https://api.weatherapi.com/v1/forecast.json"
        params = {
            "q": location,
            "key": self.api_key,
            "days": min(days, 7)  # API limit is 7 days
        }

        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()

        # Process the data
        forecast = {
            "location": data["location"]["name"],
            "country": data["location"]["country"],
            "days": []
        }

        # Extract forecast data
        for day_data in data["forecast"]["forecastday"]:
            day = {
                "date": day_data["date"],
                "forecasts": []
            }

            # Add hourly forecasts
            for hour in day_data["hour"]:
                hour_dt = datetime.fromisoformat(hour["time"].replace("Z", "+00:00"))
                day["forecasts"].append({
                    "time": hour_dt.strftime("%H:%M"),
                    "temperature": hour["temp_c"],
                    "feels_like": hour["feelslike_c"],
                    "humidity": hour["humidity"],
                    "pressure": hour["pressure_mb"],
                    "wind_speed": hour["wind_kph"],
                    "wind_direction": hour["wind_degree"],
                    "description": hour["condition"]["text"],
                    "icon": hour["condition"]["icon"]
                })

            forecast["days"].append(day)

        # Cache the result for 1 hour
        cache_key = f"forecast_{location.lower()}_{days}"
        self.cache[cache_key] = forecast
        self.cache_expiry[cache_key] = datetime.now() + timedelta(hours=1)

        return forecast

    def _get_mock_weather(self, location):
        """
        Get mock weather data when no API key is available.

        Args:
            location (str): City name or location

        Returns:
            dict: Mock weather data
        """
        return {
            "location": location,
            "country": "Unknown",
            "temperature": 20,
            "feels_like": 20,
            "humidity": 50,
            "pressure": 1013,
            "wind_speed": 5,
            "wind_direction": 180,
            "description": "Partly cloudy",
            "icon": "04d",
            "timestamp": datetime.now().isoformat(),
            "note": "This is mock data as no API key was provided."
        }

    def _get_mock_forecast(self, location, days=5):
        """
        Get mock forecast data when no API key is available.

        Args:
            location (str): City name or location
            days (int): Number of days for the forecast

        Returns:
            dict: Mock forecast data
        """
        forecast = {
            "location": location,
            "country": "Unknown",
            "days": [],
            "note": "This is mock data as no API key was provided."
        }

        # Generate mock data for each day
        today = datetime.now()
        for i in range(days):
            day_date = today + timedelta(days=i)
            day = {
                "date": day_date.strftime("%Y-%m-%d"),
                "forecasts": []
            }

            # Generate mock hourly forecasts
            for hour in range(0, 24, 3):  # Every 3 hours
                hour_time = day_date.replace(hour=hour, minute=0)
                day["forecasts"].append({
                    "time": hour_time.strftime("%H:%M"),
                    "temperature": 20 + i - (12 - hour) ** 2 / 50,  # Simulate daily temperature curve
                    "feels_like": 20 + i - (12 - hour) ** 2 / 40,
                    "humidity": 50 + (hour // 6) * 5,
                    "pressure": 1013,
                    "wind_speed": 5 + hour / 5,
                    "wind_direction": 180,
                    "description": "Partly cloudy",
                    "icon": "04d"
                })

            forecast["days"].append(day)

        return forecast
