"""
Calendar API integration for NoahAI

This module provides integration with calendar services like Google Calendar.
"""

import os
import json
import logging
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
logger = logging.getLogger("calendar_api")

# Try to import Google Calendar API
try:
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from google.auth.transport.requests import Request
    from googleapiclient.discovery import build
    GOOGLE_CALENDAR_AVAILABLE = True
except ImportError:
    GOOGLE_CALENDAR_AVAILABLE = False
    logger.warning("Google Calendar API not available. Calendar functionality will be limited.")

class CalendarAPI:
    """
    Calendar API integration for NoahAI.

    This class provides methods to interact with calendar services
    like Google Calendar.
    """

    def __init__(self, credentials_file=None, token_file=None, calendar_id="primary"):
        """
        Initialize the Calendar API integration.

        Args:
            credentials_file (str, optional): Path to the credentials file
            token_file (str, optional): Path to the token file
            calendar_id (str): Calendar ID to use (default: "primary")
        """
        self.credentials_file = credentials_file or os.environ.get("GOOGLE_CALENDAR_CREDENTIALS")
        self.token_file = token_file or "data/token.json"
        self.calendar_id = calendar_id
        self.service = None

        # Check if Google Calendar API is available
        if not GOOGLE_CALENDAR_AVAILABLE:
            logger.warning("Google Calendar API not available. Calendar functionality will be limited.")
            return

        # Initialize the service
        self._initialize_service()

    def _initialize_service(self):
        """
        Initialize the Google Calendar API service.
        """
        if not GOOGLE_CALENDAR_AVAILABLE:
            return

        # Check if credentials file exists
        if not self.credentials_file or not os.path.exists(self.credentials_file):
            logger.warning("Google Calendar credentials file not found.")
            return

        try:
            # Define the scopes
            SCOPES = ['https://www.googleapis.com/auth/calendar']

            # Get credentials
            creds = None
            if os.path.exists(self.token_file):
                with open(self.token_file, 'r') as token:
                    creds = Credentials.from_authorized_user_info(json.load(token), SCOPES)

            # If credentials are not valid, refresh them or get new ones
            if not creds or not creds.valid:
                if creds and creds.expired and creds.refresh_token:
                    creds.refresh(Request())
                else:
                    flow = InstalledAppFlow.from_client_secrets_file(self.credentials_file, SCOPES)
                    creds = flow.run_local_server(port=0)

                # Save the credentials for the next run
                with open(self.token_file, 'w') as token:
                    token.write(creds.to_json())

            # Build the service
            self.service = build('calendar', 'v3', credentials=creds)
            logger.info("Google Calendar API service initialized.")
        except Exception as e:
            logger.error(f"Error initializing Google Calendar API service: {e}")
            self.service = None

    def get_events(self, start_date=None, end_date=None, max_results=10):
        """
        Get events from the calendar.

        Args:
            start_date (datetime, optional): Start date for events (default: now)
            end_date (datetime, optional): End date for events (default: 7 days from now)
            max_results (int): Maximum number of events to return

        Returns:
            list: List of events or error message
        """
        # If Google Calendar API is not available, return mock data
        if not GOOGLE_CALENDAR_AVAILABLE or not self.service:
            return self._get_mock_events(start_date, end_date, max_results)

        try:
            # Set default dates if not provided
            if not start_date:
                start_date = datetime.now()
            if not end_date:
                end_date = start_date + timedelta(days=7)

            # Format dates for the API
            start_date_str = start_date.isoformat() + 'Z'  # 'Z' indicates UTC time
            end_date_str = end_date.isoformat() + 'Z'

            # Call the API
            events_result = self.service.events().list(
                calendarId=self.calendar_id,
                timeMin=start_date_str,
                timeMax=end_date_str,
                maxResults=max_results,
                singleEvents=True,
                orderBy='startTime'
            ).execute()

            # Process the events
            events = events_result.get('items', [])

            # Format the events
            formatted_events = []
            for event in events:
                start = event['start'].get('dateTime', event['start'].get('date'))
                end = event['end'].get('dateTime', event['end'].get('date'))

                formatted_events.append({
                    'id': event['id'],
                    'summary': event.get('summary', 'No title'),
                    'description': event.get('description', ''),
                    'location': event.get('location', ''),
                    'start': start,
                    'end': end,
                    'link': event.get('htmlLink', '')
                })

            return formatted_events
        except Exception as e:
            logger.error(f"Error getting events: {e}")
            return {"error": f"Error getting events: {str(e)}"}

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
        # If Google Calendar API is not available, return mock data
        if not GOOGLE_CALENDAR_AVAILABLE or not self.service:
            return {"error": "Google Calendar API not available."}

        try:
            # Create the event
            event = {
                'summary': summary,
                'start': {
                    'dateTime': start_time.isoformat(),
                    'timeZone': 'UTC',
                },
                'end': {
                    'dateTime': end_time.isoformat(),
                    'timeZone': 'UTC',
                }
            }

            # Add optional fields
            if description:
                event['description'] = description
            if location:
                event['location'] = location

            # Call the API
            created_event = self.service.events().insert(
                calendarId=self.calendar_id,
                body=event
            ).execute()

            # Format the response
            return {
                'id': created_event['id'],
                'summary': created_event.get('summary', 'No title'),
                'description': created_event.get('description', ''),
                'location': created_event.get('location', ''),
                'start': created_event['start'].get('dateTime', created_event['start'].get('date')),
                'end': created_event['end'].get('dateTime', created_event['end'].get('date')),
                'link': created_event.get('htmlLink', '')
            }
        except Exception as e:
            logger.error(f"Error creating event: {e}")
            return {"error": f"Error creating event: {str(e)}"}

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
        # If Google Calendar API is not available, return error
        if not GOOGLE_CALENDAR_AVAILABLE or not self.service:
            return {"error": "Google Calendar API not available."}

        try:
            # Get the existing event
            event = self.service.events().get(calendarId=self.calendar_id, eventId=event_id).execute()

            # Update the fields
            if summary:
                event['summary'] = summary
            if start_time:
                event['start'] = {
                    'dateTime': start_time.isoformat(),
                    'timeZone': 'UTC',
                }
            if end_time:
                event['end'] = {
                    'dateTime': end_time.isoformat(),
                    'timeZone': 'UTC',
                }
            if description:
                event['description'] = description
            if location:
                event['location'] = location

            # Call the API
            updated_event = self.service.events().update(
                calendarId=self.calendar_id,
                eventId=event_id,
                body=event
            ).execute()

            # Format the response
            return {
                'id': updated_event['id'],
                'summary': updated_event.get('summary', 'No title'),
                'description': updated_event.get('description', ''),
                'location': updated_event.get('location', ''),
                'start': updated_event['start'].get('dateTime', updated_event['start'].get('date')),
                'end': updated_event['end'].get('dateTime', updated_event['end'].get('date')),
                'link': updated_event.get('htmlLink', '')
            }
        except Exception as e:
            logger.error(f"Error updating event: {e}")
            return {"error": f"Error updating event: {str(e)}"}

    def delete_event(self, event_id):
        """
        Delete an event from the calendar.

        Args:
            event_id (str): ID of the event to delete

        Returns:
            dict: Success message or error message
        """
        # If Google Calendar API is not available, return error
        if not GOOGLE_CALENDAR_AVAILABLE or not self.service:
            return {"error": "Google Calendar API not available."}

        try:
            # Call the API
            self.service.events().delete(
                calendarId=self.calendar_id,
                eventId=event_id
            ).execute()

            return {"success": True, "message": f"Event {event_id} deleted successfully."}
        except Exception as e:
            logger.error(f"Error deleting event: {e}")
            return {"error": f"Error deleting event: {str(e)}"}

    def _get_mock_events(self, start_date=None, end_date=None, max_results=10):
        """
        Get mock events when Google Calendar API is not available.

        Args:
            start_date (datetime, optional): Start date for events
            end_date (datetime, optional): End date for events
            max_results (int): Maximum number of events to return

        Returns:
            list: List of mock events
        """
        # Set default dates if not provided
        if not start_date:
            start_date = datetime.now()
        if not end_date:
            end_date = start_date + timedelta(days=7)

        # Generate mock events
        events = []
        current_date = start_date

        # Generate a few mock events
        event_titles = [
            "Team Meeting",
            "Project Review",
            "Client Call",
            "Lunch with Team",
            "Weekly Planning",
            "Training Session",
            "Product Demo",
            "Code Review",
            "Brainstorming Session",
            "One-on-One Meeting"
        ]

        for i in range(min(max_results, 10)):
            # Generate a random time on the current day
            event_start = current_date.replace(
                hour=9 + (i % 8),
                minute=0,
                second=0,
                microsecond=0
            )
            event_end = event_start + timedelta(hours=1)

            # Skip if outside the requested range
            if event_end > end_date:
                break

            # Create the event
            events.append({
                'id': f"mock-event-{i}",
                'summary': event_titles[i],
                'description': f"This is a mock event #{i}",
                'location': "Virtual Meeting",
                'start': event_start.isoformat(),
                'end': event_end.isoformat(),
                'link': "#",
                'note': "This is mock data as Google Calendar API is not available."
            })

            # Move to the next day for the next event
            if (i + 1) % 3 == 0:
                current_date += timedelta(days=1)

        return events
