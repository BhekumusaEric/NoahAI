"""
Messaging API integration for NoahAI

This module provides integration with messaging services like email and SMS.
"""

import os
import json
import logging
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
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
logger = logging.getLogger("messaging_api")

# Try to import Twilio for SMS
try:
    from twilio.rest import Client as TwilioClient
    TWILIO_AVAILABLE = True
except ImportError:
    TWILIO_AVAILABLE = False
    logger.warning("Twilio not available. SMS functionality will be limited.")

class EmailAPI:
    """
    Email API integration for NoahAI.
    
    This class provides methods to send emails using SMTP.
    """
    
    def __init__(self, smtp_server=None, smtp_port=None, username=None, password=None, use_tls=True):
        """
        Initialize the Email API integration.
        
        Args:
            smtp_server (str, optional): SMTP server address
            smtp_port (int, optional): SMTP server port
            username (str, optional): SMTP username
            password (str, optional): SMTP password
            use_tls (bool): Whether to use TLS
        """
        self.smtp_server = smtp_server or os.environ.get("SMTP_SERVER")
        self.smtp_port = smtp_port or int(os.environ.get("SMTP_PORT", 587))
        self.username = username or os.environ.get("SMTP_USERNAME")
        self.password = password or os.environ.get("SMTP_PASSWORD")
        self.use_tls = use_tls
        
        # Check if SMTP credentials are available
        if not all([self.smtp_server, self.smtp_port, self.username, self.password]):
            logger.warning("SMTP credentials not complete. Email functionality will be limited.")
    
    def send_email(self, to_email, subject, body, from_email=None, html_body=None):
        """
        Send an email.
        
        Args:
            to_email (str): Recipient email address
            subject (str): Email subject
            body (str): Email body (plain text)
            from_email (str, optional): Sender email address (default: SMTP username)
            html_body (str, optional): HTML version of the email body
            
        Returns:
            dict: Success message or error message
        """
        # Check if SMTP credentials are available
        if not all([self.smtp_server, self.smtp_port, self.username, self.password]):
            return {"error": "SMTP credentials not complete."}
        
        try:
            # Set default from_email if not provided
            from_email = from_email or self.username
            
            # Create the email message
            msg = MIMEMultipart('alternative')
            msg['Subject'] = subject
            msg['From'] = from_email
            msg['To'] = to_email
            
            # Add plain text body
            msg.attach(MIMEText(body, 'plain'))
            
            # Add HTML body if provided
            if html_body:
                msg.attach(MIMEText(html_body, 'html'))
            
            # Connect to the SMTP server
            server = smtplib.SMTP(self.smtp_server, self.smtp_port)
            
            # Use TLS if requested
            if self.use_tls:
                server.starttls()
            
            # Login to the SMTP server
            server.login(self.username, self.password)
            
            # Send the email
            server.sendmail(from_email, to_email, msg.as_string())
            
            # Close the connection
            server.quit()
            
            return {
                "success": True,
                "message": f"Email sent to {to_email}",
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error sending email: {e}")
            return {"error": f"Error sending email: {str(e)}"}

class SMSAPI:
    """
    SMS API integration for NoahAI.
    
    This class provides methods to send SMS messages using Twilio.
    """
    
    def __init__(self, account_sid=None, auth_token=None, from_number=None):
        """
        Initialize the SMS API integration.
        
        Args:
            account_sid (str, optional): Twilio account SID
            auth_token (str, optional): Twilio auth token
            from_number (str, optional): Twilio phone number to send from
        """
        self.account_sid = account_sid or os.environ.get("TWILIO_ACCOUNT_SID")
        self.auth_token = auth_token or os.environ.get("TWILIO_AUTH_TOKEN")
        self.from_number = from_number or os.environ.get("TWILIO_PHONE_NUMBER")
        self.client = None
        
        # Check if Twilio is available
        if not TWILIO_AVAILABLE:
            logger.warning("Twilio not available. SMS functionality will be limited.")
            return
        
        # Check if Twilio credentials are available
        if not all([self.account_sid, self.auth_token, self.from_number]):
            logger.warning("Twilio credentials not complete. SMS functionality will be limited.")
            return
        
        # Initialize the Twilio client
        try:
            self.client = TwilioClient(self.account_sid, self.auth_token)
            logger.info("Twilio client initialized.")
        except Exception as e:
            logger.error(f"Error initializing Twilio client: {e}")
            self.client = None
    
    def send_sms(self, to_number, message):
        """
        Send an SMS message.
        
        Args:
            to_number (str): Recipient phone number
            message (str): SMS message
            
        Returns:
            dict: Success message or error message
        """
        # Check if Twilio is available
        if not TWILIO_AVAILABLE:
            return {"error": "Twilio not available."}
        
        # Check if Twilio client is initialized
        if not self.client:
            return {"error": "Twilio client not initialized."}
        
        try:
            # Send the SMS
            sms = self.client.messages.create(
                body=message,
                from_=self.from_number,
                to=to_number
            )
            
            return {
                "success": True,
                "message": f"SMS sent to {to_number}",
                "sid": sms.sid,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error sending SMS: {e}")
            return {"error": f"Error sending SMS: {str(e)}"}
    
    def get_sms_status(self, sms_sid):
        """
        Get the status of an SMS message.
        
        Args:
            sms_sid (str): Twilio SMS SID
            
        Returns:
            dict: SMS status or error message
        """
        # Check if Twilio is available
        if not TWILIO_AVAILABLE:
            return {"error": "Twilio not available."}
        
        # Check if Twilio client is initialized
        if not self.client:
            return {"error": "Twilio client not initialized."}
        
        try:
            # Get the SMS
            sms = self.client.messages(sms_sid).fetch()
            
            return {
                "sid": sms.sid,
                "status": sms.status,
                "to": sms.to,
                "from": sms.from_,
                "body": sms.body,
                "date_sent": sms.date_sent.isoformat() if sms.date_sent else None,
                "date_created": sms.date_created.isoformat() if sms.date_created else None
            }
        except Exception as e:
            logger.error(f"Error getting SMS status: {e}")
            return {"error": f"Error getting SMS status: {str(e)}"}
