"""
Safety Filter for NoahAI

This module provides safety mechanisms to prevent harmful behaviors
and ensure responsible AI usage.
"""

import os
import re
import json
import logging
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/safety_filter.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("safety_filter")

class SafetyFilter:
    """
    Safety Filter for NoahAI.
    
    This class provides safety mechanisms to prevent harmful behaviors
    and ensure responsible AI usage.
    """
    
    def __init__(self, config=None):
        """
        Initialize the Safety Filter.
        
        Args:
            config (dict, optional): Configuration for the safety filter
        """
        self.config = config or {}
        
        # Initialize safety parameters
        self.safety_level = self.config.get("safety_level", "medium")  # "low", "medium", "high"
        self.enable_content_filtering = self.config.get("enable_content_filtering", True)
        self.enable_personal_info_protection = self.config.get("enable_personal_info_protection", True)
        self.enable_harmful_response_detection = self.config.get("enable_harmful_response_detection", True)
        self.enable_reward_hacking_prevention = self.config.get("enable_reward_hacking_prevention", True)
        self.enable_safety_monitoring = self.config.get("enable_safety_monitoring", True)
        
        # Load safety resources
        self.unsafe_patterns = self._load_unsafe_patterns()
        self.sensitive_topics = self._load_sensitive_topics()
        self.personal_info_patterns = self._load_personal_info_patterns()
        
        # Initialize safety monitoring
        self.safety_violations = []
        self.safety_warnings = []
        self.safety_logs_dir = self.config.get("safety_logs_dir", "logs/safety")
        os.makedirs(self.safety_logs_dir, exist_ok=True)
        
        # Initialize safety thresholds based on safety level
        self._set_safety_thresholds()
        
        logger.info(f"Safety Filter initialized with safety level: {self.safety_level}")
    
    def _set_safety_thresholds(self):
        """
        Set safety thresholds based on the safety level.
        """
        if self.safety_level == "low":
            self.content_filter_threshold = 0.8  # Higher threshold means less filtering
            self.personal_info_threshold = 0.7
            self.harmful_response_threshold = 0.8
            self.reward_hacking_threshold = 0.8
        elif self.safety_level == "medium":
            self.content_filter_threshold = 0.6
            self.personal_info_threshold = 0.5
            self.harmful_response_threshold = 0.6
            self.reward_hacking_threshold = 0.6
        else:  # high
            self.content_filter_threshold = 0.4  # Lower threshold means more filtering
            self.personal_info_threshold = 0.3
            self.harmful_response_threshold = 0.4
            self.reward_hacking_threshold = 0.4
    
    def _load_unsafe_patterns(self):
        """
        Load patterns for unsafe content.
        
        Returns:
            dict: Dictionary of unsafe patterns
        """
        # In a production environment, these would be loaded from a file or database
        # and would be much more comprehensive
        return {
            "hate_speech": [
                r"\b(hate|despise|detest)\b.*\b(group|community|people)\b",
                r"\b(racial|ethnic|religious)\b.*\b(slur|insult|attack)\b"
            ],
            "violence": [
                r"\b(kill|murder|harm|hurt|attack)\b.*\b(person|people|individual|group)\b",
                r"\b(weapon|gun|bomb|explosive)\b.*\b(use|create|build|make)\b"
            ],
            "self_harm": [
                r"\b(suicide|self-harm|hurt myself|kill myself)\b",
                r"\b(cut|harm|hurt)\b.*\b(myself|yourself)\b"
            ],
            "sexual_content": [
                r"\b(explicit|graphic)\b.*\b(sexual|content)\b",
                r"\b(pornography|porn|adult content)\b"
            ],
            "illegal_activities": [
                r"\b(hack|steal|illegal|crime)\b.*\b(system|data|money|property)\b",
                r"\b(drug|narcotic)\b.*\b(manufacture|produce|sell|distribute)\b"
            ]
        }
    
    def _load_sensitive_topics(self):
        """
        Load sensitive topics that require careful handling.
        
        Returns:
            list: List of sensitive topics
        """
        # In a production environment, these would be loaded from a file or database
        return [
            "politics",
            "religion",
            "race",
            "gender",
            "sexuality",
            "disability",
            "mental health",
            "terrorism",
            "extremism",
            "child abuse"
        ]
    
    def _load_personal_info_patterns(self):
        """
        Load patterns for personal information.
        
        Returns:
            dict: Dictionary of personal information patterns
        """
        # In a production environment, these would be loaded from a file or database
        return {
            "email": [r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'],
            "phone": [r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', r'\b\+\d{1,3}[-. ]?\d{3}[-. ]?\d{3}[-. ]?\d{4}\b'],
            "ssn": [r'\b\d{3}[-]?\d{2}[-]?\d{4}\b'],
            "credit_card": [r'\b\d{4}[-. ]?\d{4}[-. ]?\d{4}[-. ]?\d{4}\b'],
            "address": [r'\b\d+\s+[A-Za-z]+\s+[A-Za-z]+\.?\s+(?:Apt|Unit|Suite)?\s*(?:#|No\.?)?\s*\d*\b'],
            "ip_address": [r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b'],
            "password": [r'\b(?:password|passwd|pwd)\s*[=:]\s*\S+\b', r'\bmy\s+password\s+is\s+\S+\b']
        }
    
    def filter_unsafe_content(self, text):
        """
        Filter unsafe content from text.
        
        Args:
            text (str): Text to filter
            
        Returns:
            tuple: (filtered_text, safety_score, category, matched_patterns)
        """
        if not self.enable_content_filtering:
            return text, 1.0, None, []
        
        # Initialize safety score (1.0 = completely safe, 0.0 = completely unsafe)
        safety_score = 1.0
        matched_category = None
        matched_patterns = []
        
        # Check each category of unsafe patterns
        for category, patterns in self.unsafe_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    matched_text = match.group(0)
                    matched_patterns.append({
                        "category": category,
                        "pattern": pattern,
                        "matched_text": matched_text,
                        "start": match.start(),
                        "end": match.end()
                    })
                    
                    # Reduce safety score based on category
                    if category == "hate_speech":
                        safety_score -= 0.3
                    elif category == "violence":
                        safety_score -= 0.3
                    elif category == "self_harm":
                        safety_score -= 0.4
                    elif category == "sexual_content":
                        safety_score -= 0.2
                    elif category == "illegal_activities":
                        safety_score -= 0.25
                    
                    # Set matched category to the most severe one
                    if matched_category is None or safety_score < 0.5:
                        matched_category = category
        
        # Ensure safety score is between 0 and 1
        safety_score = max(0.0, min(1.0, safety_score))
        
        # If safety score is below threshold, redact or modify the text
        if safety_score < self.content_filter_threshold:
            # Log safety violation
            self._log_safety_violation("unsafe_content", text, safety_score, matched_category, matched_patterns)
            
            # Redact matched patterns
            filtered_text = text
            for match in sorted(matched_patterns, key=lambda x: x["start"], reverse=True):
                start, end = match["start"], match["end"]
                filtered_text = filtered_text[:start] + "[FILTERED CONTENT]" + filtered_text[end:]
            
            return filtered_text, safety_score, matched_category, matched_patterns
        
        return text, safety_score, matched_category, matched_patterns
    
    def detect_personal_information(self, text):
        """
        Detect and redact personal information from text.
        
        Args:
            text (str): Text to check for personal information
            
        Returns:
            tuple: (redacted_text, contains_personal_info, detected_info)
        """
        if not self.enable_personal_info_protection:
            return text, False, []
        
        # Initialize
        redacted_text = text
        contains_personal_info = False
        detected_info = []
        
        # Check each type of personal information
        for info_type, patterns in self.personal_info_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    matched_text = match.group(0)
                    start, end = match.start(), match.end()
                    
                    # Add to detected info
                    detected_info.append({
                        "type": info_type,
                        "matched_text": matched_text,
                        "start": start,
                        "end": end
                    })
                    
                    contains_personal_info = True
                    
                    # Redact the personal information
                    redacted_text = redacted_text[:start] + f"[REDACTED {info_type.upper()}]" + redacted_text[end:]
        
        # Log if personal information was detected
        if contains_personal_info:
            self._log_safety_warning("personal_information", text, detected_info)
        
        return redacted_text, contains_personal_info, detected_info
    
    def check_harmful_response(self, response_text):
        """
        Check if an AI response could be harmful.
        
        Args:
            response_text (str): AI response to check
            
        Returns:
            tuple: (is_harmful, harm_score, harm_category, explanation)
        """
        if not self.enable_harmful_response_detection:
            return False, 0.0, None, None
        
        # Initialize
        is_harmful = False
        harm_score = 0.0
        harm_category = None
        explanation = None
        
        # Check for unsafe content
        _, safety_score, category, matched_patterns = self.filter_unsafe_content(response_text)
        
        # Convert safety score to harm score (inverse)
        harm_score = 1.0 - safety_score
        
        # Check if harm score exceeds threshold
        if harm_score > self.harmful_response_threshold:
            is_harmful = True
            harm_category = category
            
            # Generate explanation
            if matched_patterns:
                explanation = f"Response contains potentially harmful content in the category: {category}. "
                explanation += f"Specific concerns: {', '.join(p['matched_text'] for p in matched_patterns[:3])}"
            else:
                explanation = f"Response contains potentially harmful content in the category: {category}."
            
            # Log harmful response
            self._log_safety_violation("harmful_response", response_text, harm_score, harm_category, matched_patterns)
        
        return is_harmful, harm_score, harm_category, explanation
    
    def detect_reward_hacking(self, action, reward, state=None, previous_actions=None):
        """
        Detect potential reward hacking in reinforcement learning.
        
        Args:
            action: Action taken by the agent
            reward: Reward received for the action
            state (optional): Current state
            previous_actions (optional): Previous actions taken
            
        Returns:
            tuple: (is_hacking, hacking_score, explanation)
        """
        if not self.enable_reward_hacking_prevention:
            return False, 0.0, None
        
        # Initialize
        is_hacking = False
        hacking_score = 0.0
        explanation = None
        
        # Check for unusually high rewards
        if reward > 10.0:  # Assuming normal rewards are in the range [-1, 1]
            hacking_score += 0.5
            explanation = "Unusually high reward detected."
        
        # Check for repeated actions if previous actions are provided
        if previous_actions and len(previous_actions) >= 3:
            last_three = previous_actions[-3:]
            if all(a == last_three[0] for a in last_three) and reward > 0:
                hacking_score += 0.3
                if explanation:
                    explanation += " Repeated actions with positive rewards detected."
                else:
                    explanation = "Repeated actions with positive rewards detected."
        
        # Check for reward spikes
        if previous_actions and len(previous_actions) >= 5:
            # Calculate average reward for previous actions
            avg_reward = sum(previous_actions) / len(previous_actions)
            
            # Check if current reward is significantly higher
            if reward > avg_reward * 3:
                hacking_score += 0.2
                if explanation:
                    explanation += " Significant reward spike detected."
                else:
                    explanation = "Significant reward spike detected."
        
        # Determine if hacking is occurring
        if hacking_score > self.reward_hacking_threshold:
            is_hacking = True
            
            # Log reward hacking
            self._log_safety_warning("reward_hacking", str(action), {
                "action": action,
                "reward": reward,
                "hacking_score": hacking_score,
                "explanation": explanation
            })
        
        return is_hacking, hacking_score, explanation
    
    def _log_safety_violation(self, violation_type, content, score, category, details):
        """
        Log a safety violation.
        
        Args:
            violation_type (str): Type of violation
            content (str): Content that triggered the violation
            score (float): Safety or harm score
            category (str): Category of the violation
            details: Additional details about the violation
        """
        if not self.enable_safety_monitoring:
            return
        
        # Create violation record
        violation = {
            "timestamp": datetime.now().isoformat(),
            "type": violation_type,
            "category": category,
            "score": score,
            "content": content[:100] + "..." if len(content) > 100 else content,  # Truncate for log
            "details": details
        }
        
        # Add to violations list
        self.safety_violations.append(violation)
        
        # Log to file
        log_file = os.path.join(self.safety_logs_dir, f"violations_{datetime.now().strftime('%Y%m%d')}.json")
        
        try:
            # Load existing log if it exists
            existing_log = []
            if os.path.exists(log_file):
                with open(log_file, 'r') as f:
                    existing_log = json.load(f)
            
            # Add new violation
            existing_log.append(violation)
            
            # Save log
            with open(log_file, 'w') as f:
                json.dump(existing_log, f, indent=2)
        except Exception as e:
            logger.error(f"Error logging safety violation: {e}")
        
        # Log to console
        logger.warning(f"Safety violation detected: {violation_type}, category: {category}, score: {score:.2f}")
    
    def _log_safety_warning(self, warning_type, content, details):
        """
        Log a safety warning.
        
        Args:
            warning_type (str): Type of warning
            content (str): Content that triggered the warning
            details: Additional details about the warning
        """
        if not self.enable_safety_monitoring:
            return
        
        # Create warning record
        warning = {
            "timestamp": datetime.now().isoformat(),
            "type": warning_type,
            "content": content[:100] + "..." if len(content) > 100 else content,  # Truncate for log
            "details": details
        }
        
        # Add to warnings list
        self.safety_warnings.append(warning)
        
        # Log to file
        log_file = os.path.join(self.safety_logs_dir, f"warnings_{datetime.now().strftime('%Y%m%d')}.json")
        
        try:
            # Load existing log if it exists
            existing_log = []
            if os.path.exists(log_file):
                with open(log_file, 'r') as f:
                    existing_log = json.load(f)
            
            # Add new warning
            existing_log.append(warning)
            
            # Save log
            with open(log_file, 'w') as f:
                json.dump(existing_log, f, indent=2)
        except Exception as e:
            logger.error(f"Error logging safety warning: {e}")
        
        # Log to console
        logger.info(f"Safety warning detected: {warning_type}")
    
    def filter_and_check(self, user_input, ai_response):
        """
        Filter user input and check AI response for safety issues.
        
        Args:
            user_input (str): User input
            ai_response (str): AI response
            
        Returns:
            tuple: (filtered_input, filtered_response, is_safe, safety_info)
        """
        # Filter user input
        filtered_input, input_safety_score, input_category, input_patterns = self.filter_unsafe_content(user_input)
        
        # Check for personal information in user input
        filtered_input, contains_personal_info, detected_info = self.detect_personal_information(filtered_input)
        
        # Check AI response for harmful content
        is_harmful, harm_score, harm_category, harm_explanation = self.check_harmful_response(ai_response)
        
        # Filter AI response if needed
        filtered_response = ai_response
        if is_harmful:
            filtered_response, _, _, _ = self.filter_unsafe_content(ai_response)
        
        # Check for personal information in AI response
        filtered_response, response_contains_personal_info, response_detected_info = self.detect_personal_information(filtered_response)
        
        # Determine overall safety
        is_safe = (input_safety_score >= self.content_filter_threshold and 
                  not is_harmful and 
                  not response_contains_personal_info)
        
        # Compile safety information
        safety_info = {
            "input_safety_score": input_safety_score,
            "input_category": input_category,
            "input_contains_personal_info": contains_personal_info,
            "response_is_harmful": is_harmful,
            "response_harm_score": harm_score,
            "response_harm_category": harm_category,
            "response_harm_explanation": harm_explanation,
            "response_contains_personal_info": response_contains_personal_info,
            "is_safe": is_safe
        }
        
        return filtered_input, filtered_response, is_safe, safety_info
    
    def get_safety_report(self, start_date=None, end_date=None):
        """
        Get a safety report for a specified time period.
        
        Args:
            start_date (str, optional): Start date in ISO format
            end_date (str, optional): End date in ISO format
            
        Returns:
            dict: Safety report
        """
        # Filter violations and warnings by date if specified
        violations = self.safety_violations
        warnings = self.safety_warnings
        
        if start_date:
            violations = [v for v in violations if v["timestamp"] >= start_date]
            warnings = [w for w in warnings if w["timestamp"] >= start_date]
        
        if end_date:
            violations = [v for v in violations if v["timestamp"] <= end_date]
            warnings = [w for w in warnings if w["timestamp"] <= end_date]
        
        # Count violations by type
        violation_counts = {}
        for violation in violations:
            violation_type = violation["type"]
            violation_counts[violation_type] = violation_counts.get(violation_type, 0) + 1
        
        # Count warnings by type
        warning_counts = {}
        for warning in warnings:
            warning_type = warning["type"]
            warning_counts[warning_type] = warning_counts.get(warning_type, 0) + 1
        
        # Create report
        report = {
            "total_violations": len(violations),
            "total_warnings": len(warnings),
            "violation_counts": violation_counts,
            "warning_counts": warning_counts,
            "safety_level": self.safety_level,
            "content_filter_threshold": self.content_filter_threshold,
            "personal_info_threshold": self.personal_info_threshold,
            "harmful_response_threshold": self.harmful_response_threshold,
            "reward_hacking_threshold": self.reward_hacking_threshold,
            "generated_at": datetime.now().isoformat()
        }
        
        return report
