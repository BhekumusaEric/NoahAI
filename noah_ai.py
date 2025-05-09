#!/usr/bin/env python3
"""
NoahAI - An advanced AI assistant with deep learning capabilities

This is the main entry point for the NoahAI assistant. It provides a command-line
interface for interacting with the AI, with advanced reinforcement learning,
distributed training, explainable AI, continual learning, and safety mechanisms.
"""

import os
import time
import argparse
import logging
from colorama import Fore, Style, init

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/noah_ai.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("noah_ai")

# Import our custom modules
from src.utils.nlp_utils import NLPUtils
from src.utils.learning_utils import LearningUtils
from models.simple_model import SimpleResponseModel

# Try to import advanced reinforcement learning components
try:
    from src.utils.advanced_rl_manager import AdvancedRLManager
    ADVANCED_RL_AVAILABLE = True
except ImportError:
    ADVANCED_RL_AVAILABLE = False
    logger.warning("Advanced reinforcement learning components not available.")

# Try to import distributed training
try:
    from src.utils.distributed_training import DistributedTrainingManager
    DISTRIBUTED_TRAINING_AVAILABLE = True
except ImportError:
    DISTRIBUTED_TRAINING_AVAILABLE = False
    logger.warning("Distributed training not available.")

# Try to import explainable AI
try:
    from src.utils.explainable_ai import ExplainableAI
    EXPLAINABLE_AI_AVAILABLE = True
except ImportError:
    EXPLAINABLE_AI_AVAILABLE = False
    logger.warning("Explainable AI components not available.")

# Try to import continual learning
try:
    from src.utils.continual_learning import ContinualLearningManager
    CONTINUAL_LEARNING_AVAILABLE = True
except ImportError:
    CONTINUAL_LEARNING_AVAILABLE = False
    logger.warning("Continual learning components not available.")

# Try to import safety filter
try:
    from src.utils.safety_filter import SafetyFilter
    SAFETY_FILTER_AVAILABLE = True
except ImportError:
    SAFETY_FILTER_AVAILABLE = False
    logger.warning("Safety filter not available.")

# Try to import integrations
try:
    from src.integrations.integration_manager import IntegrationManager
    INTEGRATIONS_AVAILABLE = True
except ImportError:
    INTEGRATIONS_AVAILABLE = False
    logger.warning("Integrations not available.")

# Initialize colorama for colored terminal output
init(autoreset=True)

# Try to import the deep learning model, but don't fail if TensorFlow is not installed
try:
    from models.deep_learning_model import DeepLearningModel
    TENSORFLOW_AVAILABLE = True
except ImportError:
    logger.warning("TensorFlow not available. Using simple model instead.")
    from models.temp_model import SimpleResponseModel as DeepLearningModel
    TENSORFLOW_AVAILABLE = False

class NoahAI:
    """
    An advanced AI assistant that can respond to queries and commands.
    """

    def __init__(self, name="Noah", responses_file="data/responses.json",
                 use_deep_learning=True, verbose=False, model_type="advanced_lstm",
                 use_reinforcement_learning=False, use_advanced_rl=False,
                 use_distributed_training=False, use_explainable_ai=False,
                 use_continual_learning=False, use_safety_filter=False,
                 safety_level="medium", user_id="default_user"):
        """
        Initialize the NoahAI assistant.

        Args:
            name (str): The name of the AI assistant
            responses_file (str): Path to the responses JSON file
            use_deep_learning (bool): Whether to use the deep learning model
            verbose (bool): Whether to print verbose debug information
            model_type (str): Type of model to use. Options:
                - "simple_lstm": Basic LSTM model
                - "advanced_lstm": Advanced LSTM with bidirectional layers (default)
                - "cnn_lstm": CNN-LSTM hybrid model
                - "transfer_learning": Model using transfer learning
            use_reinforcement_learning (bool): Whether to use reinforcement learning
            use_advanced_rl (bool): Whether to use advanced reinforcement learning components
            use_distributed_training (bool): Whether to use distributed training
            use_explainable_ai (bool): Whether to use explainable AI components
            use_continual_learning (bool): Whether to use continual learning
            use_safety_filter (bool): Whether to use safety filter
            safety_level (str): Safety level for the safety filter ("low", "medium", "high")
            user_id (str): User ID for personalized learning
        """
        self.name = name
        self.responses_file = responses_file
        self.verbose = verbose
        self.use_deep_learning = use_deep_learning and TENSORFLOW_AVAILABLE
        self.model_type = model_type
        self.use_reinforcement_learning = use_reinforcement_learning
        self.use_advanced_rl = use_advanced_rl and ADVANCED_RL_AVAILABLE
        self.use_distributed_training = use_distributed_training and DISTRIBUTED_TRAINING_AVAILABLE
        self.use_explainable_ai = use_explainable_ai and EXPLAINABLE_AI_AVAILABLE
        self.use_continual_learning = use_continual_learning and CONTINUAL_LEARNING_AVAILABLE
        self.use_safety_filter = use_safety_filter and SAFETY_FILTER_AVAILABLE
        self.safety_level = safety_level
        self.user_id = user_id
        self.user_name = None
        self.conversation_history = []
        self.last_user_input = None
        self.last_ai_response = None

        # Create data directories
        os.makedirs("data/model", exist_ok=True)
        os.makedirs("data/logs", exist_ok=True)
        os.makedirs("data/rl_models", exist_ok=True)
        os.makedirs("data/explanations", exist_ok=True)
        os.makedirs("data/continual_learning", exist_ok=True)
        os.makedirs("logs/safety", exist_ok=True)
        os.makedirs("config", exist_ok=True)

        # Initialize the model
        if self.use_deep_learning:
            self._print_debug(f"Using deep learning model (type: {self.model_type})")
            self.model = DeepLearningModel(
                responses_file=responses_file,
                model_type=self.model_type,
                use_reinforcement_learning=self.use_reinforcement_learning
            )
        else:
            self._print_debug("Using simple response model")
            self.model = SimpleResponseModel(responses_file)

        # Initialize NLP utilities
        self.nlp_utils = NLPUtils()

        # Initialize learning utilities if using deep learning
        if self.use_deep_learning:
            # Define reinforcement learning configuration with optimized hyperparameters
            if self.use_reinforcement_learning:
                if self.model_type == "simple_lstm":
                    # Simpler model needs more aggressive learning
                    rl_config = {
                        "actor_lr": 0.0005,  # Higher learning rate for simpler model
                        "critic_lr": 0.0015,
                        "gamma": 0.97,       # Slightly lower discount factor
                        "clip_ratio": 0.2,
                        "entropy_coef": 0.02, # More exploration
                        "batch_size": 32,     # Smaller batch size
                        "epochs": 8,
                        "lam": 0.92,          # GAE lambda parameter
                        "target_kl": 0.015,   # KL divergence threshold
                        # Reward function weights
                        "reward_weights": {
                            "rating_weight": 1.0,
                            "length_weight": 0.2,
                            "sentiment_weight": 0.4,
                            "engagement_weight": 0.6,
                            "coherence_weight": 0.5,
                            "informativeness_weight": 0.4,
                            "helpfulness_weight": 0.7,
                            "consistency_weight": 0.3
                        }
                    }
                elif self.model_type == "cnn_lstm":
                    # CNN-LSTM hybrid model hyperparameters
                    rl_config = {
                        "actor_lr": 0.0002,   # Lower learning rate for complex model
                        "critic_lr": 0.0008,
                        "gamma": 0.99,        # Higher discount factor
                        "clip_ratio": 0.15,   # More conservative updates
                        "entropy_coef": 0.01,
                        "batch_size": 64,
                        "epochs": 12,
                        "lam": 0.95,
                        "target_kl": 0.01,
                        # Reward function weights
                        "reward_weights": {
                            "rating_weight": 1.0,
                            "length_weight": 0.15,
                            "sentiment_weight": 0.35,
                            "engagement_weight": 0.5,
                            "coherence_weight": 0.45,
                            "informativeness_weight": 0.5,
                            "helpfulness_weight": 0.6,
                            "consistency_weight": 0.4
                        }
                    }
                elif self.model_type == "transfer_learning":
                    # Transfer learning model hyperparameters
                    rl_config = {
                        "actor_lr": 0.00015,  # Very low learning rate for fine-tuning
                        "critic_lr": 0.0006,
                        "gamma": 0.995,       # Very high discount factor
                        "clip_ratio": 0.1,    # Very conservative updates
                        "entropy_coef": 0.005, # Less exploration
                        "batch_size": 128,     # Larger batch size
                        "epochs": 15,
                        "lam": 0.97,
                        "target_kl": 0.008,
                        # Reward function weights
                        "reward_weights": {
                            "rating_weight": 1.0,
                            "length_weight": 0.1,
                            "sentiment_weight": 0.3,
                            "engagement_weight": 0.4,
                            "coherence_weight": 0.6,
                            "informativeness_weight": 0.7,
                            "helpfulness_weight": 0.8,
                            "consistency_weight": 0.5
                        }
                    }
                else:  # Default for advanced_lstm
                    # Advanced LSTM model hyperparameters (default)
                    rl_config = {
                        "actor_lr": 0.0003,
                        "critic_lr": 0.001,
                        "gamma": 0.99,
                        "clip_ratio": 0.2,
                        "entropy_coef": 0.01,
                        "batch_size": 64,
                        "epochs": 10,
                        "lam": 0.95,
                        "target_kl": 0.01,
                        # Reward function weights
                        "reward_weights": {
                            "rating_weight": 1.0,
                            "length_weight": 0.2,
                            "sentiment_weight": 0.3,
                            "engagement_weight": 0.5,
                            "coherence_weight": 0.4,
                            "informativeness_weight": 0.4,
                            "helpfulness_weight": 0.6,
                            "consistency_weight": 0.3
                        }
                    }

                # Log the selected configuration
                self._print_debug(f"Using optimized RL hyperparameters for model type: {self.model_type}")
            else:
                rl_config = None

            self.learning_utils = LearningUtils(
                model=self.model,
                use_reinforcement_learning=self.use_reinforcement_learning,
                rl_algorithm="ppo" if self.use_reinforcement_learning else None,
                rl_config=rl_config
            )

        # Initialize advanced reinforcement learning if enabled
        self.advanced_rl_manager = None
        if self.use_advanced_rl:
            try:
                # Configure advanced RL
                advanced_rl_config = {
                    "use_transformer_state": True,
                    "use_hierarchical_rl": True,
                    "use_multi_objective_rl": True,
                    "use_meta_learning": True,
                    "use_ab_testing": True,
                    "state_size": 128,
                    "action_size": 10,
                    "agent_type": "hierarchical",  # "hierarchical", "multi_objective", or "meta_learning"
                    "transformer_model": "distilbert-base-uncased",
                    "experiment_id": f"experiment_{int(time.time())}"
                }

                self.advanced_rl_manager = AdvancedRLManager(
                    model_dir="data/rl_models",
                    config=advanced_rl_config
                )

                # Set user ID for personalized learning
                if self.user_id:
                    self.advanced_rl_manager.set_user_id(self.user_id, self.user_name)

                self._print_debug("Initialized advanced reinforcement learning manager.")
            except Exception as e:
                self._print_debug(f"Error initializing advanced RL manager: {e}")
                self.use_advanced_rl = False

        # Initialize distributed training if enabled
        self.distributed_training_manager = None
        if self.use_distributed_training:
            try:
                # Configure distributed training
                distributed_training_config = {
                    "role": "worker",  # "worker" or "parameter_server"
                    "worker_index": 0,
                    "cluster_config": {
                        "workers": ["localhost:2222"],
                        "parameter_servers": ["localhost:2223"]
                    },
                    "sync_frequency": 10,
                    "batch_size": 32,
                    "learning_rate": 0.001
                }

                self.distributed_training_manager = DistributedTrainingManager(
                    model_dir="data/distributed",
                    config=distributed_training_config
                )

                self._print_debug("Initialized distributed training manager.")
            except Exception as e:
                self._print_debug(f"Error initializing distributed training manager: {e}")
                self.use_distributed_training = False

        # Initialize explainable AI if enabled
        self.explainable_ai = None
        if self.use_explainable_ai:
            try:
                # Configure explainable AI
                explainable_ai_config = {
                    "explanation_level": "detailed",  # "simple", "detailed", "technical"
                    "use_feature_importance": True,
                    "use_counterfactuals": True,
                    "use_attention_visualization": True,
                    "output_dir": "data/explanations"
                }

                self.explainable_ai = ExplainableAI(
                    model=self.model,
                    config=explainable_ai_config
                )

                self._print_debug("Initialized explainable AI component.")
            except Exception as e:
                self._print_debug(f"Error initializing explainable AI: {e}")
                self.use_explainable_ai = False

        # Initialize continual learning if enabled
        self.continual_learning_manager = None
        if self.use_continual_learning:
            try:
                # Configure continual learning
                continual_learning_config = {
                    "use_replay_memory": True,
                    "use_elastic_weight_consolidation": True,
                    "use_knowledge_distillation": True,
                    "use_model_snapshots": True,
                    "replay_buffer_size": 1000,
                    "ewc_lambda": 0.1,
                    "distillation_temp": 2.0,
                    "snapshot_frequency": 10
                }

                self.continual_learning_manager = ContinualLearningManager(
                    model=self.model,
                    model_dir="data/continual_learning",
                    config=continual_learning_config
                )

                self._print_debug("Initialized continual learning manager.")
            except Exception as e:
                self._print_debug(f"Error initializing continual learning manager: {e}")
                self.use_continual_learning = False

        # Initialize safety filter if enabled
        self.safety_filter = None
        if self.use_safety_filter:
            try:
                # Configure safety filter
                safety_filter_config = {
                    "safety_level": self.safety_level,  # "low", "medium", "high"
                    "enable_content_filtering": True,
                    "enable_personal_info_protection": True,
                    "enable_harmful_response_detection": True,
                    "enable_reward_hacking_prevention": True,
                    "enable_safety_monitoring": True,
                    "safety_logs_dir": "logs/safety"
                }

                self.safety_filter = SafetyFilter(config=safety_filter_config)

                self._print_debug(f"Initialized safety filter with level: {self.safety_level}")
            except Exception as e:
                self._print_debug(f"Error initializing safety filter: {e}")
                self.use_safety_filter = False

        # Initialize integrations if available
        self.integration_manager = None
        if INTEGRATIONS_AVAILABLE:
            try:
                self.integration_manager = IntegrationManager()
                self._print_debug("Initialized integration manager.")
            except Exception as e:
                self._print_debug(f"Error initializing integration manager: {e}")

        # Greeting
        self._print_ai(f"Hello! I'm {self.name}, your AI assistant.")
        if self.use_deep_learning:
            self._print_ai(f"I'm using deep learning to improve my responses over time.")
            if self.use_reinforcement_learning:
                self._print_ai("I'm also using reinforcement learning to learn from your feedback.")

        # Advanced features greeting
        advanced_features = []
        if self.use_advanced_rl:
            advanced_features.append("advanced reinforcement learning")
        if self.use_distributed_training:
            advanced_features.append("distributed training")
        if self.use_explainable_ai:
            advanced_features.append("explainable AI")
        if self.use_continual_learning:
            advanced_features.append("continual learning")
        if self.use_safety_filter:
            advanced_features.append("safety mechanisms")

        if advanced_features:
            features_str = ", ".join(advanced_features[:-1])
            if len(advanced_features) > 1:
                features_str += f", and {advanced_features[-1]}"
            else:
                features_str = advanced_features[0]
            self._print_ai(f"I'm enhanced with {features_str} capabilities.")

        self._print_ai("What's your name?")

    def _print_ai(self, message):
        """
        Print a message from the AI with formatting.

        Args:
            message (str): The message to print
        """
        print(f"{Fore.GREEN}{self.name}:{Style.RESET_ALL} {message}")

    def _print_user(self, message):
        """
        Print a user message with formatting.

        Args:
            message (str): The message to print
        """
        name = self.user_name if self.user_name else "You"
        print(f"{Fore.BLUE}{name}:{Style.RESET_ALL} {message}")

    def _print_debug(self, message):
        """
        Print a debug message if verbose mode is enabled.

        Args:
            message (str): The debug message to print
        """
        if self.verbose:
            print(f"{Fore.YELLOW}[DEBUG]:{Style.RESET_ALL} {message}")

    def process_input(self, user_input):
        """
        Process user input and generate a response.

        Args:
            user_input (str): The user's input text

        Returns:
            str: The AI's response
        """
        # Store the user input for potential feedback
        self.last_user_input = user_input

        # Apply safety filter to user input if enabled
        if self.use_safety_filter and self.safety_filter:
            try:
                filtered_input, _, is_safe, safety_info = self.safety_filter.filter_and_check("", user_input)

                # If input is unsafe, warn the user
                if not is_safe and safety_info.get("input_safety_score", 1.0) < 0.5:
                    self._print_debug(f"Safety filter detected unsafe content: {safety_info}")
                    response = "I'm sorry, but I detected potentially unsafe content in your message. Please rephrase your request."
                    self.conversation_history.append(("user", user_input))
                    self.conversation_history.append(("ai", response))
                    self.last_ai_response = response
                    return response

                # Use filtered input
                user_input = filtered_input
            except Exception as e:
                self._print_debug(f"Error applying safety filter: {e}")

        # If we don't have the user's name yet, get it
        if self.user_name is None:
            self.user_name = user_input

            # Update user ID for meta-learning if using advanced RL
            if self.use_advanced_rl and self.advanced_rl_manager:
                self.advanced_rl_manager.set_user_id(self.user_id, self.user_name)

            response = f"Nice to meet you, {self.user_name}! How can I help you today?"
            self.conversation_history.append(("user", user_input))
            self.conversation_history.append(("ai", response))
            self.last_ai_response = response
            return response

        # Check for feedback commands
        if self.last_ai_response and any(word in user_input.lower() for word in ["rate", "feedback"]):
            try:
                # Extract rating from input (e.g., "rate 5" or "feedback 3")
                words = user_input.lower().split()
                for i, word in enumerate(words):
                    if word in ["rate", "feedback"] and i + 1 < len(words):
                        try:
                            rating = int(words[i + 1])
                            if 1 <= rating <= 5:
                                feedback_response = self._add_feedback(rating)
                                self.conversation_history.append(("user", user_input))
                                self.conversation_history.append(("ai", feedback_response))
                                self.last_ai_response = feedback_response
                                return feedback_response
                        except ValueError:
                            pass
            except Exception as e:
                self._print_debug(f"Error processing feedback: {e}")

        # Check for training command
        if self.use_deep_learning and "train" in user_input.lower():
            # Check if transfer learning is requested
            use_transfer_learning = "transfer" in user_input.lower()

            # Check for continual learning
            use_continual_learning = "continual" in user_input.lower() and self.use_continual_learning

            # Check for distributed training
            use_distributed_training = "distributed" in user_input.lower() and self.use_distributed_training

            response = self._train_model(
                use_transfer_learning=use_transfer_learning,
                use_continual_learning=use_continual_learning,
                use_distributed_training=use_distributed_training
            )

            self.conversation_history.append(("user", user_input))
            self.conversation_history.append(("ai", response))
            self.last_ai_response = response
            return response

        # Check for explanation request
        if self.use_explainable_ai and self.explainable_ai and any(word in user_input.lower() for word in ["explain", "why", "how come", "reasoning"]):
            if self.last_ai_response:
                try:
                    # Generate explanation for the last response
                    explanation = self.explainable_ai.explain_text_decision(
                        self.last_user_input,
                        self.last_ai_response,
                        None  # No attention weights available
                    )

                    explanation_text = "Here's my explanation for my previous response:\n\n"
                    explanation_text += explanation.get("explanation_text", "I don't have a specific explanation for this response.")

                    # Add important words if available
                    important_words = explanation.get("important_words", [])
                    if important_words:
                        explanation_text += "\n\nKey words that influenced my response:\n"
                        for word, importance in important_words[:5]:  # Top 5 words
                            explanation_text += f"- {word} (importance: {importance:.2f})\n"

                    # Add visualization path if available
                    vis_path = explanation.get("visualization_path")
                    if vis_path:
                        explanation_text += f"\nI've also created a visualization of my decision process at {vis_path}"

                    self.conversation_history.append(("user", user_input))
                    self.conversation_history.append(("ai", explanation_text))
                    self.last_ai_response = explanation_text
                    return explanation_text
                except Exception as e:
                    self._print_debug(f"Error generating explanation: {e}")

        # Debug information
        if self.verbose:
            keywords = self.nlp_utils.extract_keywords(user_input)
            sentiment = self.nlp_utils.sentiment_analysis(user_input)
            self._print_debug(f"Keywords: {keywords}")
            self._print_debug(f"Sentiment: {sentiment}")

        # Check for integration-related intents
        if self.integration_manager:
            # Extract entities
            entities = self.nlp_utils.extract_entities(user_input)
            if self.verbose:
                self._print_debug(f"Entities: {entities}")

            # Recognize intent
            intent_result = self.nlp_utils.recognize_intent(user_input)
            intent = intent_result["intent"]
            confidence = intent_result["confidence"]

            if self.verbose:
                self._print_debug(f"Intent: {intent} (confidence: {confidence:.2f})")

            # Handle integration intents with sufficient confidence
            if confidence > 0.6 and intent in ["weather", "forecast", "calendar", "email", "sms"]:
                try:
                    # Handle the intent with the integration manager
                    result = self.integration_manager.handle_intent(intent, entities, user_input)

                    # Format the response based on the intent and result
                    if intent == "weather" and "error" not in result:
                        response = (f"The current weather in {result['location']} is {result['description']} "
                                   f"with a temperature of {result['temperature']}°C.")
                    elif intent == "forecast" and "error" not in result:
                        response = f"Here's the forecast for {result['location']}:\n"
                        for day in result["days"][:3]:  # Show first 3 days
                            day_forecasts = day["forecasts"]
                            if day_forecasts:
                                mid_day = day_forecasts[len(day_forecasts)//2]
                                response += (f"- {day['date']}: {mid_day['description']}, "
                                           f"temperature around {mid_day['temperature']}°C\n")
                    elif intent == "calendar" and "error" not in result:
                        if isinstance(result, list) and result:
                            response = "Here are your upcoming events:\n"
                            for event in result[:3]:  # Show first 3 events
                                response += f"- {event['summary']} on {event['start']}\n"
                        else:
                            response = "You don't have any upcoming events."
                    else:
                        # For other intents or errors, use the model's response
                        response = self.model.generate_response(user_input, self.user_name)

                    # Apply safety filter to response if enabled
                    if self.use_safety_filter and self.safety_filter:
                        try:
                            _, filtered_response, is_safe, safety_info = self.safety_filter.filter_and_check(user_input, response)
                            response = filtered_response
                        except Exception as e:
                            self._print_debug(f"Error applying safety filter to response: {e}")

                    self.last_ai_response = response
                    self.conversation_history.append(("user", user_input))
                    self.conversation_history.append(("ai", response))
                    return response
                except Exception as e:
                    self._print_debug(f"Error handling integration intent: {e}")

        # Use advanced RL if enabled
        if self.use_advanced_rl and self.advanced_rl_manager:
            try:
                # Get state representation
                state = self.advanced_rl_manager.get_state(
                    user_input,
                    self.last_ai_response or "",
                    self.conversation_history
                )

                # Get action from advanced RL
                action = self.advanced_rl_manager.get_action(state)

                # Use action to influence response generation
                # For now, we'll just use it as a seed for the model
                response = self.model.generate_response(user_input, self.user_name, seed=action)

                # Apply safety filter to response if enabled
                if self.use_safety_filter and self.safety_filter:
                    try:
                        _, filtered_response, is_safe, safety_info = self.safety_filter.filter_and_check(user_input, response)
                        response = filtered_response
                    except Exception as e:
                        self._print_debug(f"Error applying safety filter to response: {e}")

                # Store the response
                self.last_ai_response = response
                self.conversation_history.append(("user", user_input))
                self.conversation_history.append(("ai", response))

                # Generate explanation if explainable AI is enabled
                if self.use_explainable_ai and self.explainable_ai:
                    try:
                        # Get explanation from advanced RL
                        explanation = self.advanced_rl_manager.get_explanation(state, action, user_input, response)
                        self._print_debug(f"RL explanation: {explanation}")
                    except Exception as e:
                        self._print_debug(f"Error getting RL explanation: {e}")

                return response
            except Exception as e:
                self._print_debug(f"Error using advanced RL: {e}")
                # Fall back to standard response generation

        # Generate response using the model
        response = self.model.generate_response(user_input, self.user_name)

        # Apply safety filter to response if enabled
        if self.use_safety_filter and self.safety_filter:
            try:
                _, filtered_response, is_safe, safety_info = self.safety_filter.filter_and_check(user_input, response)
                response = filtered_response
            except Exception as e:
                self._print_debug(f"Error applying safety filter to response: {e}")

        # Store the response
        self.last_ai_response = response
        self.conversation_history.append(("user", user_input))
        self.conversation_history.append(("ai", response))

        # Add to continual learning if enabled
        if self.use_continual_learning and self.continual_learning_manager:
            try:
                # Add to replay buffer
                self.continual_learning_manager.add_to_replay_buffer((user_input, response))
            except Exception as e:
                self._print_debug(f"Error adding to continual learning: {e}")

        # Check for exit command
        if any(word in user_input.lower() for word in ["bye", "goodbye", "exit", "quit"]):
            return response

        return response

    def _add_feedback(self, rating):
        """
        Add feedback for the last conversation.

        Args:
            rating (int): User rating (1-5, where 5 is best)

        Returns:
            str: Feedback acknowledgment message
        """
        if not self.last_user_input or not self.last_ai_response:
            return "I can't process feedback without a previous conversation."

        try:
            self._print_debug(f"Adding feedback with rating: {rating}")

            # Initialize reward variable
            reward = None

            # Add feedback with standard learning utils if deep learning is enabled
            if self.use_deep_learning and self.learning_utils:
                # Add feedback with conversation history for context
                reward = self.learning_utils.add_feedback(
                    user_input=self.last_user_input,
                    ai_response=self.last_ai_response,
                    rating=rating,
                    category=None,
                    conversation_history=self.conversation_history
                )

            # Add feedback with advanced RL if enabled
            if self.use_advanced_rl and self.advanced_rl_manager:
                try:
                    # Process feedback with advanced RL
                    advanced_reward = self.advanced_rl_manager.process_feedback(
                        self.last_user_input,
                        self.last_ai_response,
                        rating,
                        self.conversation_history
                    )

                    # If standard reward is None, use advanced reward
                    if reward is None:
                        reward = advanced_reward

                    # Train the advanced RL model if we have enough feedback
                    if len(self.conversation_history) >= 10:
                        training_results = self.advanced_rl_manager.train()
                        self._print_debug(f"Advanced RL training results: {training_results}")
                except Exception as e:
                    self._print_debug(f"Error processing feedback with advanced RL: {e}")

            # Create feedback message
            feedback_message = f"Thank you for your feedback! You rated my response as {rating}/5."

            # Add reinforcement learning details if enabled
            if (self.use_reinforcement_learning or self.use_advanced_rl) and reward is not None:
                self._print_debug(f"Reinforcement learning reward: {reward:.2f}")

                # If we have enough feedback, suggest training
                if self.use_deep_learning:
                    stats = self.learning_utils.get_feedback_stats()
                    count = stats["count"]
                    if count >= 10:
                        feedback_message += "\nI now have enough feedback to train. You can type 'train' to improve my responses."

                # Add advanced RL training suggestion
                if self.use_advanced_rl:
                    feedback_message += "\nI'm continuously learning from your feedback using advanced reinforcement learning."

            # Add to continual learning if enabled
            if self.use_continual_learning and self.continual_learning_manager:
                try:
                    # Add feedback to replay buffer
                    self.continual_learning_manager.add_to_replay_buffer((
                        self.last_user_input,
                        self.last_ai_response,
                        rating
                    ))

                    feedback_message += "\nYour feedback is also being used for continual learning."
                except Exception as e:
                    self._print_debug(f"Error adding feedback to continual learning: {e}")

            # Reset last input/response after feedback
            self.last_user_input = None
            self.last_ai_response = None

            return feedback_message

        except Exception as e:
            self._print_debug(f"Error adding feedback: {e}")
            return "Sorry, there was an error processing your feedback."

    def _train_model(self, use_transfer_learning=False, use_continual_learning=False, use_distributed_training=False):
        """
        Train the model using collected feedback.

        Args:
            use_transfer_learning (bool): Whether to use transfer learning techniques
            use_continual_learning (bool): Whether to use continual learning
            use_distributed_training (bool): Whether to use distributed training

        Returns:
            str: Training result message
        """
        if not self.use_deep_learning and not (self.use_advanced_rl or self.use_continual_learning or self.use_distributed_training):
            return "Sorry, I'm not using any learning models that can be trained."

        try:
            # Initialize training results
            training_results = {}
            training_messages = []

            # Get feedback stats if using standard learning
            count = 0
            if self.use_deep_learning and self.learning_utils:
                stats = self.learning_utils.get_feedback_stats()
                count = stats["count"]

                if count < 10 and not (self.use_advanced_rl or use_continual_learning or use_distributed_training):
                    return f"I need more feedback to train. Currently have {count}/10 feedback entries."

            self._print_ai(f"Training in progress... This might take a moment.")

            # Prepare training message
            if self.use_deep_learning:
                training_messages.append(f"Training with {self.model_type} model...")

                if self.use_reinforcement_learning:
                    rl_algorithm = self.learning_utils.rl_algorithm
                    training_messages.append(f"Using {rl_algorithm.upper()} reinforcement learning to optimize based on your feedback ratings.")

                    # Add algorithm-specific details
                    if rl_algorithm == "dqn":
                        training_messages.append("Using Deep Q-Network for better action selection.")
                    elif rl_algorithm == "a2c":
                        training_messages.append("Using Advantage Actor-Critic for improved policy learning.")
                    elif rl_algorithm == "ppo":
                        training_messages.append("Using Proximal Policy Optimization for stable policy updates.")

                if use_transfer_learning:
                    training_messages.append("Using transfer learning techniques for better results.")

            # Add advanced RL message if enabled
            if self.use_advanced_rl and self.advanced_rl_manager:
                agent_type = self.advanced_rl_manager.current_agent_type
                training_messages.append(f"Training advanced reinforcement learning agent (type: {agent_type}).")

            # Add continual learning message if enabled
            if use_continual_learning and self.use_continual_learning:
                training_messages.append("Using continual learning to prevent catastrophic forgetting.")

            # Add distributed training message if enabled
            if use_distributed_training and self.use_distributed_training:
                training_messages.append("Using distributed training across multiple processes.")

            # Print training message
            self._print_ai("\n".join(training_messages))

            # Start timing
            start_time = time.time()

            # Train with standard learning utils
            if self.use_deep_learning and self.learning_utils:
                # Train the model
                standard_result = self.learning_utils.train_model(
                    epochs=5,
                    batch_size=2,
                    use_early_stopping=True,
                    use_transfer_learning=use_transfer_learning,
                    train_rl=self.use_reinforcement_learning,
                    save_rl_model=self.use_reinforcement_learning
                )

                training_results["standard"] = standard_result

            # Train with advanced RL if enabled
            if self.use_advanced_rl and self.advanced_rl_manager:
                try:
                    self._print_debug("Training with advanced RL...")
                    advanced_rl_result = self.advanced_rl_manager.train()
                    training_results["advanced_rl"] = advanced_rl_result

                    # Save the advanced RL model
                    self.advanced_rl_manager.save(f"model_{int(time.time())}")
                except Exception as e:
                    self._print_debug(f"Error training with advanced RL: {e}")

            # Train with continual learning if enabled
            if use_continual_learning and self.use_continual_learning and self.continual_learning_manager:
                try:
                    self._print_debug("Training with continual learning...")

                    # In a real implementation, you would create a proper dataset
                    # This is a simplified example
                    import tensorflow as tf
                    dummy_dataset = tf.data.Dataset.from_tensor_slices(
                        (np.random.rand(10, 128), np.random.randint(0, 10, size=(10,)))
                    ).batch(2)

                    continual_result = self.continual_learning_manager.train_with_continual_learning(
                        dataset=dummy_dataset,
                        new_task_name=f"task_{int(time.time())}",
                        epochs=2
                    )

                    training_results["continual"] = continual_result
                except Exception as e:
                    self._print_debug(f"Error training with continual learning: {e}")

            # Train with distributed training if enabled
            if use_distributed_training and self.use_distributed_training and self.distributed_training_manager:
                try:
                    self._print_debug("Training with distributed training...")

                    # In a real implementation, you would create a proper distributed dataset
                    # This is a simplified example
                    import tensorflow as tf
                    dummy_dataset = tf.data.Dataset.from_tensor_slices(
                        (np.random.rand(10, 128), np.random.randint(0, 10, size=(10,)))
                    ).batch(2)

                    distributed_result = self.distributed_training_manager.train_distributed(
                        dataset=dummy_dataset,
                        epochs=2
                    )

                    training_results["distributed"] = distributed_result
                except Exception as e:
                    self._print_debug(f"Error training with distributed training: {e}")

            # Calculate total training time
            training_time = time.time() - start_time

            # Create result message
            success = any(result for result in training_results.values())

            if success:
                message = f"Training complete in {training_time:.2f} seconds!\n\n"

                # Add standard training details if available
                if "standard" in training_results and training_results["standard"]:
                    message += f"Processed {count} feedback entries with the standard model.\n"

                # Add advanced RL details if available
                if "advanced_rl" in training_results and training_results["advanced_rl"]:
                    advanced_rl_result = training_results["advanced_rl"]
                    if isinstance(advanced_rl_result, dict) and "mean_reward" in advanced_rl_result:
                        message += f"Advanced RL training achieved mean reward: {advanced_rl_result['mean_reward']:.4f}\n"

                # Add continual learning details if available
                if "continual" in training_results and training_results["continual"]:
                    continual_result = training_results["continual"]
                    if isinstance(continual_result, dict) and "success" in continual_result and continual_result["success"]:
                        message += "Continual learning training completed successfully.\n"

                # Add distributed training details if available
                if "distributed" in training_results and training_results["distributed"]:
                    message += "Distributed training completed successfully.\n"

                message += "\nI should be smarter now!"
                return message
            else:
                return "Training failed. Please check the logs for details."

        except Exception as e:
            self._print_debug(f"Error training model: {e}")
            return f"An error occurred during training: {str(e)}"

    def run(self):
        """
        Run the AI assistant in an interactive loop.
        """
        try:
            while True:
                # Get user input
                user_input = input(f"{Fore.BLUE}{self.user_name if self.user_name else 'You'}:{Style.RESET_ALL} ")

                # Echo the input
                self._print_user(user_input)

                # Process the input
                response = self.process_input(user_input)

                # Print the response with a slight delay to seem more natural
                time.sleep(0.5)
                self._print_ai(response)

                # Check for exit
                if any(word in user_input.lower() for word in ["bye", "goodbye", "exit", "quit"]):
                    break

        except KeyboardInterrupt:
            # Handle Ctrl+C gracefully
            print("\n")
            self._print_ai("Goodbye! Have a great day!")

        except Exception as e:
            print(f"\nAn error occurred: {e}")
            if self.verbose:
                import traceback
                traceback.print_exc()

        finally:
            print("\nThank you for using NoahAI!")

def parse_arguments():
    """
    Parse command line arguments.

    Returns:
        argparse.Namespace: The parsed arguments
    """
    parser = argparse.ArgumentParser(description="NoahAI - An advanced AI assistant with deep learning capabilities")
    parser.add_argument("--name", type=str, default="Noah", help="Name of the AI assistant")
    parser.add_argument("--responses", type=str, default="data/responses.json", help="Path to responses JSON file")
    parser.add_argument("--simple", action="store_true", help="Use simple model instead of deep learning")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose debug output")
    parser.add_argument("--user-id", type=str, default="default_user", help="User ID for personalized learning")

    # Add model type argument
    parser.add_argument("--model-type", type=str, default="advanced_lstm",
                        choices=["simple_lstm", "advanced_lstm", "cnn_lstm", "transfer_learning"],
                        help="Type of deep learning model to use")

    # Add reinforcement learning argument
    parser.add_argument("--reinforcement", action="store_true",
                        help="Enable reinforcement learning")

    # Add advanced reinforcement learning argument
    parser.add_argument("--advanced-rl", action="store_true",
                        help="Enable advanced reinforcement learning components")

    # Add distributed training argument
    parser.add_argument("--distributed", action="store_true",
                        help="Enable distributed training")

    # Add explainable AI argument
    parser.add_argument("--explainable", action="store_true",
                        help="Enable explainable AI components")

    # Add continual learning argument
    parser.add_argument("--continual", action="store_true",
                        help="Enable continual learning")

    # Add safety filter argument
    parser.add_argument("--safety", action="store_true",
                        help="Enable safety filter")

    # Add safety level argument
    parser.add_argument("--safety-level", type=str, default="medium",
                        choices=["low", "medium", "high"],
                        help="Safety level for the safety filter")

    return parser.parse_args()

if __name__ == "__main__":
    # Parse command line arguments
    args = parse_arguments()

    # Create and run the AI assistant
    noah = NoahAI(
        name=args.name,
        responses_file=args.responses,
        use_deep_learning=not args.simple,
        verbose=args.verbose,
        model_type=args.model_type,
        use_reinforcement_learning=args.reinforcement,
        use_advanced_rl=args.advanced_rl,
        use_distributed_training=args.distributed,
        use_explainable_ai=args.explainable,
        use_continual_learning=args.continual,
        use_safety_filter=args.safety,
        safety_level=args.safety_level,
        user_id=args.user_id
    )
    noah.run()
