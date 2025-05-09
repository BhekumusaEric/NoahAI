# NoahAI - Advanced AI Assistant

NoahAI is an advanced AI assistant with deep learning capabilities, natural language processing, and various integrations. It can understand user queries, learn from feedback, and provide helpful responses through multiple interfaces.

## Table of Contents
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Customization](#customization)
- [Contributing](#contributing)
- [License](#license)

## Features

### Advanced NLP Capabilities
- **Intent Recognition**: Understands the purpose of user queries
- **Entity Extraction**: Identifies important information in text
- **Sentiment Analysis**: Detects the emotional tone of messages
- **Keyword Extraction**: Identifies key topics in conversations

### Deep Learning
- **TensorFlow-based Model**: Uses neural networks for understanding and generating responses
- **Transfer Learning**: Leverages pre-trained models for better performance
- **Reinforcement Learning**: Improves over time based on user feedback
- **Continuous Learning**: Gets smarter with each interaction

### User Interface
- **Command-line Interface**: Simple text-based interaction
- **Web Interface**: Modern, responsive web application
- **Voice Capabilities**: Speech recognition and text-to-speech
- **Mobile-friendly Design**: Works well on all devices

### External Integrations
- **Weather API**: Get current weather and forecasts
- **News API**: Access the latest news and articles
- **Calendar Integration**: Manage events and appointments
- **Email Capabilities**: Send and manage emails

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/NoahAI.git
cd NoahAI
```

2. Create a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Run the setup script to install dependencies and download required models:
```bash
python setup.py
```

4. Configure API keys:
   - Edit the `.env` file with your API keys for external services
   - For Google Calendar and Gmail integration, follow the instructions to set up OAuth credentials

## Usage

### Command-line Interface

Run NoahAI with the command-line interface:

```bash
python noah_ai.py
```

Options:
- `--name NAME`: Set a custom name for the AI assistant (default: "Noah")
- `--responses PATH`: Specify a custom responses JSON file (default: "data/responses.json")
- `--simple`: Use the simple model instead of deep learning
- `--verbose`: Enable verbose debug output

Examples:
```bash
# Run with deep learning (default)
python noah_ai.py --name Jarvis --verbose

# Run with simple model (no deep learning)
python noah_ai.py --simple
```

### Web Interface

Run NoahAI with the web interface:

```bash
python run_web_app.py
```

Options:
- `--host`: Host to run the server on (default: 0.0.0.0)
- `--port`: Port to run the server on (default: 5000)
- `--debug`: Run in debug mode
- `--simple`: Use simple model instead of deep learning

Then open your browser and navigate to `http://localhost:5000`

### Voice Commands

In the web interface, enable voice input/output in the settings panel. You can then:
- Click the microphone button to start voice input
- Hear responses spoken aloud

### Deep Learning Features

NoahAI includes deep learning capabilities that allow it to improve over time:

1. **Feedback Collection**: Rate responses to help the AI learn
   ```
   You: How are you today?
   Noah: I'm doing well, thanks for asking! How about you?
   You: rate 5
   Noah: Thank you for your feedback! I'll use it to improve.
   ```

2. **Model Training**: Train the model with collected feedback
   ```
   You: train
   Noah: Training in progress... This might take a moment.
   Noah: Training complete! Processed 15 feedback entries. I should be smarter now!
   ```

The AI requires at least 10 feedback entries before it can train the model.

## Project Structure

```
noah-ai/
├── data/                  # Data files
│   ├── model/             # Model files
│   ├── cache/             # Cache files
│   ├── models/            # NLP model files
│   │   ├── intents.json   # Intent definitions
│   │   └── entities.json  # Entity definitions
│   └── responses.json     # Response templates
├── logs/                  # Log files
├── models/                # AI models
│   ├── deep_learning_model.py  # TensorFlow model
│   └── simple_model.py    # Simple response model
├── src/                   # Source code
│   ├── integrations/      # External API integrations
│   │   └── external_apis.py  # Weather, news, calendar, email APIs
│   └── utils/             # Utility modules
│       ├── __init__.py    # Package initialization
│       ├── nlp_utils.py   # Advanced NLP utilities
│       ├── learning_utils.py # Learning and feedback utilities
│       └── voice_utils.py # Speech recognition and synthesis
├── web/                   # Web interface
│   ├── static/            # Static files
│   │   ├── css/           # CSS stylesheets
│   │   └── js/            # JavaScript files
│   └── templates/         # HTML templates
├── .env                   # Environment variables
├── noah_ai.py             # Command-line interface
├── run_web_app.py         # Web application runner
├── setup.py               # Setup script
├── web_app.py             # Web application
└── requirements.txt       # Python dependencies
```

## Customization

### Response Templates

You can customize the AI's responses by editing the `data/responses.json` file. The file contains categories of responses that the AI can use in different contexts.

Example:
```json
{
  "greeting": [
    "Hello, {user_name}! How can I help you today?",
    "Hi there, {user_name}! What can I do for you?"
  ],
  "farewell": [
    "Goodbye, {user_name}! Have a great day!",
    "See you later, {user_name}!"
  ]
}
```

### Adding New Intents

Edit `data/models/intents.json` to add new intents with patterns and responses:

```json
{
  "new_intent": {
    "patterns": [
      "pattern one",
      "pattern two",
      "another pattern"
    ],
    "responses": [
      "Response one",
      "Response two"
    ]
  }
}
```

### Adding New Entities

Edit `data/models/entities.json` to add new entity types with recognition patterns:

```json
{
  "new_entity_type": {
    "patterns": [
      "regex_pattern_one",
      "regex_pattern_two"
    ]
  }
}
```

### Adding New Integrations

Create new integration classes in `src/integrations/` following the pattern of existing integrations:

```python
class NewIntegration:
    def __init__(self, api_key=None):
        self.api_key = api_key or os.getenv('NEW_API_KEY')
        # Initialize the integration

    def get_data(self, query):
        # Implement the integration logic
        return {"result": "data"}
```

### Improving the Deep Learning Model

The deep learning model can be improved in several ways:

1. **Model Architecture**: Modify the neural network architecture in `_create_model()` method
2. **Training Parameters**: Adjust epochs, batch size, and other parameters in the `train()` method
3. **Data Preprocessing**: Enhance text preprocessing in the `preprocess_text()` method
4. **Transfer Learning**: Implement transfer learning with pre-trained models
5. **Reinforcement Learning**: Add reinforcement learning capabilities

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
