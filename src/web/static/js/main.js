/**
 * NoahAI Web Interface - Main JavaScript
 * 
 * This file contains the client-side logic for the NoahAI web interface.
 */

// Global variables
let sessionId = null;
let socket = null;
let lastAiMessage = null;
let isTyping = false;
let speechSynthesis = window.speechSynthesis;
let speechRecognition = null;
let voiceEnabled = false;

// DOM elements
const welcomeScreen = document.getElementById('welcome-screen');
const chatContainer = document.getElementById('chat-container');
const userNameInput = document.getElementById('user-name');
const startChatButton = document.getElementById('start-chat-button');
const chatMessages = document.getElementById('chat-messages');
const messageInput = document.getElementById('message-input');
const sendButton = document.getElementById('send-button');
const connectionStatus = document.getElementById('connection-status');
const settingsButton = document.getElementById('settings-button');
const settingsPanel = document.getElementById('settings-panel');
const closeSettings = document.getElementById('close-settings');
const saveSettings = document.getElementById('save-settings');
const trainButton = document.getElementById('train-button');
const voiceButton = document.getElementById('voice-button');
const useVoiceToggle = document.getElementById('use-voice');

// Initialize the application
document.addEventListener('DOMContentLoaded', () => {
    // Initialize event listeners
    initEventListeners();
    
    // Initialize speech recognition if available
    initSpeechRecognition();
});

/**
 * Initialize event listeners
 */
function initEventListeners() {
    // Start chat button
    startChatButton.addEventListener('click', startChat);
    
    // Send message button
    sendButton.addEventListener('click', sendMessage);
    
    // Message input enter key
    messageInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            sendMessage();
        }
    });
    
    // Settings button
    settingsButton.addEventListener('click', toggleSettings);
    
    // Close settings button
    closeSettings.addEventListener('click', toggleSettings);
    
    // Save settings button
    saveSettings.addEventListener('click', saveSettingsHandler);
    
    // Train button
    trainButton.addEventListener('click', trainModel);
    
    // Voice button
    voiceButton.addEventListener('click', toggleVoice);
    
    // Voice toggle
    useVoiceToggle.addEventListener('change', (e) => {
        voiceEnabled = e.target.checked;
        if (voiceEnabled) {
            voiceButton.innerHTML = '<i class="fas fa-microphone me-1"></i>Voice (On)';
        } else {
            voiceButton.innerHTML = '<i class="fas fa-microphone me-1"></i>Voice';
            stopSpeechRecognition();
        }
    });
}

/**
 * Initialize speech recognition
 */
function initSpeechRecognition() {
    // Check if speech recognition is available
    if ('webkitSpeechRecognition' in window) {
        speechRecognition = new webkitSpeechRecognition();
        speechRecognition.continuous = false;
        speechRecognition.interimResults = false;
        
        speechRecognition.onresult = (event) => {
            const transcript = event.results[0][0].transcript;
            messageInput.value = transcript;
            sendMessage();
        };
        
        speechRecognition.onerror = (event) => {
            console.error('Speech recognition error:', event.error);
            stopSpeechRecognition();
        };
        
        speechRecognition.onend = () => {
            voiceButton.innerHTML = '<i class="fas fa-microphone me-1"></i>Voice';
            voiceButton.classList.remove('active');
        };
    } else {
        console.warn('Speech recognition not available');
        voiceButton.style.display = 'none';
        useVoiceToggle.disabled = true;
    }
}

/**
 * Start speech recognition
 */
function startSpeechRecognition() {
    if (speechRecognition && voiceEnabled) {
        try {
            speechRecognition.start();
            voiceButton.innerHTML = '<i class="fas fa-microphone-slash me-1"></i>Listening...';
            voiceButton.classList.add('active');
        } catch (error) {
            console.error('Error starting speech recognition:', error);
        }
    }
}

/**
 * Stop speech recognition
 */
function stopSpeechRecognition() {
    if (speechRecognition) {
        try {
            speechRecognition.stop();
            voiceButton.innerHTML = '<i class="fas fa-microphone me-1"></i>Voice';
            voiceButton.classList.remove('active');
        } catch (error) {
            console.error('Error stopping speech recognition:', error);
        }
    }
}

/**
 * Toggle voice input
 */
function toggleVoice() {
    if (!voiceEnabled) {
        useVoiceToggle.checked = true;
        voiceEnabled = true;
        voiceButton.innerHTML = '<i class="fas fa-microphone me-1"></i>Voice (On)';
    }
    
    if (voiceButton.classList.contains('active')) {
        stopSpeechRecognition();
    } else {
        startSpeechRecognition();
    }
}

/**
 * Speak text using speech synthesis
 * 
 * @param {string} text - Text to speak
 */
function speakText(text) {
    if (speechSynthesis && voiceEnabled) {
        // Stop any current speech
        speechSynthesis.cancel();
        
        // Create utterance
        const utterance = new SpeechSynthesisUtterance(text);
        
        // Set rate
        const rateInput = document.getElementById('voice-rate');
        utterance.rate = parseFloat(rateInput.value);
        
        // Speak
        speechSynthesis.speak(utterance);
    }
}

/**
 * Start chat session
 */
function startChat() {
    const userName = userNameInput.value.trim() || 'User';
    
    // Create session
    fetch('/api/session', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ user_name: userName })
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            // Store session ID
            sessionId = data.session_id;
            
            // Show chat interface
            welcomeScreen.classList.add('hidden');
            chatContainer.classList.remove('hidden');
            
            // Add welcome message
            addMessage('ai', data.message);
            
            // Initialize Socket.IO
            initSocketIO();
            
            // Focus message input
            messageInput.focus();
        } else {
            alert('Error creating session: ' + data.message);
        }
    })
    .catch(error => {
        console.error('Error creating session:', error);
        alert('Error creating session. Please try again.');
    });
}

/**
 * Initialize Socket.IO connection
 */
function initSocketIO() {
    // Connect to Socket.IO server
    socket = io();
    
    // Connection events
    socket.on('connect', () => {
        connectionStatus.textContent = 'Connected';
        connectionStatus.classList.remove('bg-danger');
        connectionStatus.classList.add('bg-success');
        
        // Join room
        socket.emit('join', { session_id: sessionId });
    });
    
    socket.on('disconnect', () => {
        connectionStatus.textContent = 'Disconnected';
        connectionStatus.classList.remove('bg-success');
        connectionStatus.classList.add('bg-danger');
    });
    
    // Message events
    socket.on('ai_response', (data) => {
        addMessage('ai', data.response);
        lastAiMessage = data.response;
        
        // Speak response if voice is enabled
        speakText(data.response);
    });
    
    socket.on('message_received', (data) => {
        // Show typing indicator
        showTypingIndicator();
    });
    
    socket.on('error', (data) => {
        alert('Error: ' + data.message);
    });
}

/**
 * Send message to the server
 */
function sendMessage() {
    const message = messageInput.value.trim();
    
    if (!message) {
        return;
    }
    
    // Add message to chat
    addMessage('user', message);
    
    // Clear input
    messageInput.value = '';
    
    // Send message via Socket.IO if available
    if (socket && socket.connected) {
        socket.emit('chat', {
            session_id: sessionId,
            message: message
        });
    } else {
        // Fallback to REST API
        fetch('/api/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ message: message })
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                addMessage('ai', data.response);
                lastAiMessage = data.response;
                
                // Speak response if voice is enabled
                speakText(data.response);
            } else {
                alert('Error: ' + data.message);
            }
        })
        .catch(error => {
            console.error('Error sending message:', error);
            alert('Error sending message. Please try again.');
        });
    }
    
    // Show typing indicator
    showTypingIndicator();
}

/**
 * Add message to chat
 * 
 * @param {string} sender - Message sender ('user' or 'ai')
 * @param {string} text - Message text
 */
function addMessage(sender, text) {
    // Remove typing indicator if present
    removeTypingIndicator();
    
    // Create message element
    const messageElement = document.createElement('div');
    messageElement.classList.add('message');
    messageElement.classList.add(sender === 'user' ? 'user-message' : 'ai-message');
    
    // Add message text
    messageElement.textContent = text;
    
    // Add timestamp
    const timestampElement = document.createElement('div');
    timestampElement.classList.add('message-time');
    timestampElement.textContent = new Date().toLocaleTimeString();
    messageElement.appendChild(timestampElement);
    
    // Add feedback buttons for AI messages
    if (sender === 'ai') {
        const feedbackElement = document.createElement('div');
        feedbackElement.classList.add('feedback-buttons');
        
        for (let i = 1; i <= 5; i++) {
            const button = document.createElement('button');
            button.classList.add('feedback-button');
            button.innerHTML = '★';
            button.setAttribute('data-rating', i);
            button.addEventListener('click', () => provideFeedback(i, button, feedbackElement));
            feedbackElement.appendChild(button);
        }
        
        messageElement.appendChild(feedbackElement);
    }
    
    // Add message to chat
    chatMessages.appendChild(messageElement);
    
    // Scroll to bottom
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

/**
 * Show typing indicator
 */
function showTypingIndicator() {
    if (isTyping) {
        return;
    }
    
    isTyping = true;
    
    // Create typing indicator
    const typingElement = document.createElement('div');
    typingElement.classList.add('typing-indicator');
    typingElement.id = 'typing-indicator';
    
    // Add dots
    typingElement.innerHTML = 'Noah is typing <span></span><span></span><span></span>';
    
    // Add to chat
    chatMessages.appendChild(typingElement);
    
    // Scroll to bottom
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

/**
 * Remove typing indicator
 */
function removeTypingIndicator() {
    const typingElement = document.getElementById('typing-indicator');
    
    if (typingElement) {
        typingElement.remove();
        isTyping = false;
    }
}

/**
 * Provide feedback for an AI message
 * 
 * @param {number} rating - Rating (1-5)
 * @param {HTMLElement} button - Clicked button
 * @param {HTMLElement} feedbackElement - Feedback buttons container
 */
function provideFeedback(rating, button, feedbackElement) {
    // Highlight buttons
    const buttons = feedbackElement.querySelectorAll('.feedback-button');
    buttons.forEach((btn, index) => {
        if (index < rating) {
            btn.classList.add('active');
        } else {
            btn.classList.remove('active');
        }
    });
    
    // Send feedback to server
    fetch('/api/feedback', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ rating: rating })
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            addMessage('ai', data.response);
        } else {
            alert('Error: ' + data.message);
        }
    })
    .catch(error => {
        console.error('Error providing feedback:', error);
        alert('Error providing feedback. Please try again.');
    });
    
    // Disable feedback buttons
    buttons.forEach(btn => {
        btn.disabled = true;
    });
}

/**
 * Toggle settings panel
 */
function toggleSettings() {
    settingsPanel.classList.toggle('open');
}

/**
 * Save settings
 */
function saveSettingsHandler() {
    // Get settings
    const settings = {
        use_reinforcement: document.getElementById('use-reinforcement').checked,
        use_advanced_rl: document.getElementById('use-advanced-rl').checked,
        use_continual: document.getElementById('use-continual').checked,
        use_transfer_learning: document.getElementById('use-transfer-learning').checked,
        use_distributed: document.getElementById('use-distributed').checked,
        use_safety: document.getElementById('use-safety').checked,
        safety_level: document.getElementById('safety-level').value,
        use_voice: document.getElementById('use-voice').checked
    };
    
    // Update voice setting
    voiceEnabled = settings.use_voice;
    
    // Close settings panel
    toggleSettings();
    
    // Show confirmation
    alert('Settings saved!');
}

/**
 * Train the model
 */
function trainModel() {
    // Get training options
    const useTransferLearning = document.getElementById('use-transfer-learning').checked;
    const useContinualLearning = document.getElementById('use-continual').checked;
    const useDistributedTraining = document.getElementById('use-distributed').checked;
    
    // Send training request
    fetch('/api/train', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            use_transfer_learning: useTransferLearning,
            use_continual_learning: useContinualLearning,
            use_distributed_training: useDistributedTraining
        })
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            addMessage('ai', data.response);
        } else {
            alert('Error: ' + data.message);
        }
    })
    .catch(error => {
        console.error('Error training model:', error);
        alert('Error training model. Please try again.');
    });
}
