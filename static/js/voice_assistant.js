/**
 * Voice Assistant for Medical Recommendation System
 * Converts speech to text for symptom input
 * Uses Web Speech API (built-in browser support)
 */

class VoiceAssistant {
    constructor(options = {}) {
        this.language = options.language || 'en-US';
        this.continuous = options.continuous || false;
        this.interimResults = options.interimResults || true;

        // Check browser support
        if (!('webkitSpeechRecognition' in window) && !('SpeechRecognition' in window)) {
            console.error('Speech Recognition not supported in this browser');
            this.supported = false;
            return;
        }

        this.supported = true;

        // Initialize Speech Recognition
        const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
        this.recognition = new SpeechRecognition();

        // Configuration
        this.recognition.continuous = this.continuous;
        this.recognition.interimResults = this.interimResults;
        this.recognition.lang = this.language;
        this.recognition.maxAlternatives = 1;

        // State
        this.isListening = false;
        this.transcript = '';
        this.finalTranscript = '';

        // Callbacks
        this.onStart = null;
        this.onEnd = null;
        this.onResult = null;
        this.onError = null;

        // Setup event listeners
        this.setupEventListeners();
    }

    setupEventListeners() {
        // Start event
        this.recognition.onstart = () => {
            this.isListening = true;
            console.log('Voice recognition started');
            if (this.onStart) this.onStart();
        };

        // Result event
        this.recognition.onresult = (event) => {
            let interimTranscript = '';

            for (let i = event.resultIndex; i < event.results.length; i++) {
                const transcript = event.results[i][0].transcript;

                if (event.results[i].isFinal) {
                    this.finalTranscript += transcript + ' ';
                } else {
                    interimTranscript += transcript;
                }
            }

            this.transcript = this.finalTranscript + interimTranscript;

            if (this.onResult) {
                this.onResult({
                    transcript: this.transcript,
                    finalTranscript: this.finalTranscript,
                    interimTranscript: interimTranscript,
                    isFinal: event.results[event.results.length - 1].isFinal
                });
            }
        };

        // End event
        this.recognition.onend = () => {
            this.isListening = false;
            console.log('Voice recognition ended');
            if (this.onEnd) this.onEnd();
        };

        // Error event
        this.recognition.onerror = (event) => {
            console.error('Speech recognition error:', event.error);
            this.isListening = false;

            if (this.onError) {
                this.onError({
                    error: event.error,
                    message: this.getErrorMessage(event.error)
                });
            }
        };
    }

    getErrorMessage(error) {
        const messages = {
            'no-speech': 'No speech detected. Please try again.',
            'audio-capture': 'No microphone found. Please check your device.',
            'not-allowed': 'Microphone access denied. Please allow microphone access.',
            'network': 'Network error occurred. Please check your connection.',
            'aborted': 'Speech recognition was aborted.',
            'language-not-supported': 'Language not supported.',
            'service-not-allowed': 'Speech recognition service not allowed.'
        };

        return messages[error] || 'An unknown error occurred.';
    }

    start() {
        if (!this.supported) {
            alert('Speech Recognition is not supported in your browser. Please use Chrome, Edge, or Safari.');
            return;
        }

        if (this.isListening) {
            console.warn('Already listening');
            return;
        }

        // Reset transcripts
        this.transcript = '';
        this.finalTranscript = '';

        try {
            this.recognition.start();
        } catch (error) {
            console.error('Failed to start recognition:', error);
        }
    }

    stop() {
        if (!this.isListening) {
            return;
        }

        try {
            this.recognition.stop();
        } catch (error) {
            console.error('Failed to stop recognition:', error);
        }
    }

    toggle() {
        if (this.isListening) {
            this.stop();
        } else {
            this.start();
        }
    }

    getTranscript() {
        return this.finalTranscript.trim();
    }

    clearTranscript() {
        this.transcript = '';
        this.finalTranscript = '';
    }

    isSupported() {
        return this.supported;
    }
}

// Export for use in other scripts
if (typeof module !== 'undefined' && module.exports) {
    module.exports = VoiceAssistant;
}
