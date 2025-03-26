import azure.cognitiveservices.speech as speechsdk
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("AzureSTT")

# Azure Speech API credentials (Replace with your actual values)
SPEECH_KEY = "Fj1KPt7grC6bAkNja7daZUstpP8wZTXsV6Zjr2FOxkO7wsBQ5SzQJQQJ99BCACHYHv6XJ3w3AAAAACOGL3Xg"
SPEECH_ENDPOINT = "https://ai-aihackthonhub282549186415.cognitiveservices.azure.com"
SPEECH_HOST = "https://ai-aihackthonhub282549186415.cognitiveservices.azure.com"

def recognize_from_microphone():
    """Captures audio from the microphone and sends it to Azure Speech API."""
    
    # Set up speech configuration (using endpoint authentication)
    speech_config = speechsdk.SpeechConfig(
        subscription=SPEECH_KEY,
        endpoint=SPEECH_ENDPOINT
    )
    
    # Set up audio input from the default microphone
    audio_config = speechsdk.audio.AudioConfig(use_default_microphone=True)
    
    # Create a SpeechRecognizer
    speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)
    
    logger.info("Speak into your microphone...")

    # Start speech recognition and wait for a result
    result = speech_recognizer.recognize_once()

    # Handle result
    if result.reason == speechsdk.ResultReason.RecognizedSpeech:
        logger.info(f"Recognized: {result.text}")
    elif result.reason == speechsdk.ResultReason.NoMatch:
        logger.warning("No speech recognized.")
    elif result.reason == speechsdk.ResultReason.Canceled:
        cancellation_details = result.cancellation_details
        logger.error(f"Speech recognition canceled: {cancellation_details.reason}")
        if cancellation_details.reason == speechsdk.CancellationReason.Error:
            logger.error(f"Error details: {cancellation_details.error_details}")

if __name__ == "_main_":
    recognize_from_microphone()