from vibes_ttv.analyzers.stt.base import BaseTranscriber
from vibes_ttv.analyzers.stt.whisper_transcriber import WhisperTranscriber
from vibes_ttv.analyzers.stt.google_transcriber import GoogleSpeechTranscriber
from vibes_ttv.analyzers.stt.gemini_transcriber import GeminiSpeechTranscriber

# Register transcribers here for extensibility
# Why use a registry dictionary?
# A dictionary-based registry makes it trivially easy to add new STT engines.
# Developers only need to import their new Transcriber class and register it here
# without modifying any factory instantiation logic.
TRANSCRIBER_REGISTRY = {
    "whisper": WhisperTranscriber,
    "google_stt": GoogleSpeechTranscriber,
    "gemini": GeminiSpeechTranscriber,
}

def get_transcriber(engine_type: str, **kwargs) -> BaseTranscriber:
    # Why check registry dynamically?
    # Keeping instantiation decoupled from hardcoded if-else statements improves code quality,
    # allows default fallback logic, and ensures errors are raised early if an unsupported 
    # engine is selected.
    transcriber_class = TRANSCRIBER_REGISTRY.get(engine_type)
    if not transcriber_class:
        supported = list(TRANSCRIBER_REGISTRY.keys())
        raise ValueError(
            f"Unsupported STT engine type '{engine_type}'. Supported engines: {supported}"
        )
    
    return transcriber_class(**kwargs)
