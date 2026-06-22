from abc import ABC, abstractmethod

class BaseTranscriber(ABC):
    # Why use an abstract base class (ABC)?
    # Enforcing a common interface via ABC ensures that any new STT engine (like Whisper, Google Cloud STT, etc.)
    # implements the required 'transcribe' method and behaves predictably within the application,
    # preventing runtime attribute errors.

    @classmethod
    def start_preload(cls):
        # Why have start_preload in the base class?
        # Some local models (like Whisper) benefit from preloading model files into memory 
        # asynchronously during application startup. Cloud-based speech recognizers typically 
        # do not require this, so we provide a default empty implementation.
        pass

    @abstractmethod
    def transcribe(self, audio_path: str, **kwargs) -> list[dict]:
        # Why define this return type?
        # The VOD processing pipeline expects a list of dictionaries with 'start', 'end', and 'text' keys
        # to align timestamps with chat messages. All subclasses must return this exact format.
        pass
