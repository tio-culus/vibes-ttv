import whisper
import torch
import threading

class WhisperTranscriber:
    # Why not faster-whisper?
    # While faster-whisper is highly optimized, it requires specific C++ libraries (ctranslate2) 
    # which can be complex to compile or configure on Windows. 
    # Standard openai-whisper is more robust, installs out-of-the-box via pip, 
    # and RTX 4070 has more than enough power to run it at high speeds.
    
    _cached_model = None
    _is_loading = False
    _lock = threading.Lock()

    @classmethod
    def start_preload(cls):
        # Why run preload on background thread?
        # Loading the 1.6GB Whisper model from disk (or downloading it initially) is slow.
        # Starting this process in a daemon thread on app import leverages user idle time (inputting URL/API key)
        # and audio download time, completely hiding the load latency from the user.
        with cls._lock:
            if cls._cached_model is not None or cls._is_loading:
                return
            cls._is_loading = True
        
        thread = threading.Thread(target=cls._preload_target, daemon=True)
        thread.start()

    @classmethod
    def _preload_target(cls):
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"🧠 [Background] Preloading Whisper 'turbo' model on {device}...")
            # We fix the model to 'turbo' as agreed for maximum speed and accuracy.
            model = whisper.load_model("turbo", device=device)
            with cls._lock:
                cls._cached_model = model
            print(f"✨ [Background] Whisper 'turbo' model loaded successfully into memory.")
        except Exception as e:
            print(f"❌ [Background] Failed to preload Whisper model: {e}")
        finally:
            with cls._lock:
                cls._is_loading = False

    @classmethod
    def get_model(cls) -> whisper.Whisper:
        # Why return a cached singleton?
        # Re-loading the Whisper model on every transaction is highly inefficient and creates 
        # long delays between VOD downloads and transcriptions. Caching the model instance in memory 
        # reduces load time to 0ms for subsequent analyses.
        with cls._lock:
            if cls._cached_model is not None:
                return cls._cached_model
        
        # Synchronous fallback if preload hasn't finished or wasn't triggered
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading Whisper model 'turbo' on device: {device} (synchronous fallback)...")
        model = whisper.load_model("turbo", device=device)
        with cls._lock:
            cls._cached_model = model
        return model

    def transcribe(self, audio_path: str, model_name: str = "turbo") -> list[dict]:
        # Why ignore model_name argument and use cached turbo model?
        # Standardizing on the 'turbo' model ensures a consistent, high-performance experience 
        # and allows us to safely leverage the cached memory singleton.
        model = self.get_model()
        
        # Why language="ja" and condition_on_previous_text=False?
        # Specifying the language prevents the model from misidentifying Japanese speech 
        # as English during silent pauses, which improves initial timestamp accuracy.
        # Setting condition_on_previous_text=False disables context sharing between 30-second windows.
        # This completely breaks context-loop hallucinations (e.g. repeated "ご視聴ありがとうございました" 
        # or "チャンネル登録お願いします") commonly occurring in silent intervals, preventing speech omission.
        result = model.transcribe(audio_path, language="ja", condition_on_previous_text=False)
        
        segments = []
        for seg in result.get('segments', []):
            segments.append({
                "start": seg.get('start'),
                "end": seg.get('end'),
                "text": seg.get('text')
            })
            
        return segments
