import whisper
import torch

class WhisperTranscriber:
    # Why not faster-whisper?
    # While faster-whisper is highly optimized, it requires specific C++ libraries (ctranslate2) 
    # which can be complex to compile or configure on Windows. 
    # Standard openai-whisper is more robust, installs out-of-the-box via pip, 
    # and RTX 4070 has more than enough power to run it at high speeds.
    
    def transcribe(self, audio_path: str, model_name: str = "small") -> list[dict]:
        # Why check for CUDA availability dynamically?
        # A hardcoded "cuda" device might crash the application if PyTorch CUDA dependencies 
        # are misconfigured on a target host. Falling back to CPU ensures robustness.
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading Whisper model '{model_name}' on device: {device} (CUDA is_available={torch.cuda.is_available()})")
        
        # Why reload the model every transcribe session?
        # Keeping models in VRAM permanently might crash other GPU-intensive tasks.
        # However, for this dashboard, we load it dynamically. In the future, we could cache the model reference.
        model = whisper.load_model(model_name, device=device)
        
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
