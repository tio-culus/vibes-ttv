import pytest
from unittest.mock import MagicMock, patch
from vibes_ttv.analyzers.stt.base import BaseTranscriber
from vibes_ttv.analyzers.stt.factory import get_transcriber, TRANSCRIBER_REGISTRY
from vibes_ttv.analyzers.stt.whisper_transcriber import WhisperTranscriber
from vibes_ttv.analyzers.stt.google_transcriber import GoogleSpeechTranscriber

def test_stt_factory():
    # Why test factory lookup?
    # Verifying that get_transcriber correctly instantiates the registered transcribers
    # and throws a ValueError for unsupported engines keeps the backend registry robust.
    whisper_t = get_transcriber("whisper")
    assert isinstance(whisper_t, WhisperTranscriber)

    google_t = get_transcriber("google_stt", project_id="test-proj", bucket_name="test-bucket")
    assert isinstance(google_t, GoogleSpeechTranscriber)
    assert google_t.project_id == "test-proj"
    assert google_t.bucket_name == "test-bucket"

    with pytest.raises(ValueError):
        get_transcriber("invalid_engine")

@patch("vibes_ttv.analyzers.stt.google_transcriber.storage.Client")
@patch("vibes_ttv.analyzers.stt.google_transcriber.speech.SpeechClient")
def test_google_stt_transcription(mock_speech_client_cls, mock_storage_client_cls):
    # Why mock GCP services?
    # Testing Cloud STT without sending requests to live GCP endpoints prevents auth errors,
    # network dependency during testing, and GCS cost overhead, while verifying 
    # the entire GCS upload, speech API invocation, and result parsing logic.
    mock_speech_client = MagicMock()
    mock_speech_client_cls.return_value = mock_speech_client
    
    mock_storage_client = MagicMock()
    mock_storage_client_cls.return_value = mock_storage_client
    
    # Mock GCS blob upload
    mock_bucket = MagicMock()
    mock_storage_client.bucket.return_value = mock_bucket
    mock_blob = MagicMock()
    mock_bucket.blob.return_value = mock_blob
    
    # Mock Speech API long running operation
    mock_operation = MagicMock()
    mock_speech_client.long_running_recognize.return_value = mock_operation
    
    # Mock Speech API response
    mock_response = MagicMock()
    mock_operation.result.return_value = mock_response
    
    # Mock results
    mock_result_final = MagicMock()
    mock_alternative = MagicMock()
    
    mock_word_1 = MagicMock()
    mock_word_1.speaker_tag = 1
    mock_word_1.word = "こんにちは"
    mock_word_1.start_time.total_seconds.return_value = 1.0
    mock_word_1.end_time.total_seconds.return_value = 3.0
    
    mock_word_2 = MagicMock()
    mock_word_2.speaker_tag = 2
    mock_word_2.word = "テストです"
    mock_word_2.start_time.total_seconds.return_value = 4.0
    mock_word_2.end_time.total_seconds.return_value = 6.0
    
    mock_word_3 = MagicMock()
    mock_word_3.speaker_tag = 2
    mock_word_3.word = "よろしく"
    mock_word_3.start_time.total_seconds.return_value = 7.0
    mock_word_3.end_time.total_seconds.return_value = 9.0
    
    # Word 4 has a long pause (> 2.0s) from word 3
    mock_word_4 = MagicMock()
    mock_word_4.speaker_tag = 2
    mock_word_4.word = "お願いします"
    mock_word_4.start_time.total_seconds.return_value = 12.0
    mock_word_4.end_time.total_seconds.return_value = 14.0
    
    mock_alternative.words = [mock_word_1, mock_word_2, mock_word_3, mock_word_4]
    mock_result_final.alternatives = [mock_alternative]
    
    mock_response.results = [mock_result_final]
    
    # Instantiate transcriber and transcribe dummy path
    transcriber = GoogleSpeechTranscriber(project_id="test-proj", bucket_name="test-bucket")
    segments = transcriber.transcribe("dummy_audio.mp3")
    
    # Verify calls
    mock_storage_client.bucket.assert_called_once_with("test-bucket")
    mock_blob.upload_from_filename.assert_called_once_with("dummy_audio.mp3")
    mock_speech_client.long_running_recognize.assert_called_once()
    mock_blob.delete.assert_called_once()  # Assert GCS file cleanup was triggered
    
    # Verify parsed segments
    assert len(segments) == 3
    assert segments[0] == {"start": 1.0, "end": 3.0, "text": "[Streamer 1] こんにちは"}
    assert segments[1] == {"start": 4.0, "end": 9.0, "text": "[Streamer 2] テストですよろしく"}
    assert segments[2] == {"start": 12.0, "end": 14.0, "text": "[Streamer 2] お願いします"}
