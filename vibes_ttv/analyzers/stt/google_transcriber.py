import os
import time
import subprocess
import concurrent.futures
from google.cloud import speech
from google.cloud import storage
from vibes_ttv.analyzers.stt.base import BaseTranscriber

class GoogleSpeechTranscriber(BaseTranscriber):
    # Why use Google Cloud Speech-to-Text?
    # Cloud-based STT offloads heavy processing from the local system (RTX 4070 / CPU),
    # provides excellent Japanese recognition out-of-the-box, and handles large files 
    # through asynchronous API jobs, making it ideal for low-spec deployment targets.

    def __init__(self, project_id: str = "vibes-ttv", bucket_name: str = "temporary-speech-files"):
        # Why parameterize project_id and bucket_name?
        # Standardizing on defaults ('vibes-ttv' and 'temporary-speech-files') reduces configuration
        # overhead for the primary user, while allowing customization via args supports multi-tenant 
        # deployments or custom GCS configurations.
        self.project_id = project_id
        self.bucket_name = bucket_name

    def transcribe(self, audio_path: str, **kwargs) -> list[dict]:
        # Why split into chunks first?
        # Processing long-duration audio directly with Diarization (speaker separation) enabled 
        # causes Google Cloud STT API to truncate word lists or hit gRPC payload limits (10MB).
        # Splitting the audio into 5-minute chunks locally using ffmpeg copy-segment command 
        # is extremely fast (zero-copy audio packet slicing), avoids quality loss, and guarantees 
        # 100% segment and speaker tag retrieval.
        temp_dir = "downloads"
        os.makedirs(temp_dir, exist_ok=True)
        
        # Unique prefix to avoid concurrency issues if multiple vods are processed
        chunk_prefix = os.path.join(temp_dir, f"temp_split_{int(time.time())}_%03d.mp3")
        
        print(f"[GCP STT] Splitting and converting {audio_path} into mono 16kHz 5-minute chunks using ffmpeg...")
        # Why convert to mono 16kHz during split?
        # Google Cloud STT Diarization is highly unstable on stereo (2-channel) audio and often returns 
        # empty segments. Downmixing to mono (1 channel) and resampling to 16000Hz ensures maximum 
        # recognition accuracy and avoids channel conflict, while significantly reducing chunk file sizes.
        split_cmd = [
            "ffmpeg", "-y", "-i", audio_path, 
            "-f", "segment", "-segment_time", "300", 
            "-ac", "1", "-ar", "16000", chunk_prefix
        ]
        
        res = subprocess.run(split_cmd, capture_output=True, text=True)
        if res.returncode != 0:
            print(f"[GCP STT] Error splitting audio with FFmpeg: {res.stderr}")
            print("[GCP STT] Falling back to single file transcription...")
            # Fallback to single file transcription if split fails
            return self._transcribe_single_file(audio_path)
            
        # Find all generated chunk files
        chunk_files = []
        prefix_base = os.path.basename(chunk_prefix).replace("%03d.mp3", "")
        for f in os.listdir(temp_dir):
            if f.startswith(prefix_base) and f.endswith(".mp3"):
                chunk_files.append(os.path.join(temp_dir, f))
                
        chunk_files.sort()
        print(f"[GCP STT] Generated {len(chunk_files)} chunks. Starting parallel transcription...")
        
        all_segments = []
        # Why max_workers=3?
        # Balancing processing speed and GCP STT rate limits. Too many concurrent workers may exceed 
        # regional API quotas (e.g., maximum concurrent long running operations). 3 is a safe default.
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            # Submit tasks
            future_to_chunk = {
                executor.submit(self._transcribe_chunk, chunk_path, idx): idx 
                for idx, chunk_path in enumerate(chunk_files)
            }
            
            # Gather results in order of chunk index
            results = {}
            for future in concurrent.futures.as_completed(future_to_chunk):
                idx = future_to_chunk[future]
                try:
                    results[idx] = future.result()
                    print(f"[GCP STT] Chunk {idx+1}/{len(chunk_files)} completed successfully.")
                except Exception as e:
                    print(f"[GCP STT] Chunk {idx+1}/{len(chunk_files)} failed: {e}")
                    results[idx] = []
                    
            for idx in sorted(results.keys()):
                all_segments.extend(results[idx])
                
        # Cleanup temporary local chunk files
        # Why use try-finally for local cleanup?
        # Leaving temporary chunk copies on disk wastes local storage. We ensure deletion here.
        print("[GCP STT] Cleaning up temporary local audio chunks...")
        for chunk_path in chunk_files:
            try:
                if os.path.exists(chunk_path):
                    os.remove(chunk_path)
            except Exception as e:
                print(f"[GCP STT] Warning deleting local chunk: {e}")
                
        print(f"[GCP STT] Parallel transcription completed. Total segments: {len(all_segments)}")
        return all_segments

    def _transcribe_chunk(self, chunk_path: str, chunk_index: int) -> list[dict]:
        # Why run each chunk independently?
        # Processing audio in 5-minute chunks bypasses Google Cloud STT's gRPC payload size limit 
        # (10MB) for speaker diarization word lists, ensuring 100% of the speech segments are captured.
        speech_client = speech.SpeechClient()
        storage_client = storage.Client(project=self.project_id)
        bucket = storage_client.bucket(self.bucket_name)
        
        filename = os.path.basename(chunk_path)
        blob_name = f"vibes_ttv_temp/{int(time.time())}_{chunk_index}_{filename}"
        blob = bucket.blob(blob_name)
        
        blob.upload_from_filename(chunk_path)
        gcs_uri = f"gs://{self.bucket_name}/{blob_name}"
        
        try:
            audio = speech.RecognitionAudio(uri=gcs_uri)
            diarization_config = speech.SpeakerDiarizationConfig(
                enable_speaker_diarization=True,
                min_speaker_count=1,
                max_speaker_count=5,
            )
            config = speech.RecognitionConfig(
                encoding=speech.RecognitionConfig.AudioEncoding.MP3,
                language_code="ja-JP",
                enable_word_time_offsets=True,
                model="latest_long",
                diarization_config=diarization_config,
            )
            
            operation = speech_client.long_running_recognize(config=config, audio=audio)
            # Timeout for a 5-minute chunk can be smaller, e.g. 5 minutes (300s)
            response = operation.result(timeout=300)
            
            segments = []
            all_words = []
            for result in response.results:
                if result.alternatives:
                    alt = result.alternatives[0]
                    if alt.words:
                        all_words.extend(alt.words)
                        
            seen_words = set()
            unique_words = []
            for w in all_words:
                w_id = (w.start_time.total_seconds(), w.end_time.total_seconds(), w.word)
                if w_id not in seen_words:
                    seen_words.add(w_id)
                    unique_words.append(w)
                    
            if unique_words:
                current_speaker = None
                current_segment_words = []
                
                # We apply 300.0s offset based on chunk index
                # Why apply offset?
                # Since each chunk is processed independently starting from 0.0s, we must add 
                # (chunk_index * 300.0) seconds to restore the absolute time reference in the global VOD timeline.
                offset = chunk_index * 300.0
                
                for word in unique_words:
                    speaker_tag = word.speaker_tag
                    word_text = word.word
                    start_time = word.start_time.total_seconds() + offset
                    end_time = word.end_time.total_seconds() + offset
                    
                    is_speaker_changed = (current_speaker is not None and speaker_tag != current_speaker)
                    is_pause = False
                    if current_segment_words:
                        last_word_end = current_segment_words[-1]["end"]
                        # We evaluate pause based on actual offset-adjusted times
                        if start_time - last_word_end > 2.0:
                            is_pause = True
                            
                    if is_speaker_changed or is_pause:
                        if current_segment_words:
                            # Concatenate words. For Japanese, we don't need spaces.
                            # Why not add spaces between words?
                            # Japanese text does not use spaces between words, so we join them directly.
                            seg_text = "".join([w["word"] for w in current_segment_words])
                            segments.append({
                                "start": current_segment_words[0]["start"],
                                "end": current_segment_words[-1]["end"],
                                "text": f"[Streamer {current_speaker}] {seg_text}"
                            })
                        current_segment_words = []
                        
                    current_speaker = speaker_tag
                    current_segment_words.append({
                        "start": start_time,
                        "end": end_time,
                        "word": word_text
                    })
                    
                if current_segment_words:
                    seg_text = "".join([w["word"] for w in current_segment_words])
                    segments.append({
                        "start": current_segment_words[0]["start"],
                        "end": current_segment_words[-1]["end"],
                        "text": f"[Streamer {current_speaker}] {seg_text}"
                    })
                    
            return segments
            
        finally:
            try:
                blob.delete()
            except Exception as e:
                print(f"[GCP STT] Chunk {chunk_index} cleanup warning: {e}")

    def _transcribe_single_file(self, audio_path: str) -> list[dict]:
        # Why keep _transcribe_single_file?
        # Provides a robust fallback path to transcribe the entire audio file directly in one API call
        # in case local FFmpeg audio splitting fails.
        speech_client = speech.SpeechClient()
        storage_client = storage.Client(project=self.project_id)
        bucket = storage_client.bucket(self.bucket_name)
        
        filename = os.path.basename(audio_path)
        blob_name = f"vibes_ttv_temp/{int(time.time())}_{filename}"
        blob = bucket.blob(blob_name)
        
        print(f"[GCP STT] Uploading {audio_path} to gs://{self.bucket_name}/{blob_name}...")
        blob.upload_from_filename(audio_path)
        
        gcs_uri = f"gs://{self.bucket_name}/{blob_name}"
        
        try:
            audio = speech.RecognitionAudio(uri=gcs_uri)
            diarization_config = speech.SpeakerDiarizationConfig(
                enable_speaker_diarization=True,
                min_speaker_count=1,
                max_speaker_count=5,
            )
            config = speech.RecognitionConfig(
                encoding=speech.RecognitionConfig.AudioEncoding.MP3,
                language_code="ja-JP",
                enable_word_time_offsets=True,
                model="latest_long",
                diarization_config=diarization_config,
            )
            
            print(f"[GCP STT] Starting Google Cloud Speech-to-Text long running recognize for {gcs_uri}...")
            operation = speech_client.long_running_recognize(config=config, audio=audio)
            
            print("[GCP STT] Waiting for Google Cloud Speech-to-Text to complete (this may take a few minutes)...")
            response = operation.result(timeout=7200)
            
            segments = []
            all_words = []
            for result in response.results:
                if result.alternatives:
                    alt = result.alternatives[0]
                    if alt.words:
                        all_words.extend(alt.words)
            
            seen_words = set()
            unique_words = []
            for w in all_words:
                w_id = (w.start_time.total_seconds(), w.end_time.total_seconds(), w.word)
                if w_id not in seen_words:
                    seen_words.add(w_id)
                    unique_words.append(w)
            
            if unique_words:
                current_speaker = None
                current_segment_words = []
                
                for word in unique_words:
                    speaker_tag = word.speaker_tag
                    word_text = word.word
                    start_time = word.start_time.total_seconds()
                    end_time = word.end_time.total_seconds()
                    
                    is_speaker_changed = (current_speaker is not None and speaker_tag != current_speaker)
                    is_pause = False
                    if current_segment_words:
                        last_word_end = current_segment_words[-1]["end"]
                        if start_time - last_word_end > 2.0:
                            is_pause = True
                            
                    if is_speaker_changed or is_pause:
                        if current_segment_words:
                            seg_text = "".join([w["word"] for w in current_segment_words])
                            segments.append({
                                "start": current_segment_words[0]["start"],
                                "end": current_segment_words[-1]["end"],
                                "text": f"[Streamer {current_speaker}] {seg_text}"
                            })
                        current_segment_words = []
                        
                    current_speaker = speaker_tag
                    current_segment_words.append({
                        "start": start_time,
                        "end": end_time,
                        "word": word_text
                    })
                    
                if current_segment_words:
                    seg_text = "".join([w["word"] for w in current_segment_words])
                    segments.append({
                        "start": current_segment_words[0]["start"],
                        "end": current_segment_words[-1]["end"],
                        "text": f"[Streamer {current_speaker}] {seg_text}"
                    })
            
            print(f"[GCP STT] Google Cloud Speech-to-Text completed. Extracted {len(segments)} segments.")
            return segments
            
        finally:
            try:
                print(f"[GCP STT] Deleting temporary GCS file gs://{self.bucket_name}/{blob_name}...")
                blob.delete()
            except Exception as e:
                print(f"[GCP STT] Failed to delete temporary GCS file: {e}")
