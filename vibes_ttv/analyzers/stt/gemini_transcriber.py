import os
import time
import subprocess
import concurrent.futures
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from typing import List
from vibes_ttv.analyzers.stt.base import BaseTranscriber

# Define structured output schemas using Pydantic
class SpeechSegment(BaseModel):
    start: float = Field(description="The start time of the speech segment in seconds relative to the chunk start.")
    end: float = Field(description="The end time of the speech segment in seconds relative to the chunk start.")
    text: str = Field(description="The transcribed text. Must start with speaker prefix, e.g. '[Streamer 0] こんにちは' or '[Streamer 1] そうですね'. Max speaker index should be 4 (representing 5 speakers).")

class TranscriptionResult(BaseModel):
    segments: List[SpeechSegment] = Field(description="List of all transcribed speech segments in chronological order.")

class GeminiSpeechTranscriber(BaseTranscriber):
    # Why use Gemini 3.5 Flash with Sliding Window?
    # While Gemini handles large files natively, its output token limit (8,192 tokens) cuts off transcripts 
    # for long audio (e.g. at 39 minutes for a 96-minute vod).
    # On the other hand, simple chunking creates context-cuts at chunk boundaries.
    # The sliding window approach (e.g. 15-minute segments with a 3-minute overlap) uploaded individually:
    # 1. Bypasses output token limits completely.
    # 2. Avoids context cutting by letting the model see the surrounding 3 minutes of context.
    # 3. Keeps tasks stateless, preventing self-imitation loops common in multi-turn Chat histories.
    # 4. Allows a mathematical deduplication step to reconstruct a single, continuous, non-overlapping transcript.

    def __init__(self, model_name: str = "gemini-3.1-flash-lite", api_key: str = None):
        self.model_name = model_name
        self.api_key = api_key

    def _get_audio_duration(self, audio_path: str) -> float:
        # Why call ffprobe to query duration?
        # Dynamically computing the sliding intervals requires knowing the exact duration of the source file.
        # Calling ffprobe is fast and precise.
        cmd = [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nocut=1", audio_path
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode == 0:
            try:
                return float(res.stdout.strip())
            except ValueError:
                pass
        return 5760.0 # Fallback default (96 minutes) if ffprobe fails

    def transcribe(self, audio_path: str, **kwargs) -> list[dict]:
        duration = self._get_audio_duration(audio_path)
        
        window_size = 240.0  # 4 minutes
        overlap = 60.0      # 1 minute
        
        # Calculate sliding window intervals
        intervals = []
        start = 0.0
        while start < duration:
            end = min(start + window_size, duration)
            intervals.append((start, end))
            if end >= duration:
                break
            start = end - overlap
            
        print(f"[Gemini STT] Audio Duration: {duration:.2f}s. Generated {len(intervals)} sliding windows.")
        
        temp_dir = "downloads"
        os.makedirs(temp_dir, exist_ok=True)
        
        # Why re-encode to stereo MP3 instead of copy?
        # Using copy stream slicing can lead to inaccurate timestamp alignments (slicing at keyframe boundaries).
        # Re-encoding the short 15-minute slices to high quality stereo MP3 ensures exact start-to-end alignment 
        # while keeping file size small (~15MB per slice) and preserving spatial audio separation.
        chunk_files = []
        for idx, (start_sec, end_sec) in enumerate(intervals):
            slice_path = os.path.join(temp_dir, f"temp_gemini_window_{int(time.time())}_{idx:03d}.mp3")
            print(f"[Gemini STT] Slicing chunk {idx+1}/{len(intervals)} [{start_sec:.1f}s - {end_sec:.1f}s]...")
            slice_cmd = [
                "ffmpeg", "-y", "-ss", str(start_sec), "-to", str(end_sec), "-i", audio_path,
                "-c:a", "libmp3lame", "-q:a", "2", slice_path
            ]
            subprocess.run(slice_cmd, capture_output=True)
            chunk_files.append(slice_path)
            
        print(f"[Gemini STT] Slicing completed. Starting parallel transcription of {len(chunk_files)} windows...")
        
        all_results = {}
        # Why use max_workers=16?
        # Increased to 16 as a reliable multiple of 8. It strikes a great balance 
        # between faster upload concurrency and keeping under the default 4M TPM API quota.
        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
            future_to_chunk = {
                executor.submit(self._transcribe_chunk, chunk_path, idx): idx 
                for idx, chunk_path in enumerate(chunk_files)
            }
            
            for future in concurrent.futures.as_completed(future_to_chunk):
                idx = future_to_chunk[future]
                try:
                    all_results[idx] = future.result()
                    print(f"[Gemini STT] Window {idx+1}/{len(intervals)} completed successfully.")
                except Exception as e:
                    print(f"[Gemini STT] Window {idx+1}/{len(intervals)} failed: {e}")
                    all_results[idx] = []
                    
        # Cleanup temporary local sliced audio files
        print("[Gemini STT] Cleaning up temporary local window files...")
        for chunk_path in chunk_files:
            try:
                if os.path.exists(chunk_path):
                    os.remove(chunk_path)
            except Exception as e:
                print(f"[Gemini STT] Warning deleting local window: {e}")
                
        # Merge and Deduplicate results using overlapping midpoints
        # Why use deduplication ranges?
        # Overlapping slices transcribe the same speech multiple times. 
        # Defining non-overlapping acceptance ranges [limit_start, limit_end) for each chunk 
        # based on mid-overlap points ensures that each spoken sentence is captured exactly once, 
        # utilizing the context of surrounding audio without duplicate printouts.
        merged_segments = []
        for idx, (start_sec, end_sec) in enumerate(intervals):
            chunk_segments = all_results.get(idx, [])
            
            # Determine acceptance boundaries relative to global audio timeline
            if idx == 0:
                limit_start = 0.0
                limit_end = end_sec - (overlap / 2.0)
            elif idx == len(intervals) - 1:
                limit_start = start_sec + (overlap / 2.0)
                limit_end = float('inf')
            else:
                limit_start = start_sec + (overlap / 2.0)
                limit_end = end_sec - (overlap / 2.0)
                
            for seg in chunk_segments:
                # Convert relative chunk time to global timeline time
                global_start = seg["start"] + start_sec
                global_end = seg["end"] + start_sec
                
                # Check if the segment start time falls within this chunk's designated range
                if limit_start <= global_start < limit_end:
                    merged_segments.append({
                        "start": global_start,
                        "end": global_end,
                        "text": seg["text"]
                    })
                    
        merged_segments.sort(key=lambda x: x["start"])
        print(f"[Gemini STT] Sliding window transcription completed. Total deduplicated segments: {len(merged_segments)}")
        return merged_segments

    def _transcribe_chunk(self, chunk_path: str, chunk_index: int) -> list[dict]:
        # Why pass self.api_key?
        # If the user enters a specific API key in the UI, we should utilize it rather than 
        # falling back to environmental variables, allowing seamless runtime key switching.
        client = genai.Client(api_key=self.api_key) if self.api_key else genai.Client()
        
        print(f"[Gemini STT] Uploading window {chunk_index+1} ({os.path.basename(chunk_path)})...")
        uploaded_file = client.files.upload(file=chunk_path)
        
        while uploaded_file.state.name == "PROCESSING":
            time.sleep(2)
            uploaded_file = client.files.get(name=uploaded_file.name)
            
        if uploaded_file.state.name == "FAILED":
            raise RuntimeError(f"File upload failed for window {chunk_index}")

        try:
            system_instruction = (
                "あなたは高性能な音声文字起こしおよび話者分離（Diarization）のアシスタントです。\n"
                "与えられた音声ファイルに含まれるすべての発話を正確に文字起こししてください。\n"
                "音声は配信アーカイブの一部を切り出したものです。背景にゲーム音、効果音、またはBGMが流れています。\n"
                "また、チャットを読み上げるボイスボット（ゆっくりボイスなど）の音声が入ることもあります。\n"
                "読み上げBotの声も含め、配信者およびコラボ相手などすべての発話を別々の話者として聴き分けて書き起こし、話者タグを付与してください。\n"
                "各セグメントのテキストは、必ず『[Streamer 0] 』、『[Streamer 1] 』などのプレフィックスから開始してください。\n"
                "（配信の主担当者を[Streamer 0]、読み上げBotやコラボ相手などを[Streamer 1]〜[Streamer 4]として区別して判定してください）\n"
                "音声の無音部分や、ゲーム音・ノイズのみの区間は無視し、文字起こしを出力しないでください。"
            )
            
            prompt = (
                "この音声ファイルを文字起こししてください。\n"
                "時間（startとend）は、与えられた切り出し音声ファイルの開始（0.0秒）からの相対時間（実数）で正確に取得してください。\n"
                "結果は定義されたスキーマに厳密に沿って出力してください。"
            )
            
            config = types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=TranscriptionResult,
                system_instruction=system_instruction,
                temperature=0.0
            )
            
            response = client.models.generate_content(
                model=self.model_name,
                contents=[uploaded_file, prompt],
                config=config
            )
            
            # Why handle parsing with a regex fallback?
            # If the response.parsed is None, it means validation failed (e.g. truncated JSON due to token limits, 
            # or malformed empty structures). Falling back to raw text regex parsing allows us to salvage 
            # as many transcribed segments as possible instead of discarding the entire window.
            try:
                if response.parsed is not None:
                    result: TranscriptionResult = response.parsed
                    segments = []
                    for seg in result.segments:
                        segments.append({
                            "start": seg.start,
                            "end": seg.end,
                            "text": seg.text
                        })
                    return segments
                else:
                    print(f"[Gemini STT] Warning: Window {chunk_index+1} parsed as None. Trying regex fallback...")
                    if response.text:
                        return self._parse_fallback_json(response.text)
                    return []
            except Exception as parse_err:
                print(f"[Gemini STT] Warning: Window {chunk_index+1} Pydantic parsing threw exception: {parse_err}. Trying regex fallback...")
                if response.text:
                    try:
                        return self._parse_fallback_json(response.text)
                    except Exception as fallback_err:
                        print(f"[Gemini STT] Error: Regex fallback failed: {fallback_err}")
                return []
            
        finally:
            try:
                client.files.delete(name=uploaded_file.name)
            except Exception as e:
                print(f"[Gemini STT] Window {chunk_index} remote cleanup warning: {e}")

    def _parse_fallback_json(self, raw_text: str) -> list[dict]:
        # Why write a custom regex-based JSON segment parser?
        # If Gemini's response is cut off at the token limit, the JSON will be truncated and standard json.loads fails.
        # By scanning for valid dictionary-like segment patterns using regex, we can salvage all segments 
        # generated before the cut-off point, ensuring we don't lose up to 15 minutes of transcription data.
        import re
        import json
        segments = []
        # Find all string blocks enclosed in curly braces
        raw_blocks = re.findall(r'\{[^{}]*\}', raw_text)
        for block in raw_blocks:
            try:
                obj = json.loads(block)
                if "start" in obj and "end" in obj and "text" in obj:
                    segments.append({
                        "start": float(obj["start"]),
                        "end": float(obj["end"]),
                        "text": str(obj["text"])
                    })
            except Exception:
                # If json.loads fails, extract individual fields using regex
                start_match = re.search(r'"start"\s*:\s*([0-9.]+)', block)
                end_match = re.search(r'"end"\s*:\s*([0-9.]+)', block)
                text_match = re.search(r'"text"\s*:\s*"([^"\\]*(?:\\.[^"\\]*)*)"', block)
                if start_match and end_match and text_match:
                    try:
                        text_val = text_match.group(1)
                        # Decode escape characters if any
                        text_val = json.loads(f'"{text_val}"')
                        segments.append({
                            "start": float(start_match.group(1)),
                            "end": float(end_match.group(1)),
                            "text": text_val
                        })
                    except Exception:
                        pass
        return segments
