import sys
import os

# Why append parent directory to sys.path programmatically?
# When executing 'streamlit run vibes_ttv/app.py', Streamlit resolves paths relative 
# to the script's directory, which breaks top-level package imports of 'vibes_ttv'.
# Programmatically adding the project root resolves this without requiring the user 
# to manually configure the PYTHONPATH environment variable on Windows.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import re
from datetime import datetime

# Import project modules
from vibes_ttv.database.db_manager import DBManager
from vibes_ttv.database.models import VOD, Streamer, Topic, VODListenerStats
from vibes_ttv.collectors.chat_collector import ChatCollector
from vibes_ttv.collectors.audio_collector import AudioCollector
from vibes_ttv.analyzers.whisper_transcriber import WhisperTranscriber
from vibes_ttv.analyzers.timeline_merger import TimelineMerger
from vibes_ttv.analyzers.comment_analyzer import CommentAnalyzer
from vibes_ttv.analyzers.topic_analyzer import TopicAnalyzer

# ---------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------
def format_seconds(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"

def format_twitch_offset(seconds: int) -> str:
    # Why format as XhYmZs instead of raw seconds?
    # Twitch's query parameters natively parsing jump times requires time segments (e.g. ?t=1h2m3s).
    # Sending raw seconds (?t=300) is deprecated on Twitch VOD player and fails on mobile web views,
    # so standard segmented format guarantees reliable playback navigation.
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    parts = []
    if h > 0:
        parts.append(f"{h}h")
    if m > 0 or h > 0:
        parts.append(f"{m}m")
    parts.append(f"{s}s")
    return "".join(parts)

def calculate_chat_velocities(chat_data: list[dict], duration_seconds: int) -> tuple[float, int, str]:
    import json
    if not chat_data:
        return 0.0, 0, "[]"
    df = pd.DataFrame(chat_data)
    total_chats = len(df)
    hours = max(duration_seconds / 3600.0, 0.01)
    avg_velocity_hour = total_chats / hours
    
    df['minute_bin'] = (df['offset_seconds'] // 60).astype(int)
    chats_per_minute = df.groupby('minute_bin').size()
    max_velocity_min = int(chats_per_minute.max()) if not chats_per_minute.empty else 0
    
    # Why fill zero for missing minutes?
    # Skipping minutes without chats would make the line chart discontinuous 
    # and skew the timeline representation. Generating a complete range of minutes 
    # filled with zeros preserves temporal fidelity.
    duration_minutes = int(duration_seconds // 60)
    max_minute = duration_minutes
    if not chats_per_minute.empty:
        max_minute = max(max_minute, int(chats_per_minute.index.max()))
        
    velocity_list = []
    for m in range(max_minute + 1):
        count = int(chats_per_minute.get(m, 0))
        velocity_list.append({"minute": m, "count": count})
        
    velocity_json = json.dumps(velocity_list)
    return avg_velocity_hour, max_velocity_min, velocity_json

def extract_vod_id(url: str) -> str:
    # Why use regex for VOD ID extraction?
    # Twitch VOD URLs consistently contain the numeric ID after '/videos/'.
    # Extracting this locally avoids hitting the network or calling external packages.
    match = re.search(r"/videos/(\d+)", url)
    if match:
        return match.group(1)
    # Fallback to alphanumeric cleaning if format differs slightly
    return re.sub(r'\W+', '', url.split('/')[-1])

import threading

class AnalysisRunner:
    # Why use a thread-safe AnalysisRunner?
    # Streamlit execution is stateless and synchronous, which makes direct UI button interaction 
    # during long processes impossible. Running the analysis pipeline in a background thread 
    # and tracking progress metrics inside a thread-safe runner object allows the UI to poll 
    # status and render action buttons (pause, resume, stop) reliably.
    def __init__(self, db, vod_url: str, api_key: str, model_name: str, batch_size: int):
        self.db = db
        self.vod_url = vod_url
        self.api_key = api_key
        self.model_name = model_name
        self.batch_size = batch_size
        
        self.progress_val = 0
        self.message = "初期化中..."
        self.is_pausable = True
        self.is_done = False
        self.error = None
        self.vod_id = None
        
        # Why track start_time in the runner?
        # Storing start_time allows the main thread (UI loop) to calculate elapsed time dynamically 
        # during st.rerun() polling, preventing the timer from freezing when the background thread 
        # is executing blocking operations (like Whisper) and not pushing callback updates.
        import time
        self.start_time = time.time()
        
        self.chat_collection_time = 0
        self.extraction_time = 0
        self.transcription_time = 0
        self.ai_analysis_time = 0
        self.total_analysis_time = 0
        
        self.pause_event = threading.Event()
        self.pause_event.set()  # True means "Running"
        self.stop_event = threading.Event()
        
        self._thread = None
        
    def start(self):
        self._thread = threading.Thread(target=self._run)
        self._thread.daemon = True
        self._thread.start()
        
    def _run(self):
        try:
            self.vod_id = run_real_analysis_thread(self)
            self.is_done = True
        except Exception as e:
            self.error = str(e)
            self.is_done = True
            
    def check_pause(self):
        # Why raise custom Exception on stop?
        # Raising an exception immediately halts the execution of loops in submodules 
        # (like chat collection or Gemini batching) and ensures that execution is stopped safely.
        if self.stop_event.is_set():
            raise Exception("分析が中止されました。")
        while not self.pause_event.is_set():
            if self.stop_event.is_set():
                raise Exception("分析が中止されました。")
            import time
            time.sleep(0.5)
            
    def set_pausable(self, pausable: bool):
        self.is_pausable = pausable


def run_real_analysis_thread(runner: AnalysisRunner) -> str:
    db = runner.db
    vod_url = runner.vod_url
    api_key = runner.api_key
    model_name = runner.model_name
    batch_size = runner.batch_size
    
    import time
    start_time = time.time()
    
    t_chat_collection = 0
    t_extraction = 0
    t_transcription = 0
    t_ai_analysis = 0
    
    def progress_callback(message: str, progress_val: int):
        runner.check_pause()
        runner.message = message
        runner.progress_val = progress_val
        
    try:
        # Step 0: Get VOD metadata
        runner.set_pausable(True)
        progress_callback("🔍 [0/5] Twitch VOD メタデータを取得中...", 5)
        collector = ChatCollector()
        metadata = collector.get_video_metadata(vod_url)
        
        # Step 1: Collect chat logs
        runner.set_pausable(True)
        progress_callback("🤖 [1/5] Twitchチャットログを収集中...", 10)
        t_chat_start = time.time()
        chat_data = collector.collect_chat(vod_url, progress_callback=progress_callback)
        t_chat_collection = int(time.time() - t_chat_start)
        if not chat_data:
            raise Exception("チャットログの取得に失敗しました。URLが正しいか、VODが公開されているかご確認ください。")
            
        # Step 2: Download and extract audio
        # Why disable pause during download?
        # Audio download runs as a subprocess via yt-dlp. Stopping a subprocess in PyTorch/Python 
        # is unpredictable and prone to resource leaks. Disabling pause is safer.
        runner.set_pausable(False)
        progress_callback("🎵 [2/5] VODから音声トラックを抽出中 (yt-dlp)...", 30)
        t_extract_start = time.time()
        audio_coll = AudioCollector()
        audio_path = audio_coll.collect_audio(vod_url, progress_callback=progress_callback)
        t_extraction = int(time.time() - t_extract_start)
        vod_id = os.path.basename(audio_path).replace(".mp3", "")
        
        # Step 3: Transcription using Local Whisper
        # Why disable pause during Whisper?
        # Local model transcription runs highly optimized blocking PyTorch code. Forcing 
        # it to stop requires complex multi-processing hacks. Completing it without pause is more robust.
        runner.set_pausable(False)
        progress_callback(f"✍️ [3/5] Whisper ({model_name}) で音声を文字起こし中... (数分かかる場合があります)", 50)
        t_transcribe_start = time.time()
        transcriber = WhisperTranscriber()
        segments = transcriber.transcribe(audio_path, model_name=model_name)
        t_transcription = int(time.time() - t_transcribe_start)
        
        # Step 4: Merge chats and text
        runner.set_pausable(True)
        progress_callback("🔗 [4/5] チャットログと音声認識タイムスタンプをアラインメント中...", 75)
        merger = TimelineMerger()
        merged_events = merger.merge(segments, chat_data)
        timeline_txt = merger.format_to_text(merged_events)
        
        # Step 5: AI analysis using Gemini API
        runner.set_pausable(True)
        progress_callback("🧠 [5/5] Gemini API で話題のコンテキストを抽出中...", 80)
        t_ai_start = time.time()
        
        if metadata:
            streamer_id = metadata.get("streamer_id") or "twitch_streamer"
            streamer_name = metadata.get("streamer_name") or "Twitch Streamer"
            title = metadata.get("title") or f"Twitch配信アーカイブ (ID: {vod_id})"
            duration = metadata.get("duration_seconds") or int(merged_events[-1]["offset_seconds"]) if merged_events else 3600
            
            streamed_at = datetime.now()
            created_at_str = metadata.get("created_at")
            if created_at_str:
                try:
                    streamed_at = datetime.fromisoformat(created_at_str.replace("Z", "+00:00"))
                except Exception:
                    pass
        else:
            streamer_id = "twitch_streamer"
            streamer_name = "Twitch Streamer"
            title = f"Twitch配信アーカイブ (ID: {vod_id})"
            duration = int(merged_events[-1]["offset_seconds"]) if merged_events else 3600
            streamed_at = datetime.now()
            
        streamer = db.get_or_create_streamer(streamer_id, streamer_name)
        avg_vel, max_vel, vel_json = calculate_chat_velocities(chat_data, duration)
        
        vod = VOD(
            vod_id=vod_id,
            streamer_id=streamer.streamer_id,
            title=title,
            duration_seconds=duration,
            streamed_at=streamed_at,
            average_viewers=0,
            avg_chat_velocity_hour=avg_vel,
            max_chat_velocity_min=max_vel,
            merged_timeline_text=timeline_txt,
            chat_velocity_json=vel_json
        )
        
        topic_analyzer = TopicAnalyzer(api_key=api_key)
        topics_data = topic_analyzer.analyze_topics(timeline_txt)
        db_topics = [
            Topic(
                vod_id=vod_id,
                start_offset_seconds=t["start_offset_seconds"],
                end_offset_seconds=t["end_offset_seconds"],
                category=t["category"],
                description=t["description"],
                is_high_context=t["is_high_context"]
            )
            for t in topics_data
        ]
        
        comment_analyzer = CommentAnalyzer(api_key=api_key)
        listener_stats = comment_analyzer.analyze_listeners(chat_data, batch_size=batch_size, progress_callback=progress_callback)
        
        db_stats = []
        for s in listener_stats:
            import json
            # Why serialize comment_details to JSON string?
            # Storing the list of specific comments with their offsets and categories as a JSON string 
            # avoids table duplication while retaining the detail view for UI inspection.
            details_json = json.dumps(s.get("comment_details", []), ensure_ascii=False)
            db_stats.append(
                VODListenerStats(
                    vod_id=vod_id,
                    listener_username=s["username"],
                    total_comments=s["total_comments"],
                    reaction_comments_count=s["reaction_comments_count"],
                    question_comments_count=s["question_comments_count"],
                    insight_comments_count=s["insight_comments_count"],
                    instruction_comments_count=s["instruction_comments_count"],
                    other_comments_count=s["other_comments_count"],
                    persona_type=s["persona_type"],
                    comment_details_json=details_json
                )
            )
            
        t_ai_analysis = int(time.time() - t_ai_start)
        total_time = int(time.time() - start_time)
        
        # Database transaction for atomic replacement
        # Why run deletions and insertions in a single transaction block?
        # If the re-analysis fails halfway (e.g. VOD already deleted on Twitch), 
        # the legacy data remains completely intact and safe. We only commit changes 
        # when all new dataset objects are fully compiled.
        session_db = db.get_session()
        try:
            # Delete legacy associated tables to prevent duplicate records accumulation
            # Why delete both v-prefixed and raw numeric VOD IDs?
            # When re-analyzing, Twitch or external libs (like yt-dlp) might return a different ID format 
            # (e.g. raw numeric vs v-prefixed). Checking for both ensures that legacy orphan records 
            # are fully cleared, and only the newest analyzed VOD datasets are kept.
            target_ids = [vod_id]
            if vod_id.startswith('v'):
                target_ids.append(vod_id[1:])
            else:
                target_ids.append(f"v{vod_id}")
                
            for tid in target_ids:
                session_db.query(Topic).filter_by(vod_id=tid).delete()
                session_db.query(VODListenerStats).filter_by(vod_id=tid).delete()
                if tid != vod_id:
                    session_db.query(VOD).filter_by(vod_id=tid).delete()
            
            # Save or update VOD record
            vod.chat_collection_time_seconds = t_chat_collection
            vod.extraction_time_seconds = t_extraction
            vod.transcription_time_seconds = t_transcription
            vod.ai_analysis_time_seconds = t_ai_analysis
            vod.total_analysis_time_seconds = total_time
            session_db.merge(vod)
            
            # Save new topics
            for t in db_topics:
                session_db.add(t)
                
            # Save new listener stats
            for s in db_stats:
                session_db.merge(s)
                
            session_db.commit()
        except Exception as e:
            session_db.rollback()
            raise e
        finally:
            db.remove_session()
            
        if os.path.exists(audio_path):
            os.remove(audio_path)
        
        # Expose stats to the runner
        runner.chat_collection_time = t_chat_collection
        runner.extraction_time = t_extraction
        runner.transcription_time = t_transcription
        runner.ai_analysis_time = t_ai_analysis
        runner.total_analysis_time = total_time
        
        progress_callback("✨ 分析完了！", 100)
        return vod_id
        
    except Exception as e:
        try:
            if 'audio_path' in locals() and os.path.exists(audio_path):
                os.remove(audio_path)
        except Exception:
            pass
        raise e


# ---------------------------------------------------------
# Mock Data Generator (for Quick Validation)
# ---------------------------------------------------------


# ---------------------------------------------------------
# Real Pipeline Runner
# ---------------------------------------------------------
def run_real_analysis(db: DBManager, vod_url: str, api_key: str, model_name: str, batch_size: int = 30) -> str:
    status_text = st.empty()
    progress_bar = st.progress(0)
    
    # Why track start_time?
    # Knowing the elapsed time helps reassure the user that the pipeline is active 
    # even when processing heavy steps (like Whisper transcription).
    import time
    start_time = time.time()
    
    t_chat_collection = 0
    t_extraction = 0
    t_transcription = 0
    t_ai_analysis = 0
    
    def progress_callback(message: str, progress_val: int):
        elapsed = int(time.time() - start_time)
        status_text.text(f"⏱️ 経過時間: {elapsed}秒 | {message}")
        progress_bar.progress(progress_val)
        
    try:
        # Step 0: Get VOD metadata
        progress_callback("🔍 [0/5] Twitch VOD メタデータを取得中...", 5)
        collector = ChatCollector()
        metadata = collector.get_video_metadata(vod_url)
        
        # Step 1: Collect chat logs
        progress_callback("🤖 [1/5] Twitchチャットログを収集中...", 10)
        t_chat_start = time.time()
        chat_data = collector.collect_chat(vod_url, progress_callback=progress_callback)
        t_chat_collection = int(time.time() - t_chat_start)
        if not chat_data:
            st.error("チャットログの取得に失敗しました。URLが正しいか、VODが公開されているかご確認ください。")
            return None
            
        # Step 2: Download and extract audio
        progress_callback("🎵 [2/5] VODから音声トラックを抽出中 (yt-dlp)...", 30)
        t_extract_start = time.time()
        audio_coll = AudioCollector()
        audio_path = audio_coll.collect_audio(vod_url, progress_callback=progress_callback)
        t_extraction = int(time.time() - t_extract_start)
        # Extract VOD ID from audio output filename
        vod_id = os.path.basename(audio_path).replace(".mp3", "")
        
        # Step 3: Transcription using Local Whisper
        progress_callback(f"✍️ [3/5] Whisper ({model_name}) で音声を文字起こし中... (数分かかる場合があります)", 50)
        t_transcribe_start = time.time()
        transcriber = WhisperTranscriber()
        segments = transcriber.transcribe(audio_path, model_name=model_name)
        t_transcription = int(time.time() - t_transcribe_start)
        
        # Step 4: Merge chats and text
        progress_callback("🔗 [4/5] チャットログと音声認識タイムスタンプをアラインメント中...", 75)
        merger = TimelineMerger()
        merged_events = merger.merge(segments, chat_data)
        timeline_txt = merger.format_to_text(merged_events)
        
        # Step 5: AI analysis using Gemini API
        progress_callback("🧠 [5/5] Gemini API で話題のコンテキストを抽出中...", 80)
        t_ai_start = time.time()
        
        # Why fetch from metadata or fallback?
        # Using real Twitch metadata yields accurate titles, streamer IDs, and creation dates,
        # fallback to placeholders only if GQL query fails due to network issues.
        if metadata:
            streamer_id = metadata.get("streamer_id") or "twitch_streamer"
            streamer_name = metadata.get("streamer_name") or "Twitch Streamer"
            title = metadata.get("title") or f"Twitch配信アーカイブ (ID: {vod_id})"
            duration = metadata.get("duration_seconds") or int(merged_events[-1]["offset_seconds"]) if merged_events else 3600
            
            streamed_at = datetime.now()
            created_at_str = metadata.get("created_at")
            if created_at_str:
                try:
                    streamed_at = datetime.fromisoformat(created_at_str.replace("Z", "+00:00"))
                except Exception:
                    pass
        else:
            streamer_id = "twitch_streamer"
            streamer_name = "Twitch Streamer"
            title = f"Twitch配信アーカイブ (ID: {vod_id})"
            duration = int(merged_events[-1]["offset_seconds"]) if merged_events else 3600
            streamed_at = datetime.now()
            
        streamer = db.get_or_create_streamer(streamer_id, streamer_name)
        avg_vel, max_vel, vel_json = calculate_chat_velocities(chat_data, duration)
        
        vod = VOD(
            vod_id=vod_id,
            streamer_id=streamer.streamer_id,
            title=title,
            duration_seconds=duration,
            streamed_at=streamed_at,
            average_viewers=0,  # Viewer count input is removed from UI
            avg_chat_velocity_hour=avg_vel,
            max_chat_velocity_min=max_vel,
            merged_timeline_text=timeline_txt,
            chat_velocity_json=vel_json
        )
        
        # Topic analysis
        topic_analyzer = TopicAnalyzer(api_key=api_key)
        topics_data = topic_analyzer.analyze_topics(timeline_txt)
        db_topics = [
            Topic(
                vod_id=vod_id,
                start_offset_seconds=t["start_offset_seconds"],
                end_offset_seconds=t["end_offset_seconds"],
                category=t["category"],
                description=t["description"],
                is_high_context=t["is_high_context"]
            )
            for t in topics_data
        ]
        
        # Comment persona analysis
        comment_analyzer = CommentAnalyzer(api_key=api_key)
        # Why pass configurable batch_size?
        # Different batch sizes strike different balances between Gemini analysis accuracy and rate limits.
        # Allowing it to be passed from the UI dynamically gives flexibility to the analysis process.
        listener_stats = comment_analyzer.analyze_listeners(chat_data, batch_size=batch_size, progress_callback=progress_callback)
        
        t_ai_analysis = int(time.time() - t_ai_start)
        total_time = int(time.time() - start_time)
        
        db_stats = []
        for s in listener_stats:
            import json
            # Why serialize comment_details to JSON string?
            # Storing the list of specific comments with their offsets and categories as a JSON string 
            # avoids table duplication while retaining the detail view for UI inspection.
            details_json = json.dumps(s.get("comment_details", []), ensure_ascii=False)
            db_stats.append(
                VODListenerStats(
                    vod_id=vod_id,
                    listener_username=s["username"],
                    total_comments=s["total_comments"],
                    reaction_comments_count=s["reaction_comments_count"],
                    question_comments_count=s["question_comments_count"],
                    insight_comments_count=s["insight_comments_count"],
                    instruction_comments_count=s["instruction_comments_count"],
                    other_comments_count=s["other_comments_count"],
                    persona_type=s["persona_type"],
                    comment_details_json=details_json
                )
            )
            
        # Database transaction for atomic replacement
        # Why run deletions and insertions in a single transaction block?
        # If the analysis fails halfway (e.g. VOD already deleted on Twitch), 
        # the legacy data remains completely intact and safe. We only commit changes 
        # when all new dataset objects are fully compiled.
        session_db = db.get_session()
        try:
            # Delete legacy associated tables to prevent duplicate records accumulation
            # Why delete both v-prefixed and raw numeric VOD IDs?
            # When re-analyzing, Twitch or external libs (like yt-dlp) might return a different ID format 
            # (e.g. raw numeric vs v-prefixed). Checking for both ensures that legacy orphan records 
            # are fully cleared, and only the newest analyzed VOD datasets are kept.
            target_ids = [vod_id]
            if vod_id.startswith('v'):
                target_ids.append(vod_id[1:])
            else:
                target_ids.append(f"v{vod_id}")
                
            for tid in target_ids:
                session_db.query(Topic).filter_by(vod_id=tid).delete()
                session_db.query(VODListenerStats).filter_by(vod_id=tid).delete()
                if tid != vod_id:
                    session_db.query(VOD).filter_by(vod_id=tid).delete()
            
            # Save or update VOD record
            vod.chat_collection_time_seconds = t_chat_collection
            vod.extraction_time_seconds = t_extraction
            vod.transcription_time_seconds = t_transcription
            vod.ai_analysis_time_seconds = t_ai_analysis
            vod.total_analysis_time_seconds = total_time
            session_db.merge(vod)
            
            # Save new topics
            for t in db_topics:
                session_db.add(t)
                
            # Save new listener stats
            for s in db_stats:
                session_db.merge(s)
                
            session_db.commit()
        except Exception as e:
            session_db.rollback()
            raise e
        finally:
            db.remove_session()
            
        # Clean up audio file to save disk space
        # Why delete the file?
        # MP3 files from multi-hour streams take up massive disk space.
        # Once transcribed, they are no longer needed, so deletion prevents storage leak.
        if os.path.exists(audio_path):
            os.remove(audio_path)
            
        progress_callback("✨ 分析完了！", 100)
        return vod_id
        
    except Exception as e:
        try:
            if 'audio_path' in locals() and os.path.exists(audio_path):
                os.remove(audio_path)
        except Exception:
            pass
        st.error(f"分析パイプライン実行中にエラーが発生しました: {e}")
        return None

# ---------------------------------------------------------
# Streamlit Dashboard UI
# ---------------------------------------------------------
def main():
    st.set_page_config(
        page_title="vibes-ttv | Twitch VOD Analyzer",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize DB
    # Why db_uri in memory fallback?
    # DB initialization should be safe; we default to a local sqlite file.
    db = DBManager()
    db.create_tables()
    
    # Premium CSS injection
    # Why use Custom CSS?
    # Streamlit's default UI can look plain and generic. 
    # Injecting curated modern design features (dark mode theme, Outfit fonts, linear-gradients, 
    # glassmorphism card layouts, and subtle shadows) makes the product look extremely premium.
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&family=Noto+Sans+JP:wght@300;400;700&display=swap');
        
        html, body, [class*="css"] {
            font-family: 'Outfit', 'Noto Sans JP', sans-serif;
        }
        
        .main-header {
            font-size: 2.8rem;
            background: linear-gradient(135deg, #a855f7 0%, #3b82f6 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: 800;
            margin-bottom: 0.5rem;
        }
        
        .sub-header {
            color: #9ca3af;
            font-size: 1.1rem;
            margin-bottom: 2rem;
        }
        
        .dashboard-card {
            background-color: #1f2937;
            border: 1px solid #374151;
            border-radius: 12px;
            padding: 1.5rem;
            margin-bottom: 1rem;
            box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1), 0 2px 4px -1px rgba(0,0,0,0.06);
            transition: transform 0.2s ease, border-color 0.2s ease;
        }
        
        .dashboard-card:hover {
            transform: translateY(-2px);
            border-color: #6366f1;
        }
        
        .metric-title {
            color: #9ca3af;
            font-size: 0.85rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        
        .metric-value {
            font-size: 2rem;
            font-weight: 700;
            color: #f3f4f6;
            margin-top: 0.25rem;
        }
        
        .topic-row {
            padding: 1rem;
            border-left: 4px solid #3b82f6;
            background-color: #111827;
            border-radius: 0 8px 8px 0;
            margin-bottom: 0.75rem;
        }
        
        .topic-row-hc {
            border-left: 4px solid #ef4444 !important;
            background-color: rgba(239, 68, 68, 0.05) !important;
        }
        
        .hc-badge {
            background-color: #ef4444;
            color: white;
            padding: 2px 6px;
            border-radius: 4px;
            font-size: 0.75rem;
            font-weight: 700;
            display: inline-block;
            margin-left: 8px;
        }
        
        .lc-badge {
            background-color: #10b981;
            color: white;
            padding: 2px 6px;
            border-radius: 4px;
            font-size: 0.75rem;
            font-weight: 700;
            display: inline-block;
            margin-left: 8px;
        }
        
        /* Prevent stale components from flickering during analysis runner reruns */
        [data-stale="true"] {
            opacity: 0.7 !important;
            transition: none !important;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # ---------------------------------------------------------
    # Sidebar Configuration
    # ---------------------------------------------------------
    st.sidebar.markdown("### ⚙️ 分析設定")
    
    # Define variables with default values to prevent NameError in subsequent checks
    api_key = os.getenv("GEMINI_API_KEY", "")
    whisper_model = "turbo"
    listener_batch_size = 30
    
    # Mode selection
    mode = st.sidebar.radio(
        "分析モードを選択してください",
        ["過去の分析結果を閲覧", "実際のVODを分析"]
    )
    
    selected_vod_id = None
    
    if mode == "過去の分析結果を閲覧":
        st.sidebar.markdown("#### 📁 保存されたデータ")
        # Why not query all streamers dynamically?
        # Loading all streamers from SQLite lets the user navigate historical analyses cleanly.
        streamers = db.get_all_streamers()
        if not streamers:
            st.sidebar.warning("⚠️ データベースに保存された分析データがありません。新規分析を行ってください。")
        else:
            selected_streamer = st.sidebar.selectbox(
                "配信者を選択",
                options=streamers,
                format_func=lambda s: f"{s.display_name} ({s.streamer_id})"
            )
            
            if selected_streamer:
                # Why group VODs by streamer?
                # A streamer can have many VODs. Grouping them hierarchically prevents the drop-down 
                # from becoming cluttered and simplifies database lookups.
                vods = db.get_vods_by_streamer(selected_streamer.streamer_id)
                if not vods:
                    st.sidebar.info("この配信者のアーカイブデータはありません。")
                else:
                    selected_vod = st.sidebar.selectbox(
                        "配信アーカイブを選択",
                        options=vods,
                        format_func=lambda v: f"{v.title} ({v.vod_id})"
                    )
                    if selected_vod:
                        selected_vod_id = selected_vod.vod_id
                        st.session_state["vod_id"] = selected_vod_id
                        
    # Common analysis configuration (always show)
    # Why show configuration even in view mode?
    # Keeping the API Key and model configurations visible at all times allows re-triggering analysis
    # from the main dashboard when viewing legacy incomplete datasets.
    st.sidebar.markdown("---")
    st.sidebar.markdown("#### ⚙️ 分析パラメーター")
    api_key = st.sidebar.text_input("Gemini API Key", value=api_key)
    whisper_model = st.sidebar.selectbox(
        "Whisperモデルサイズ",
        ["tiny", "base", "small", "medium", "large", "turbo"],
        index=5 # default turbo
    )
    listener_batch_size = st.sidebar.slider(
        "リスナー分析バッチサイズ",
        min_value=10,
        max_value=100,
        value=listener_batch_size,
        step=10,
        help="一度にGemini APIに送信するリスナーの数です。値を小さくすると品質が向上する可能性がありますが、API呼び出し回数が増加します。"
    )
    
    if mode == "実際のVODを分析":
        st.sidebar.markdown("---")
        vod_url = st.sidebar.text_input("Twitch VOD URL", value="https://www.twitch.tv/videos/123456789")
        
        # Check if analysis is currently running
        is_running = "analysis_runner" in st.session_state and not st.session_state["analysis_runner"].is_done
        
        if st.sidebar.button("分析を実行する", width="stretch", disabled=is_running):
            if not api_key:
                st.sidebar.error("Gemini API Key を入力してください。")
            else:
                # Why extract VOD ID and check database?
                # Analyzing a Twitch VOD is a heavy process involving downloading, Whisper transcription, and Gemini API calls.
                # Checking if the VOD already exists in SQLite allows us to serve the dashboard instantly, saving time and API quota.
                vod_id = extract_vod_id(vod_url)
                existing_vod = db.get_vod(vod_id)
                if existing_vod:
                    st.session_state["vod_id"] = vod_id
                    st.sidebar.success("データベースから既存の分析結果を読み込みました！")
                else:
                    runner = AnalysisRunner(db, vod_url, api_key, whisper_model, listener_batch_size)
                    st.session_state["analysis_runner"] = runner
                    runner.start()
                    st.rerun()
                        
    # Load default session state if exists
    if "vod_id" in st.session_state:
        selected_vod_id = st.session_state["vod_id"]
        
    # Main content rendering
    st.markdown("<h1 class='main-header'>vibes-ttv</h1>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>Twitch配信アーカイブ（VOD）の態度・話題分析ダッシュボード</div>", unsafe_allow_html=True)
    
    # Progress rendering and controls if analysis is running
    if "analysis_runner" in st.session_state:
        runner = st.session_state["analysis_runner"]
        
        # UI card container for analysis progress
        import time
        elapsed = int(time.time() - runner.start_time)
        st.markdown(f"""
        <div class="dashboard-card" style="border-color: #a855f7; background-color: rgba(168, 85, 247, 0.05);">
            <h4 style="margin-top:0; color:#a855f7; display:flex; justify-content:space-between; align-items:center;">
                <span>⚙️ バックグラウンド分析を実行中</span>
                <span style="font-size:0.85rem; padding:2px 8px; border-radius:12px; background-color:#a855f7; color:white;">
                    {runner.progress_val}%
                </span>
            </h4>
            <p style="color:#d1d5db; font-size:0.95rem; margin-bottom:0.5rem;">⏱️ 経過時間: {elapsed}秒 | {runner.message}</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.progress(runner.progress_val / 100.0 if 0 <= runner.progress_val <= 100 else 0.0)
        
        col_ctrl1, col_ctrl2 = st.columns(2)
        with col_ctrl1:
            # Why check runner.pause_event?
            # pause_event.is_set() is True when the analysis is running, and False when paused.
            # Showing the corresponding button dynamically allows clear control.
            if runner.pause_event.is_set():
                is_pausable = runner.is_pausable
                # Why disable pause during yt-dlp/Whisper?
                # Subprocesses and local C/C++ model execution block CPU execution in a way 
                # that cannot be paused safely without causing memory leaks or lockouts.
                st.button(
                    "一時停止", 
                    disabled=not is_pausable, 
                    key="pause_btn", 
                    width="stretch",
                    help="音声抽出中およびWhisper文字起こし中は一時停止できません。" if not is_pausable else None
                )
            else:
                st.button("再開", key="resume_btn", width="stretch")
                
        with col_ctrl2:
            st.button("分析を中止", key="stop_btn", width="stretch")
            
        # Handle button actions
        if st.session_state.get("pause_btn"):
            runner.pause_event.clear()
            st.rerun()
        if st.session_state.get("resume_btn"):
            runner.pause_event.set()
            st.rerun()
        if st.session_state.get("stop_btn"):
            runner.stop_event.set()
            runner.pause_event.set()  # resume if paused to exit thread quickly
            st.rerun()
            
        if runner.is_done:
            if runner.error:
                st.error(f"分析中にエラーが発生しました: {runner.error}")
            elif runner.vod_id:
                st.session_state["vod_id"] = runner.vod_id
                st.success(f"分析が完了しました！ (合計処理時間: {runner.total_analysis_time}秒)")
            del st.session_state["analysis_runner"]
            st.rerun()
        else:
            # Why sleep and rerun?
            # To keep the UI reactive and update the progress bar in near real-time, 
            # we sleep for 1 second and trigger st.rerun(). This acts as a client-side polling loop.
            import time
            time.sleep(1.0)
            st.rerun()
    
    if not selected_vod_id:
        st.info("👈 左側のサイドバーから「過去の分析結果を閲覧」するか、「実際のVOD」を分析してください。")
        return
        
    # Fetch data from DB
    vod = db.get_vod(selected_vod_id)
    if not vod:
        st.error("分析結果データの取得に失敗しました。")
        return
        
    # ---------------------------------------------------------
    # 1. Dashboard Metrics Summary
    # ---------------------------------------------------------
    # Fetch related structures
    session = db.get_session()
    topics = session.query(Topic).filter_by(vod_id=selected_vod_id).all()
    stats = session.query(VODListenerStats).filter_by(vod_id=selected_vod_id).all()
    
    st.markdown(f"### 📊 分析結果の概要: *{vod.title}*")
    
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    
    with m_col1:
        st.markdown(f"""
        <div class="dashboard-card">
            <div class="metric-title">配信時間</div>
            <div class="metric-value">{format_seconds(vod.duration_seconds)}</div>
        </div>
        """, unsafe_allow_html=True)
        
    with m_col2:
        st.markdown(f"""
        <div class="dashboard-card">
            <div class="metric-title">コメントリスナー数</div>
            <div class="metric-value">{len(stats)} 人</div>
        </div>
        """, unsafe_allow_html=True)
        
    with m_col3:
        st.markdown(f"""
        <div class="dashboard-card">
            <div class="metric-title">平均コメント時速</div>
            <div class="metric-value">{vod.avg_chat_velocity_hour:.1f} 回/h</div>
        </div>
        """, unsafe_allow_html=True)
        
    with m_col4:
        st.markdown(f"""
        <div class="dashboard-card">
            <div class="metric-title">最大瞬間コメント分速</div>
            <div class="metric-value">{vod.max_chat_velocity_min} 回/分</div>
        </div>
        """, unsafe_allow_html=True)
        
    # Why check performance attributes?
    # Older analyzed VODs in the database might not have performance execution times.
    # We display the panel only if the total time is stored, ensuring a clean dashboard layout.
    if hasattr(vod, "total_analysis_time_seconds") and vod.total_analysis_time_seconds:
        with st.expander("⌛ 分析処理のパフォーマンス (処理時間サマリー)", expanded=False):
            t_trans = vod.transcription_time_seconds or 1
            ratio = vod.duration_seconds / t_trans
            
            p_col1, p_col2, p_col3 = st.columns(3)
            with p_col1:
                st.metric("全体の所要時間", f"{vod.total_analysis_time_seconds} 秒")
                st.metric("音声抽出 (yt-dlp)", f"{vod.extraction_time_seconds or 0} 秒")
            with p_col2:
                st.metric("Whisper文字起こし", f"{vod.transcription_time_seconds or 0} 秒")
                st.metric("チャットログ収集", f"{vod.chat_collection_time_seconds or 0} 秒")
            with p_col3:
                st.metric("Whisper推論倍速", f"{ratio:.1f} 倍速")
                st.metric("Gemini AI分析", f"{vod.ai_analysis_time_seconds or 0} 秒")
        
    # ---------------------------------------------------------
    # Tabs layout for detailed analysis
    # ---------------------------------------------------------
    tab_attitude, tab_topics, tab_timeline = st.tabs(["💬 視聴者の態度分析", "🗣️ 配信中の話題分析", "📝 統合タイムライン"])
    
    # ---------------------------------------------------------
    # Tab 1: Viewer Attitude Analysis
    # ---------------------------------------------------------
    with tab_attitude:
        st.markdown("#### 態度とコメントの内訳")
        
        # Calculate overall comments count by type
        total_reaction = sum(s.reaction_comments_count for s in stats)
        total_question = sum(s.question_comments_count for s in stats)
        total_insight = sum(s.insight_comments_count for s in stats)
        total_instruction = sum(s.instruction_comments_count for s in stats)
        total_other = sum(s.other_comments_count for s in stats)
        total_chats = sum(s.total_comments for s in stats)
        
        col_chart1, col_chart2 = st.columns(2)
        
        with col_chart1:
            # Comment type distribution donut chart
            comment_breakdown = pd.DataFrame({
                "分類": ["感想・リアクション", "質問", "考察", "指示・提案", "その他"],
                "件数": [total_reaction, total_question, total_insight, total_instruction, total_other]
            })
            
            # Donut chart via Altair
            donut_comment = alt.Chart(comment_breakdown).mark_arc(innerRadius=60, outerRadius=110).encode(
                theta=alt.Theta(field="件数", type="quantitative"),
                color=alt.Color(
                    field="分類", 
                    type="nominal",
                    scale=alt.Scale(domain=comment_breakdown["分類"].tolist(), range=["#a855f7", "#3b82f6", "#10b981", "#f59e0b", "#6b7280"])
                ),
                tooltip=["分類", "件数"]
            ).properties(title="コメントの内訳（全体）", height=280)
            
            # Why width="stretch" instead of use_container_width=True?
            # Setting width="stretch" resolves Streamlit deprecation warnings for charts.
            st.altair_chart(donut_comment, width="stretch")
            
        with col_chart2:
            # Listener persona type distribution
            persona_counts = {}
            for s in stats:
                persona_counts[s.persona_type] = persona_counts.get(s.persona_type, 0) + 1
                
            # Maps persona names to Japanese
            p_jp_map = {
                "reaction": "リアクション型",
                "question": "質問型",
                "insight": "考察型",
                "instruction": "指示・提案型",
                "other": "その他雑談型"
            }
            
            persona_breakdown = pd.DataFrame({
                "ペルソナ": [p_jp_map.get(k, k) for k in persona_counts.keys()],
                "人数": list(persona_counts.values())
            })
            
            donut_persona = alt.Chart(persona_breakdown).mark_arc(innerRadius=60, outerRadius=110).encode(
                theta=alt.Theta(field="人数", type="quantitative"),
                color=alt.Color(
                    field="ペルソナ", 
                    type="nominal",
                    scale=alt.Scale(domain=persona_breakdown["ペルソナ"].tolist(), range=["#c084fc", "#60a5fa", "#34d399", "#fbbf24", "#9ca3af"])
                ),
                tooltip=["ペルソナ", "人数"]
            ).properties(title="コメントリスナーのペルソナ内訳", height=280)
            
            # Why width="stretch" instead of use_container_width=True?
            # Resolves Streamlit deprecation warnings for charts.
            st.altair_chart(donut_persona, width="stretch")
            

            
        # Chart: Velocity line chart
        st.markdown("#### コメント速度の推移")
        # Why check chat_velocity_json?
        # Older analyzed records or mock data might not have the chat velocity time-series text.
        # Handling None safely prevents Streamlit rendering exceptions.
        if hasattr(vod, 'chat_velocity_json') and vod.chat_velocity_json:
            import json
            try:
                vel_data = json.loads(vod.chat_velocity_json)
                if vel_data:
                    vel_df = pd.DataFrame(vel_data)
                    
                    # Why use Altair layered line and area chart instead of st.line_chart?
                    # Streamlit's built-in charts offer minimal customization. Overlaying a semi-transparent 
                    # area chart with a solid line styled in Twitch-theme violet (#a855f7) creates a beautiful 
                    # glowing trend graph that aligns perfectly with our dark mode theme and supports interactive tooltips.
                    base = alt.Chart(vel_df).encode(
                        x=alt.X('minute:Q', title='経過時間 (分)'),
                        y=alt.Y('count:Q', title='コメント分速 (件/分)'),
                        tooltip=[
                            alt.Tooltip('minute:Q', title='経過時間 (分)'),
                            alt.Tooltip('count:Q', title='コメント数')
                        ]
                    )
                    
                    area = base.mark_area(
                        color='#a855f7',
                        opacity=0.2
                    )
                    
                    line = base.mark_line(
                        color='#a855f7',
                        strokeWidth=2
                    )
                    
                    chart = alt.layer(area, line).properties(
                        height=280
                    )
                    st.altair_chart(chart, width="stretch")
                else:
                    st.info("コメント速度のデータが空です。")
            except Exception as e:
                st.error(f"コメント速度グラフの描画中にエラーが発生しました: {e}")
        else:
            st.info("このアーカイブには時系列のコメント速度データが保存されていません（古いデータなどのため）。")
            
        # Table: Listener detail list
        st.markdown("#### リスナー詳細一覧")
        stats_data = []
        for s in stats:
            stats_data.append({
                "リスナー名": s.listener_username,
                "総コメント数": s.total_comments,
                "リアクション": s.reaction_comments_count,
                "質問": s.question_comments_count,
                "考察": s.insight_comments_count,
                "指示・提案": s.instruction_comments_count,
                "その他": s.other_comments_count,
                "ペルソナ種類": p_jp_map.get(s.persona_type, s.persona_type)
            })
        stats_df = pd.DataFrame(stats_data)
        # Why width="stretch" instead of use_container_width=True?
        # Resolves Streamlit deprecation warnings for dataframes.
        st.dataframe(stats_df, width="stretch")
        
        # Why filter by hasattr and non-empty?
        # Checking for comment_details_json ensures that we only show the detailed logs 
        # when the data is available (for newer runs), maintaining backward compatibility.
        listeners_with_details = [s for s in stats if hasattr(s, "comment_details_json") and s.comment_details_json]
        if listeners_with_details:
            st.markdown("---")
            st.markdown("#### 💬 コメントの分類詳細")
            
            listener_names = sorted([s.listener_username for s in listeners_with_details])
            selected_user = st.selectbox("詳細を表示するリスナーを選択", listener_names)
            
            user_stat = next(s for s in listeners_with_details if s.listener_username == selected_user)
            
            try:
                import json
                comment_details = json.loads(user_stat.comment_details_json)
            except Exception:
                comment_details = []
                
            if comment_details:
                cat_options = {
                    "reaction": "リアクション",
                    "question": "質問",
                    "insight": "考察",
                    "instruction": "指示・提案",
                    "other": "その他"
                }
                rev_cat_map = {v: k for k, v in cat_options.items()}
                
                # Filter categories
                selected_cats = st.multiselect(
                    "表示する態度カテゴリ（複数選択可）",
                    options=list(cat_options.values()),
                    default=list(cat_options.values())
                )
                
                filter_keys = [rev_cat_map[c] for c in selected_cats]
                filtered_details = [c for c in comment_details if c.get("category") in filter_keys]
                
                if filtered_details:
                    badge_styles = {
                        "reaction": "background-color: rgba(168, 85, 247, 0.15); color: #c084fc; border: 1px solid rgba(168, 85, 247, 0.3);",
                        "question": "background-color: rgba(59, 130, 246, 0.15); color: #60a5fa; border: 1px solid rgba(59, 130, 246, 0.3);",
                        "insight": "background-color: rgba(234, 179, 8, 0.15); color: #facc15; border: 1px solid rgba(234, 179, 8, 0.3);",
                        "instruction": "background-color: rgba(239, 68, 68, 0.15); color: #f87171; border: 1px solid rgba(239, 68, 68, 0.3);",
                        "other": "background-color: rgba(156, 163, 175, 0.15); color: #9ca3af; border: 1px solid rgba(156, 163, 175, 0.3);"
                    }
                    
                    html_rows = []
                    for c in filtered_details:
                        time_str = format_seconds(c.get("offset_seconds", 0))
                        msg = c.get("message", "")
                        cat = c.get("category", "other")
                        badge_style = badge_styles.get(cat, badge_styles["other"])
                        cat_jp = cat_options.get(cat, "その他")
                        
                        html_rows.append(f"""
                        <tr style="border-bottom: 1px solid rgba(255, 255, 255, 0.05);">
                            <td style="padding: 0.6rem 0.8rem; color: #9ca3af; font-family: monospace; font-size: 0.9rem; white-space: nowrap;">🕒 {time_str}</td>
                            <td style="padding: 0.6rem 0.8rem; color: #f3f4f6; text-align: left; font-size: 0.95rem;">{msg}</td>
                            <td style="padding: 0.6rem 0.8rem; text-align: right; white-space: nowrap;">
                                <span style="padding: 0.2rem 0.5rem; border-radius: 6px; font-size: 0.78rem; font-weight: 700; {badge_style}">
                                    {cat_jp}
                                </span>
                            </td>
                        </tr>
                        """)
                        
                    table_html = f"""
                    <table style="width: 100%; border-collapse: collapse; background-color: rgba(30, 41, 59, 0.2); border: 1px solid rgba(255, 255, 255, 0.05); border-radius: 8px; overflow: hidden; margin-top: 1rem;">
                        <thead>
                            <tr style="background-color: rgba(255, 255, 255, 0.02); border-bottom: 1px solid rgba(255, 255, 255, 0.08);">
                                <th style="padding: 0.6rem 0.8rem; text-align: left; color: #9ca3af; font-size: 0.85rem; font-weight: 600; width: 100px;">時間</th>
                                <th style="padding: 0.6rem 0.8rem; text-align: left; color: #9ca3af; font-size: 0.85rem; font-weight: 600;">コメント内容</th>
                                <th style="padding: 0.6rem 0.8rem; text-align: right; color: #9ca3af; font-size: 0.85rem; font-weight: 600; width: 120px;">判定された態度</th>
                            </tr>
                        </thead>
                        <tbody>
                            {"".join(html_rows)}
                        </tbody>
                    </table>
                    """
                    st.markdown(table_html, unsafe_allow_html=True)
                else:
                    st.info("選択された態度カテゴリに該当するコメントはありません。")
            else:
                st.info("このリスナーの個別コメント詳細がありません。")
        else:
            st.info("このアーカイブには個別のコメント分類詳細データが保存されていません（古いデータなどのため）。")
            
            # Check if analysis is currently running
            # Why check is_running?
            # Preventing double-triggering of background runners avoids SQLite write conflicts.
            is_running = "analysis_runner" in st.session_state and not st.session_state["analysis_runner"].is_done
            
            # Why use a unique key?
            # Streamlit requires unique widget keys when multiple buttons can potentially exist in the DOM.
            if st.button("🔄 このアーカイブを再分析する", disabled=is_running, key="reanalyze_btn", width="stretch"):
                if not api_key:
                    st.error("Gemini API Key を入力してください（サイドバーに入力欄があります）。")
                else:
                    # Reconstruct Twitch VOD URL from stored ID
                    # Why reconstruct?
                    # The VOD URL is needed for the scraper and yt-dlp, but only the ID is stored in the DB.
                    # Appending the ID to the base Twitch videos URL cleanly reconstructs the URL.
                    # We strip the optional 'v' prefix because yt-dlp fails if the URL path contains a non-numeric ID.
                    clean_id = selected_vod_id[1:] if selected_vod_id.startswith('v') else selected_vod_id
                    vod_url = f"https://www.twitch.tv/videos/{clean_id}"
                    runner = AnalysisRunner(db, vod_url, api_key, whisper_model, listener_batch_size)
                    st.session_state["analysis_runner"] = runner
                    runner.start()
                    st.rerun()

    # ---------------------------------------------------------
    # Tab 2: Topic Analysis
    # ---------------------------------------------------------
    with tab_topics:
        st.markdown("#### 配信中の話題タイムライン")
        
        cat_jp_map = {
            "game": "ゲーム内容",
            "daily_news": "時事・日常会話",
            "past_stream": "過去配信の話題",
            "other": "その他・雑談"
        }
        
        # Parse velocity data if exists
        vel_list = []
        if hasattr(vod, "chat_velocity_json") and vod.chat_velocity_json:
            import json
            try:
                vel_list = json.loads(vod.chat_velocity_json)
            except Exception:
                pass
                
        # Collect velocity for all topics to generate rankings
        topic_velocity_pairs = []
        for t in topics:
            start_min = int(t.start_offset_seconds // 60)
            end_min = int(t.end_offset_seconds // 60)
            topic_counts = [item["count"] for item in vel_list if start_min <= item["minute"] <= end_min]
            topic_max_vel = max(topic_counts) if topic_counts else 0
            topic_velocity_pairs.append((t, topic_max_vel))
            
        if topics:
            # Sort by velocity descending
            # Why sort topics by velocity?
            # Sorting allows us to easily slice the top 3 (best) and bottom 3 (worst) topics.
            # This gives a quick summary of stream highlights and lowlights.
            sorted_pairs = sorted(topic_velocity_pairs, key=lambda x: x[1], reverse=True)
            best_3 = sorted_pairs[:3]
            worst_3 = sorted(topic_velocity_pairs, key=lambda x: x[1])[:3]
            
            # Why calculate occupy durations for category percentages?
            # Calculating category ratios by duration sum (start-to-end seconds) instead of simple topic count
            # accurately reflects how much stream time was spent on each topic type.
            st.markdown("##### 📊 話題の分類比率")
            cat_durations = {}
            for t in topics:
                dur = t.end_offset_seconds - t.start_offset_seconds
                cat_label = cat_jp_map.get(t.category, t.category)
                cat_durations[cat_label] = cat_durations.get(cat_label, 0) + dur
                
            topic_breakdown = pd.DataFrame([
                {"カテゴリ": k, "時間(分)": round(v / 60, 1)} for k, v in cat_durations.items()
            ])
            
            # Donut chart rendering with Altair
            # Why use donut chart?
            # Donut charts match the dashboard's premium styling and visualize category distributions cleanly.
            donut_topic = alt.Chart(topic_breakdown).mark_arc(innerRadius=50, outerRadius=90).encode(
                theta=alt.Theta(field="時間(分)", type="quantitative"),
                color=alt.Color(
                    field="カテゴリ",
                    type="nominal",
                    scale=alt.Scale(
                        domain=["ゲーム内容", "時事・日常会話", "過去配信の話題", "その他・雑談"],
                        range=["#a855f7", "#3b82f6", "#10b981", "#6b7280"]
                    )
                ),
                tooltip=["カテゴリ", "時間(分)"]
            ).properties(
                height=220
            )
            st.altair_chart(donut_topic, width="stretch")
            st.markdown("<br/>", unsafe_allow_html=True)
            
            st.markdown("##### 📈 話題の盛り上がりランキング")
            r_col1, r_col2 = st.columns(2)
            
            with r_col1:
                # Why use custom inline HTML/CSS?
                # Streamlit's default container elements lack premium styling features. 
                # Implementing a custom semi-transparent green background with Outfit-aligned 
                # typography creates a visually distinct best-3 ranking card.
                st.markdown("""
                <div style="background-color: rgba(16, 185, 129, 0.05); border: 1px solid rgba(16, 185, 129, 0.2); border-radius: 12px; padding: 1rem; margin-bottom: 1.5rem;">
                    <h5 style="margin-top:0; color:#10b981; font-weight:700;">🔥 盛り上がった話題ベスト3</h5>
                """, unsafe_allow_html=True)
                
                medals = ["🥇 1位", "🥈 2位", "🥉 3位"]
                for idx, (t, vel) in enumerate(best_3):
                    time_range = f"{format_seconds(t.start_offset_seconds)} 〜 {format_seconds(t.end_offset_seconds)}"
                    desc = t.description[:40] + "..." if len(t.description) > 40 else t.description
                    twitch_url = f"https://www.twitch.tv/videos/{vod.vod_id}?t={format_twitch_offset(t.start_offset_seconds)}"
                    
                    st.markdown(f"""
                    <div style="margin-bottom:0.75rem; border-bottom: 1px dashed rgba(255,255,255,0.1); padding-bottom:0.5rem; text-align:left;">
                        <strong style="color:#f3f4f6;">{medals[idx]} | <a href="{twitch_url}" target="_blank" style="color: #c084fc; text-decoration: none; border-bottom: 1px dashed rgba(192, 132, 252, 0.4);">{time_range} 🔗</a></strong> <span style="color:#10b981; font-weight:700;">({vel} 件/分)</span><br/>
                        <span style="color:#9ca3af; font-size:0.85rem;">{desc}</span>
                    </div>
                    """, unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
                
            with r_col2:
                # Why use custom inline HTML/CSS?
                # Creating a red themed companion card to represent the lowest-activity segments 
                # gives the user a clear comparative view of silent stream moments.
                st.markdown("""
                <div style="background-color: rgba(239, 68, 68, 0.05); border: 1px solid rgba(239, 68, 68, 0.2); border-radius: 12px; padding: 1rem; margin-bottom: 1.5rem;">
                    <h5 style="margin-top:0; color:#f87171; font-weight:700;">💤 静かだった話題ワースト3</h5>
                """, unsafe_allow_html=True)
                
                slugs = ["🐌 1位", "🥈 2位", "🥉 3位"]
                for idx, (t, vel) in enumerate(worst_3):
                    time_range = f"{format_seconds(t.start_offset_seconds)} 〜 {format_seconds(t.end_offset_seconds)}"
                    desc = t.description[:40] + "..." if len(t.description) > 40 else t.description
                    twitch_url = f"https://www.twitch.tv/videos/{vod.vod_id}?t={format_twitch_offset(t.start_offset_seconds)}"
                    
                    st.markdown(f"""
                    <div style="margin-bottom:0.75rem; border-bottom: 1px dashed rgba(255,255,255,0.1); padding-bottom:0.5rem; text-align:left;">
                        <strong style="color:#f3f4f6;">{slugs[idx]} | <a href="{twitch_url}" target="_blank" style="color: #c084fc; text-decoration: none; border-bottom: 1px dashed rgba(192, 132, 252, 0.4);">{time_range} 🔗</a></strong> <span style="color:#f87171; font-weight:700;">({vel} 件/分)</span><br/>
                        <span style="color:#9ca3af; font-size:0.85rem;">{desc}</span>
                    </div>
                    """, unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
                
            st.markdown("<hr/>", unsafe_allow_html=True)
        
        # Sort topics chronologically
        sorted_topics = sorted(topics, key=lambda x: x.start_offset_seconds)
        
        for t in sorted_topics:
            time_range = f"{format_seconds(t.start_offset_seconds)} 〜 {format_seconds(t.end_offset_seconds)}"
            cat_label = cat_jp_map.get(t.category, t.category)
            
            # Calculate maximum velocity in the topic's time range
            start_min = int(t.start_offset_seconds // 60)
            end_min = int(t.end_offset_seconds // 60)
            
            # Why filter and get max?
            # Instead of performing complex queries or recalculations, we can scan the VOD's 
            # pre-calculated chat_velocity_json. This delivers maximum instant lookup performance 
            # and works transparently for all historical data without database re-creation.
            topic_counts = [item["count"] for item in vel_list if start_min <= item["minute"] <= end_min]
            topic_max_vel = max(topic_counts) if topic_counts else 0
            
            hc_badge_html = "<span class='hc-badge'>⚠ ハイコンテクスト</span>" if t.is_high_context else "<span class='lc-badge'>✓ ローコンテクスト</span>"
            hc_class = "topic-row-hc" if t.is_high_context else ""
            
            twitch_url = f"https://www.twitch.tv/videos/{vod.vod_id}?t={format_twitch_offset(t.start_offset_seconds)}"
            
            st.markdown(f"""
            <div class="topic-row {hc_class}" style="display: flex; justify-content: space-between; align-items: stretch;">
                <div style="flex: 1; padding-right: 1.5rem;">
                    <strong><a href="{twitch_url}" target="_blank" style="color: #c084fc; text-decoration: none; border-bottom: 1px dashed rgba(192, 132, 252, 0.4); padding-bottom: 1px;">🕒 {time_range} | 分類: {cat_label} 🔗</a></strong>
                    <div style="margin-top: 0.5rem; color: #d1d5db; font-size: 0.95rem;">
                        {t.description}
                    </div>
                </div>
                <div style="display: flex; flex-direction: column; justify-content: space-between; align-items: flex-end; min-width: 145px; text-align: right;">
                    {hc_badge_html}
                    <div style="color: #c084fc; font-size: 0.85rem; font-weight: 700; margin-top: 0.5rem; white-space: nowrap;">
                        🚀 最大瞬間分速: {topic_max_vel} 件/分
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown("<hr/>", unsafe_allow_html=True)
        st.markdown("#### 💡 話題の文脈解説（ハイコンテクスト判定について）")
        st.markdown(
            "- **ハイコンテクスト (⚠)**: 配信者が過去の配信や、リスナーとの内輪の文脈、または時事ネタを「説明なしで」ピックしている状態です。"
            "新規視聴者は話についていきづらい可能性があるため、定期的に補足説明を入れるなどの工夫が推奨されます。\n"
            "- **ローコンテクスト (✓)**: 現在の画面内の状況だけで理解できる話や、時事ネタ等に対して丁寧な前提説明がある状態です。新規視聴者でも安心して楽しむことができます。"
        )

    # ---------------------------------------------------------
    # Tab 3: Merged Timeline Log
    # ---------------------------------------------------------
    with tab_timeline:
        st.markdown("#### 💬 配信音声書き起こし ＆ チャット統合ログ")
        st.markdown(
            "配信者の発言テキストと、その発言前後に投稿されたチャットコメントを時系列順に統合したログです。"
            "配信全体の文脈や、どの瞬間にどのような会話が行われていたかを詳細に振り返ることができます。"
        )
        
        # Why check merged_timeline_text?
        # Older analyzed records or mock data might not have the merged timeline text.
        # Handling None safely prevents Streamlit exceptions.
        if hasattr(vod, 'merged_timeline_text') and vod.merged_timeline_text:
            st.text_area(
                label="統合タイムラインログ (コピー・スクロール可能)",
                value=vod.merged_timeline_text,
                height=500,
                disabled=True
            )
        else:
            st.info("このアーカイブには統合タイムラインテキストが保存されていません（デモデータ等のため）。")

    # Close DB session safely
    db.remove_session()

if __name__ == "__main__":
    main()
