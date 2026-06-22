import sys
import os
import json # Why not import json at file level? To serialize the merged events for the database cleanly.

# Why append parent directory to sys.path programmatically?
# When executing 'streamlit run vibes_ttv/app.py', Streamlit resolves paths relative 
# to the script's directory, which breaks top-level package imports of 'vibes_ttv'.
# Programmatically adding the project root resolves this without requiring the user 
# to manually configure the PYTHONPATH environment variable on Windows.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# pyrefly: ignore [missing-import]
import streamlit as st
import pandas as pd
# pyrefly: ignore [missing-import]
import altair as alt
import re
from datetime import datetime

# Import project modules
from vibes_ttv.database.db_manager import DBManager
from vibes_ttv.database.models import VOD, Streamer, Topic, VODListenerStats
from vibes_ttv.collectors.chat_collector import ChatCollector
from vibes_ttv.collectors.audio_collector import AudioCollector
from vibes_ttv.analyzers.stt.factory import get_transcriber
from vibes_ttv.analyzers.stt.whisper_transcriber import WhisperTranscriber
from vibes_ttv.analyzers.timeline_merger import TimelineMerger
from vibes_ttv.analyzers.comment_analyzer import CommentAnalyzer, CommentCategory
from vibes_ttv.analyzers.topic_analyzer import TopicAnalyzer

# Why not start preload on import?
# Preloading the heavy Whisper model automatically consumes 1.6GB+ of RAM/VRAM even if the user
# selects Google Cloud STT. We defer preloading until Whisper is confirmed as the active engine in the UI.


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

def get_twitch_vod_url(vod_id: str, offset_seconds: float = None) -> str:
    # Why strip 'v' prefix?
    # The database stores VOD IDs with a 'v' prefix (e.g. 'v2786816848') for historical consistency, 
    # but the actual Twitch VOD URL only accepts raw numerical IDs.
    clean_id = vod_id.lstrip('v')
    url = f"https://www.twitch.tv/videos/{clean_id}"
    if offset_seconds is not None:
        url += f"?t={format_twitch_offset(int(offset_seconds))}"
    return url

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

# ---------------------------------------------------------
# Mock Data Generator (for Quick Validation)
# ---------------------------------------------------------


# ---------------------------------------------------------
# Real Pipeline Runner
# ---------------------------------------------------------
# Why add optional progress_callback?
# Adding progress_callback parameter allows callers (like st.status flow in UI) to capture 
# and render execution steps natively, while keeping a fallback st.empty() logic for standalone runner cases.
def run_real_analysis(
    db: DBManager, 
    vod_url: str, 
    api_key: str, 
    batch_size: int = 30, 
    progress_callback=None,
    stt_engine: str = "whisper",
    google_project_id: str = "vibes-ttv",
    google_bucket_name: str = "temporary-speech-files"
) -> str:
    
    # Why track start_time?
    # Knowing the elapsed time helps reassure the user that the pipeline is active 
    # even when processing heavy steps (like STT transcription).
    import time
    start_time = time.time()
    
    t_chat_collection = 0
    t_extraction = 0
    t_transcription = 0
    t_ai_analysis = 0
    
    # Why define callback helper wrapper?
    # By assigning progress_callback to local variable name or defining it locally,
    # we avoid modifying every single downstream call site in the function while
    # supporting custom callback integration (like st.status flow).
    if progress_callback is None:
        status_text = st.empty()
        progress_bar = st.progress(0)
        def local_progress_callback(message: str, progress_val: int):
            elapsed = int(time.time() - start_time)
            status_text.text(f"⏱️ 経過時間: {elapsed}秒 | {message}")
            progress_bar.progress(progress_val)
        actual_callback = local_progress_callback
    else:
        actual_callback = progress_callback
        
    try:
        # Step 0: Get VOD metadata
        actual_callback("🔍 [0/5] Twitch VOD メタデータを取得中...", 5)
        collector = ChatCollector()
        metadata = collector.get_video_metadata(vod_url)
        
        # Step 1: Collect chat logs
        actual_callback("🤖 [1/5] Twitchチャットログを収集中...", 10)
        t_chat_start = time.time()
        chat_data = collector.collect_chat(vod_url, progress_callback=actual_callback)
        t_chat_collection = int(time.time() - t_chat_start)
        if not chat_data:
            st.error("チャットログの取得に失敗しました。URLが正しいか、VODが公開されているかご確認ください。")
            return None
            
        # Step 2: Download and extract audio
        actual_callback("🎵 [2/5] VODから音声トラックを抽出中 (yt-dlp)...", 30)
        t_extract_start = time.time()
        audio_coll = AudioCollector()
        audio_path = audio_coll.collect_audio(vod_url, progress_callback=actual_callback)
        t_extraction = int(time.time() - t_extract_start)
        # Extract VOD ID from audio output filename
        vod_id = os.path.basename(audio_path).replace(".mp3", "")
        
        # Step 3: Transcription using selected STT engine
        engine_label = "Google Cloud STT" if stt_engine == "google_stt" else "Whisper (turbo)"
        actual_callback(f"✍️ [3/5] {engine_label} で音声を文字起こし中... (数分かかる場合があります)", 50)
        t_transcribe_start = time.time()
        
        # Why parameterize via factory?
        # Decoupling transcription implementation from the main analysis loop allows 
        # switching backends (e.g. Whisper, Google Cloud STT, or custom mock transcribers) 
        # transparently by changing config arguments.
        transcriber_kwargs = {}
        if stt_engine == "google_stt":
            transcriber_kwargs["project_id"] = google_project_id
            transcriber_kwargs["bucket_name"] = google_bucket_name
            
        transcriber = get_transcriber(stt_engine, **transcriber_kwargs)
        segments = transcriber.transcribe(audio_path)
        t_transcription = int(time.time() - t_transcribe_start)
        
        # Step 4: Merge chats and text
        actual_callback("🔗 [4/5] チャットログと音声認識タイムスタンプをアラインメント中...", 75)
        merger = TimelineMerger()
        merged_events = merger.merge(segments, chat_data)
        timeline_txt = merger.format_to_text(merged_events)
        
        # Step 5: AI analysis using Gemini API
        actual_callback("🧠 [5/5] Gemini API で話題 of コンテキストを抽出中...", 80)
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
            merged_timeline_json=None,
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
        # Why pass configurable slice_size?
        # Setting the slice_size allows the user to balance API rate-limits vs context window range.
        # Why not hardcode slice_size?
        # Exposing it to the UI via batch_size allows the user to optimize request speed 
        # and prevent "Resource Exhausted" Gemini API errors during analysis.
        # Why pass merged_events?
        # Providing the chronological timeline events list enables context-aware (streamer talk + other chats)
        # comment classification, yielding superior semantic accuracy.
        listener_stats = comment_analyzer.analyze_listeners(
            merged_events=merged_events,
            slice_size=batch_size,
            progress_callback=actual_callback
        )
        
        # Why save timeline as serialized JSON in the VOD record?
        # Overwriting with a JSON string of merged_events (which now has classification categories) 
        # avoids text parsing during reload and facilitates rich formatting in the UI.
        vod.merged_timeline_json = json.dumps(merged_events, ensure_ascii=False)
        
        t_ai_analysis = int(time.time() - t_ai_start)
        total_time = int(time.time() - start_time)
        
        db_stats = []
        for s in listener_stats:
            # Why not rebuild counts dictionary manually?
            # comment_analyzer now returns a clean "category_counts" dictionary directly, 
            # so we can serialize and store it without redundant mapping logic.
            db_stats.append(
                VODListenerStats(
                    vod_id=vod_id,
                    listener_username=s["username"],
                    total_comments=s["total_comments"],
                    category_counts_json=json.dumps(s["category_counts"], ensure_ascii=False),
                    persona_type=s["persona_type"]
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
            # Why delete by vod_id?
            # Since both selected_vod_id and the newly analyzed vod_id maintain 'v'-prefixed consistency,
            # we can safely delete old records by the exact same vod_id in a single transaction block.
            session_db.query(Topic).filter_by(vod_id=vod_id).delete()
            session_db.query(VODListenerStats).filter_by(vod_id=vod_id).delete()
            
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
            
        actual_callback("✨ 分析完了！", 100)
        return vod_id
        
    except Exception as e:
        try:
            if 'audio_path' in locals() and os.path.exists(audio_path):
                os.remove(audio_path)
        except Exception:
            pass
        # Why raise the exception?
        # Propagating the error to the caller (main loop) ensures that the exact stack trace 
        # is stored in st.session_state and shown in the UI, rather than swallowed here.
        raise e


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
    stt_engine = "whisper"
    google_project_id = "vibes-ttv"
    google_bucket_name = "temporary-speech-files"
    listener_batch_size = 30
    
    # Mode selection
    mode = st.sidebar.radio(
        "分析モードを選択してください",
        ["過去の分析結果を閲覧", "実際のVODを分析"],
        key="analysis_mode"
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
    api_key = st.sidebar.text_input("Gemini API Key", value=api_key, key="gemini_api_key")
    
    # Why make STT engine configurable?
    # Exposing STT engines allows switching between local Whisper (high GPU/CPU utilization but free)
    # and Google Cloud STT (zero local compute overhead, requires Google Cloud account/GCS).
    stt_option = st.sidebar.selectbox(
        "音声文字起こし (STT) エンジンの選択",
        options=["ローカル Whisper (turbo)", "Google Cloud Speech-to-Text"],
        index=0,
        key="stt_option_select"
    )
    stt_engine = "whisper" if stt_option == "ローカル Whisper (turbo)" else "google_stt"
    
    if stt_engine == "google_stt":
        # Why expose project_id and bucket_name settings?
        # Streamlit sidebar input fields allow runtime configuration of GCP assets 
        # without hardcoding project specifics, increasing usability.
        google_project_id = st.sidebar.text_input(
            "GCP プロジェクトID",
            value=google_project_id,
            key="gcp_project_id_input"
        )
        google_bucket_name = st.sidebar.text_input(
            "GCS バケット名",
            value=google_bucket_name,
            key="gcs_bucket_name_input"
        )
    else:
        # Why trigger preload here?
        # Initializing Whisper model preload asynchronously when Whisper is active 
        # avoids loading the heavy model if the user plans to use Google Cloud STT, 
        # saving RAM and VRAM.
        WhisperTranscriber.start_preload()

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
        vod_url = st.sidebar.text_input("Twitch VOD URL", value="https://www.twitch.tv/videos/123456789", key="vod_url")
        
        # Check if analysis trigger is armed
        is_running = bool(st.session_state.get("start_analysis"))
        
        if st.sidebar.button("分析を実行する", width="stretch", disabled=is_running, key="run_analysis_btn"):
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
                    # Why set trigger state?
                    # Streamlit execution is synchronous. Setting a trigger state and rerunning 
                    # allows us to intercept execution at the top of the main area, showing 
                    # an isolated st.status container for synchronous visual updates without flickering.
                    st.session_state["start_analysis"] = True
                    st.session_state["analysis_vod_url"] = vod_url
                    st.rerun()
                        
    # Load default session state if exists
    if "vod_id" in st.session_state:
        selected_vod_id = st.session_state["vod_id"]
        
    # Main content rendering
    st.markdown("<h1 class='main-header'>vibes-ttv</h1>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>Twitch配信アーカイブ（VOD）の態度・話題分析ダッシュボード</div>", unsafe_allow_html=True)
    
    # ---------------------------------------------------------
    # Notification Messages Rendering
    # ---------------------------------------------------------
    # Why check and render message here?
    # This renders the success/error messages compiled during the pipeline execution
    # immediately after the page reruns, then clears them from session state to prevent repeated displays.
    if "analysis_message" in st.session_state:
        msg = st.session_state["analysis_message"]
        if msg["type"] == "success":
            st.success(msg["text"])
        elif msg["type"] == "error":
            st.error(msg["text"])
        del st.session_state["analysis_message"]
        
    # ---------------------------------------------------------
    # Active Analysis Flow using st.status
    # ---------------------------------------------------------
    # Why run analysis synchronously inside st.status?
    # Keeping execution synchronous avoids multi-threading race conditions and memory leaks,
    # while the st.status container dynamically updates progress logs in place without 
    # triggering page-wide reruns (stale flickering).
    if st.session_state.get("start_analysis"):
        analysis_url = st.session_state.get("analysis_vod_url")
        
        with st.status("🔍 Twitch VOD の分析を実行中...", expanded=True) as status:
            # Why use st.empty()?
            # st.empty() creates a single placeholder slot inside the st.status container.
            # Calling progress_placeholder.write() repeatedly overwrites the previous message in-place,
            # preventing log duplication and keeping the visual display clean.
            progress_placeholder = st.empty()
            
            def sync_progress_callback(message: str, progress_val: int):
                progress_placeholder.write(message)
                
            try:
                vod_id = run_real_analysis(
                    db, 
                    analysis_url, 
                    api_key, 
                    batch_size=listener_batch_size, 
                    progress_callback=sync_progress_callback,
                    stt_engine=stt_engine,
                    google_project_id=google_project_id,
                    google_bucket_name=google_bucket_name
                )
                if vod_id:
                    status.update(label="✨ 分析が完了しました！", state="complete", expanded=False)
                    st.session_state["vod_id"] = vod_id
                    st.session_state["analysis_message"] = {
                        "type": "success",
                        "text": "分析が正常に完了しました！ダッシュボードを表示します。"
                    }
                else:
                    status.update(label="⚠️ 分析に失敗しました。", state="error", expanded=True)
                    st.session_state["analysis_message"] = {
                        "type": "error",
                        "text": "分析処理に失敗しました。チャットログまたは音声抽出エラーをご確認ください。"
                    }
            except Exception as e:
                status.update(label="❌ エラーが発生しました。", state="error", expanded=True)
                st.session_state["analysis_message"] = {
                    "type": "error",
                    "text": f"分析実行中に想定外のエラーが発生しました: {e}"
                }
                
        # Reset trigger flags and reload page to render dashboard
        del st.session_state["start_analysis"]
        if "analysis_vod_url" in st.session_state:
            del st.session_state["analysis_vod_url"]
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
        
        # Why not define total_counts dynamically using CommentCategory?
        # Extracting counts directly using the CommentCategory Enum loop ensures that if a new category 
        # is added, it will automatically be calculated and displayed without changing UI code.
        category_totals = {
            cat: sum(s.category_counts.get(cat.value, 0) for s in stats)
            for cat in CommentCategory
        }
        total_chats = sum(s.total_comments for s in stats)
        
        col_chart1, col_chart2 = st.columns(2)
        
        with col_chart1:
            # Comment type distribution donut chart
            # Why not hardcode category names and totals?
            # Building the dataframe dynamically from the CommentCategory Enum and category_totals 
            # prevents UI discrepancy and makes it future-proof against enum definition updates.
            comment_breakdown = pd.DataFrame({
                "分類": [cat.display_label for cat in CommentCategory],
                "件数": [category_totals[cat] for cat in CommentCategory]
            })
            
            # Donut chart via Altair
            # Why not hardcode chart colors?
            # Syncing the Altair Scale range directly with CommentCategory color_hex properties 
            # maintains a unified theme color palette across both charts and HTML badges.
            donut_comment = alt.Chart(comment_breakdown).mark_arc(innerRadius=60, outerRadius=110).encode(
                theta=alt.Theta(field="件数", type="quantitative"),
                color=alt.Color(
                    field="分類", 
                    type="nominal",
                    scale=alt.Scale(
                        domain=comment_breakdown["分類"].tolist(), 
                        range=[cat.color_hex for cat in CommentCategory]
                    )
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
                
            # Why not hardcode persona names?
            # Creating the mapping from CommentCategory dynamically ensures 
            # that any persona references across the dashboard stay updated.
            p_jp_map = {cat.value: cat.persona_label for cat in CommentCategory}
            
            persona_breakdown = pd.DataFrame({
                "ペルソナ": [p_jp_map.get(k, k) for k in persona_counts.keys()],
                "人数": list(persona_counts.values())
            })
            
            donut_persona = alt.Chart(persona_breakdown).mark_arc(innerRadius=60, outerRadius=110).encode(
                theta=alt.Theta(field="人数", type="quantitative"),
                color=alt.Color(
                    field="ペルソナ", 
                    type="nominal",
                    scale=alt.Scale(domain=persona_breakdown["ペルソナ"].tolist(), 
                    range=[cat.color_hex for cat in CommentCategory])
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
            counts = s.category_counts
            # Why not hardcode table columns?
            # Building rows dynamically from CommentCategory maps UI columns cleanly 
            # and prevents broken displays when schema modifications happen.
            row = {
                "リスナー名": s.listener_username,
                "総コメント数": s.total_comments,
            }
            for cat in CommentCategory:
                row[cat.display_label] = counts.get(cat.value, 0)
            row["ペルソナ種類"] = p_jp_map.get(s.persona_type, s.persona_type)
            stats_data.append(row)
        stats_df = pd.DataFrame(stats_data)
        # Why width="stretch" instead of use_container_width=True?
        # Resolves Streamlit deprecation warnings for dataframes.
        st.dataframe(stats_df, width="stretch")
        
        # Why fetch comment details from merged_timeline_json?
        # Storing all comments in a single merged_timeline_json array allows us to filter details 
        # dynamically at query time, keeping the database schema normalized and lightweight.
        if hasattr(vod, "merged_timeline_json") and vod.merged_timeline_json:
            st.markdown("---")
            st.markdown("#### 💬 コメントの分類詳細")
            
            listener_names = sorted([s.listener_username for s in stats])
            selected_user = st.selectbox("詳細を表示するリスナーを選択", listener_names)
            
            try:
                events = json.loads(vod.merged_timeline_json)
                comment_details = [
                    {
                        "message": ev["text"],
                        "offset_seconds": ev["offset_seconds"],
                        "category": ev.get("category", "other")
                    }
                    for ev in events if ev["type"] == "listener" and ev["name"] == selected_user
                ]
            except Exception:
                comment_details = []
                
            if comment_details:
                # Why not hardcode filter categories?
                # Building option mappings dynamically from CommentCategory ensures 
                # that if categories evolve, filters update automatically without broken keys.
                cat_options = {cat.value: cat.display_label for cat in CommentCategory}
                rev_cat_map = {v: k for k, v in cat_options.items()}
                
                # Filter categories
                selected_cats = st.multiselect(
                    "表示する態度カテゴリ（複数選択可）",
                    options=list(cat_options.values()),
                    default=[]
                )
                
                filter_keys = [rev_cat_map[c] for c in selected_cats]
                # Why check filter_keys?
                # If no filters are selected (empty filter_keys), we show all comments by default.
                # If filters are selected, only show comments that match the OR criteria.
                if filter_keys:
                    filtered_details = [c for c in comment_details if c.get("category") in filter_keys]
                else:
                    filtered_details = comment_details
                
                if filtered_details:
                    # Why not hardcode badge CSS styles?
                    # Generating CSS dynamically from CommentCategory color_hex keeps badge styles 
                    # perfectly in sync with the category's theme color while defining alpha opacities.
                    badge_styles = {
                        cat.value: f"background-color: {cat.color_hex}26; color: {cat.color_hex}; border: 1px solid {cat.color_hex}4d;"
                        for cat in CommentCategory
                    }
                    
                    html_rows = []
                    for c in filtered_details:
                        time_str = format_seconds(c.get("offset_seconds", 0))
                        msg = c.get("message", "")
                        cat = c.get("category", "other")
                        badge_style = badge_styles.get(cat, badge_styles["other"])
                        cat_jp = cat_options.get(cat, "その他")
                        
                        # Why avoid multi-line string indentation?
                        # In Streamlit st.markdown, lines starting with 4 or more spaces 
                        # are treated as indented code blocks, which escapes and prints HTML raw text.
                        # Using explicit string concatenation with no leading spaces avoids code block detection.
                        html_rows.append(
                            '<tr style="border-bottom: 1px solid rgba(255, 255, 255, 0.05);">'
                            f'<td style="padding: 0.6rem 0.8rem; color: #9ca3af; font-family: monospace; font-size: 0.9rem; white-space: nowrap;">🕒 {time_str}</td>'
                            f'<td style="padding: 0.6rem 0.8rem; color: #f3f4f6; text-align: left; font-size: 0.95rem;">{msg}</td>'
                            '<td style="padding: 0.6rem 0.8rem; text-align: right; white-space: nowrap;">'
                            f'<span style="padding: 0.2rem 0.5rem; border-radius: 6px; font-size: 0.78rem; font-weight: 700; {badge_style}">'
                            f'{cat_jp}'
                            '</span>'
                            '</td>'
                            '</tr>'
                        )
                        
                    # Why avoid multi-line string indentation here?
                    # Using flat concatenated strings keeps table elements out of Markdown pre/code block parsing.
                    table_html = (
                        '<table style="width: 100%; border-collapse: collapse; background-color: rgba(30, 41, 59, 0.2); border: 1px solid rgba(255, 255, 255, 0.05); border-radius: 8px; overflow: hidden; margin-top: 1rem;">'
                        '<thead>'
                        '<tr style="background-color: rgba(255, 255, 255, 0.02); border-bottom: 1px solid rgba(255, 255, 255, 0.08);">'
                        '<th style="padding: 0.6rem 0.8rem; text-align: left; color: #9ca3af; font-size: 0.85rem; font-weight: 600; width: 100px;">時間</th>'
                        '<th style="padding: 0.6rem 0.8rem; text-align: left; color: #9ca3af; font-size: 0.85rem; font-weight: 600;">コメント内容</th>'
                        '<th style="padding: 0.6rem 0.8rem; text-align: right; color: #9ca3af; font-size: 0.85rem; font-weight: 600; width: 120px;">判定された態度</th>'
                        '</tr>'
                        '</thead>'
                        '<tbody>'
                        f'{"".join(html_rows)}'
                        '</tbody>'
                        '</table>'
                    )
                    st.markdown(table_html, unsafe_allow_html=True)
                else:
                    st.info("選択された態度カテゴリに該当するコメントはありません。")
            else:
                st.info("このリスナーの個別コメント詳細がありません。")
        else:
            st.info("このアーカイブには個別のコメント分類詳細データが保存されていません（古いデータなどのため）。")
            
            # Check if analysis is currently running
            # Why not check session_state["analysis_runner"]?
            # AnalysisRunner has been deprecated and removed in favor of the synchronous st.status flow,
            # so we check start_analysis state directly to verify execution status and prevent double-triggering.
            is_running = bool(st.session_state.get("start_analysis"))
            
            # Why use a unique key?
            # Streamlit requires unique widget keys when multiple buttons can potentially exist in the DOM.
            if st.button("🔄 このアーカイブを再分析する", disabled=is_running, key="reanalyze_btn", width="stretch"):
                if not api_key:
                    st.error("Gemini API Key を入力してください（サイドバーに入力欄があります）。")
                else:
                    # Reconstruct Twitch VOD URL from stored ID
                    # Why reconstruct?
                    # The VOD URL is needed for the scraper and yt-dlp, but only the ID is stored in the DB.
                    # We utilize get_twitch_vod_url to cleanly format the URL and strip 'v' prefix automatically.
                    vod_url = get_twitch_vod_url(selected_vod_id)
                    st.session_state["start_analysis"] = True
                    st.session_state["analysis_vod_url"] = vod_url
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
                    twitch_url = get_twitch_vod_url(vod.vod_id, t.start_offset_seconds)
                    
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
                    twitch_url = get_twitch_vod_url(vod.vod_id, t.start_offset_seconds)
                    
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
            
            twitch_url = get_twitch_vod_url(vod.vod_id, t.start_offset_seconds)
            
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
        
        # Why check merged_timeline_json?
        # Loading the structured JSON array and rendering it dynamically via HTML table
        # allows users to filter events by speaker or comment category seamlessly in the UI.
        if hasattr(vod, 'merged_timeline_json') and vod.merged_timeline_json:
            try:
                events = json.loads(vod.merged_timeline_json)
                
                # Get unique speakers list
                names = sorted(list(set(ev["name"] for ev in events)))
                
                # Filter widgets
                col1, col2 = st.columns(2)
                with col1:
                    # Why set unique key?
                    # Streamlit throws DuplicateWidgetID error if keys are omitted and same widget exists elsewhere.
                    selected_user = st.selectbox(
                        "表示する発言者を選択",
                        ["すべての発言者"] + names,
                        key="timeline_user_selectbox"
                    )
                with col2:
                    cat_options = {cat.value: cat.display_label for cat in CommentCategory}
                    cat_options["streamer"] = "配信者の発言"
                    # Why set unique key?
                    # Clarifying the key prevents widget ID collisions with other multiselect items.
                    selected_cats = st.multiselect(
                        "表示する分類（複数選択可）",
                        options=list(cat_options.values()),
                        default=[],
                        key="timeline_cat_multiselect"
                    )
                
                rev_cat_map = {v: k for k, v in cat_options.items()}
                filter_keys = [rev_cat_map[c] for c in selected_cats]
                
                filtered_events = []
                for ev in events:
                    if selected_user != "すべての発言者" and ev["name"] != selected_user:
                        continue
                    
                    if ev["type"] == "streamer":
                        cat_key = "streamer"
                    else:
                        cat_key = ev.get("category", "other")
                        
                    # Why check filter_keys?
                    # If no filters are selected, show all events (no filter applied).
                    # Otherwise, only show events that match one of the selected categories (OR logic).
                    if not filter_keys or cat_key in filter_keys:
                        filtered_events.append(ev)
                
                if filtered_events:
                    badge_styles = {
                        cat.value: f"background-color: {cat.color_hex}26; color: {cat.color_hex}; border: 1px solid {cat.color_hex}4d;"
                        for cat in CommentCategory
                    }
                    # Why specify custom style for streamer?
                    # Highlighting the streamer with distinct violet branding theme makes it much easier
                    # to identify conversation turnpoints when scrolling through long chronological threads.
                    badge_styles["streamer"] = "background-color: #a855f726; color: #a855f7; border: 1px solid #a855f74d;"
                    
                    html_rows = []
                    for ev in filtered_events:
                        time_str = format_seconds(ev.get("offset_seconds", 0))
                        name = ev.get("name", "")
                        msg = ev.get("text", "")
                        
                        if ev["type"] == "streamer":
                            cat_key = "streamer"
                            name_style = 'color: #c084fc; font-weight: bold;'
                        else:
                            cat_key = ev.get("category", "other")
                            name_style = 'color: #f3f4f6;'
                            
                        badge_style = badge_styles.get(cat_key, badge_styles["other"])
                        cat_jp = cat_options.get(cat_key, "その他")
                        
                        # Why avoid indentation in HTML strings?
                        # Preventing leading spaces in markdown multiline string prevents Streamlit's parser
                        # from mistakenly escaping tags into standard markdown blockquotes or pre blocks.
                        html_rows.append(
                            '<tr style="border-bottom: 1px solid rgba(255, 255, 255, 0.05);">'
                            f'<td style="padding: 0.6rem 0.8rem; color: #9ca3af; font-family: monospace; font-size: 0.9rem; white-space: nowrap;">🕒 {time_str}</td>'
                            f'<td style="padding: 0.6rem 0.8rem; text-align: left; font-size: 0.95rem; {name_style}">{name}</td>'
                            f'<td style="padding: 0.6rem 0.8rem; color: #f3f4f6; text-align: left; font-size: 0.95rem;">{msg}</td>'
                            '<td style="padding: 0.6rem 0.8rem; text-align: right; white-space: nowrap;">'
                            f'<span style="padding: 0.2rem 0.5rem; border-radius: 6px; font-size: 0.78rem; font-weight: 700; {badge_style}">'
                            f'{cat_jp}'
                            '</span>'
                            '</td>'
                            '</tr>'
                        )
                        
                    # Why render HTML instead of using st.dataframe?
                    # The default dataframe component does not allow custom HTML color-coded badges
                    # or custom text styles for different speakers, reducing user readability.
                    table_html = (
                        '<table style="width: 100%; border-collapse: collapse; background-color: rgba(30, 41, 59, 0.2); border: 1px solid rgba(255, 255, 255, 0.05); border-radius: 8px; overflow: hidden; margin-top: 1rem;">'
                        '<thead>'
                        '<tr style="background-color: rgba(255, 255, 255, 0.02); border-bottom: 1px solid rgba(255, 255, 255, 0.08);">'
                        '<th style="padding: 0.6rem 0.8rem; text-align: left; color: #9ca3af; font-size: 0.85rem; font-weight: 600; width: 100px;">時間</th>'
                        '<th style="padding: 0.6rem 0.8rem; text-align: left; color: #9ca3af; font-size: 0.85rem; font-weight: 600; width: 150px;">発言者</th>'
                        '<th style="padding: 0.6rem 0.8rem; text-align: left; color: #9ca3af; font-size: 0.85rem; font-weight: 600;">コメント内容</th>'
                        '<th style="padding: 0.6rem 0.8rem; text-align: right; color: #9ca3af; font-size: 0.85rem; font-weight: 600; width: 150px;">分類</th>'
                        '</tr>'
                        '</thead>'
                        '<tbody>'
                        f'{"".join(html_rows)}'
                        '</tbody>'
                        '</table>'
                    )
                    st.markdown(table_html, unsafe_allow_html=True)
                else:
                    st.info("選択されたフィルタに一致する発言はありません。")
            except Exception as e:
                st.error(f"統合タイムラインの読み込み中にエラーが発生しました: {e}")
        elif hasattr(vod, 'merged_timeline_text') and vod.merged_timeline_text:
            # Fallback for old database rows (though we drop DB in tests/dev)
            st.text_area(
                label="統合タイムラインログ (コピー・スクロール可能)",
                value=vod.merged_timeline_text,
                height=500,
                disabled=True
            )
        else:
            st.info("このアーカイブには統合タイムラインデータが保存されていません（デモデータ等のため）。")

    # Close DB session safely
    db.remove_session()

if __name__ == "__main__":
    main()
