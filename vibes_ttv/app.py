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

def calculate_chat_velocities(chat_data: list[dict], duration_seconds: int) -> tuple[float, int]:
    if not chat_data:
        return 0.0, 0
    df = pd.DataFrame(chat_data)
    total_chats = len(df)
    hours = max(duration_seconds / 3600.0, 0.01)
    avg_velocity_hour = total_chats / hours
    
    df['minute_bin'] = (df['offset_seconds'] // 60).astype(int)
    chats_per_minute = df.groupby('minute_bin').size()
    max_velocity_min = int(chats_per_minute.max()) if not chats_per_minute.empty else 0
    return avg_velocity_hour, max_velocity_min

def extract_vod_id(url: str) -> str:
    # Why use regex for VOD ID extraction?
    # Twitch VOD URLs consistently contain the numeric ID after '/videos/'.
    # Extracting this locally avoids hitting the network or calling external packages.
    match = re.search(r"/videos/(\d+)", url)
    if match:
        return match.group(1)
    # Fallback to alphanumeric cleaning if format differs slightly
    return re.sub(r'\W+', '', url.split('/')[-1])

# ---------------------------------------------------------
# Mock Data Generator (for Quick Validation)
# ---------------------------------------------------------
def create_mock_data(db: DBManager) -> str:
    # Why not generate dynamically every page load?
    # Keeping mock data persistent in SQLite mirrors real usage 
    # and tests the DB fetch logic correctly on subsequent loads.
    streamer = db.get_or_create_streamer("tio_vtuber", "ティオ Ch.")
    
    vod_id = "mock_vod_001"
    existing_vod = db.get_vod(vod_id)
    if existing_vod:
        return vod_id
        
    vod = VOD(
        vod_id=vod_id,
        streamer_id=streamer.streamer_id,
        title="【大感謝】雑談しながら難関ボス攻略！新アプデも確認していくぞ【ティオ】",
        duration_seconds=5400,
        streamed_at=datetime.now(),
        average_viewers=150,
        avg_chat_velocity_hour=800.0,
        max_chat_velocity_min=75
    )
    db.save_vod(vod)
    
    topics = [
        Topic(
            vod_id=vod_id,
            start_offset_seconds=0,
            end_offset_seconds=900,
            category="daily_news",
            description="最近オープンした美味しいカフェの話と、昨日のニュースについての雑談。ニュースの前置きや前提説明が省かれているため、初見リスナーには少し難解。",
            is_high_context=True
        ),
        Topic(
            vod_id=vod_id,
            start_offset_seconds=900,
            end_offset_seconds=3600,
            category="game",
            description="アプデ後の高難易度ボス『真・魔王』への挑戦。ボスのギミックやアプデ内容を初見リスナー向けに解説しながらプレイ。",
            is_high_context=False
        ),
        Topic(
            vod_id=vod_id,
            start_offset_seconds=3600,
            end_offset_seconds=4500,
            category="past_stream",
            description="先週のホラゲ配信で発生したバグや、その時のリスナーとの身内ネタについてのトーク。過去枠を見ていないと伝わりづらいハイコンテクストな話題。",
            is_high_context=True
        ),
        Topic(
            vod_id=vod_id,
            start_offset_seconds=4500,
            end_offset_seconds=5400,
            category="other",
            description="配信のエンディング、今後の配信スケジュールの告知、及びスパチャ・支援者への感謝読み上げ。",
            is_high_context=False
        ),
    ]
    db.save_topics(topics)
    
    # Generate Mock Listener Stats
    stats_list = []
    # 45 reaction users
    for i in range(45):
        stats_list.append(VODListenerStats(
            vod_id=vod_id,
            listener_username=f"listener_react_{i:02d}",
            total_comments=15,
            reaction_comments_count=12,
            question_comments_count=1,
            insight_comments_count=0,
            instruction_comments_count=0,
            other_comments_count=2,
            persona_type="reaction"
        ))
    # 15 question users
    for i in range(15):
        stats_list.append(VODListenerStats(
            vod_id=vod_id,
            listener_username=f"listener_quest_{i:02d}",
            total_comments=8,
            reaction_comments_count=1,
            question_comments_count=6,
            insight_comments_count=0,
            instruction_comments_count=0,
            other_comments_count=1,
            persona_type="question"
        ))
    # 10 insight users
    for i in range(10):
        stats_list.append(VODListenerStats(
            vod_id=vod_id,
            listener_username=f"listener_insight_{i:02d}",
            total_comments=5,
            reaction_comments_count=0,
            question_comments_count=1,
            insight_comments_count=4,
            instruction_comments_count=0,
            other_comments_count=0,
            persona_type="insight"
        ))
    # 5 instruction users
    for i in range(5):
        stats_list.append(VODListenerStats(
            vod_id=vod_id,
            listener_username=f"listener_inst_{i:02d}",
            total_comments=6,
            reaction_comments_count=1,
            question_comments_count=0,
            insight_comments_count=0,
            instruction_comments_count=4,
            other_comments_count=1,
            persona_type="instruction"
        ))
    # 25 other users
    for i in range(25):
        stats_list.append(VODListenerStats(
            vod_id=vod_id,
            listener_username=f"listener_other_{i:02d}",
            total_comments=10,
            reaction_comments_count=2,
            question_comments_count=1,
            insight_comments_count=0,
            instruction_comments_count=0,
            other_comments_count=7,
            persona_type="other"
        ))
    db.save_listener_stats(stats_list)
    return vod_id

def get_mock_velocity_data() -> pd.DataFrame:
    minutes = list(range(90))
    np.random.seed(42)
    base = 15 + np.sin(np.array(minutes) / 5) * 10
    # Add peak for boss fight (minutes 15 to 60)
    for i in range(15, 60):
        base[i] += 25 + np.random.randint(0, 25)
    base[50] = 75  # absolute peak when boss dies
    velocities = [max(int(v), 2) for v in base]
    return pd.DataFrame({"時間 (分)": minutes, "コメント分速": velocities})

# ---------------------------------------------------------
# Real Pipeline Runner
# ---------------------------------------------------------
def run_real_analysis(db: DBManager, vod_url: str, avg_viewers: int, api_key: str, model_name: str, batch_size: int = 30) -> str:
    status_text = st.empty()
    progress_bar = st.progress(0)
    
    # Why track start_time?
    # Knowing the elapsed time helps reassure the user that the pipeline is active 
    # even when processing heavy steps (like Whisper transcription).
    import time
    start_time = time.time()
    
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
        chat_data = collector.collect_chat(vod_url, progress_callback=progress_callback)
        if not chat_data:
            st.error("チャットログの取得に失敗しました。URLが正しいか、VODが公開されているかご確認ください。")
            return None
            
        # Step 2: Download and extract audio
        progress_callback("🎵 [2/5] VODから音声トラックを抽出中 (yt-dlp)...", 30)
        audio_coll = AudioCollector()
        audio_path = audio_coll.collect_audio(vod_url, progress_callback=progress_callback)
        # Extract VOD ID from audio output filename
        vod_id = os.path.basename(audio_path).replace(".mp3", "")
        
        # Step 3: Transcription using Local Whisper
        progress_callback(f"✍️ [3/5] Whisper ({model_name}) で音声を文字起こし中... (数分かかる場合があります)", 50)
        transcriber = WhisperTranscriber()
        segments = transcriber.transcribe(audio_path, model_name=model_name)
        
        # Step 4: Merge chats and text
        progress_callback("🔗 [4/5] チャットログと音声認識タイムスタンプをアラインメント中...", 75)
        merger = TimelineMerger()
        merged_events = merger.merge(segments, chat_data)
        timeline_txt = merger.format_to_text(merged_events)
        
        # Step 5: AI analysis using Gemini API
        progress_callback("🧠 [5/5] Gemini API で話題のコンテキストを抽出中...", 80)
        
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
        avg_vel, max_vel = calculate_chat_velocities(chat_data, duration)
        
        vod = VOD(
            vod_id=vod_id,
            streamer_id=streamer.streamer_id,
            title=title,
            duration_seconds=duration,
            streamed_at=streamed_at,
            average_viewers=avg_viewers,
            avg_chat_velocity_hour=avg_vel,
            max_chat_velocity_min=max_vel,
            merged_timeline_text=timeline_txt
        )
        db.save_vod(vod)
        
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
        db.save_topics(db_topics)
        
        # Comment persona analysis
        comment_analyzer = CommentAnalyzer(api_key=api_key)
        # Why pass configurable batch_size?
        # Different batch sizes strike different balances between Gemini analysis accuracy and rate limits.
        # Allowing it to be passed from the UI dynamically gives flexibility to the analysis process.
        listener_stats = comment_analyzer.analyze_listeners(chat_data, batch_size=batch_size, progress_callback=progress_callback)
        db_stats = [
            VODListenerStats(
                vod_id=vod_id,
                listener_username=s["username"],
                total_comments=s["total_comments"],
                reaction_comments_count=s["reaction_comments_count"],
                question_comments_count=s["question_comments_count"],
                insight_comments_count=s["insight_comments_count"],
                instruction_comments_count=s["instruction_comments_count"],
                other_comments_count=s["other_comments_count"],
                persona_type=s["persona_type"]
            )
            for s in listener_stats
        ]
        db.save_listener_stats(db_stats)
        
        # Clean up audio file to save disk space
        # Why delete the file?
        # MP3 files from multi-hour streams take up massive disk space.
        # Once transcribed, they are no longer needed, so deletion prevents storage leak.
        if os.path.exists(audio_path):
            os.remove(audio_path)
            
        progress_callback("✨ 分析完了！", 100)
        return vod_id
        
    except Exception as e:
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
    </style>
    """, unsafe_allow_html=True)
    
    # ---------------------------------------------------------
    # Sidebar Configuration
    # ---------------------------------------------------------
    st.sidebar.markdown("### ⚙️ 分析設定")
    
    # Mode selection
    mode = st.sidebar.radio(
        "分析モードを選択してください",
        ["過去の分析結果を閲覧", "デモ・モックデータ", "実際のVODを分析"]
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
                        
    elif mode == "デモ・モックデータ":
        st.sidebar.info("💡 ティオさん、デモモードではRTX 4070やGeminiのAPIキーを使用せず、プリセットのデータで即座にダッシュボードを確認できます。")
        # Why width="stretch" instead of use_container_width=True?
        # In newer Streamlit versions (1.58.0+), use_container_width is deprecated and throws console warnings.
        # Replacing it with width="stretch" eliminates warnings while keeping the responsive button layout.
        if st.sidebar.button("デモデータを読み込む", width="stretch"):
            selected_vod_id = create_mock_data(db)
            st.session_state["vod_id"] = selected_vod_id
            st.sidebar.success("デモデータをデータベースに作成しました！")
    else:
        # Real Mode Config
        # Why not type="password"?
        # Using type="password" causes browser password managers (like Google Password Manager)
        # to misinterpret it as a login credential form and offer auto-generation/save popups.
        # Standard text type provides a better user experience for developer API keys.
        api_key = st.sidebar.text_input("Gemini API Key", value=os.getenv("GEMINI_API_KEY", ""))
        # Why add 'turbo' model option?
        # OpenAI Whisper's 'turbo' model provides a very fast transcription speed (similar to tiny/base) 
        # while keeping high accuracy comparable to large-v3. It is ideal for local high-speed processing.
        whisper_model = st.sidebar.selectbox(
            "Whisperモデルサイズ",
            ["tiny", "base", "small", "medium", "large", "turbo"],
            index=5 # default turbo
        )
        vod_url = st.sidebar.text_input("Twitch VOD URL", value="https://www.twitch.tv/videos/123456789")
        avg_viewers = st.sidebar.number_input("平均同接数 (コメント比率計算用)", min_value=1, value=150)
        # Why not hardcode batch size?
        # Different batch sizes affect the accuracy of the user persona grouping and Gemini API rate usage.
        # Letting the user configure it via a sidebar slider from 10 to 100 provides manual optimization.
        listener_batch_size = st.sidebar.slider(
            "リスナー分析バッチサイズ",
            min_value=10,
            max_value=100,
            value=30,
            step=10,
            help="一度にGemini APIに送信するリスナーの数です。値を小さくすると品質が向上する可能性がありますが、API呼び出し回数が増加します。"
        )
        
        # Why width="stretch" instead of use_container_width=True?
        # In newer Streamlit versions, use_container_width is deprecated. Replacing it with width="stretch" resolves warnings.
        if st.sidebar.button("分析を実行する", width="stretch"):
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
                    with st.spinner("配信データを分析中..."):
                        res_id = run_real_analysis(db, vod_url, avg_viewers, api_key, whisper_model, batch_size=listener_batch_size)
                        if res_id:
                            st.session_state["vod_id"] = res_id
                            st.sidebar.success("分析が完了しました！")
                        
    # Load default session state if exists
    if "vod_id" in st.session_state:
        selected_vod_id = st.session_state["vod_id"]
        
    # Main content rendering
    st.markdown("<h1 class='main-header'>vibes-ttv</h1>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>Twitch配信アーカイブ（VOD）の態度・話題分析ダッシュボード</div>", unsafe_allow_html=True)
    
    if not selected_vod_id:
        st.info("👈 左側のサイドバーから「デモ・モックデータ」を読み込むか、「実際のVOD」を分析してください。")
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
            <div class="metric-title">平均同接数 / コメントリスナー数</div>
            <div class="metric-value">{vod.average_viewers} 人 / {len(stats)} 人</div>
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
            
        # Metric: Comment Ratio
        # Unique chatters / average viewers
        unique_chatters = len(stats)
        comment_ratio = unique_chatters / max(vod.average_viewers, 1)
        
        st.markdown("<hr/>", unsafe_allow_html=True)
        col_metric_r, col_desc_r = st.columns([1, 3])
        with col_metric_r:
            st.metric("コメント比率 (リスナー数/平均同接)", f"{comment_ratio:.1%}")
        with col_desc_r:
            st.markdown(
                "**コメント比率の評価:**\n"
                "- **20%以上**: 視聴者のエンゲージメントが非常に高く、一体感がある配信環境です。\n"
                "- **10%〜20%**: 標準的な配信で、適度なコミュニケーションが行われています。\n"
                "- **10%以下**: 視聴者の多くはROM（見るだけ）であり、配信者主体の進行になっています。"
            )
            
        # Chart: Velocity line chart
        st.markdown("#### コメント速度の推移")
        if selected_vod_id == "mock_vod_001":
            vel_df = get_mock_velocity_data()
            line_chart = alt.Chart(vel_df).mark_line(color="#a855f7", strokeWidth=2).encode(
                x="時間 (分):Q",
                y="コメント分速:Q",
                tooltip=["時間 (分)", "コメント分速"]
            ).properties(height=250)
            # Why width="stretch" instead of use_container_width=True?
            # Resolves Streamlit deprecation warnings for charts.
            st.altair_chart(line_chart, width="stretch")
        else:
            st.info("リアル分析時のコメント分速時系列チャートは、タイムスタンプベースで集計したものが表示されます。")
            
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
        
        # Sort topics chronologically
        sorted_topics = sorted(topics, key=lambda x: x.start_offset_seconds)
        
        for t in sorted_topics:
            time_range = f"{format_seconds(t.start_offset_seconds)} 〜 {format_seconds(t.end_offset_seconds)}"
            cat_label = cat_jp_map.get(t.category, t.category)
            
            hc_badge_html = "<span class='hc-badge'>⚠ ハイコンテクスト</span>" if t.is_high_context else "<span class='lc-badge'>✓ ローコンテクスト</span>"
            hc_class = "topic-row-hc" if t.is_high_context else ""
            
            st.markdown(f"""
            <div class="topic-row {hc_class}">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <strong>🕒 {time_range} | 分類: {cat_label}</strong>
                    {hc_badge_html}
                </div>
                <div style="margin-top: 0.5rem; color: #d1d5db; font-size: 0.95rem;">
                    {t.description}
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
