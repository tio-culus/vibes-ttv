# pyrefly: ignore [missing-import]
import pytest
import os
import tempfile
import json
from vibes_ttv.database.db_manager import DBManager
from vibes_ttv.database.models import Streamer, VOD, Topic, VODListenerStats
from vibes_ttv.analyzers.timeline_merger import TimelineMerger
from vibes_ttv.analyzers.comment_analyzer import CommentAnalyzer
from vibes_ttv.collectors.chat_collector import ChatCollector

# Why test database inside memory instead of using local file?
# In-memory SQLite (sqlite:///:memory:) isolates test states,
# runs extremely fast, and leaves no residual temp files on the system, 
# ensuring clean, repeatable test runs.

def test_database_manager():
    db = DBManager("sqlite:///:memory:")
    db.create_tables()
    
    # Test Streamer creation
    streamer = db.get_or_create_streamer("test_user", "Test Streamer")
    assert streamer.streamer_id == "test_user"
    assert streamer.display_name == "Test Streamer"
    
    # Test VOD saving
    vod = VOD(
        vod_id="test_vod_01",
        streamer_id="test_user",
        title="Test Stream Title",
        duration_seconds=3600,
        chat_velocity_json='[{"minute": 0, "count": 10}]',
        chat_collection_time_seconds=10,
        extraction_time_seconds=20,
        transcription_time_seconds=300,
        ai_analysis_time_seconds=40,
        total_analysis_time_seconds=370,
        merged_timeline_json='[{"type": "streamer", "offset_seconds": 10.0, "text": "Hello"}]'
    )
    db.save_vod(vod)
    
    fetched_vod = db.get_vod("test_vod_01")
    assert fetched_vod is not None
    assert fetched_vod.title == "Test Stream Title"
    assert fetched_vod.chat_velocity_json == '[{"minute": 0, "count": 10}]'
    assert fetched_vod.chat_collection_time_seconds == 10
    assert fetched_vod.extraction_time_seconds == 20
    assert fetched_vod.transcription_time_seconds == 300
    assert fetched_vod.ai_analysis_time_seconds == 40
    assert fetched_vod.total_analysis_time_seconds == 370
    assert fetched_vod.merged_timeline_json == '[{"type": "streamer", "offset_seconds": 10.0, "text": "Hello"}]'
    
    # Why verify VODListenerStats fields?
    # Ensuring that VODListenerStats correctly stores basic listener counts and persona type
    # protects the UI features from schema regression or mapping failures in production.
    stats = VODListenerStats(
        vod_id="test_vod_01",
        listener_username="listener_alpha",
        total_comments=5,
        category_counts_json=json.dumps({
            "reaction": 3,
            "question": 1,
            "insight": 0,
            "instruction": 1,
            "other": 0
        }),
        persona_type="reaction"
    )
    db.save_listener_stats([stats])
    
    session = db.get_session()
    fetched_stats = session.query(VODListenerStats).filter_by(vod_id="test_vod_01", listener_username="listener_alpha").first()
    assert fetched_stats is not None
    assert fetched_stats.total_comments == 5
    assert fetched_stats.category_counts.get("reaction") == 3
    db.remove_session()


def test_timeline_merger():
    merger = TimelineMerger()
    whisper_segs = [
        {"start": 10.0, "end": 15.0, "text": "こんにちは！"}
    ]
    chat_data = [
        {"offset_seconds": 12.0, "username": "user1", "message": "きたー！"},
        {"offset_seconds": 5.0, "username": "user2", "message": "待機"}
    ]
    
    merged = merger.merge(whisper_segs, chat_data)
    assert len(merged) == 3
    
    # Sorted order validation
    assert merged[0]["offset_seconds"] == 5.0
    assert merged[1]["type"] == "streamer"
    assert merged[2]["name"] == "user1"
    
    # Format verification
    text = merger.format_to_text(merged)
    assert "[00:00:05] user2: 待機" in text
    assert "[00:00:10] Streamer: こんにちは！" in text
    assert "[00:00:12] user1: きたー！" in text


def test_comment_analyzer_local_filter():
    # Why mock key?
    # We test only local regular expressions and parser rules here without issuing HTTP requests 
    # to Gemini API endpoints. Using a mock key avoids credential requirements.
    analyzer = CommentAnalyzer(api_key="mock_key")
    
    # Simple reactions
    assert analyzer._is_simple_reaction("www") is True
    assert analyzer._is_simple_reaction("草") is True
    assert analyzer._is_simple_reaction("8888") is True
    assert analyzer._is_simple_reaction("👍") is True
    
    # Complex sentences
    assert analyzer._is_simple_reaction("昨日の配信すごく面白かったです！") is False
    assert analyzer._is_simple_reaction("どうしてここで右に進んだんですか？") is False


def test_chat_collector_file_cache():
    collector = ChatCollector()
    mock_chat = [
        {"offset_seconds": 10.0, "username": "user1", "message": "hello", "timestamp": 12345678}
    ]
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as tmp:
        tmp_path = tmp.name
        
    try:
        collector.save_to_file(mock_chat, tmp_path)
        loaded = collector.load_from_file(tmp_path)
        assert len(loaded) == 1
        assert loaded[0]["username"] == "user1"
        assert loaded[0]["message"] == "hello"
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def test_chat_collector_url_parsing():
    # Why verify URL parsing with optional 'v' prefix?
    # Ensuring that the ChatCollector correctly parses VOD IDs from URLs 
    # with or without a 'v' prefix prevents pipeline crashes on legacy IDs.
    collector = ChatCollector()
    
    # Standard URL
    assert collector._extract_vod_id("https://www.twitch.tv/videos/123456789") == "123456789"
    # URL with 'v' prefix in ID
    assert collector._extract_vod_id("https://www.twitch.tv/videos/v2786816848") == "2786816848"
    # Fallback/alternative format
    assert collector._extract_vod_id("https://www.twitch.tv/v/987654321") == "987654321"
    # Invalid URL should raise ValueError
    with pytest.raises(ValueError):
        collector._extract_vod_id("https://www.twitch.tv/invalid_url")


def test_calculate_chat_velocities():
    # Why import locally?
    # Importing locally in test functions isolates module loading errors 
    # and ensures that test cases remain modular.
    from vibes_ttv.app import calculate_chat_velocities
    chat_data = [
        {"offset_seconds": 10.0, "username": "user1", "message": "hello"},
        {"offset_seconds": 70.0, "username": "user2", "message": "world"},
        {"offset_seconds": 130.0, "username": "user1", "message": "again"}
    ]
    avg_vel, max_vel, vel_json = calculate_chat_velocities(chat_data, 180)
    assert avg_vel == 60.0  # 3 chats / 0.05 hours = 60.0
    assert max_vel == 1     # max chats in a single minute bin is 1
    
    import json
    parsed = json.loads(vel_json)
    # duration is 180 seconds, so max_minute is 3. range(4) -> 0, 1, 2, 3
    assert len(parsed) == 4
    assert parsed[0] == {"minute": 0, "count": 1}
    assert parsed[1] == {"minute": 1, "count": 1}
    assert parsed[2] == {"minute": 2, "count": 1}
    assert parsed[3] == {"minute": 3, "count": 0}


def test_timeline_merger_complex():
    # Why not test with live Whisper model?
    # Feeding a complex pre-defined list of segments to the TimelineMerger allows us to verify 
    # the chronological merging logic and formatting robustness directly without actual model 
    # load overhead. This is faster and isolates merging correctness from transcription variables.
    from vibes_ttv.analyzers.timeline_merger import TimelineMerger
    merger = TimelineMerger()
    
    # Dummy transcription data representing streamer talk segments
    whisper_segs = [
        {"start": 30.0, "end": 35.0, "text": "配信開始しました"},
        {"start": 120.0, "end": 125.0, "text": "PoE2おもしろいね"},
        {"start": 300.0, "end": 305.0, "text": "ご視聴ありがとうございました"}
    ]
    
    # Chat data from viewers
    chat_data = [
        {"offset_seconds": 10.0, "username": "user1", "message": "待機画面"},
        {"offset_seconds": 32.0, "username": "user2", "message": "きた！"},
        {"offset_seconds": 122.0, "username": "user1", "message": "神ゲー"},
        {"offset_seconds": 310.0, "username": "user3", "message": "おつかれさまでした"}
    ]
    
    merged = merger.merge(whisper_segs, chat_data)
    
    # Event count: 3 segments + 4 chats = 7 events
    assert len(merged) == 7
    
    # Chronological sort order validation
    offsets = [event["offset_seconds"] for event in merged]
    assert offsets == sorted(offsets)
    
    # Format and content verification
    assert merged[0]["type"] == "listener"
    assert merged[0]["text"] == "待機画面"
    
    assert merged[1]["type"] == "streamer"
    assert merged[1]["text"] == "配信開始しました"
    
    assert merged[-1]["type"] == "listener"
    assert merged[-1]["text"] == "おつかれさまでした"
    
    text = merger.format_to_text(merged)
    assert "[00:00:10] user1: 待機画面" in text
    assert "[00:00:30] Streamer: 配信開始しました" in text
    assert "[00:02:00] Streamer: PoE2おもしろいね" in text
    assert "[00:05:00] Streamer: ご視聴ありがとうございました" in text
    assert "[00:05:10] user3: おつかれさまでした" in text


def test_atomic_transaction_behavior():
    # Why test transaction isolation?
    # Confirming that database changes are rolled back completely upon failure 
    # guarantees that we never leave a broken or partially deleted state in SQLite 
    # if a Twitch download or Gemini API call fails during re-analysis.
    db = DBManager("sqlite:///:memory:")
    db.create_tables()
    
    # Setup legacy dataset
    db.get_or_create_streamer("streamer_X", "Streamer X")
    vod = VOD(
        vod_id="vod_X",
        streamer_id="streamer_X",
        title="Legacy Broadcast Title",
        duration_seconds=1000,
    )
    db.save_vod(vod)
    
    legacy_topic = Topic(
        vod_id="vod_X",
        start_offset_seconds=10,
        end_offset_seconds=20,
        category="other",
        description="Legacy Topic",
        is_high_context=False
    )
    db.save_topics([legacy_topic])
    
    legacy_stats = VODListenerStats(
        vod_id="vod_X",
        listener_username="listener_X",
        total_comments=10,
        category_counts_json=json.dumps({
            "reaction": 10,
            "question": 0,
            "insight": 0,
            "instruction": 0,
            "other": 0
        }),
        persona_type="reaction"
    )
    db.save_listener_stats([legacy_stats])
    
    # Verify legacy data exists
    session = db.get_session()
    assert session.query(Topic).filter_by(vod_id="vod_X").count() == 1
    assert session.query(VODListenerStats).filter_by(vod_id="vod_X").count() == 1
    db.remove_session()
    
    # Setup new dataset objects
    new_vod = VOD(
        vod_id="vod_X",
        streamer_id="streamer_X",
        title="New Upgraded Title",
        duration_seconds=2000,
    )
    
    new_topic = Topic(
        vod_id="vod_X",
        start_offset_seconds=50,
        end_offset_seconds=60,
        category="game",
        description="New Topic",
        is_high_context=True
    )
    
    new_stats = VODListenerStats(
        vod_id="vod_X",
        listener_username="listener_Y",
        total_comments=5,
        category_counts_json=json.dumps({
            "reaction": 0,
            "question": 5,
            "insight": 0,
            "instruction": 0,
            "other": 0
        }),
        persona_type="question"
    )
    
    # Run transaction that FAILS
    session_db = db.get_session()
    try:
        session_db.query(Topic).filter_by(vod_id="vod_X").delete()
        session_db.query(VODListenerStats).filter_by(vod_id="vod_X").delete()
        
        session_db.merge(new_vod)
        session_db.add(new_topic)
        session_db.merge(new_stats)
        
        # Trigger mock exception before commit
        raise ValueError("Simulated network or API error during compilation")
        
        session_db.commit()
    except Exception:
        session_db.rollback()
    finally:
        db.remove_session()
        
    # Verify legacy data is STILL intact and untouched
    session = db.get_session()
    fetched_vod = session.query(VOD).filter_by(vod_id="vod_X").first()
    assert fetched_vod.title == "Legacy Broadcast Title"  # Not upgraded
    assert session.query(Topic).filter_by(vod_id="vod_X").count() == 1
    assert session.query(Topic).filter_by(vod_id="vod_X").first().description == "Legacy Topic"
    assert session.query(VODListenerStats).filter_by(vod_id="vod_X").count() == 1
    assert session.query(VODListenerStats).filter_by(vod_id="vod_X").first().listener_username == "listener_X"
    db.remove_session()
    
    # Run transaction that SUCCEEDS
    session_db = db.get_session()
    try:
        session_db.query(Topic).filter_by(vod_id="vod_X").delete()
        session_db.query(VODListenerStats).filter_by(vod_id="vod_X").delete()
        
        session_db.merge(new_vod)
        session_db.add(new_topic)
        session_db.merge(new_stats)
        
        session_db.commit()
    except Exception:
        session_db.rollback()
    finally:
        db.remove_session()
        
    # Verify data has been correctly swapped
    session = db.get_session()
    fetched_vod = session.query(VOD).filter_by(vod_id="vod_X").first()
    assert fetched_vod.title == "New Upgraded Title"  # Upgraded!
    assert session.query(Topic).filter_by(vod_id="vod_X").count() == 1
    assert session.query(Topic).filter_by(vod_id="vod_X").first().description == "New Topic"
    assert session.query(VODListenerStats).filter_by(vod_id="vod_X").count() == 1
    assert session.query(VODListenerStats).filter_by(vod_id="vod_X").first().listener_username == "listener_Y"
    db.remove_session()


def test_comment_analyzer_sliced_context():
    # Why mock Gemini client for generate_content?
    # We want to isolate the timeline-slicing parser, local pre-classification, 
    # and persona tie-breaker logic from actual network calls. 
    # Mocking the generate_content call to return a validated SliceClassificationResponse
    # makes the test deterministic, robust, and fast.
    from unittest.mock import MagicMock
    from vibes_ttv.analyzers.comment_analyzer import CommentAnalyzer, SliceClassificationResponse, LineClassification
    
    analyzer = CommentAnalyzer(api_key="mock_key")
    
    # Mocking the client's generate_content call
    mock_response = MagicMock()
    mock_response.parsed = SliceClassificationResponse(
        results=[
            LineClassification(line_id="L2", category="insight")
        ]
    )
    analyzer.client.models.generate_content = MagicMock(return_value=mock_response)
    
    # Setup chat_data and merged_events
    # Line 0: Streamer comment (context only, ignored for client classification)
    # Line 1: 'www' is a simple reaction (pre-classified locally, no Gemini call)
    # Line 2: 'このボスは火属性に弱いと思います' is a complex comment (Gemini classified as 'insight')
    merged_events = [
        {"type": "streamer", "offset_seconds": 10.0, "text": "ゲームを開始します"},
        {"type": "listener", "name": "user_a", "offset_seconds": 12.0, "text": "www"},
        {"type": "listener", "name": "user_a", "offset_seconds": 15.0, "text": "このボスは火属性に弱いと思います"},
    ]
    
    results = analyzer.analyze_listeners(merged_events=merged_events)
    
    # Assertions
    assert len(results) == 1
    stats = results[0]
    assert stats["username"] == "user_a"
    assert stats["total_comments"] == 2
    assert stats["category_counts"]["reaction"] == 1
    assert stats["category_counts"]["insight"] == 1
    assert stats["category_counts"].get("other", 0) == 0
    
    # Tie-breaker logic: 'insight' (1) vs 'reaction' (1) -> 'insight' should win
    assert stats["persona_type"] == "insight"
    
    # Check detail content order
    details = stats["comment_details"]
    assert len(details) == 2
    assert details[0]["message"] == "www"
    assert details[0]["category"] == "reaction"
    assert details[1]["message"] == "このボスは火属性に弱いと思います"
    assert details[1]["category"] == "insight"


def test_get_twitch_vod_url():
    # Why import locally?
    # Keeping imports scoped to the test cases prevents cluttering the module-level namespace.
    from vibes_ttv.app import get_twitch_vod_url
    
    # Standard numerical ID
    assert get_twitch_vod_url("2786816848") == "https://www.twitch.tv/videos/2786816848"
    # ID with 'v' prefix
    assert get_twitch_vod_url("v2786816848") == "https://www.twitch.tv/videos/2786816848"
    # With offset_seconds
    assert get_twitch_vod_url("v2786816848", 3661) == "https://www.twitch.tv/videos/2786816848?t=1h1m1s"


def test_comment_serialization_flow():
    import json
    from unittest.mock import MagicMock
    from vibes_ttv.database.db_manager import DBManager
    from vibes_ttv.database.models import VOD
    from vibes_ttv.analyzers.comment_analyzer import CommentAnalyzer, SliceClassificationResponse, LineClassification
    from vibes_ttv.analyzers.timeline_merger import TimelineMerger
    
    # 1. DB setup
    db = DBManager("sqlite:///:memory:")
    db.create_tables()
    
    # 2. Setup mock CommentAnalyzer and output
    analyzer = CommentAnalyzer(api_key="mock_key")
    mock_response = MagicMock()
    mock_response.parsed = SliceClassificationResponse(
        results=[
            LineClassification(line_id="L2", category="insight")
        ]
    )
    analyzer.client.models.generate_content = MagicMock(return_value=mock_response)
    
    # 3. Setup events
    merged_events = [
        {"type": "streamer", "offset_seconds": 10.0, "text": "配信開始"},
        {"type": "listener", "name": "user_a", "offset_seconds": 12.0, "text": "www"},
        {"type": "listener", "name": "user_a", "offset_seconds": 15.0, "text": "ここは火属性ですね"},
    ]
    
    # 4. Run analyzer
    results = analyzer.analyze_listeners(merged_events=merged_events)
    
    # Verify categories are assigned in-place
    assert merged_events[1]["category"] == "reaction"
    assert merged_events[2]["category"] == "insight"
    
    # 5. Format to text with categories
    merger = TimelineMerger()
    formatted = merger.format_to_text(merged_events, show_categories=True)
    assert "[00:00:12] [reaction] user_a: www" in formatted
    assert "[00:00:15] [insight] user_a: ここは火属性ですね" in formatted
    
    # 6. JSON serialization check
    serialized = json.dumps(merged_events, ensure_ascii=False)
    assert isinstance(serialized, str)
    
    # 7. DB Save and Load check
    db.get_or_create_streamer("streamer_test", "Tester")
    vod = VOD(
        vod_id="vod_test_serial",
        streamer_id="streamer_test",
        title="Test Serialization",
        duration_seconds=100,
        merged_timeline_json=serialized
    )
    db.save_vod(vod)
    
    fetched = db.get_vod("vod_test_serial")
    assert fetched is not None
    assert fetched.merged_timeline_json == serialized
    
    # 8. Dynamic filtering check
    loaded_events = json.loads(fetched.merged_timeline_json)
    user_comments = [
        {"message": ev["text"], "offset_seconds": ev["offset_seconds"], "category": ev.get("category", "other")}
        for ev in loaded_events if ev["type"] == "listener" and ev["name"] == "user_a"
    ]
    assert len(user_comments) == 2
    assert user_comments[0]["category"] == "reaction"
    assert user_comments[1]["category"] == "insight"
