import pytest
import os
import tempfile
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
        duration_seconds=3600
    )
    db.save_vod(vod)
    
    fetched_vod = db.get_vod("test_vod_01")
    assert fetched_vod is not None
    assert fetched_vod.title == "Test Stream Title"


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
