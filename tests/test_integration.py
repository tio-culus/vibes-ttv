import pytest
from vibes_ttv.collectors.chat_collector import ChatCollector

# Why separate integration tests from unit tests?
# Integration tests execute real network requests to Twitch. 
# While valuable for verifying Twitch's GQL hashes, they can be slow and fail 
# due to network environment issues or Twitch-side rate limits.
# Isolating them in a separate test module prevents them from cluttering standard unit tests.

def test_twitch_chat_download_integration():
    collector = ChatCollector()
    
    # Use a known public VOD URL to run the integration check
    test_vod_url = "https://www.twitch.tv/videos/2795732905"
    
    print("\n[Integration Test] Connecting to Twitch and fetching comments...")
    chat_data = collector.collect_chat(test_vod_url)
    
    # Verify we got a non-empty list of chats
    assert isinstance(chat_data, list), "Fetched chat data must be a list."
    assert len(chat_data) > 0, "Failed to retrieve any chat messages. GQL hash or auth might be blocked."
    
    # Verify that the schema structure maps correctly to database requirements
    first_chat = chat_data[0]
    assert "offset_seconds" in first_chat
    assert "username" in first_chat
    assert "message" in first_chat
    assert "timestamp" in first_chat
    
    print(f"[Integration Test] Success! Retrieved {len(chat_data)} comments.")
    print(f"Sample comment: [{first_chat['offset_seconds']}s] {first_chat['username']}: {first_chat['message']}")


def test_twitch_video_metadata_integration():
    collector = ChatCollector()
    test_vod_url = "https://www.twitch.tv/videos/2795732905"
    
    print("\n[Integration Test] Connecting to Twitch and fetching metadata...")
    metadata = collector.get_video_metadata(test_vod_url)
    
    assert isinstance(metadata, dict), "Fetched metadata must be a dictionary."
    assert metadata.get("vod_id") == "2795732905"
    assert metadata.get("streamer_id") == "alfrea"
    assert "title" in metadata
    assert metadata.get("duration_seconds") > 0
    print(f"[Integration Test] Metadata fetched successfully: {metadata}")
