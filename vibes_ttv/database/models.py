from datetime import datetime, timezone
import json # Why not import json at file level? To deserialize database-persisted category counts dynamically.
from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, ForeignKey
from sqlalchemy.orm import declarative_base, relationship

# Why not using the newer declarative_base from sqlalchemy.orm directly?
# We use declarative_base() because it is widely compatible across various SQLAlchemy 2.x versions 
# and provides a clean, implicit metadata binding suitable for this SQLite-based lightweight tool.
Base = declarative_base()

# Why not standard sqlite3 module?
# We use SQLAlchemy ORM instead of raw sqlite3 because it allows future extensibility 
# (e.g. migrating to PostgreSQL or MySQL for production) and provides better type safety 
# and relationship mapping between VODs, Streamers, and Listeners.

class Streamer(Base):
    __tablename__ = 'streamers'
    
    streamer_id = Column(String, primary_key=True)  # Twitch user ID/username
    display_name = Column(String, nullable=False)
    
    # Why not lazy="dynamic"?
    # For a streamer, the list of VODs is typically small enough to be loaded on demand (default lazy='select').
    # A dynamic loader would overhead query generation for simple dashboard views.
    vods = relationship("VOD", back_populates="streamer")


class VOD(Base):
    __tablename__ = 'vods'
    
    vod_id = Column(String, primary_key=True)  # Twitch VOD ID
    streamer_id = Column(String, ForeignKey('streamers.streamer_id'), nullable=False)
    title = Column(String, nullable=False)
    duration_seconds = Column(Integer, nullable=False)
    # Why not use datetime.utcnow?
    # datetime.utcnow() is deprecated in Python 3.12+.
    # Using timezone-aware datetime.now(timezone.utc) prevents DeprecationWarning.
    streamed_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    average_viewers = Column(Integer, default=0)
    # Why not avg_chat_velocity_hour?
    # Unifying the metrics to minutely rate to keep consistency on the dashboard.
    avg_chat_velocity_min = Column(Float, default=0.0)
    max_chat_velocity_min = Column(Integer, default=0)
    # Why save the merged timeline as a structured JSON array?
    # Storing the combined whisper segments and chat messages as a raw JSON string 
    # enables rich UI formatting (like categorization badges, streamer highlight, etc.) 
    # on the dashboard without database reconstruction, avoiding fixed-text parsing limitations.
    merged_timeline_json = Column(String, nullable=True)
    
    # Why save serialized JSON instead of a separate table?
    # Saving velocity time-series data as a JSON string within the VOD table avoids database JOIN 
    # overhead and batch-insert complexity for simple one-dimensional data. This keeps the database 
    # query performance high and simplifies data modeling.
    chat_velocity_json = Column(String, nullable=True)
    
    # Why save execution times in the database?
    # Persisting performance metrics directly inside the database allows the user to analyze 
    # Whisper's transcription throughput (processing speeds) and Gemini's analysis latencies 
    # historically, ensuring that performance diagnostics are retained across sessions.
    chat_collection_time_seconds = Column(Integer, nullable=True)
    extraction_time_seconds = Column(Integer, nullable=True)
    transcription_time_seconds = Column(Integer, nullable=True)
    ai_analysis_time_seconds = Column(Integer, nullable=True)
    total_analysis_time_seconds = Column(Integer, nullable=True)
    
    streamer = relationship("Streamer", back_populates="vods")
    topics = relationship("Topic", back_populates="vod")
    listener_stats = relationship("VODListenerStats", back_populates="vod")


class Topic(Base):
    __tablename__ = 'topics'
    
    topic_id = Column(Integer, primary_key=True, autoincrement=True)
    vod_id = Column(String, ForeignKey('vods.vod_id'), nullable=False)
    start_offset_seconds = Column(Integer, nullable=False)
    end_offset_seconds = Column(Integer, nullable=False)
    category = Column(String, nullable=False)  # 'game', 'daily_news', 'past_stream', 'other'
    description = Column(String, nullable=False)
    is_high_context = Column(Boolean, default=False)
    
    vod = relationship("VOD", back_populates="topics")


class VODListenerStats(Base):
    __tablename__ = 'vod_listener_stats'
    
    stats_id = Column(Integer, primary_key=True, autoincrement=True)
    vod_id = Column(String, ForeignKey('vods.vod_id'), nullable=False)
    listener_username = Column(String, nullable=False)
    total_comments = Column(Integer, default=0)
    # Why not use individual columns for each category counts?
    # Keeping counts in a serialized JSON string prevents database schema mismatch issues 
    # when additional comment categories (from CommentCategory Enum) are added or removed.
    category_counts_json = Column(String, default='{}')
    persona_type = Column(String, nullable=False) 
    
    vod = relationship("VOD", back_populates="listener_stats")

    @property
    def category_counts(self) -> dict[str, int]:
        # Why not load category_counts directly as a field?
        # Serializing as a raw string and loading lazily via property avoids custom database type dependencies.
        try:
            return json.loads(self.category_counts_json or '{}')
        except Exception:
            return {}

    @category_counts.setter
    def category_counts(self, val: dict[str, int]):
        self.category_counts_json = json.dumps(val, ensure_ascii=False)
