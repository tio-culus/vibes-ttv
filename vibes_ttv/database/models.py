from datetime import datetime, timezone
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
    avg_chat_velocity_hour = Column(Float, default=0.0)
    max_chat_velocity_min = Column(Integer, default=0)
    # Why save the raw merged timeline text?
    # Keeping the generated timeline text (combining transcriber text and chats) in the database 
    # allows displaying it on the dashboard without needing to re-run transcription or chat collectors,
    # and lets us show a historical log to the user.
    merged_timeline_text = Column(String, nullable=True)
    
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
    reaction_comments_count = Column(Integer, default=0)
    question_comments_count = Column(Integer, default=0)
    insight_comments_count = Column(Integer, default=0)
    instruction_comments_count = Column(Integer, default=0)
    other_comments_count = Column(Integer, default=0)
    persona_type = Column(String, nullable=False)  # reaction, question, insight, instruction, other
    
    vod = relationship("VOD", back_populates="listener_stats")
