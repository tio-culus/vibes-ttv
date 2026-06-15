import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, scoped_session
from vibes_ttv.database.models import Base, Streamer, VOD, Topic, VODListenerStats

class DBManager:
    # Why not hardcode the DB path?
    # We allow injecting a custom database URI (e.g. sqlite:///:memory: for testing) 
    # to maintain clean separation of concerns and facilitate automated testing.
    def __init__(self, db_uri: str = "sqlite:///vibes_ttv.db"):
        # Why check_same_thread=False?
        # SQLite by default raises exceptions when accessed across multiple threads.
        # Streamlit utilizes a multi-threaded architecture, so this arg is required to prevent thread errors.
        connect_args = {"check_same_thread": False} if "sqlite" in db_uri else {}
        self.engine = create_engine(db_uri, connect_args=connect_args)
        
        # Why not a simple sessionmaker?
        # scoped_session guarantees thread-safety by providing a thread-local session,
        # which is crucial for Streamlit where each user interaction runs in a separate thread.
        self.session_factory = sessionmaker(bind=self.engine)
        self.Session = scoped_session(self.session_factory)
        
    def create_tables(self):
        # Why not migration tools like Alembic?
        # For the initial scope of this application, programmatic creation is simpler.
        # Alembic can be introduced later if the schema requires continuous evolution.
        Base.metadata.create_all(self.engine)
        
        # Why run manual ALTER TABLE migrations?
        # SQLite metadata.create_all() does not automatically add new columns to existing tables.
        # Running a manual column check and ALTER TABLE statement handles lightweight schema updates
        # smoothly, keeping backward compatibility without deleting the user's historical database.
        try:
            from sqlalchemy import text
            with self.engine.connect() as conn:
                # Why use text() helper?
                # SQLAlchemy 2.0+ requires raw SQL strings to be wrapped in the text() construct
                # to be executed, preventing "Not an executable object" query exceptions.
                cursor = conn.execute(text("PRAGMA table_info(vod_listener_stats)"))
                columns = [row[1] for row in cursor.fetchall()]
                if "comment_details_json" not in columns:
                    # Execute raw SQL to dynamically add the text field for serialization
                    conn.execute(text("ALTER TABLE vod_listener_stats ADD COLUMN comment_details_json TEXT"))
                    conn.commit()
        except Exception as e:
            print(f"Migration error: {e}")
        
    def get_session(self):
        return self.Session()
        
    def remove_session(self):
        # Why call remove()?
        # We must clean up thread-local sessions to prevent connection leaks in multi-threaded environments.
        self.Session.remove()

    def get_or_create_streamer(self, streamer_id: str, display_name: str = None) -> Streamer:
        session = self.get_session()
        streamer = session.query(Streamer).filter_by(streamer_id=streamer_id).first()
        if not streamer:
            streamer = Streamer(streamer_id=streamer_id, display_name=display_name or streamer_id)
            session.add(streamer)
            session.commit()
            # Refresh to bind the object to the session correctly
            session.refresh(streamer)
        return streamer

    def save_vod(self, vod: VOD):
        session = self.get_session()
        # Why merge instead of add?
        # Merge handles upsert (update-or-insert) logic seamlessly.
        # If the VOD already exists, it updates it, avoiding unique key violations.
        session.merge(vod)
        session.commit()

    def get_vod(self, vod_id: str) -> VOD:
        session = self.get_session()
        return session.query(VOD).filter_by(vod_id=vod_id).first()

    def save_topics(self, topics: list[Topic]):
        session = self.get_session()
        # Why not commit individually?
        # Batch inserting and single commit reduces SQLite disk write cycles,
        # dramatically speeding up database operations.
        for topic in topics:
            session.add(topic)
        session.commit()

    def save_listener_stats(self, stats_list: list[VODListenerStats]):
        session = self.get_session()
        for stats in stats_list:
            session.merge(stats)
        session.commit()

    def get_all_streamers(self) -> list[Streamer]:
        session = self.get_session()
        # Why not select specific fields?
        # Fetching the entire Streamer entity provides straightforward access to display_name 
        # and streamer_id, which is cleaner than converting tuples to custom dictionary lists.
        return session.query(Streamer).all()

    def get_vods_by_streamer(self, streamer_id: str) -> list[VOD]:
        session = self.get_session()
        # Why filter by streamer_id?
        # Setting a filter constraints query results to the selected streamer,
        # which is required to feed the hierarchical drop-down navigation in the sidebar.
        return session.query(VOD).filter_by(streamer_id=streamer_id).all()
