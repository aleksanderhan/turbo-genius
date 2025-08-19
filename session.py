from sqlalchemy import create_engine, Column, Integer, String, Text, LargeBinary, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship

# Database setup
SQLALCHEMY_DATABASE_URL = "sqlite:///./sessions.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class SessionDB(Base):
    __tablename__ = "sessions"
    id = Column(Integer, primary_key=True, autoincrement=True, index=True)
    title = Column(String, index=True, nullable=True)
    messages = Column(Text, nullable=True)
    images = relationship("SessionImageDB", back_populates="session")  # New relationship

class SessionImageDB(Base):
    __tablename__ = "session_images"
    id = Column(Integer, primary_key=True, autoincrement=True, index=True)
    session_id = Column(Integer, ForeignKey("sessions.id"))
    image = Column(LargeBinary, nullable=False)
    session = relationship("SessionDB", back_populates="images")

Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

class Session:
    def __init__(self, session_id=None, title=None, messages=None):
        self.id = session_id
        self.title = title
        self.messages = messages or []

    def add_user_message(self, message):
        self.messages.append({"role": "user", "content": message})

    def add_assistant_message(self, message):
        self.messages.append({"role": "assistant", "content": message})

    def add_system_message(self, message):
        self.messages.append({"role": "system", "content": message})

    def get_messages(self):
        return [{"role": message["role"], "content": message["content"]} for message in self.messages if message["role"] != "image"]
    
    def truncate_messages(self):
        self.messages = self.messages[2:]

class SessionManager:
    def __init__(self):
        pass

    def get_session(self, session_id, db):
        session_db = db.query(SessionDB).filter(SessionDB.id == session_id).first()
        if session_db:
            session = Session(session_db.id, session_db.title, eval(session_db.messages))
            return session
        else:
            raise KeyError(f"Session {session_id} not found")

    def get_session_list(self, db):
        return [{"id": session.id, "title": session.title} for session in db.query(SessionDB).all()]

    def get_new_session(self, db, system_message="You are a helpful assistant."):
        session_db = SessionDB(messages="[{'role': 'system', 'content': '" + system_message + "'}]")
        db.add(session_db)
        db.commit()
        db.refresh(session_db)  # Refresh to get the auto-generated ID
        session = self.get_session(session_db.id, db)
        #session.add_system_message(system_message)
        #self.save_session(session, db)
        return session

    def remove_session(self, session_id, db):
        session_db = db.query(SessionDB).filter(SessionDB.id == session_id).first()
        if session_db:
            db.delete(session_db)
            db.commit()

    def save_session(self, session, db):
        session_db = db.query(SessionDB).filter(SessionDB.id == session.id).first()
        if session_db:
            session_db.messages = str(session.messages)
            db.add(session_db)
            db.commit()

