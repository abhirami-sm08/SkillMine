import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from database.models import Base

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scamshield.db")
DATABASE_URL = f"sqlite:///{DB_PATH}"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False}, echo=False)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def init_db():
    """Creates tables in SQLite database if they do not exist."""
    Base.metadata.create_all(bind=engine)

def get_db():
    """Dependency / context helper for DB sessions."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
