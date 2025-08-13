from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
from sqlalchemy.orm import declarative_base

DATABASE_URL = "sqlite+aiosqlite:///./chat_database.db"

engine = create_async_engine(
    DATABASE_URL, 
    connect_args={"check_same_thread": False},
)

AsyncSessionLocal = async_sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()