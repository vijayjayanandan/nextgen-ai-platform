from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, declarative_base
from typing import AsyncGenerator

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

# SQLAlchemy models base class
from sqlalchemy.orm import registry, DeclarativeMeta

# Create a registry
metadata = registry()

# Create a base class using the registry
Base = metadata.generate_base()

# Create async engine
engine = create_async_engine(
    str(settings.SQLALCHEMY_DATABASE_URI),
    echo=settings.DEBUG,
    future=True,
    pool_size=20,
    max_overflow=20,
)

# Create async session factory
AsyncSessionLocal = sessionmaker(
    engine, 
    class_=AsyncSession, 
    expire_on_commit=False, 
    autoflush=False
)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    Dependency that provides an async database session.
    
    Yields:
        AsyncSession: Database session
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception as e:
            await session.rollback()
            logger.error(f"Database session error: {str(e)}")
            raise
        finally:
            await session.close()


async def create_db_and_tables() -> None:
    """
    Create all database tables defined in the models.
    """
    try:
        async with engine.begin() as conn:
            # Import all models to ensure they are registered
            from app.models.user import User
            from app.models.document import Document, DocumentChunk
            from app.models.embedding import Embedding
            from app.models.chat import Conversation, Message
            from app.models.model import Model, ModelVersion
            from app.models.audit import AuditLog

            # Create all tables
            await conn.run_sync(Base.metadata.create_all)
            
        logger.info("Database tables created successfully")
    except Exception as e:
        logger.error(f"Error creating database tables: {str(e)}")
        raise