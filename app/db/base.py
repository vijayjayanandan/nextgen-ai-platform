import uuid
from datetime import datetime
from sqlalchemy import Column, DateTime, String, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.ext.declarative import declared_attr

from app.db.session import Base


class TimestampMixin:
    """
    Mixin that adds created_at and updated_at columns to models.
    """
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)


class TableNameMixin:
    """
    Mixin that automatically derives the table name from the class name.
    """
    @declared_attr
    def __tablename__(cls) -> str:
        # Convert CamelCase to snake_case
        name = cls.__name__
        return ''.join(['_' + c.lower() if c.isupper() else c for c in name]).lstrip('_')


class UUIDMixin:
    """
    Mixin that adds a UUID primary key column.
    """
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, unique=True, nullable=False)


class BaseModel(UUIDMixin, TimestampMixin, TableNameMixin, Base):
    """
    Base model class with common fields and behaviors.
    All models should inherit from this class.
    """
    __abstract__ = True

    def __repr__(self):
        return f"<{self.__class__.__name__} {self.id}>"

    def dict(self):
        """Convert model instance to dictionary."""
        result = {}
        for column in self.__table__.columns:
            result[column.name] = getattr(self, column.name)
        return result

    # Explicitly remove metadata to avoid conflicts
    __metadata__ = None


# Import and register all models here to ensure they're discovered by SQLAlchemy
# Note: Import models in app/db/session.py to avoid circular imports
