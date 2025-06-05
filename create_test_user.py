import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.core.config import settings
from app.core.security import get_password_hash
from app.models.user import User  # Correct import path
import uuid
from datetime import datetime

# Create database URL
DATABASE_URL = f"postgresql://{settings.POSTGRES_USER}:{settings.POSTGRES_PASSWORD}@{settings.POSTGRES_SERVER}:{settings.POSTGRES_PORT}/{settings.POSTGRES_DB}"

# Create engine and session
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create database session
db = SessionLocal()

try:
    # Check if user already exists
    existing_user = db.query(User).filter(User.email == "test@example.com").first()
    if existing_user:
        print("User already exists!")
    else:
        # Create test user
        test_user = User(
            id=uuid.uuid4(),
            email="test@example.com",
            full_name="Test User",
            hashed_password=get_password_hash("testpassword123"),
            is_active=True,
            is_superuser=True,
            roles=["admin", "user"],
            preferences={},
            api_keys={},
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        
        db.add(test_user)
        db.commit()
        print("Test user created successfully!")
    
    print("\nLogin credentials:")
    print("Email: test@example.com")
    print("Password: testpassword123")
    
except Exception as e:
    print(f"Error: {e}")
    db.rollback()
finally:
    db.close()
