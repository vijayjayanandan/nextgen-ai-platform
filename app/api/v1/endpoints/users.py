from typing import Any, List, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
import uuid

from app.core.security import get_current_active_superuser, get_current_user, get_password_hash
from app.db.session import get_db
from app.models.user import User
from app.schemas.user import UserCreate, UserUpdate, UserInDB, UserResponse
from app.core.logging import get_logger, audit_log

router = APIRouter()
logger = get_logger(__name__)


@router.get("/me", response_model=UserResponse)
async def read_current_user(
    current_user: UserInDB = Depends(get_current_user),
) -> Any:
    """
    Get current user information.
    """
    return current_user


@router.put("/me", response_model=UserResponse)
async def update_current_user(
    user_update: UserUpdate,
    current_user: UserInDB = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> Any:
    """
    Update current user information.
    """
    try:
        # Get the user from the database
        db_user = await db.get(User, current_user.id)
        if not db_user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        # Update user fields
        update_data = user_update.model_dump(exclude_unset=True)
        
        # Hash the password if it's being updated
        if "password" in update_data:
            update_data["hashed_password"] = get_password_hash(update_data.pop("password"))
        
        # Update user attributes
        for field, value in update_data.items():
            setattr(db_user, field, value)
        
        # Save changes
        db.add(db_user)
        await db.commit()
        await db.refresh(db_user)
        
        # Audit log
        audit_log(
            user_id=str(current_user.id),
            action="update",
            resource_type="user",
            resource_id=str(current_user.id),
            details={"fields_updated": list(update_data.keys())}
        )
        
        return UserResponse.model_validate(db_user)
    except Exception as e:
        logger.error(f"Error updating user: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error updating user: {str(e)}"
        )


@router.get("/", response_model=List[UserResponse])
async def read_users(
    skip: int = 0,
    limit: int = 100,
    current_user: UserInDB = Depends(get_current_active_superuser),
    db: AsyncSession = Depends(get_db),
) -> Any:
    """
    Retrieve users. Only accessible to superusers.
    """
    try:
        result = await db.execute(select(User).offset(skip).limit(limit))
        users = result.scalars().all()
        
        # Audit log
        audit_log(
            user_id=str(current_user.id),
            action="read",
            resource_type="users",
            resource_id="all",
            details={"skip": skip, "limit": limit}
        )
        
        return users
    except Exception as e:
        logger.error(f"Error retrieving users: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving users: {str(e)}"
        )


@router.post("/", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def create_user(
    user_create: UserCreate,
    current_user: UserInDB = Depends(get_current_active_superuser),
    db: AsyncSession = Depends(get_db),
) -> Any:
    """
    Create a new user. Only accessible to superusers.
    """
    try:
        # Check if user with this email already exists
        result = await db.execute(select(User).filter(User.email == user_create.email))
        existing_user = result.scalars().first()
        
        if existing_user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already registered"
            )
        
        # Create new user
        user = User(
            email=user_create.email,
            hashed_password=get_password_hash(user_create.password),
            full_name=user_create.full_name,
            is_active=user_create.is_active,
            is_superuser=user_create.is_superuser,
            roles=user_create.roles
        )
        
        db.add(user)
        await db.commit()
        await db.refresh(user)
        
        # Audit log
        audit_log(
            user_id=str(current_user.id),
            action="create",
            resource_type="user",
            resource_id=str(user.id),
            details={"email": user.email, "roles": user.roles}
        )
        
        return user
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating user: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error creating user: {str(e)}"
        )


@router.get("/{user_id}", response_model=UserResponse)
async def read_user(
    user_id: str,
    current_user: UserInDB = Depends(get_current_active_superuser),
    db: AsyncSession = Depends(get_db),
) -> Any:
    """
    Get a specific user by ID. Only accessible to superusers.
    """
    try:
        user = await db.get(User, uuid.UUID(user_id))
        
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        # Audit log
        audit_log(
            user_id=str(current_user.id),
            action="read",
            resource_type="user",
            resource_id=user_id,
            details={}
        )
        
        return user
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid user ID format"
        )
    except Exception as e:
        logger.error(f"Error retrieving user: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving user: {str(e)}"
        )


@router.put("/{user_id}", response_model=UserResponse)
async def update_user(
    user_id: str,
    user_update: UserUpdate,
    current_user: UserInDB = Depends(get_current_active_superuser),
    db: AsyncSession = Depends(get_db),
) -> Any:
    """
    Update a user. Only accessible to superusers.
    """
    try:
        # Get the user from the database
        user = await db.get(User, uuid.UUID(user_id))
        
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        # Update user fields
        update_data = user_update.model_dump(exclude_unset=True)
        
        # Hash the password if it's being updated
        if "password" in update_data:
            update_data["hashed_password"] = get_password_hash(update_data.pop("password"))
        
        # Update user attributes
        for field, value in update_data.items():
            setattr(user, field, value)
        
        # Save changes
        db.add(user)
        await db.commit()
        await db.refresh(user)
        
        # Audit log
        audit_log(
            user_id=str(current_user.id),
            action="update",
            resource_type="user",
            resource_id=user_id,
            details={"fields_updated": list(update_data.keys())}
        )
        
        return user
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid user ID format"
        )
    except Exception as e:
        logger.error(f"Error updating user: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error updating user: {str(e)}"
        )


@router.delete("/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_user(
    user_id: str,
    current_user: UserInDB = Depends(get_current_active_superuser),
    db: AsyncSession = Depends(get_db),
) -> None:
    """
    Delete a user. Only accessible to superusers.
    """
    try:
        # Get the user from the database
        user = await db.get(User, uuid.UUID(user_id))
        
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        # Delete the user
        await db.delete(user)
        await db.commit()
        
        # Audit log
        audit_log(
            user_id=str(current_user.id),
            action="delete",
            resource_type="user",
            resource_id=user_id,
            details={"email": user.email}
        )
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid user ID format"
        )
    except Exception as e:
        logger.error(f"Error deleting user: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error deleting user: {str(e)}"
        )
