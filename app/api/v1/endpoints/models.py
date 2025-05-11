from typing import Dict, List, Optional, Any
import uuid
from fastapi import APIRouter, Depends, HTTPException, status, Path, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from app.core.security import get_current_user, require_role
from app.schemas.user import UserInDB
from app.db.session import get_db
from app.models.model import Model, ModelVersion, ModelStatus

router = APIRouter()


@router.get("/", response_model=List[Dict[str, Any]])
async def list_models(
    status: Optional[str] = Query(None, description="Filter by model status"),
    type: Optional[str] = Query(None, description="Filter by model type"),
    provider: Optional[str] = Query(None, description="Filter by provider"),
    current_user: UserInDB = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    List all available models.
    """
    try:
        # Build query
        query = select(Model)
        
        # Apply filters
        if status:
            query = query.filter(Model.status == status)
        if type:
            query = query.filter(Model.type == type)
        if provider:
            query = query.filter(Model.provider == provider)
        
        # Execute query
        result = await db.execute(query)
        models = result.scalars().all()
        
        # Format response
        return [
            {
                "id": str(model.id),
                "name": model.name,
                "display_name": model.display_name,
                "description": model.description,
                "type": model.type.value,
                "provider": model.provider.value,
                "deployment_type": model.deployment_type.value,
                "status": model.status.value,
                "max_tokens": model.max_tokens,
                "supports_functions": model.supports_functions,
                "allowed_for_protected_b": model.allowed_for_protected_b
            }
            for model in models
        ]
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error listing models: {str(e)}"
        )


@router.get("/{model_name}", response_model=Dict[str, Any])
async def get_model(
    model_name: str = Path(..., description="Name of the model"),
    current_user: UserInDB = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Get details for a specific model.
    """
    try:
        # Get model
        result = await db.execute(select(Model).filter(Model.name == model_name))
        model = result.scalars().first()
        
        if not model:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Model {model_name} not found"
            )
        
        # Get model versions
        result = await db.execute(
            select(ModelVersion)
            .filter(ModelVersion.model_id == model.id)
            .order_by(ModelVersion.released_at.desc())
        )
        versions = result.scalars().all()
        
        # Format response
        response = {
            "id": str(model.id),
            "name": model.name,
            "display_name": model.display_name,
            "description": model.description,
            "type": model.type.value,
            "provider": model.provider.value,
            "base_model": model.base_model,
            "deployment_type": model.deployment_type.value,
            "endpoint_url": model.endpoint_url,
            "max_tokens": model.max_tokens,
            "supports_functions": model.supports_functions,
            "supported_languages": model.supported_languages,
            "security_classification": model.security_classification,
            "allowed_for_protected_b": model.allowed_for_protected_b,
            "status": model.status.value,
            "default_parameters": model.default_parameters,
            "versions": [
                {
                    "id": str(version.id),
                    "version": version.version,
                    "status": version.status.value,
                    "is_default": version.is_default,
                    "released_at": version.released_at.isoformat() if version.released_at else None,
                    "release_notes": version.release_notes,
                    "accuracy": version.accuracy
                }
                for version in versions
            ]
        }
        
        return response
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting model details: {str(e)}"
        )


@router.post("/", response_model=Dict[str, Any])
async def create_model(
    model_data: Dict[str, Any],
    current_user: UserInDB = Depends(require_role("admin")),
    db: AsyncSession = Depends(get_db)
):
    """
    Create a new model (admin only).
    """
    try:
        # Create model
        model = Model(
            name=model_data["name"],
            display_name=model_data["display_name"],
            description=model_data.get("description"),
            type=model_data["type"],
            provider=model_data["provider"],
            base_model=model_data.get("base_model"),
            deployment_type=model_data["deployment_type"],
            endpoint_url=model_data.get("endpoint_url"),
            api_key_variable=model_data.get("api_key_variable"),
            max_tokens=model_data.get("max_tokens"),
            supports_functions=model_data.get("supports_functions", False),
            supported_languages=model_data.get("supported_languages", ["en"]),
            security_classification=model_data.get("security_classification", "unclassified"),
            allowed_for_protected_b=model_data.get("allowed_for_protected_b", False),
            allowed_roles=model_data.get("allowed_roles", []),
            status=model_data.get("status", ModelStatus.TESTING),
            default_parameters=model_data.get("default_parameters", {})
        )
        
        db.add(model)
        await db.commit()
        await db.refresh(model)
        
        # Create initial version if provided
        if "version" in model_data:
            version = ModelVersion(
                model_id=model.id,
                version=model_data["version"],
                model_uri=model_data.get("model_uri"),
                trained_by=uuid.UUID(str(current_user.id)),
                is_default=True,
                status=model_data.get("status", ModelStatus.TESTING)
            )
            
            db.add(version)
            await db.commit()
        
        return {
            "id": str(model.id),
            "name": model.name,
            "display_name": model.display_name,
            "type": model.type.value,
            "provider": model.provider.value,
            "status": model.status.value,
            "message": "Model created successfully"
        }
    except Exception as e:
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error creating model: {str(e)}"
        )


@router.patch("/{model_name}", response_model=Dict[str, Any])
async def update_model(
    model_data: Dict[str, Any],
    model_name: str = Path(..., description="Name of the model"),
    current_user: UserInDB = Depends(require_role("admin")),
    db: AsyncSession = Depends(get_db)
):
    """
    Update a model (admin only).
    """
    try:
        # Get model
        result = await db.execute(select(Model).filter(Model.name == model_name))
        model = result.scalars().first()
        
        if not model:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Model {model_name} not found"
            )
        
        # Update model fields
        update_fields = [
            "display_name", "description", "type", "provider", "base_model",
            "deployment_type", "endpoint_url", "api_key_variable", "max_tokens",
            "supports_functions", "supported_languages", "security_classification",
            "allowed_for_protected_b", "allowed_roles", "status", "default_parameters"
        ]
        
        for field in update_fields:
            if field in model_data:
                setattr(model, field, model_data[field])
        
        await db.commit()
        
        return {
            "id": str(model.id),
            "name": model.name,
            "display_name": model.display_name,
            "status": model.status.value,
            "message": "Model updated successfully"
        }
    except HTTPException:
        raise
    except Exception as e:
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error updating model: {str(e)}"
        )


@router.post("/{model_name}/versions", response_model=Dict[str, Any])
async def add_model_version(
    version_data: Dict[str, Any],
    model_name: str = Path(..., description="Name of the model"),
    current_user: UserInDB = Depends(require_role("admin")),
    db: AsyncSession = Depends(get_db)
):
    """
    Add a new version to a model (admin only).
    """
    try:
        # Get model
        result = await db.execute(select(Model).filter(Model.name == model_name))
        model = result.scalars().first()
        
        if not model:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Model {model_name} not found"
            )
        
        # Check if version already exists
        result = await db.execute(
            select(ModelVersion)
            .filter(ModelVersion.model_id == model.id)
            .filter(ModelVersion.version == version_data["version"])
        )
        existing_version = result.scalars().first()
        
        if existing_version:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Version {version_data['version']} already exists for model {model_name}"
            )
        
        # Create version
        version = ModelVersion(
            model_id=model.id,
            version=version_data["version"],
            model_uri=version_data.get("model_uri"),
            trained_by=uuid.UUID(str(current_user.id)),
            training_dataset=version_data.get("training_dataset"),
            training_params=version_data.get("training_params", {}),
            evaluation_metrics=version_data.get("evaluation_metrics", {}),
            accuracy=version_data.get("accuracy"),
            is_default=version_data.get("is_default", False),
            release_notes=version_data.get("release_notes"),
            released_at=version_data.get("released_at"),
            status=version_data.get("status", ModelStatus.TESTING)
        )
        
        db.add(version)
        
        # If this is the default version, update other versions
        if version.is_default:
            result = await db.execute(
                select(ModelVersion)
                .filter(ModelVersion.model_id == model.id)
                .filter(ModelVersion.id != version.id)
                .filter(ModelVersion.is_default == True)
            )
            other_default_versions = result.scalars().all()
            
            for other_version in other_default_versions:
                other_version.is_default = False
        
        await db.commit()
        await db.refresh(version)
        
        return {
            "id": str(version.id),
            "model_id": str(model.id),
            "model_name": model.name,
            "version": version.version,
            "is_default": version.is_default,
            "status": version.status.value,
            "message": "Model version added successfully"
        }
    except HTTPException:
        raise
    except Exception as e:
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error adding model version: {str(e)}"
        )