from sqlalchemy import Column, String, ForeignKey, Text, Boolean, DateTime, Enum, Float, Integer
from sqlalchemy.dialects.postgresql import JSONB, UUID, ARRAY
from sqlalchemy.orm import relationship
import enum

from app.db.base import BaseModel


class ModelType(str, enum.Enum):
    """Enum for model types."""
    LLM = "llm"
    EMBEDDING = "embedding"
    CLASSIFIER = "classifier"
    CUSTOM = "custom"


class ModelProvider(str, enum.Enum):
    """Enum for model providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    AZURE = "azure"
    ON_PREM = "on_prem"
    OTHER = "other"


class ModelDeploymentType(str, enum.Enum):
    """Enum for model deployment types."""
    API = "api"
    ON_PREM = "on_prem"
    HYBRID = "hybrid"


class ModelStatus(str, enum.Enum):
    """Enum for model status."""
    ACTIVE = "active"
    INACTIVE = "inactive"
    DEPRECATED = "deprecated"
    TESTING = "testing"


class Model(BaseModel):
    """
    Model representing an AI model in the platform.
    """
    # Basic model information
    name = Column(String, nullable=False, unique=True, index=True)
    display_name = Column(String, nullable=False)
    description = Column(Text, nullable=True)
    
    # Model type and provider
    type = Column(Enum(ModelType), nullable=False, index=True)
    provider = Column(Enum(ModelProvider), nullable=False, index=True)
    base_model = Column(String, nullable=True)  # For fine-tuned models
    
    # Deployment information
    deployment_type = Column(Enum(ModelDeploymentType), nullable=False)
    endpoint_url = Column(String, nullable=True)
    api_key_variable = Column(String, nullable=True)  # Environment variable name for the API key
    
    # Model capabilities and limitations
    max_tokens = Column(Integer, nullable=True)
    supports_functions = Column(Boolean, default=False, nullable=False)
    supported_languages = Column(ARRAY(String), default=["en"], nullable=False)
    
    # Security and access control
    security_classification = Column(String, nullable=False, default="unclassified")
    allowed_for_protected_b = Column(Boolean, default=False, nullable=False)
    allowed_roles = Column(JSONB, default=[], nullable=False)
    
    # Status
    status = Column(Enum(ModelStatus), default=ModelStatus.TESTING, nullable=False)
    
    # Configuration
    default_parameters = Column(JSONB, default={}, nullable=False)
    
    # Relationships
    versions = relationship("ModelVersion", back_populates="model", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<Model {self.id}: {self.name}>"


class ModelVersion(BaseModel):
    """
    ModelVersion representing a specific version of a model.
    """
    # Foreign key to the parent model
    model_id = Column(UUID(as_uuid=True), ForeignKey("model.id", ondelete="CASCADE"), nullable=False, index=True)
    
    # Version information
    version = Column(String, nullable=False)
    
    # Model file or reference
    model_uri = Column(String, nullable=True)
    
    # Training information
    trained_by = Column(UUID(as_uuid=True), ForeignKey("user.id"), nullable=True)
    training_dataset = Column(String, nullable=True)
    training_params = Column(JSONB, default={}, nullable=False)
    
    # Evaluation metrics
    evaluation_metrics = Column(JSONB, default={}, nullable=False)
    accuracy = Column(Float, nullable=True)
    
    # Release information
    is_default = Column(Boolean, default=False, nullable=False)
    release_notes = Column(Text, nullable=True)
    released_at = Column(DateTime, nullable=True)
    
    # Status
    status = Column(Enum(ModelStatus), default=ModelStatus.TESTING, nullable=False)
    
    # Relationships
    model = relationship("Model", back_populates="versions")
    
    def __repr__(self):
        return f"<ModelVersion {self.id}: Model {self.model_id}, Version {self.version}>"
