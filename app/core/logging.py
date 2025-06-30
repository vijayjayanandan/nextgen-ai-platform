import logging
import sys
import json
from datetime import datetime
from typing import Dict, Any

from app.core.config import settings


class JSONFormatter(logging.Formatter):
    """
    Custom JSON formatter for structured logging.
    This is helpful for integration with log analytics services.
    """
    def format(self, record: logging.LogRecord) -> str:
        log_record: Dict[str, Any] = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "service": settings.PROJECT_NAME,
            "environment": settings.ENVIRONMENT,
        }
        
        # Include exception info if available
        if record.exc_info:
            log_record["exception"] = self.formatException(record.exc_info)
            
        # Include any extra attributes
        for key, value in record.__dict__.items():
            if key not in {
                "args", "asctime", "created", "exc_info", "exc_text", "filename",
                "funcName", "id", "levelname", "levelno", "lineno", "module",
                "msecs", "message", "msg", "name", "pathname", "process",
                "processName", "relativeCreated", "stack_info", "thread", "threadName"
            }:
                log_record[key] = value
                
        return json.dumps(log_record)


def configure_logging() -> logging.Logger:
    """
    Configure the root logger with appropriate settings based on environment.
    
    Returns:
        Logger instance for the application
    """
    log_level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)
    
    # Determine formatter based on environment
    if settings.ENVIRONMENT == "production":
        formatter = JSONFormatter()
    else:
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
    
    # Configure handlers - both console and file
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    
    file_handler = logging.FileHandler("app.log", mode='w')
    file_handler.setFormatter(formatter)
    
    # Get the logger
    logger = logging.getLogger("ircc_ai_platform")
    logger.setLevel(log_level)
    
    # Remove existing handlers to avoid duplicates during hot reloads
    if logger.hasHandlers():
        logger.handlers.clear()
        
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    # Avoid propagation to the root logger
    logger.propagate = False
    
    return logger


def get_logger(module_name: str) -> logging.Logger:
    """
    Get a named logger for a specific module.
    
    Args:
        module_name: The name of the module requesting a logger
        
    Returns:
        A configured logger instance
    """
    return logging.getLogger(f"ircc_ai_platform.{module_name}")


# Configure audit logging
def audit_log(user_id: str, action: str, resource_type: str, resource_id: str, details: Dict[str, Any] = None) -> None:
    """
    Create an audit log entry.
    
    Args:
        user_id: ID of the user performing the action
        action: The action being performed (e.g., "create", "view", "update", "delete")
        resource_type: Type of resource being acted upon (e.g., "document", "chat")
        resource_id: ID of the specific resource
        details: Additional audit details
    """
    if not settings.ENABLE_AUDIT_LOGGING:
        return
        
    logger = logging.getLogger("ircc_ai_platform.audit")
    
    audit_data = {
        "user_id": user_id,
        "action": action,
        "resource_type": resource_type,
        "resource_id": resource_id,
        "details": details or {},
        "timestamp": datetime.utcnow().isoformat()
    }
    
    logger.info("Audit entry", extra={"audit": audit_data})
    
    # In a production environment, we might want to store these in a database
    # as well for longer retention and better querying capabilities
