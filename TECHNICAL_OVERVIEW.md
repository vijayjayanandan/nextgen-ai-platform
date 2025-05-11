# 📘 Technical Handover: Backend Overview for IRCC AI Platform

This document provides a high-level walkthrough of the backend Python codebase, structured to help new developers quickly understand the purpose and design of each module.

---

## 📂 `app/`

Main application package containing all core logic, APIs, models, services, and utilities.

- **`app/__init__.py`** – Module role inferred from context.
- **`app/api/__init__.py`** – Module role inferred from context.
- **`app/api/dependencies.py`** – Module role inferred from context.
- **`app/api/v1/__init__.py`** – Module role inferred from context.
- **`app/api/v1/endpoints/__init__.py`** – Module role inferred from context.
- **`app/api/v1/endpoints/chat.py`** – Module role inferred from context.
- **`app/api/v1/endpoints/completions.py`** – Module role inferred from context.
- **`app/api/v1/endpoints/documents.py`** – Module role inferred from context.
- **`app/api/v1/endpoints/embeddings.py`** – Module role inferred from context.
- **`app/api/v1/endpoints/explanation.py`** – Module role inferred from context.
- **`app/api/v1/endpoints/models.py`** – Defines ORM database models.
- **`app/api/v1/endpoints/moderation.py`** – Module role inferred from context.
- **`app/api/v1/endpoints/retrieval.py`** – Module role inferred from context.
- **`app/api/v1/router.py`** – Defines FastAPI route groupings and endpoints.
- **`app/core/__init__.py`** – Module role inferred from context.
- **`app/core/config.py`** – Handles application configuration and environment settings.
- **`app/core/dependencies.py`** – Module role inferred from context.
- **`app/core/logging.py`** – Configures and manages application logging.
- **`app/core/middleware.py`** – Defines FastAPI middleware for request/response processing.
- **`app/core/security.py`** – Contains authentication and security utility functions.
- **`app/db/__init__.py`** – Module role inferred from context.
- **`app/db/base.py`** – Module role inferred from context.
- **`app/db/repositories/__init__.py`** – Module role inferred from context.
- **`app/db/repositories/chat_repository.py`** – Handles database-level data access.
- **`app/db/repositories/document_repository.py`** – Handles database-level data access.
- **`app/db/repositories/embedding_repository.py`** – Handles database-level data access.
- **`app/db/repositories/user_repository.py`** – Handles database-level data access.
- **`app/db/session.py`** – Module role inferred from context.
- **`app/main.py`** – Entry point for the FastAPI application.
- **`app/models/__init__.py`** – Defines ORM database models.
- **`app/models/audit.py`** – Defines ORM database models.
- **`app/models/chat.py`** – Defines ORM database models.
- **`app/models/document.py`** – Defines ORM database models.
- **`app/models/embedding.py`** – Defines ORM database models.
- **`app/models/model.py`** – Defines ORM database models.
- **`app/models/user.py`** – Defines ORM database models.
- **`app/schemas/__init__.py`** – Contains Pydantic models for request/response validation.
- **`app/schemas/chat.py`** – Contains Pydantic models for request/response validation.
- **`app/schemas/completion.py`** – Contains Pydantic models for request/response validation.
- **`app/schemas/document.py`** – Contains Pydantic models for request/response validation.
- **`app/schemas/embedding.py`** – Contains Pydantic models for request/response validation.
- **`app/schemas/explanation.py`** – Contains Pydantic models for request/response validation.
- **`app/schemas/model.py`** – Contains Pydantic models for request/response validation.
- **`app/schemas/moderation.py`** – Contains Pydantic models for request/response validation.
- **`app/schemas/retrieval.py`** – Contains Pydantic models for request/response validation.
- **`app/schemas/user.py`** – Contains Pydantic models for request/response validation.
- **`app/scripts/check_models.py`** – Defines ORM database models.
- **`app/services/__init__.py`** – Implements core business logic and orchestration.
- **`app/services/embeddings/__init__.py`** – Implements core business logic and orchestration.
- **`app/services/embeddings/base.py`** – Implements core business logic and orchestration.
- **`app/services/embeddings/embedding_service.py`** – Implements core business logic and orchestration.
- **`app/services/explanation/__init__.py`** – Implements core business logic and orchestration.
- **`app/services/explanation/attribution_service.py`** – Implements core business logic and orchestration.
- **`app/services/integrations/__init__.py`** – Implements core business logic and orchestration.
- **`app/services/integrations/dynamics_service.py`** – Implements core business logic and orchestration.
- **`app/services/integrations/gcdocs_service.py`** – Implements core business logic and orchestration.
- **`app/services/integrations/gcms_service.py`** – Implements core business logic and orchestration.
- **`app/services/llm/__init__.py`** – Implements core business logic and orchestration.
- **`app/services/llm/adapters/__init__.py`** – Implements core business logic and orchestration.
- **`app/services/llm/adapters/base.py`** – Implements core business logic and orchestration.
- **`app/services/llm/adapters/deepseek.py`** – Implements core business logic and orchestration.
- **`app/services/llm/adapters/llama.py`** – Implements core business logic and orchestration.
- **`app/services/llm/adapters/openai_compatible.py`** – Implements core business logic and orchestration.
- **`app/services/llm/anthropic_service.py`** – Implements core business logic and orchestration.
- **`app/services/llm/base.py`** – Implements core business logic and orchestration.
- **`app/services/llm/on_prem_service.py`** – Implements core business logic and orchestration.
- **`app/services/llm/openai_service.py`** – Implements core business logic and orchestration.
- **`app/services/model_router.py`** – Defines FastAPI route groupings and endpoints.
- **`app/services/moderation/__init__.py`** – Implements core business logic and orchestration.
- **`app/services/moderation/content_filter.py`** – Implements core business logic and orchestration.
- **`app/services/orchestrator.py`** – Implements core business logic and orchestration.
- **`app/services/retrieval/__init__.py`** – Implements core business logic and orchestration.
- **`app/services/retrieval/document_processor.py`** – Implements core business logic and orchestration.
- **`app/services/retrieval/vector_db_service.py`** – Implements core business logic and orchestration.
- **`app/utils/__init__.py`** – Module role inferred from context.
- **`app/utils/helpers.py`** – Module role inferred from context.
- **`app/utils/security.py`** – Contains authentication and security utility functions.
- **`app/utils/telemetry.py`** – Module role inferred from context.

## 📂 `tests/`

Test suite for validating APIs, services, and utilities using pytest.

- **`tests/__init__.py`** – Module role inferred from context.
- **`tests/conftest.py`** – Module role inferred from context.
- **`tests/test_api/__init__.py`** – Module role inferred from context.
- **`tests/test_api/test_chat.py`** – Test case for APIs, services, or components.
- **`tests/test_api/test_completions.py`** – Test case for APIs, services, or components.
- **`tests/test_services/__init__.py`** – Implements core business logic and orchestration.
- **`tests/test_services/test_on_prem_models.py`** – Implements core business logic and orchestration.
- **`tests/test_services/test_orchestrator.py`** – Implements core business logic and orchestration.
- **`tests/test_utils/__init__.py`** – Module role inferred from context.
- **`tests/test_utils/test_helpers.py`** – Test case for APIs, services, or components.
