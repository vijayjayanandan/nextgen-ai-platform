# AI Platform - Backend

This repository contains the backend implementation for the AI Platform, a secure hybrid AI system that integrates cloud-based large language models (LLMs) with on-premises deployment for handling sensitive Protected B data.

## Architecture Overview

The IRCC AI Platform is built on a modular microservices architecture with the following core components:

1. **API Gateway**: Handles authentication, rate limiting, and request routing
2. **Orchestrator**: Coordinates the flow between retrieval, LLMs, and filtering
3. **Model Router**: Intelligently routes requests to appropriate LLM services based on sensitivity
4. **Document Processing**: Ingests, chunks, and embeds documents for retrieval
5. **Retrieval Service**: Performs semantic search using vector database
6. **Content Filtering**: Ensures outputs meet ethical AI guardrails
7. **Attribution Service**: Provides explainable outputs with source citations

## Technical Stack

- **Framework**: FastAPI
- **Database**: PostgreSQL with SQLAlchemy
- **Vector Database**: Pinecone (configurable)
- **Authentication**: JWT with Azure AD integration capability
- **LLM Providers**: OpenAI, Anthropic, On-premises models
- **Deployment**: Docker and Azure VM

## Getting Started

### Prerequisites

- Python 3.10+
- Docker and Docker Compose
- PostgreSQL
- Vector Database (Pinecone, Weaviate, etc.)

### Environment Setup

1. Clone the repository
2. Copy `.env.example` to `.env` and update the values
3. Install dependencies:

```bash
pip install -r requirements.txt
```

### Local Development

Run the application locally with hot reloading:

```bash
uvicorn app.main:app --reload
```

### Docker Deployment

Build and run with Docker Compose:

```bash
docker-compose up -d
```

## API Endpoints

The platform provides the following main API endpoints:

- `/api/v1/completions`: Text completions similar to OpenAI API
- `/api/v1/chat`: Chat completions and conversation management
- `/api/v1/documents`: Document ingestion and management
- `/api/v1/retrieval`: Semantic search and retrieval
- `/api/v1/moderation`: Content filtering and moderation
- `/api/v1/models`: Model management
- `/api/v1/users`: User management

## Key Features

### Hybrid Deployment Model

The platform leverages Azure cloud for scalability while keeping sensitive data processing on-premises:

- **Cloud Components**: API Gateway, public LLM APIs, web interface
- **On-premises Components**: Secure inference environment for Protected B data

### Multi-model Support

The system can route requests to different LLM providers:

- **External APIs**: OpenAI, Anthropic Claude
- **On-premises Models**: For sensitive data requiring Protected B handling

### Retrieval Augmented Generation (RAG)

- Document chunking and embedding
- Semantic search with vector database
- Context augmentation for grounded responses

### Security & Compliance

- Authentication via Azure AD
- Role-based access control
- Comprehensive audit logging
- Content filtering and ethical guardrails

## Core Components

### Orchestrator

The orchestrator coordinates the entire pipeline:
1. Content filtering of user input
2. Retrieval of relevant documents
3. LLM inference with the appropriate model
4. Filtering and attribution of outputs

### Model Router

Routes requests to appropriate LLM services based on:
- Data sensitivity classification
- Model capabilities and permissions
- User roles and access levels

### Document Processor

Handles document ingestion with:
- Chunking strategies for various document types
- Embedding generation
- Vector storage for efficient retrieval

## Configuration

Configuration is managed through environment variables. See `.env.example` for all available options.

## Deployment

The application is designed to be deployed to an Azure VM with the following considerations:

- Azure Canada regions for data residency
- On-premises secure inference for Protected B data
- Docker containers for easy deployment

## Development Guidelines

- Follow PEP 8 style guidelines
- Use async/await for all database and external API operations
- Implement proper error handling and logging
- Write comprehensive tests

## Security Notes

- The platform is designed to handle Protected B data
- Ensure proper network segmentation for on-premises components
- Follow all GC security policies for deployment

## Open Source LLM Support

The IRCC AI Platform now explicitly supports the following open source LLM models:

1. **Llama Models**:
   - `llama-7b` - Llama 3 7B model for general-purpose use
   - `llama-70b` - Llama 3 70B model for advanced reasoning tasks

2. **Deepseek Models**:
   - `deepseek-7b` - Deepseek 7B base model for general tasks
   - `deepseek-coder` - Deepseek Coder model optimized for code generation

These models are hosted on Azure VMs in the Canada Central region to meet data residency requirements and are automatically selected for Protected B data. See `docs/on_prem_models.md` for detailed implementation information.
