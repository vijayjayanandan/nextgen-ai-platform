# Enhanced PII Filtering - Deployment Guide

## Overview

This guide provides step-by-step instructions for deploying and configuring the enhanced PII filtering system in your FastAPI + Next.js AI platform.

## 🚀 Quick Start

### 1. Environment Configuration

Copy the example environment file and configure PII filtering settings:

```bash
cp .env.example .env
```

Edit your `.env` file to include:

```bash
# Enhanced PII Filtering Configuration
ENABLE_PII_FILTERING=true
PII_RISK_THRESHOLD=0.7
ENABLE_ML_PII_DETECTION=false
DEFAULT_ANONYMIZATION_METHOD=tokenization
ENABLE_REVERSIBLE_ANONYMIZATION=true
PII_PROCESSING_TIMEOUT_SECONDS=5
ENABLE_PII_AUDIT_LOGGING=true
```

### 2. Install Dependencies

Ensure all required dependencies are installed:

```bash
pip install -r requirements.txt
```

### 3. Start the Application

```bash
# Development
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Production
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

### 4. Verify Integration

Check that PII filtering is active:

```bash
curl -X GET http://localhost:8000/api/v1/docs
```

Look for the "Enhanced Moderation" section in the API documentation.

## 📋 Configuration Options

### Risk Threshold Settings

| Threshold | Description | Use Case |
|-----------|-------------|----------|
| `0.5` | High Security | Maximum protection, some false positives |
| `0.7` | Balanced (Recommended) | Good accuracy, minimal false positives |
| `0.9` | Conservative | Only high-confidence detections |

### Anonymization Methods

| Method | Description | Reversible | Example |
|--------|-------------|------------|---------|
| `tokenization` | Replace with secure tokens | ✅ | `TOKEN_SIN_A1B2C3D4` |
| `generalization` | Replace with generic patterns | ❌ | `XXX-XXX-XXXX` |
| `redaction` | Replace with redaction tags | ❌ | `[REDACTED_SIN]` |
| `suppression` | Complete removal | ❌ | *(removed)* |

## 🔧 Advanced Configuration

### Custom Risk Thresholds by Entity Type

```python
# In app/core/config.py (advanced users)
PII_ENTITY_RISK_THRESHOLDS = {
    "sin": 0.5,           # Very strict for SIN
    "email": 0.8,         # More lenient for emails
    "phone": 0.7,         # Balanced for phone numbers
    "ircc_number": 0.6    # Strict for IRCC numbers
}
```

### Endpoint-Specific Configuration

```python
# Configure which endpoints get filtered
PII_FILTERED_ENDPOINTS = [
    "/api/v1/chat",
    "/api/v1/documents",
    "/api/v1/retrieval",
    "/api/v1/moderation"
]
```

## 🧪 Testing the Integration

### 1. Run Unit Tests

```bash
# Test PII filtering components
pytest tests/test_enhanced_pii_filtering.py -v

# Test middleware integration
pytest tests/test_pii_middleware_integration.py -v
```

### 2. Manual API Testing

Test PII detection endpoint:

```bash
curl -X POST "http://localhost:8000/api/v1/moderation/detect-pii" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -d '{
    "content": "My SIN is 123-456-789",
    "sensitivity_level": "high",
    "anonymize": true
  }'
```

Expected response:
```json
{
  "content": "My SIN is TOKEN_SIN_A1B2C3D4",
  "pii_detected": true,
  "detections": [
    {
      "entity_type": "sin",
      "text": "123-456-789",
      "confidence": 0.95,
      "risk_level": "critical"
    }
  ],
  "risk_score": 0.85,
  "is_safe": false,
  "anonymization_applied": true
}
```

### 3. Test Middleware Integration

Send a request with PII to any filtered endpoint:

```bash
curl -X POST "http://localhost:8000/api/v1/chat/send" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -d '{
    "message": "Please help me with my SIN 123-456-789",
    "conversation_id": "test_123"
  }'
```

Check logs for PII detection messages:
```
INFO: PII detected in request field 'message'
INFO: PII middleware processing time: 45.2ms
```

## 📊 Monitoring and Metrics

### Key Metrics to Monitor

1. **PII Detection Rate**: Percentage of requests with PII detected
2. **Processing Time**: Average middleware processing time
3. **False Positive Rate**: Rate of incorrect PII detections
4. **Compliance Score**: Overall compliance assessment

### Log Analysis

PII filtering generates structured logs:

```json
{
  "timestamp": "2024-06-14T14:30:00Z",
  "level": "WARNING",
  "message": "PII detected in request field 'message'",
  "user_id": "user_123",
  "pii_types": ["sin"],
  "risk_score": 0.85,
  "anonymization_applied": true,
  "processing_time_ms": 45.2
}
```

### Audit Trail

All PII detections are logged for compliance:

```json
{
  "user_id": "user_123",
  "action": "pii_middleware_detection",
  "resource_type": "content",
  "resource_id": "request_message",
  "details": {
    "pii_detections_count": 1,
    "pii_types": ["sin"],
    "risk_levels": ["critical"],
    "overall_risk_score": 0.85,
    "anonymization_applied": true
  }
}
```

## 🔒 Security Considerations

### Data Protection

1. **Encryption**: All PII tokens are encrypted at rest
2. **Access Control**: Role-based access to PII data
3. **Audit Trails**: Comprehensive logging of all operations
4. **Retention**: Automatic data retention policies

### Network Security

```bash
# Use HTTPS in production
ALLOWED_HOSTS=["https://your-domain.com"]

# Configure secure headers
SECURE_SSL_REDIRECT=true
SECURE_HSTS_SECONDS=31536000
```

### Database Security

```bash
# Use encrypted database connections
POSTGRES_SSLMODE=require
POSTGRES_SSLCERT=/path/to/client-cert.pem
POSTGRES_SSLKEY=/path/to/client-key.pem
```

## 🚨 Troubleshooting

### Common Issues

#### 1. Middleware Not Processing Requests

**Symptoms**: No PII filtering logs, requests pass through unchanged

**Solutions**:
- Check `ENABLE_PII_FILTERING=true` in environment
- Verify middleware is added to FastAPI app
- Ensure endpoints match filtered patterns

```bash
# Check configuration
python -c "from app.core.config import settings; print(f'PII Filtering: {settings.ENABLE_PII_FILTERING}')"
```

#### 2. High False Positive Rate

**Symptoms**: Too many false PII detections

**Solutions**:
- Increase risk threshold: `PII_RISK_THRESHOLD=0.8`
- Review detection patterns
- Add context validation rules

#### 3. Performance Issues

**Symptoms**: Slow API responses, high processing times

**Solutions**:
- Reduce risk threshold for faster processing
- Enable async processing
- Optimize pattern matching

```bash
# Monitor processing times
grep "PII middleware processing time" /var/log/app.log | tail -20
```

#### 4. Missing Dependencies

**Symptoms**: Import errors, module not found

**Solutions**:
```bash
# Install missing dependencies
pip install fastapi starlette pydantic

# Verify installation
python -c "from app.core.pii_middleware import PIIFilteringMiddleware; print('✅ PII middleware available')"
```

### Debug Mode

Enable debug logging for troubleshooting:

```bash
# In .env
LOG_LEVEL=DEBUG
DEBUG=true

# Check debug logs
tail -f /var/log/app.log | grep "PII"
```

## 📈 Performance Optimization

### Production Settings

```bash
# Optimized for production
ENABLE_PII_FILTERING=true
PII_RISK_THRESHOLD=0.7
PII_PROCESSING_TIMEOUT_SECONDS=3
ENABLE_ML_PII_DETECTION=false  # Disable for better performance
MAX_CONCURRENT_REQUESTS=200
```

### Scaling Considerations

1. **Horizontal Scaling**: Multiple FastAPI workers
2. **Caching**: Cache compiled regex patterns
3. **Async Processing**: Non-blocking PII detection
4. **Load Balancing**: Distribute PII processing load

```bash
# Production deployment with multiple workers
uvicorn app.main:app --workers 4 --host 0.0.0.0 --port 8000
```

## 🔄 Updates and Maintenance

### Regular Maintenance Tasks

1. **Weekly**: Review PII detection accuracy
2. **Monthly**: Update detection patterns
3. **Quarterly**: Compliance audit and review

### Updating PII Patterns

```python
# Add new Canadian immigration patterns
CUSTOM_PII_PATTERNS = {
    "work_permit": r"\bWP\d{10}\b",
    "study_permit": r"\bSP\d{10}\b"
}
```

### Version Updates

```bash
# Update to latest version
git pull origin main
pip install -r requirements.txt
pytest tests/ -v  # Run tests
uvicorn app.main:app --reload  # Restart application
```

## 📞 Support

### Getting Help

1. **Documentation**: Check `/docs/enhanced_pii_filtering.md`
2. **API Docs**: Visit `http://localhost:8000/api/v1/docs`
3. **Logs**: Check application logs for error details
4. **Tests**: Run test suite for validation

### Reporting Issues

When reporting issues, include:

1. Environment configuration
2. Error logs and stack traces
3. Steps to reproduce
4. Expected vs actual behavior

### Emergency Procedures

If PII filtering needs to be disabled immediately:

```bash
# Disable PII filtering
export ENABLE_PII_FILTERING=false

# Restart application
systemctl restart your-app-service
```

## ✅ Deployment Checklist

- [ ] Environment variables configured
- [ ] Dependencies installed
- [ ] Tests passing
- [ ] API documentation accessible
- [ ] PII filtering middleware active
- [ ] Enhanced moderation endpoints available
- [ ] Logging and monitoring configured
- [ ] Security settings applied
- [ ] Performance benchmarks met
- [ ] Compliance requirements satisfied

## 🎯 Next Steps

After successful deployment:

1. **Monitor Performance**: Track PII detection metrics
2. **Train Users**: Educate team on new capabilities
3. **Compliance Review**: Conduct privacy impact assessment
4. **Optimization**: Fine-tune based on usage patterns
5. **Integration**: Connect with existing audit systems

Your enhanced PII filtering system is now ready to provide enterprise-grade protection for your Canadian immigration AI platform! 🚀
