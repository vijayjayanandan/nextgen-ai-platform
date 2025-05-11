#!/usr/bin/env python3
"""
Script to check the status of on-premises LLM models.
This is useful for monitoring and diagnostics.

Usage:
    python check_models.py [--all] [--model MODEL_NAME]
    
Options:
    --all       Check all configured on-prem models
    --model     Check a specific model by name
"""

import os
import sys
import json
import asyncio
import argparse
import httpx
from typing import Dict, Any, List, Optional

# Add parent directory to path to be able to import app modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.config import settings


async def check_model_status(model_name: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Check the status of a specific model.
    
    Args:
        model_name: Name of the model to check
        config: Model configuration
        
    Returns:
        Dictionary with status information
    """
    endpoint_url = config.get("endpoint")
    
    if not endpoint_url:
        return {
            "name": model_name,
            "status": "error",
            "error": "No endpoint URL configured",
            "details": None
        }
    
    # Try to connect to the model endpoint
    try:
        # First try a simple health check
        health_url = f"{endpoint_url}/health"
        models_url = f"{endpoint_url}/models"
        
        async with httpx.AsyncClient(timeout=5) as client:
            # Try health endpoint first
            try:
                health_response = await client.get(health_url)
                health_status = health_response.status_code == 200
            except Exception:
                health_status = False
                
            # Try models list endpoint
            try:
                models_response = await client.get(models_url)
                if models_response.status_code == 200:
                    models_data = models_response.json()
                else:
                    models_data = None
            except Exception:
                models_data = None
            
            # If both failed, try a simple completion as last resort
            if not health_status and not models_data:
                try:
                    completion_url = f"{endpoint_url}/completions"
                    completion_response = await client.post(
                        completion_url,
                        json={
                            "model": model_name,
                            "prompt": "Hello",
                            "max_tokens": 5
                        },
                        timeout=10
                    )
                    completion_status = completion_response.status_code == 200
                except Exception:
                    completion_status = False
            else:
                completion_status = None
            
            # Determine overall status
            if health_status:
                status = "healthy"
            elif models_data is not None:
                status = "available"
            elif completion_status:
                status = "responding"
            else:
                status = "unavailable"
                
            return {
                "name": model_name,
                "status": status,
                "endpoint": endpoint_url,
                "health_check": health_status,
                "models_list": bool(models_data),
                "completion_test": completion_status,
                "type": config.get("model_type"),
                "details": models_data if models_data else None
            }
                
    except Exception as e:
        return {
            "name": model_name,
            "status": "error",
            "endpoint": endpoint_url,
            "error": str(e),
            "details": None
        }


async def check_all_models() -> List[Dict[str, Any]]:
    """
    Check the status of all configured on-prem models.
    
    Returns:
        List of status dictionaries for each model
    """
    results = []
    
    if not hasattr(settings, "ON_PREM_MODELS"):
        print("No on-premises models configured.")
        return []
    
    for model_name, config in settings.ON_PREM_MODELS.items():
        status = await check_model_status(model_name, config)
        results.append(status)
        
    return results


async def main():
    parser = argparse.ArgumentParser(description="Check on-premises LLM models status")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--all", action="store_true", help="Check all configured models")
    group.add_argument("--model", type=str, help="Check a specific model by name")
    
    args = parser.parse_args()
    
    if args.all:
        results = await check_all_models()
        print(json.dumps(results, indent=2))
    else:
        if not hasattr(settings, "ON_PREM_MODELS") or args.model not in settings.ON_PREM_MODELS:
            print(f"Model {args.model} not found in configuration.")
            sys.exit(1)
            
        result = await check_model_status(args.model, settings.ON_PREM_MODELS[args.model])
        print(json.dumps(result, indent=2))
    

if __name__ == "__main__":
    asyncio.run(main())