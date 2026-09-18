import os
import json
import time
import math
from fastapi import FastAPI, Header, HTTPException, Request
import litellm
import redis
from dotenv import load_dotenv

# Load API keys from .env file
load_dotenv()

app = FastAPI(title="Phoenix Gateway")

# Initialize Redis client targeting the Docker container
redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)

WINDOW_SECONDS = 60  # 60-second sliding window for health metrics

def calculate_percentile(data, p):
    """Calculates percentiles without requiring external libraries like NumPy."""
    if not data:
        return 0
    data.sort()
    k = (len(data) - 1) * (p / 100.0)
    f = int(k)
    c = math.ceil(k)
    if f == c:
        return data[int(k)]
    return data[f] * (c - k) + data[c] * (k - f)

def log_provider_metric(provider: str, latency: float, error_type: str = "success"):
    """Logs the request outcome into a Redis sorted set using the current timestamp as the score."""
    now = time.time()
    key = f"health:{provider}"
    
    # Store JSON string containing latency and the error taxonomy outcome
    record = json.dumps({"timestamp": now, "latency": latency, "error_type": error_type})
    
    # Add to sorted set with the timestamp as the score
    redis_client.zadd(key, {record: now})
    
    # Remove entries older than the 60-second sliding window
    redis_client.zremrangebyscore(key, 0, now - WINDOW_SECONDS)
    
    # Set an expiry on the key so it cleans itself up if unused
    redis_client.expire(key, WINDOW_SECONDS * 2)

@app.post("/v1/chat/completions")
async def chat_completions(
    request: Request,
    x_tenant_id: str = Header(..., description="Unique ID for the tenant"),
    x_feature_id: str = Header(..., description="Unique ID for the feature")
):
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")
        
    model = body.get("model")
    messages = body.get("messages")
    
    if not model or not messages:
         raise HTTPException(status_code=400, detail="Model and messages are required")

    start_time = time.time()
    
    # Extract the base provider name from the model string (e.g., "groq" from "groq/...")
    provider = model.split("/")[0] if "/" in model else "unknown"

    try:
        response = litellm.completion(
            model=model,
            messages=messages,
            metadata={
                "tenant_id": x_tenant_id,
                "feature_id": x_feature_id
            }
        )
        latency = time.time() - start_time
        log_provider_metric(provider, latency, "success")
        return response
        
    except litellm.RateLimitError:
        log_provider_metric(provider, time.time() - start_time, "rate_limit")
        raise HTTPException(status_code=429, detail="Provider Rate Limit Exceeded")
    except litellm.APITimeoutError:
        log_provider_metric(provider, time.time() - start_time, "timeout")
        raise HTTPException(status_code=504, detail="Provider Timeout")
    except litellm.AuthenticationError:
        log_provider_metric(provider, time.time() - start_time, "auth_failure")
        raise HTTPException(status_code=401, detail="Provider Authentication Failed")
    except litellm.ContentPolicyViolationError:
        log_provider_metric(provider, time.time() - start_time, "content_filter")
        raise HTTPException(status_code=400, detail="Content Policy Violation")
    except Exception as e:
        log_provider_metric(provider, time.time() - start_time, "server_error")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics")
def get_metrics():
    """Exposes real-time provider health metrics calculated from the Redis sliding window."""
    providers = ["groq", "gemini", "openrouter"]
    metrics = {}
    now = time.time()
    
    for provider in providers:
        key = f"health:{provider}"
        
        # Fetch all records from the last 60 seconds
        records = redis_client.zrangebyscore(key, now - WINDOW_SECONDS, now)
        
        if not records:
            metrics[provider] = {"status": "no_data"}
            continue
            
        latencies = []
        error_counts = {
            "success": 0, "rate_limit": 0, "timeout": 0, 
            "server_error": 0, "auth_failure": 0, "content_filter": 0
        }
        
        for r in records:
            data = json.loads(r)
            error_counts[data["error_type"]] += 1
            if data["error_type"] == "success":
                latencies.append(data["latency"])
                
        total_requests = len(records)
        success_rate = (error_counts["success"] / total_requests) * 100
        
        metrics[provider] = {
            "total_requests_last_60s": total_requests,
            "success_rate_percent": round(success_rate, 2),
            "errors": error_counts,
            "latency_ms": {
                "p50": round(calculate_percentile(latencies, 50) * 1000, 2) if latencies else 0,
                "p95": round(calculate_percentile(latencies, 95) * 1000, 2) if latencies else 0,
                "p99": round(calculate_percentile(latencies, 99) * 1000, 2) if latencies else 0,
            }
        }
        
    return metrics