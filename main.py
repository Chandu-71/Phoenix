import os
from fastapi import FastAPI, Header, HTTPException, Request
import litellm
from dotenv import load_dotenv

# Load API keys from your .env file
load_dotenv()

app = FastAPI(title="Phoenix Gateway")

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

    try:
        # LiteLLM automatically uses the API keys from the environment 
        # based on the model prefix (e.g., "gemini/...", "groq/...", "openrouter/...")
        response = litellm.completion(
            model=model,
            messages=messages,
            metadata={
                "tenant_id": x_tenant_id,
                "feature_id": x_feature_id
            }
        )
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))