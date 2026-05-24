import os
from fastapi import APIRouter, HTTPException, Depends, Security
from fastapi.security import APIKeyHeader
import httpx
from pydantic import BaseModel

# Initialize router
router = APIRouter(prefix="/realtime", tags=["realtime"])

# Setup standard API Key protection for internal service communication
API_KEY_NAME = "X-PrepHub-Auth"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

def verify_internal_auth(api_key: str = Security(api_key_header)):
    """
    Ensures that only authenticated Next.js frontend clients can request WebRTC session tokens.
    """
    expected_secret = os.environ.get("BACKEND_AUTH_TOKEN", "development-secure-token-123")
    if not api_key or api_key != expected_secret:
        raise HTTPException(
            status_code=403, 
            detail="Forbidden: Invalid or missing X-PrepHub-Auth security header."
        )
    return api_key

class SessionConfig(BaseModel):
    model: str = "gpt-4o-realtime-preview-2024-12-17"
    voice: str = "alloy"
    instructions: str = "You are a professional technical interviewer at Google. Conduct a high-rigor coding mock."

@router.post("/session")
async def create_webrtc_session(
    config: SessionConfig,
    auth: str = Depends(verify_internal_auth)
):
    """
    Generates an ephemeral client secret token for OpenAI Realtime (WebRTC) connections.
    This prevents leaking the master OPENAI_API_KEY to the client's browser.
    """
    openai_key = os.environ.get("OPENAI_API_KEY")
    if not openai_key:
        raise HTTPException(
            status_code=500, 
            detail="Internal Error: OPENAI_API_KEY environment variable is not configured on the host."
        )

    # Calling the OpenAI Realtime sessions endpoint to get the ephemeral token
    url = "https://api.openai.com/v1/realtime/sessions"
    headers = {
        "Authorization": f"Bearer {openai_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": config.model,
        "voice": config.voice,
        "modalities": ["audio", "text"],
        "instructions": config.instructions,
        "input_audio_transcription": {
            "model": "whisper-1"
        }
    }

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                url, 
                headers=headers, 
                json=payload, 
                timeout=10.0
            )
            
        if response.status_code != 200:
            raise HTTPException(
                status_code=response.status_code, 
                detail=f"OpenAI API Error: {response.text}"
            )
            
        # Returns the complete session config including the ephemeral client_secret
        return response.json()
        
    except httpx.RequestError as exc:
        raise HTTPException(
            status_code=503, 
            detail=f"Connection failure to upstream OpenAI endpoints: {str(exc)}"
        )
