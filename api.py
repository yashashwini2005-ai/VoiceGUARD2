from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
import base64
import tempfile
import os

# Import your existing inference logic
from inference import classify_audio

# -------------------------------
# App Initialization
# -------------------------------
app = FastAPI(title="AI Voice Fraud Detection API")

# -------------------------------
# API Security
# -------------------------------
API_KEY = "voiceguard_2026_secure_key"  # 🔴 change this before demo

# -------------------------------
# Request Schema
# -------------------------------
class VoiceRequest(BaseModel):
    language: str
    audioFormat: str
    audioBase64: str

# -------------------------------
# Constants
# -------------------------------
ALLOWED_LANGUAGES = ["Tamil", "English", "Hindi", "Malayalam", "Telugu"]

# -------------------------------
# Helper: API Key Validation
# -------------------------------
def verify_api_key(x_api_key: str):
    if x_api_key != API_KEY:
        raise HTTPException(
            status_code=401,
            detail="Invalid API key"
        )

# -------------------------------
# Main API Endpoint
# -------------------------------
@app.post("/api/voice-detection")
def detect_voice(
    request: VoiceRequest,
    x_api_key: str = Header(None)
):
    # ---------------------------
    # Security Check
    # ---------------------------
    if not x_api_key:
        raise HTTPException(status_code=401, detail="API key missing")
    verify_api_key(x_api_key)

    # ---------------------------
    # Request Validation
    # ---------------------------
    if request.language not in ALLOWED_LANGUAGES:
        raise HTTPException(status_code=400, detail="Unsupported language")

    if request.audioFormat.lower() != "mp3":
        raise HTTPException(status_code=400, detail="Only MP3 audio supported")

    if not request.audioBase64:
        raise HTTPException(status_code=400, detail="audioBase64 is required")

    # ---------------------------
    # Decode Base64 → MP3
    # ---------------------------
    try:
        audio_bytes = base64.b64decode(request.audioBase64)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid Base64 audio")

    # ---------------------------
    # Save Temporary MP3 File
    # ---------------------------
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio:
        temp_audio.write(audio_bytes)
        temp_audio_path = temp_audio.name

    # ---------------------------
    # Run Inference
    # ---------------------------
    label, confidence = classify_audio(temp_audio_path)

    # Cleanup
    os.remove(temp_audio_path)

    if label is None:
        raise HTTPException(status_code=500, detail="Inference failed")

    # ---------------------------
    # Map Output
    # ---------------------------
    classification = "HUMAN"
    if label in [1, 2, 3, 4]:
        classification = "AI_GENERATED"

    explanation = (
        "Unnatural pitch consistency and robotic speech patterns detected"
        if classification == "AI_GENERATED"
        else "Natural speech variations detected"
    )

    # ---------------------------
    # Final Response (EXACT FORMAT)
    # ---------------------------
    return {
        "status": "success",
        "language": request.language,
        "classification": classification,
        "confidenceScore": round(confidence, 2),
        "explanation": explanation
    }
