#!/usr/bin/env python3
"""
FastAPI version of WolofChat for modern API deployment
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
from app import WolofChatApp, WolofChatConfig
import tempfile
import os
import speech_recognition as sr

app = FastAPI(
    title="WolofChat API",
    description="Wolof Educational Q&A API with AI-powered responses",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize WolofChat
wolofchat = WolofChatApp()

# Pydantic models
class QuestionRequest(BaseModel):
    question: str
    subject: str = "General"
    language: str = "Wolof"

class QuestionResponse(BaseModel):
    question: str
    answer: str
    sources: Optional[List[str]] = None
    subject: str
    language: str

class SpeechToTextRequest(BaseModel):
    text: str
    language: str = "wo"

class StatusResponse(BaseModel):
    openai: bool
    ollama: bool
    speech_recognition: bool
    text_to_speech: bool
    web_search: bool

@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "WolofChat API is running!", "version": "1.0.0"}

@app.get("/api/status", response_model=StatusResponse)
async def get_status():
    """Get service status"""
    status = wolofchat.get_service_status()
    return StatusResponse(**status)

@app.post("/api/ask", response_model=QuestionResponse)
async def ask_question(request: QuestionRequest):
    """Ask a question and get answer"""
    try:
        if not request.question:
            raise HTTPException(status_code=400, detail="Question is required")
        
        answer, sources = wolofchat.find_answer(
            request.question, 
            request.subject, 
            request.language
        )
        
        return QuestionResponse(
            question=request.question,
            answer=answer,
            sources=sources,
            subject=request.subject,
            language=request.language
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/speech-to-text")
async def speech_to_text(
    audio: UploadFile = File(...),
    language: str = Form("Wolof")
):
    """Convert speech to text"""
    try:
        if not audio:
            raise HTTPException(status_code=400, detail="No audio file provided")
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            content = await audio.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        try:
            # Convert audio file to AudioData
            recognizer = sr.Recognizer()
            with sr.AudioFile(temp_file_path) as source:
                audio_data = recognizer.record(source)
            
            text = wolofchat.process_voice_input(audio_data, language)
            
            if text:
                return {"text": text}
            else:
                raise HTTPException(status_code=400, detail="Could not recognize speech")
        
        finally:
            # Clean up temporary file
            os.unlink(temp_file_path)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/text-to-speech")
async def text_to_speech(request: SpeechToTextRequest):
    """Convert text to speech"""
    try:
        if not request.text:
            raise HTTPException(status_code=400, detail="Text is required")
        
        audio_file_path = wolofchat.generate_voice_output(request.text, request.language)
        
        if audio_file_path:
            return FileResponse(
                audio_file_path, 
                media_type="audio/mp3",
                filename="speech.mp3"
            )
        else:
            raise HTTPException(status_code=500, detail="Could not generate speech")
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/subjects")
async def get_subjects():
    """Get available subjects"""
    return {
        "subjects": [
            "General",
            "History", 
            "Mathematics",
            "Science",
            "Geography",
            "Technology",
            "Literature"
        ]
    }

@app.get("/api/languages")
async def get_languages():
    """Get supported languages"""
    return {
        "languages": [
            {"code": "Wolof", "name": "Wolof"},
            {"code": "English", "name": "English"}
        ]
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "services": wolofchat.get_service_status()}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port) 