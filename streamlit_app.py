# Streamlit Cloud deployment entry point
# Wolof Educational QA App - Enhanced with Ollama Mistral and web search

import streamlit as st
import pandas as pd
from gtts import gTTS
import tempfile
import os
import re
import math
import requests
from bs4 import BeautifulSoup
import json
import urllib.parse
import speech_recognition as sr

# --- LLM Configuration ---
OPENAI_AVAILABLE = False
OLLAMA_AVAILABLE = False
HUGGINGFACE_AVAILABLE = False

# OpenAI API Key (must be set as an environment variable)
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

# Try to connect to OpenAI
try:
    import openai
    if OPENAI_API_KEY:
        openai.api_key = OPENAI_API_KEY
        # Test the connection with a simple request
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=10
        )
        OPENAI_AVAILABLE = True
    else:
        OPENAI_AVAILABLE = False
except Exception as e:
    OPENAI_AVAILABLE = False
    print(f"OpenAI not available: {e}")

# ... (rest of the file unchanged, as previously cleaned) ... 