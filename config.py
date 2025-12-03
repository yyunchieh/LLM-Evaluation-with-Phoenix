# config.py
from dotenv import load_dotenv
import os

load_dotenv()

def load_openai_api_key():
    """Load OpenAI API key from .env"""
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    if not OPENAI_API_KEY:
        raise ValueError("Missing OPENAI_API_KEY in .env file")
    return OPENAI_API_KEY




