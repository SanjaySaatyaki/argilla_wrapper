"""Configuration module for Argilla client."""
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Argilla configuration
ARGILLA_API_URL = os.getenv("ARGILLA_API_URL", "http://localhost:8000")
ARGILLA_API_KEY = os.getenv("ARGILLA_API_KEY", "")

if not ARGILLA_API_KEY:
    raise ValueError("ARGILLA_API_KEY is not set in .env file")

if not ARGILLA_API_URL:
    raise ValueError("ARGILLA_API_URL is not set in .env file")
