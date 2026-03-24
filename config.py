import os
from dotenv import load_dotenv
import google.generativeai as genai

# load environment variables from .env if present
load_dotenv()

API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    raise ValueError("Missing GOOGLE_API_KEY in environment. Please set it in .env file.")

# configure the gemini client
genai.configure(api_key=API_KEY)

MODEL_NAME = "gemini-1.5-flash"


def generate_response(prompt: str) -> str:
    """Send the prompt to Gemini and return the response text."""
    if not prompt:
        return ""
    try:
        # CORRECTED: Initialize the GenerativeModel
        model = genai.GenerativeModel(MODEL_NAME)
        
        # CORRECTED: Use generate_content to get the response
        response = model.generate_content(prompt)
        
        # Return the text string
        return response.text
    except Exception as e:
        raise RuntimeError(f"Error generating response: {e}")