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
        response = genai.responses.create(
            model=MODEL_NAME,
            input=prompt,
        )
        # The structure may vary; assume output[0].content[0].text
        output = response.output
        if output and isinstance(output, list):
            parts = []
            for item in output:
                if isinstance(item, dict) and "content" in item:
                    for c in item["content"]:
                        text = c.get("text")
                        if text:
                            parts.append(text)
            return "".join(parts).strip()
        # fallback
        return str(output)
    except Exception as e:
        raise RuntimeError(f"Error generating response: {e}")
