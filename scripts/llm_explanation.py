# scripts/llm_explanation.py

import os
import sys
import requests
import json
from pathlib import Path
from dotenv import load_dotenv

# Dynamically resolve root
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(ROOT_DIR))

from utils.config_loader import load_config
from logutils.logger import get_logger

# Initialize logger
logger = get_logger("llm_explanation")

# Load environment variables from .env
env_path = ROOT_DIR / ".env"
if env_path.exists():
    load_dotenv(dotenv_path=env_path)
    logger.info(f"✅ Loaded .env from: {env_path}")
else:
    logger.warning("⚠️ .env file not found; make sure EURIAI_API_KEY is set manually.")

EURIAI_API_KEY = os.getenv("EURIAI_API_KEY")

def get_sentiment_explanation(text, prediction_label, confidence, model_name="gpt-4.1-nano", temperature=0.6, max_tokens=300):
    try:
        logger.info("🧠 Sending request to EURON LLM for explanation")

        if not EURIAI_API_KEY:
            raise ValueError("Missing EURIAI_API_KEY in environment.")

        url = "https://api.euron.one/api/v1/euri/alpha/chat/completions"

        prompt = (
            f"The following sentence was classified as **{prediction_label}** "
            f"with a confidence score of {confidence:.4f}. Explain why this sentiment label "
            f"makes sense based on the text:\n\n"
            f"\"{text}\"\n\n"
            f"Use clear reasoning and highlight relevant phrases."
        )

        headers = {
            "Authorization": f"Bearer {EURIAI_API_KEY}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": "You are a sentiment analysis expert helping users understand model predictions."},
                {"role": "user", "content": prompt}
            ],
            "temperature": temperature,
            "max_tokens": max_tokens
        }

        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        explanation = response.json()["choices"][0]["message"]["content"]

        logger.info("✅ Explanation successfully received from LLM")
        return explanation.strip()

    except Exception as e:
        logger.exception(f"❌ LLM explanation failed: {e}")
        return "Explanation failed. Please check logs or retry."

# Optional test
if __name__ == "__main__":
    print(get_sentiment_explanation("I am extremely happy today!", "Positive 😊", 0.98))
