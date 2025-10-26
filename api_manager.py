import os
import itertools
import streamlit as st
from dotenv import load_dotenv

# Load .env for local development
load_dotenv()

class GeminiKeyManager:
    def __init__(self, env_var="GEMINI_KEYS"):
        # Try to load from .env first
        keys_str = os.getenv(env_var, "")

        # If not found, try Streamlit secrets (for Streamlit Cloud)
        if not keys_str:
            keys_str = st.secrets.get(env_var, "")

        # Split comma-separated keys (in case you have multiple)
        self.keys = [k.strip() for k in keys_str.split(",") if k.strip()]

        if not self.keys:
            raise ValueError("❌ No Gemini API keys found. Please set GEMINI_KEYS in .env or Streamlit secrets.")

        self.cycle = itertools.cycle(self.keys)
        self.current_key = next(self.cycle)

    def get_key(self):
        self.current_key = next(self.cycle)
        return self.current_key

