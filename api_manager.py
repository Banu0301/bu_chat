# api_manager.py
import os
import itertools
from dotenv import load_dotenv

load_dotenv()

class GeminiKeyManager:
    def __init__(self, env_var="GEMINI_KEYS"):
        keys_str = os.getenv(env_var, "")
        self.keys = [k.strip() for k in keys_str.split(",") if k.strip()]
        if not self.keys:
            raise ValueError("❌ No Gemini API keys found in .env")
        self.cycle = itertools.cycle(self.keys)
        self.current_key = next(self.cycle)

    def get_key(self):
        self.current_key = next(self.cycle)
        return self.current_key
