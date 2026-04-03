#!/usr/bin/env python3
"""List all Gemini models that support the Live API (bidiGenerateContent)."""
import os, sys
from dotenv import load_dotenv
from google import genai

load_dotenv()

api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    print("Set GEMINI_API_KEY first.", file=sys.stderr)
    sys.exit(1)

client = genai.Client(api_key=api_key)
print("Models supporting bidiGenerateContent:\n")
for m in client.models.list():
    methods = getattr(m, "supported_actions", None) or getattr(m, "supported_generation_methods", None) or []
    if "bidiGenerateContent" in str(methods):
        print(f"  {m.name}")
