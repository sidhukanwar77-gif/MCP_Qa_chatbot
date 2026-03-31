"""
config.py - The Settings Manager
"""

import os
from dotenv import load_dotenv

load_dotenv()

LLM_MODEL = os.getenv('LLM_MODEL', 'llama3-8b-8192')
GROQ_API_KEY = os.getenv('GROQ_API_KEY', '')

CHROMA_DB_PATH = os.getenv('CHROMA_DB_PATH', './data/chroma_db')
MEMORY_DB_PATH = os.getenv('MEMORY_DB_PATH', './data/memory_db')
DOCS_PATH = os.getenv('DOCS_PATH', './docs')
