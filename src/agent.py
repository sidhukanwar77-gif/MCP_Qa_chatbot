"""
agent.py - The Brain (Groq API)

Calls Groq API for fast, free LLM responses.
No local model download needed.
"""

import httpx
from src.config import LLM_MODEL, GROQ_API_KEY


class QAAgent:
    """
    Question-Answering Agent using Groq API.
    """

    def __init__(self):
        if not GROQ_API_KEY or GROQ_API_KEY == 'your_groq_api_key_here':
            raise ValueError('GROQ_API_KEY not set. Add your Groq API key to the .env file.')
        self.model = LLM_MODEL
        self.api_key = GROQ_API_KEY

    def answer(self, question: str, context: str = '', memory: str = '') -> str:
        system_prompt = """You are a helpful assistant that answers questions based on the provided documents.
If the answer is not in the context, say "I don't have that information in the documents."
Be concise and clear."""

        user_prompt = f"""Previous conversation:
{memory or 'None'}

Context from documents:
{context or 'No documents available.'}

Question: {question}"""

        try:
            response = httpx.post(
                'https://api.groq.com/openai/v1/chat/completions',
                headers={
                    'Authorization': f'Bearer {self.api_key}',
                    'Content-Type': 'application/json',
                },
                json={
                    'model': self.model,
                    'messages': [
                        {'role': 'system', 'content': system_prompt},
                        {'role': 'user', 'content': user_prompt},
                    ],
                    'temperature': 0.3,
                    'max_tokens': 512,
                },
                timeout=30.0,
            )
            response.raise_for_status()
            return response.json()['choices'][0]['message']['content'].strip()
        except httpx.HTTPStatusError as e:
            return f'Groq API error: {e.response.status_code} - {e.response.text}'
        except Exception as e:
            return f'Error generating answer: {str(e)}'
