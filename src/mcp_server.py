"""
mcp_server.py - The Tool Shop (NOW CONNECTED!)

This is the same MCP server, but now the tools are connected
to real RAG, Memory, and LangGraph systems.
"""

import asyncio
import json
import sys
import os

# Make sure we can import from src/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcp.server.fastmcp import FastMCP
from src.rag import RAGSystem
from src.memory import MemorySystem
from src.graph import QAWorkflow

# --- Create all systems ---
mcp = FastMCP('rag-memory-server')
rag = RAGSystem()
memory = MemorySystem()

# Load documents on startup
try:
    rag.load_documents()
    print('Documents loaded successfully!')
except Exception as e:
    print(f'Warning: Could not load documents: {e}')

# Create the workflow (uses RAG, Memory, and Agent internally)
workflow = QAWorkflow()


@mcp.tool()
async def search_documents(query: str) -> str:
    """Search through documents for relevant information."""
    results = rag.search(query)  # NOW USING REAL RAG!
    return '\n---\n'.join(results)


@mcp.tool()
async def ask_question(question: str, session_id: str = 'default') -> str:
    """Ask a question and get an AI-powered answer."""
    answer = workflow.run(question, session_id)  # NOW USING REAL WORKFLOW!
    return answer


@mcp.tool()
async def save_memory(session_id: str, key: str, value: str) -> str:
    """Save information to memory."""
    memory.add_message(session_id, 'system', f'{key}: {value}')  # REAL MEMORY!
    return f'Saved: {key} = {value}'


@mcp.tool()
async def get_memory(session_id: str) -> str:
    """Get conversation history for a session."""
    return memory.get_context_string(session_id)  # REAL MEMORY!


if __name__ == '__main__':
    print('Starting CONNECTED MCP Server...')
    mcp.run(transport='stdio')
