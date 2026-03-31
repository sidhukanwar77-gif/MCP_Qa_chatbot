"""
memory.py - The Brain's Diary

WHAT IT DOES:
  Stores and retrieves conversation history.
  Each conversation has a 'session_id' (like a chat room number).
  All messages in that session are stored together.

ANALOGY:
  Like a diary where each page is a different conversation.
  The session_id is the page number.
"""

import json    # For converting data to/from text
import os      # For file operations
from typing import List, Dict  # Type hints
from datetime import datetime  # For timestamps

from src.config import MEMORY_DB_PATH


class MemorySystem:
    """
    A simple file-based memory system.
    
    Stores conversations as JSON files on disk.
    Each session gets its own file.
    """

    def __init__(self):
        """Set up the memory system and create the storage folder."""
        self.memory_path = MEMORY_DB_PATH

        # Create the memory folder if it doesn't exist
        # exist_ok=True means 'don't crash if it already exists'
        os.makedirs(self.memory_path, exist_ok=True)

    def _get_session_file(self, session_id: str) -> str:
        """
        Get the file path for a specific session.
        
        The underscore _ at the start means 'this is a private helper method'
        (other code shouldn't call it directly).
        """
        return os.path.join(self.memory_path, f'{session_id}.json')

    def add_message(self, session_id: str, role: str, content: str):
        """
        Add a message to a conversation session.
        
        Args:
            session_id: Which conversation (e.g., 'user123')
            role: Who said it ('user' or 'assistant')
            content: What was said
        """
        # Load existing messages for this session
        messages = self.get_messages(session_id)

        # Create the new message with a timestamp
        new_message = {
            'role': role,           # 'user' or 'assistant'
            'content': content,     # The actual message text
            'timestamp': datetime.now().isoformat(),  # When it was said
        }

        # Add to the list
        messages.append(new_message)

        # Save back to file
        file_path = self._get_session_file(session_id)
        with open(file_path, 'w') as f:  # 'w' = write mode
            json.dump(messages, f, indent=2)  # indent=2 makes it readable

    def get_messages(self, session_id: str, last_n: int = 10) -> List[Dict]:
        """
        Get messages from a conversation session.
        
        Args:
            session_id: Which conversation
            last_n: How many recent messages to return (default 10)
        
        Returns:
            List of message dictionaries
        """
        file_path = self._get_session_file(session_id)

        # If no file exists yet, return empty list
        if not os.path.exists(file_path):
            return []

        # Read the file
        with open(file_path, 'r') as f:  # 'r' = read mode
            messages = json.load(f)  # Convert JSON text back to Python objects

        # Return only the last N messages
        return messages[-last_n:]

    def get_context_string(self, session_id: str) -> str:
        """
        Get conversation history as a formatted string.
        This is what we'll feed to the AI as context.
        
        Returns something like:
          User: How many leave days?
          Assistant: You get 20 paid leave days per year.
        """
        messages = self.get_messages(session_id)

        if not messages:
            return 'No previous conversation history.'

        # Format each message as 'Role: Content'
        lines = []
        for msg in messages:
            role = msg['role'].capitalize()  # 'user' -> 'User'
            lines.append(f"{role}: {msg['content']}")

        return '\n'.join(lines)  # Join all lines with newlines

    def clear_session(self, session_id: str):
        """Delete all messages for a session (like erasing a diary page)."""
        file_path = self._get_session_file(session_id)
        if os.path.exists(file_path):
            os.remove(file_path)

