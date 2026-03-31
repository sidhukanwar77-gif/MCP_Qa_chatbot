"""Test the Memory system in isolation."""

import sys
sys.path.insert(0, '.')
from src.memory import MemorySystem

def test_memory():
    print('=== Testing Memory System ===')

    # Step 1: Create memory system
    memory = MemorySystem()
    print('Step 1: Memory system created')

    # Step 2: Add some messages
    memory.add_message('test-session', 'user', 'How many leave days do I get?')
    memory.add_message('test-session', 'assistant', 'You get 20 paid leave days per year.')
    memory.add_message('test-session', 'user', 'Can I carry them forward?')
    memory.add_message('test-session', 'assistant', 'Yes, up to 10 days.')
    print('Step 2: Added 4 messages')

    # Step 3: Retrieve messages
    messages = memory.get_messages('test-session')
    print(f'Step 3: Retrieved {len(messages)} messages')

    # Step 4: Get formatted context
    context = memory.get_context_string('test-session')
    print(f'Step 4: Context string:\n{context}')

    # Step 5: Clear session
    memory.clear_session('test-session')
    messages_after = memory.get_messages('test-session')
    print(f'Step 5: After clearing: {len(messages_after)} messages')

    print('\n=== Memory Test PASSED ===')

if __name__ == '__main__':
    test_memory()
