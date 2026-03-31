"""Test the LangChain Agent in isolation."""

import sys
sys.path.insert(0, '.')
from src.agent import QAAgent

def test_agent():
    print('=== Testing LangChain Agent ===')
    print('(Make sure Ollama is running!)')
    print()

    # Step 1: Create agent
    agent = QAAgent()
    print('Step 1: Agent created')

    # Step 2: Test with context (simulating RAG results)
    fake_context = 'Every employee gets 20 paid leave days per year.'
    fake_memory = 'No previous conversation.'
    question = 'How many leave days do I get?'

    print(f'Step 2: Asking: "{question}"')
    print(f'  Context: {fake_context}')
    answer = agent.answer(question, fake_context, fake_memory)
    print(f'  Answer: {answer}')

    # Step 3: Test without context
    print(f'Step 3: Asking without context: "What is the salary?"')
    answer2 = agent.answer('What is the salary?')
    print(f'  Answer: {answer2}')

    print('\n=== Agent Test PASSED ===')

if __name__ == '__main__':
    test_agent()
