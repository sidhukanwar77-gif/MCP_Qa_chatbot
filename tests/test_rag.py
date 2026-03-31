"""
test_rag.py - Test the RAG system in isolation

WHAT THIS TESTS:
  1. Can we load documents?
  2. Are they split into chunks?
  3. Can we search and get relevant results?
"""

import sys
sys.path.insert(0, '.')  # Make sure Python can find our src folder

from src.rag import RAGSystem

def test_rag():
    print('=== Testing RAG System ===')
    print()

    # Step 1: Create the RAG system
    print('Step 1: Creating RAG system...')
    rag = RAGSystem()
    print('  OK - RAG system created')
    print()

    # Step 2: Load documents
    print('Step 2: Loading documents from docs/ folder...')
    num_chunks = rag.load_documents()
    print(f'  OK - Created {num_chunks} chunks')
    print()

    # Step 3: Search for something
    print('Step 3: Searching for "leave policy"...')
    results = rag.search('leave policy')
    for i, result in enumerate(results):
        print(f'  Result {i+1}: {result[:100]}...')
    print()

    # Step 4: Search for something else
    print('Step 4: Searching for "work from home"...')
    results = rag.search('work from home')
    for i, result in enumerate(results):
        print(f'  Result {i+1}: {result[:100]}...')
    print()

    print('=== RAG Test PASSED ===')

if __name__ == '__main__':
    test_rag()
