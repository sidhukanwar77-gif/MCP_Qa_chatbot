"""
chat.py - Interactive Chat Interface

Run this to chat with your document-powered AI assistant.
Type your questions, and it uses RAG + Memory + LangGraph.
"""

import sys
sys.path.insert(0, '.')

from src.rag import RAGSystem
from src.graph import QAWorkflow
from src.memory import MemorySystem

def main():
    print('=========================================')
    print('  Document Q&A Chatbot')
    print('  (RAG + Memory + LangChain + LangGraph)')
    print('=========================================')
    print()

    # Step 1: Load documents
    print('Loading documents...')
    rag = RAGSystem()
    num_chunks = rag.load_documents()
    print(f'Loaded {num_chunks} document chunks.')
    print()

    # Step 2: Create the workflow
    workflow = QAWorkflow()

    # Step 3: Start chatting!
    session_id = 'interactive-chat'
    print('Ready! Type your questions (type "quit" to exit, "clear" to reset memory)')
    print()

    while True:
        question = input('You: ').strip()

        if not question:
            continue
        if question.lower() == 'quit':
            print('Goodbye!')
            break
        if question.lower() == 'clear':
            MemorySystem().clear_session(session_id)
            print('Memory cleared!')
            continue

        print()
        answer = workflow.run(question, session_id)
        print(f'\nAssistant: {answer}\n')

if __name__ == '__main__':
    main()
