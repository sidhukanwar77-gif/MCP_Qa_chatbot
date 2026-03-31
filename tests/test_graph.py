"""
graph.py - The Assembly Line (LangGraph Workflow)

WHAT IT DOES:
  Creates a step-by-step workflow:
    1. load_memory   -> Get previous conversation history
    2. retrieve_docs  -> Search documents for relevant info
    3. generate       -> Use AI to create an answer
    4. save_memory    -> Save this Q&A for next time

ANALOGY:
  An assembly line in a factory. The 'product' (your question)
  moves from station to station, getting processed at each step.
"""

from typing import TypedDict, List  # For defining the state structure
from langgraph.graph import StateGraph, END  # Graph builder

from src.rag import RAGSystem
from src.memory import MemorySystem
from src.agent import QAAgent


# --- Define the State (the clipboard) ---
class GraphState(TypedDict):
    """
    This defines WHAT data flows through the workflow.
    TypedDict = a dictionary where each key has a specific type.
    """
    question: str           # The user's question
    session_id: str         # Which conversation session
    memory_context: str     # Previous conversation history (filled by node 1)
    document_context: str   # Relevant document chunks (filled by node 2)
    answer: str             # The AI's answer (filled by node 3)


class QAWorkflow:
    """
    The complete Q&A workflow.
    Connects RAG + Memory + Agent into a step-by-step pipeline.
    """

    def __init__(self):
        """Set up all components and build the graph."""
        # Create instances of each system
        self.rag = RAGSystem()          # The filing cabinet
        self.memory = MemorySystem()    # The diary
        self.agent = QAAgent()          # The brain

        # Build the workflow graph
        self.graph = self._build_graph()

    def _build_graph(self):
        """
        Build the LangGraph workflow.
        
        This method creates the assembly line by:
        1. Creating a new graph with our state structure
        2. Adding nodes (stations)
        3. Adding edges (connections between stations)
        4. Compiling it into a runnable workflow
        """
        # Create a new graph with GraphState as the clipboard format
        workflow = StateGraph(GraphState)

        # --- Add Nodes (stations) ---
        # Each node is a function that takes state and returns updated state
        workflow.add_node('load_memory', self._load_memory)
        workflow.add_node('retrieve_docs', self._retrieve_docs)
        workflow.add_node('generate', self._generate)
        workflow.add_node('save_memory', self._save_memory)

        # --- Add Edges (connections) ---
        # set_entry_point = 'start here'
        workflow.set_entry_point('load_memory')

        # After load_memory, go to retrieve_docs
        workflow.add_edge('load_memory', 'retrieve_docs')

        # After retrieve_docs, go to generate
        workflow.add_edge('retrieve_docs', 'generate')

        # After generate, go to save_memory
        workflow.add_edge('generate', 'save_memory')

        # After save_memory, we're done
        workflow.add_edge('save_memory', END)

        # Compile the graph into a runnable object
        return workflow.compile()

    # --- Node Functions ---
    # Each function = one station on the assembly line

    def _load_memory(self, state: GraphState) -> dict:
        """
        Station 1: Load previous conversation history.
        
        Reads the diary for this session and adds the history
        to the state clipboard.
        """
        print('  [Node 1] Loading memory...')
        memory_context = self.memory.get_context_string(state['session_id'])
        print(f'  [Node 1] Memory loaded: {len(memory_context)} characters')
        return {'memory_context': memory_context}

    def _retrieve_docs(self, state: GraphState) -> dict:
        """
        Station 2: Search documents for relevant information.
        
        Takes the question from the clipboard, searches the
        filing cabinet, and adds the results to the clipboard.
        """
        print('  [Node 2] Searching documents...')
        results = self.rag.search(state['question'])
        # Join all results into one string, separated by dividers
        context = '\n---\n'.join(results)
        print(f'  [Node 2] Found {len(results)} relevant chunks')
        return {'document_context': context}

    def _generate(self, state: GraphState) -> dict:
        """
        Station 3: Generate an answer using the AI.
        
        Takes the question, document context, and memory from
        the clipboard, and uses the brain (agent) to generate
        an answer.
        """
        print('  [Node 3] Generating answer...')
        answer = self.agent.answer(
            question=state['question'],
            context=state['document_context'],
            memory=state['memory_context'],
        )
        print(f'  [Node 3] Answer generated: {len(answer)} characters')
        return {'answer': answer}

    def _save_memory(self, state: GraphState) -> dict:
        """
        Station 4: Save this conversation to memory.
        
        Writes both the question and answer to the diary
        so we remember them next time.
        """
        print('  [Node 4] Saving to memory...')
        self.memory.add_message(state['session_id'], 'user', state['question'])
        self.memory.add_message(state['session_id'], 'assistant', state['answer'])
        print('  [Node 4] Memory saved!')
        return {}  # No new state to add

    def run(self, question: str, session_id: str = 'default') -> str:
        """
        Run the complete workflow for a question.
        
        Args:
            question: What the user is asking
            session_id: Which conversation session
        
        Returns:
            The AI's answer
        """
        print(f'\nProcessing: "{question}"')
        print('Running workflow...')

        # Create the initial state (clipboard)
        initial_state = {
            'question': question,
            'session_id': session_id,
            'memory_context': '',
            'document_context': '',
            'answer': '',
        }

        # Run the graph! It will go through all stations in order
        result = self.graph.invoke(initial_state)

        return result['answer']
