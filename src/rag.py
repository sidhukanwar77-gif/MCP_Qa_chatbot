"""
rag.py - The Filing Cabinet (Retrieval-Augmented Generation)

WHAT IT DOES:
  1. Loads your documents (text files)
  2. Splits them into small chunks
  3. Converts chunks into vectors (lists of numbers)
  4. Stores vectors in ChromaDB
  5. When asked, searches for the most relevant chunks

ANALOGY:
  Imagine a librarian who reads every book, writes a summary
  card for each chapter, and files them alphabetically.
  When you ask a question, the librarian quickly finds the
  most relevant cards and gives them to you.
"""

import os
from typing import List

from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from src.config import CHROMA_DB_PATH, DOCS_PATH


class RAGSystem:
    """
    The RAG System class.
    
    A class is like a blueprint for creating objects.
    This blueprint describes how to build a filing cabinet
    that can load, store, and search documents.
    """

    def __init__(self):
        """
        __init__ runs when you create a new RAGSystem object.
        Think of it as 'setting up the filing cabinet for the first time.'
        """
        # Create the embedding model (text-to-numbers converter)
        # Uses HuggingFace model running locally — no Ollama needed
        self.embeddings = HuggingFaceEmbeddings(
            model_name='all-MiniLM-L6-v2',  # Small, fast, good quality
        )

        # The vector store (database) - starts as None until we load documents
        self.vector_store = None

        # Text splitter configuration
        # This decides HOW to split documents into chunks
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,      # Each chunk will be ~500 characters
            chunk_overlap=50,    # Chunks overlap by 50 chars (so we don't lose context)
            length_function=len, # Use Python's len() to measure chunk size
        )

    def load_documents(self, docs_path: str = None) -> int:
        """
        Load documents from a folder and store them in the vector database.
        
        Args:
            docs_path: Path to folder with documents. Uses config default if not provided.
        
        Returns:
            Number of chunks created and stored.
        """
        # Use the provided path, or fall back to the config setting
        path = docs_path or DOCS_PATH

        # Check if the folder exists
        if not os.path.exists(path):
            raise FileNotFoundError(f"Documents folder not found: {path}")

        # Load all .txt files from the folder
        # DirectoryLoader automatically finds and reads all matching files
        loader = DirectoryLoader(
            path,                     # Where to look
            glob='**/*.txt',          # Pattern: any .txt file, even in subfolders
            loader_cls=TextLoader,    # Use TextLoader to read each file
        )
        documents = loader.load()  # Actually read the files

        if not documents:
            print('Warning: No documents found!')
            return 0

        # Split documents into chunks
        chunks = self.text_splitter.split_documents(documents)
        print(f'Split {len(documents)} documents into {len(chunks)} chunks')

        # Store chunks in ChromaDB
        # Chroma.from_documents does THREE things:
        #   1. Converts each chunk to a vector using our embedding model
        #   2. Stores the vectors in the database
        #   3. Also stores the original text (so we can return it later)
        self.vector_store = Chroma.from_documents(
            documents=chunks,              # The chunks to store
            embedding=self.embeddings,      # How to convert text to vectors
            persist_directory=CHROMA_DB_PATH,  # Where to save the database on disk
            collection_name='rag_docs',     # Name for this collection of documents
        )

        print(f'Stored {len(chunks)} chunks in vector database')
        return len(chunks)

    def load_pdf(self, pdf_path: str) -> int:
        """Load a single PDF file into the vector database."""
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()

        if not documents:
            return 0

        chunks = self.text_splitter.split_documents(documents)

        if self.vector_store is None:
            self.vector_store = Chroma.from_documents(
                documents=chunks,
                embedding=self.embeddings,
                persist_directory=CHROMA_DB_PATH,
                collection_name='rag_docs',
            )
        else:
            self.vector_store.add_documents(chunks)

        return len(chunks)

    def search(self, query: str, num_results: int = 3) -> List[str]:
        """
        Search the vector database for chunks relevant to your query.
        
        HOW IT WORKS:
          1. Your query is converted to a vector
          2. ChromaDB finds the vectors most similar to yours
          3. Returns the original text of those chunks
        
        Args:
            query: What to search for
            num_results: How many results to return (default 3)
        
        Returns:
            List of relevant text chunks
        """
        # Make sure we have documents loaded
        if self.vector_store is None:
            # Try to load from existing database on disk
            if os.path.exists(CHROMA_DB_PATH):
                self.vector_store = Chroma(
                    persist_directory=CHROMA_DB_PATH,
                    embedding_function=self.embeddings,
                    collection_name='rag_docs',
                )
            else:
                return ['No documents loaded yet. Please load documents first.']

        # Search! similarity_search finds the most similar chunks
        results = self.vector_store.similarity_search(query, k=num_results)

        # Extract just the text content from results
        return [doc.page_content for doc in results]

    def get_retriever(self):
        """
        Get a retriever object for use with LangChain.
        
        A retriever is a standardized way for LangChain to search.
        Instead of calling search() directly, LangChain uses
        the retriever interface.
        """
        if self.vector_store is None:
            if os.path.exists(CHROMA_DB_PATH):
                self.vector_store = Chroma(
                    persist_directory=CHROMA_DB_PATH,
                    embedding_function=self.embeddings,
                    collection_name='rag_docs',
                )
            else:
                raise ValueError('No documents loaded. Call load_documents() first.')

        # as_retriever() wraps the vector store in LangChain's Retriever interface
        return self.vector_store.as_retriever(
            search_kwargs={'k': 3}  # Return top 3 results
        )
