"""
RAG System for Rexx Systems Documentation
This system processes PDF documents, creates embeddings, stores them in a vector database,
and uses Ollama for question answering with retrieval augmentation.
"""

import os
import json
import numpy as np
from typing import List, Dict, Tuple
import hashlib
from datetime import datetime

# PDF processing
import pypdf
import pdfplumber

# Text processing
import re
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Embeddings and vector store
from sentence_transformers import SentenceTransformer
import chromadb
#from chromadb.config import Settings

# integration
import requests

# Progress tracking
from tqdm import tqdm

class RexxRAGSystem:
    """
    A Retrieval-Augmented Generation system for Rexx Systems documentation.
    """

    def __init__(self, pdf_folder: str, model_name: str = "llama2", embedding_model: str = "all-MiniLM-L6-v2"):
        """
        Initialize the RAG system.
        """
        self.pdf_folder = pdf_folder
        self.model_name = model_name
        self.ollama_base_url = "http://localhost:11434"

        # Use absolute path for Chroma DB storage (relative to this script's location)
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.persist_directory = os.path.join(self.script_dir, "rexx_chroma_db")

        # Initialize embedding model
        print("Loading embedding model...")
        self.embedding_model = SentenceTransformer(embedding_model)

        # Initialize ChromaDB
        print("Initializing vector database...")
        self.chroma_client = chromadb.PersistentClient(
            path=self.persist_directory
        )

        # Create or get collection
        self.collection = self.chroma_client.get_or_create_collection(
            name="rexx_documentation",
            metadata={"hnsw:space": "cosine"}
        )

        # Text splitter for chunking documents
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

        # Metadata storage (also use absolute path)
        self.metadata_file = os.path.join(self.script_dir, "rexx_rag_metadata.json")
        self.load_metadata()

    
    def load_metadata(self):
        """Load system metadata."""
        if os.path.exists(self.metadata_file):
            with open(self.metadata_file, 'r') as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {
                "processed_files": {},
                "total_chunks": 0,
                "last_update": None
            }
    
    def save_metadata(self):
        """Save system metadata."""
        self.metadata["last_update"] = datetime.now().isoformat()
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """
        Extract text from a PDF file using multiple methods.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Extracted text content
        """
        text_content = ""
        
        # Try pypdf first
        try:
            reader = pypdf.PdfReader(pdf_path)
            for page_num, page in enumerate(reader.pages):
                text = page.extract_text()
                if text.strip():
                    text_content += f"\n--- Page {page_num + 1} ---\n{text}\n"
        except Exception as e:
            print(f"pypdf extraction failed: {e}")
        
        # If pypdf fails or returns little content, try pdfplumber
        if len(text_content) < 100:
            try:
                with pdfplumber.open(pdf_path) as pdf:
                    for page_num, page in enumerate(pdf.pages):
                        text = page.extract_text()
                        if text:
                            text_content += f"\n--- Page {page_num + 1} ---\n{text}\n"
                        
                        # Also extract tables
                        tables = page.extract_tables()
                        for table_num, table in enumerate(tables):
                            if table:
                                text_content += f"\n[Table {table_num + 1} on Page {page_num + 1}]\n"
                                for row in table:
                                    text_content += " | ".join(str(cell) if cell else "" for cell in row) + "\n"
            except Exception as e:
                print(f"pdfplumber extraction failed: {e}")
        
        return text_content
    
    def clean_text(self, text: str) -> str:
        """
        Clean and preprocess text.
        
        Args:
            text: Raw text content
            
        Returns:
            Cleaned text
        """
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep important punctuation
        text = re.sub(r'[^\w\s\.\,\;\:\!\?\-\(\)\[\]\{\}\/\@\#\$\%\&\*\+\=]', '', text)
        
        # Fix common OCR errors
        text = text.replace('ﬁ', 'fi').replace('ﬂ', 'fl')
        
        return text.strip()
    
    def process_pdfs(self, force_reprocess: bool = False):
        """
        Process all PDFs in the specified folder.
        
        Args:
            force_reprocess: If True, reprocess all files even if already processed
        """
        pdf_files = [f for f in os.listdir(self.pdf_folder) if f.lower().endswith('.pdf')]
        
        if not pdf_files:
            print(f"No PDF files found in {self.pdf_folder}")
            return
        
        print(f"Found {len(pdf_files)} PDF files to process")
        
        for pdf_file in tqdm(pdf_files, desc="Processing PDFs"):
            pdf_path = os.path.join(self.pdf_folder, pdf_file)
            file_hash = self._get_file_hash(pdf_path)
            
            # Check if already processed
            if not force_reprocess and pdf_file in self.metadata["processed_files"]:
                if self.metadata["processed_files"][pdf_file]["hash"] == file_hash:
                    print(f"Skipping {pdf_file} (already processed)")
                    continue
            
            print(f"\nProcessing {pdf_file}...")
            
            # Extract text
            text = self.extract_text_from_pdf(pdf_path)
            
            if not text.strip():
                print(f"Warning: No text extracted from {pdf_file}")
                continue
            
            # Clean text
            text = self.clean_text(text)
            
            # Split into chunks
            chunks = self.text_splitter.split_text(text)
            print(f"Created {len(chunks)} chunks from {pdf_file}")
            
            # Create embeddings and store
            self._store_chunks(chunks, pdf_file)
            
            # Update metadata
            self.metadata["processed_files"][pdf_file] = {
                "hash": file_hash,
                "chunks": len(chunks),
                "processed_date": datetime.now().isoformat()
            }
            
        self.save_metadata()
        print("\nProcessing complete!")
    
    def _get_file_hash(self, file_path: str) -> str:
        """Calculate hash of file for change detection."""
        with open(file_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    
    def _store_chunks(self, chunks: List[str], source_file: str):
        """
        Create embeddings and store chunks in ChromaDB.
        
        Args:
            chunks: List of text chunks
            source_file: Source PDF filename
        """
        # Generate unique IDs for chunks
        chunk_ids = [f"{source_file}_{i}" for i in range(len(chunks))]
        
        # Create metadata for each chunk
        metadatas = [{
            "source": source_file,
            "chunk_index": i,
            "chunk_total": len(chunks)
        } for i in range(len(chunks))]
        
        # Generate embeddings
        embeddings = self.embedding_model.encode(chunks, show_progress_bar=True)
        
        # Store in ChromaDB
        self.collection.add(
            ids=chunk_ids,
            embeddings=embeddings.tolist(),
            documents=chunks,
            metadatas=metadatas
        )
        
        self.metadata["total_chunks"] += len(chunks)
    
    def retrieve(self, query: str, n_results: int = 5) -> List[Dict]:
        """
        Retrieve relevant chunks for a query.
        
        Args:
            query: User query
            n_results: Number of results to return
            
        Returns:
            List of relevant chunks with metadata
        """
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query])
        
        # Search in ChromaDB
        results = self.collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=n_results
        )
        
        # Format results
        retrieved_chunks = []
        for i in range(len(results['documents'][0])):
            retrieved_chunks.append({
                'text': results['documents'][0][i],
                'source': results['metadatas'][0][i]['source'],
                'distance': results['distances'][0][i]
            })
        
        return retrieved_chunks
    
    def generate_response(self, query: str, retrieved_chunks: List[Dict]) -> str:
        """
        Generate response using Ollama with retrieved context.
        
        Args:
            query: User query
            retrieved_chunks: Retrieved context chunks
            
        Returns:
            Generated response
        """
        # Build context from retrieved chunks
        context = "\n\n".join([
            f"[Source: {chunk['source']}]\n{chunk['text']}" 
            for chunk in retrieved_chunks
        ])
        
        # Build prompt
        prompt = f"""You are a helpful assistant answering questions about Rexx Systems Human Resource Software.
Use the following context from the documentation to answer the user's question.
If the answer cannot be found in the context, say so clearly.

Context:
{context}

Question: {query}

Answer: """
        
        # Call Ollama API
        try:
            response = requests.post(
                f"{self.ollama_base_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False
                }
            )
            
            if response.status_code == 200:
                return response.json()['response']
            else:
                return f"Error: Ollama returned status code {response.status_code}"
        
        except requests.exceptions.ConnectionError:
            return "Error: Cannot connect to Ollama. Make sure Ollama is running (ollama serve)."
        except Exception as e:
            return f"Error generating response: {str(e)}"
    
    def query(self, question: str, n_results: int = 5) -> Dict:
        """
        Main query interface for the RAG system.
        
        Args:
            question: User question
            n_results: Number of chunks to retrieve
            
        Returns:
            Dictionary with answer and metadata
        """
        # Retrieve relevant chunks
        retrieved_chunks = self.retrieve(question, n_results)
        
        # Generate response
        answer = self.generate_response(question, retrieved_chunks)
        
        return {
            "question": question,
            "answer": answer,
            "sources": [chunk['source'] for chunk in retrieved_chunks],
            "retrieved_chunks": retrieved_chunks
        }
    
    def evaluate_model(self, test_questions: List[Dict[str, str]]) -> Dict:
        """
        Evaluate the model on a set of test questions.
        
        Args:
            test_questions: List of dicts with 'question' and optionally 'expected_answer'
            
        Returns:
            Evaluation metrics
        """
        results = []
        
        for test_case in tqdm(test_questions, desc="Evaluating model"):
            response = self.query(test_case['question'])
            
            result = {
                "question": test_case['question'],
                "generated_answer": response['answer'],
                "sources": response['sources']
            }
            
            if 'expected_answer' in test_case:
                result['expected_answer'] = test_case['expected_answer']
                # Simple similarity check (you can implement more sophisticated metrics)
                result['contains_expected'] = test_case['expected_answer'].lower() in response['answer'].lower()
            
            results.append(result)
        
        # Calculate metrics
        if any('contains_expected' in r for r in results):
            accuracy = sum(r.get('contains_expected', False) for r in results) / len(results)
        else:
            accuracy = None
        
        return {
            "results": results,
            "metrics": {
                "total_questions": len(test_questions),
                "accuracy": accuracy,
                "average_sources": np.mean([len(r['sources']) for r in results])
            }
        }
    
    def get_stats(self) -> Dict:
        """Get system statistics."""
        return {
            "total_documents": len(self.metadata["processed_files"]),
            "total_chunks": self.metadata["total_chunks"],
            "last_update": self.metadata["last_update"],
            "collection_size": self.collection.count(),
            "processed_files": list(self.metadata["processed_files"].keys())
        }


def main():
    """Example usage of the RAG system."""

    # Use relative path from script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    pdf_folder = os.path.join(script_dir, "rexx_pdfs")
    
    # Initialize RAG system
    rag = RexxRAGSystem(pdf_folder, model_name="llama2")
    
    # Process PDFs (this will take some time on first run)
    print("Processing PDF documents...")
    rag.process_pdfs()
    
    # Show statistics
    stats = rag.get_stats()
    print(f"\nSystem Statistics:")
    print(f"- Total documents: {stats['total_documents']}")
    print(f"- Total chunks: {stats['total_chunks']}")
    
    # Example query
    print("\n" + "="*50)
    print("Testing the RAG system...")
    
    test_question = "What are the main features of Rexx Systems?"
    response = rag.query(test_question)
    
    print(f"\nQuestion: {response['question']}")
    print(f"\nAnswer: {response['answer']}")
    print(f"\nSources: {', '.join(response['sources'])}")
    
    # Example evaluation
    print("\n" + "="*50)
    print("Running evaluation...")
    
    test_questions = [
        {"question": "What is Rexx Systems?"},
        {"question": "How does the HR module work?"},
        {"question": "What are the system requirements?"},
        {"question": "How to configure user permissions?"},
        {"question": "What reporting features are available?"}
    ]
    
    evaluation = rag.evaluate_model(test_questions)
    
    print(f"\nEvaluation Results:")
    print(f"- Total questions: {evaluation['metrics']['total_questions']}")
    print(f"- Average sources per answer: {evaluation['metrics']['average_sources']:.2f}")


if __name__ == "__main__":
    main()
