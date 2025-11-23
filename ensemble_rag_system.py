"""
Ensemble RAG System - Combine Multiple Ollama Models
Uses multiple models and combines their outputs for better results
"""

import os
from typing import List, Dict, Optional
import numpy as np
from collections import Counter
import requests
from rexx_rag_system import RexxRAGSystem


class EnsembleRAGSystem:
    """
    Ensemble RAG system that combines multiple Ollama models.
    
    Strategies:
    1. Voting: Multiple models vote on the best answer
    2. Consensus: Use answer that most models agree on
    3. Best-of-N: Generate N answers and pick best quality
    4. Weighted: Weight models by their performance
    """
    
    def __init__(self, pdf_folder: str, models: List[str], weights: Optional[Dict[str, float]] = None):
        """
        Initialize ensemble system.
        
        Args:
            pdf_folder: Path to PDF documents
            models: List of Ollama model names
            weights: Optional weights for each model (for weighted voting)
        """
        self.pdf_folder = pdf_folder
        self.models = models
        self.weights = weights or {model: 1.0 for model in models}
        
        # Initialize a RAG system (for retrieval, which is model-agnostic)
        self.base_rag = RexxRAGSystem(pdf_folder, model_name=models[0])
        
        print(f"Initialized ensemble with {len(models)} models: {', '.join(models)}")
    
    def query_single_model(self, question: str, model_name: str, retrieved_chunks: List[Dict]) -> str:
        """Query a single model with retrieved context."""
        context = "\n\n".join([
            f"[Source: {chunk['source']}]\n{chunk['text']}" 
            for chunk in retrieved_chunks
        ])
        
        prompt = f"""You are a helpful assistant answering questions about Rexx Systems Human Resource Software.
Use the following context from the documentation to answer the user's question.
If the answer cannot be found in the context, say so clearly.

Context:
{context}

Question: {question}

Answer: """
        
        try:
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": model_name,
                    "prompt": prompt,
                    "stream": False
                }
            )
            
            if response.status_code == 200:
                return response.json()['response']
            else:
                return f"Error from {model_name}"
        except Exception as e:
            return f"Error from {model_name}: {str(e)}"
    
    def query_voting(self, question: str, n_results: int = 5) -> Dict:
        """
        Voting strategy: Each model generates an answer, return the most common one.
        
        Args:
            question: User question
            n_results: Number of chunks to retrieve
            
        Returns:
            Dict with best answer and voting details
        """
        # Retrieve context once (same for all models)
        retrieved_chunks = self.base_rag.retrieve(question, n_results)
        
        # Get answers from all models
        answers = {}
        for model in self.models:
            answer = self.query_single_model(question, model, retrieved_chunks)
            answers[model] = answer
            print(f"✓ {model}: {len(answer)} chars")
        
        # Simple voting: most common answer (by similarity)
        # In practice, you'd want more sophisticated comparison
        answer_texts = list(answers.values())
        
        # Return all answers for user to choose, or pick longest as "best"
        best_answer = max(answer_texts, key=len)
        
        return {
            "question": question,
            "answer": best_answer,
            "all_answers": answers,
            "sources": [chunk['source'] for chunk in retrieved_chunks],
            "strategy": "voting"
        }
    
    def query_consensus(self, question: str, n_results: int = 5, similarity_threshold: float = 0.7) -> Dict:
        """
        Consensus strategy: Find answer most models agree on.
        
        Args:
            question: User question
            n_results: Number of chunks to retrieve
            similarity_threshold: Threshold for considering answers similar
            
        Returns:
            Dict with consensus answer
        """
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        
        # Retrieve context
        retrieved_chunks = self.base_rag.retrieve(question, n_results)
        
        # Get answers from all models
        answers = []
        model_names = []
        
        for model in self.models:
            answer = self.query_single_model(question, model, retrieved_chunks)
            answers.append(answer)
            model_names.append(model)
            print(f"✓ {model}: Generated")
        
        # Find consensus using TF-IDF similarity
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(answers)
        similarities = cosine_similarity(tfidf_matrix)
        
        # Find answer with highest average similarity to others
        avg_similarities = similarities.mean(axis=1)
        consensus_idx = np.argmax(avg_similarities)
        
        consensus_score = avg_similarities[consensus_idx]
        
        return {
            "question": question,
            "answer": answers[consensus_idx],
            "consensus_model": model_names[consensus_idx],
            "consensus_score": float(consensus_score),
            "all_answers": dict(zip(model_names, answers)),
            "sources": [chunk['source'] for chunk in retrieved_chunks],
            "strategy": "consensus"
        }
    
    def query_best_of_n(self, question: str, n_results: int = 5, quality_metric: str = "length") -> Dict:
        """
        Best-of-N strategy: Generate N answers and pick highest quality.
        
        Args:
            question: User question
            n_results: Number of chunks to retrieve
            quality_metric: How to judge quality ('length', 'coherence', 'specificity')
            
        Returns:
            Dict with best answer
        """
        # Retrieve context
        retrieved_chunks = self.base_rag.retrieve(question, n_results)
        
        # Get answers from all models
        answers = {}
        scores = {}
        
        for model in self.models:
            answer = self.query_single_model(question, model, retrieved_chunks)
            answers[model] = answer
            
            # Calculate quality score
            if quality_metric == "length":
                scores[model] = len(answer)
            elif quality_metric == "coherence":
                scores[model] = self._calculate_coherence(answer)
            elif quality_metric == "specificity":
                scores[model] = self._calculate_specificity(answer, question)
            
            print(f"✓ {model}: Score = {scores[model]:.2f}")
        
        # Pick best
        best_model = max(scores, key=scores.get)
        
        return {
            "question": question,
            "answer": answers[best_model],
            "best_model": best_model,
            "quality_score": scores[best_model],
            "all_scores": scores,
            "all_answers": answers,
            "sources": [chunk['source'] for chunk in retrieved_chunks],
            "strategy": "best_of_n",
            "metric": quality_metric
        }
    
    def query_weighted(self, question: str, n_results: int = 5) -> Dict:
        """
        Weighted strategy: Combine answers weighted by model performance.
        
        In practice, this might mean:
        - Use high-quality model for final answer
        - Use fast model for initial screening
        - Use specialized model for technical questions
        
        Args:
            question: User question
            n_results: Number of chunks to retrieve
            
        Returns:
            Dict with weighted best answer
        """
        # Retrieve context
        retrieved_chunks = self.base_rag.retrieve(question, n_results)
        
        # Get answers from all models
        answers = {}
        weighted_scores = {}
        
        for model in self.models:
            answer = self.query_single_model(question, model, retrieved_chunks)
            answers[model] = answer
            
            # Calculate weighted score (using length as proxy for detail)
            base_score = len(answer)
            weighted_scores[model] = base_score * self.weights.get(model, 1.0)
            
            print(f"✓ {model}: Weight={self.weights[model]:.2f}, Score={weighted_scores[model]:.1f}")
        
        # Pick best weighted answer
        best_model = max(weighted_scores, key=weighted_scores.get)
        
        return {
            "question": question,
            "answer": answers[best_model],
            "best_model": best_model,
            "weighted_score": weighted_scores[best_model],
            "all_answers": answers,
            "weights_used": self.weights,
            "sources": [chunk['source'] for chunk in retrieved_chunks],
            "strategy": "weighted"
        }
    
    def _calculate_coherence(self, text: str) -> float:
        """Calculate coherence score (simplified)."""
        if not text:
            return 0
        
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        if not sentences:
            return 0
        
        # Check for complete sentences
        complete = sum(1 for s in sentences if len(s) > 10 and s[0].isupper())
        coherence = complete / len(sentences)
        
        return coherence
    
    def _calculate_specificity(self, answer: str, question: str) -> float:
        """Calculate how specific the answer is to the question."""
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        
        try:
            vectorizer = TfidfVectorizer()
            tfidf = vectorizer.fit_transform([question.lower(), answer.lower()])
            similarity = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]
            return similarity
        except:
            return 0.5


def example_ensemble_usage():
    """Example: Using ensemble RAG system."""
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    pdf_folder = os.path.join(script_dir, "rexx_pdfs")
    
    print("="*80)
    print("ENSEMBLE RAG SYSTEM DEMO")
    print("="*80)
    
    # Check available models
    print("\nChecking available models...")
    try:
        response = requests.get("http://localhost:11434/api/tags")
        if response.status_code == 200:
            available = [m['name'] for m in response.json().get("models", [])]
            print(f"Available: {', '.join(available)}")
        else:
            print("⚠️  Could not get model list")
            available = []
    except:
        print("❌ Ollama not running")
        return
    
    # Setup ensemble (using models you have)
    models_to_use = ["llama2", "mistral"]  # Adjust based on what you have
    models_to_use = [m for m in models_to_use if any(m in a for a in available)]
    
    if len(models_to_use) < 2:
        print("\n❌ Need at least 2 models for ensemble")
        print("   Install with: ollama pull llama2 && ollama pull mistral")
        return
    
    print(f"\n🎯 Using ensemble: {', '.join(models_to_use)}")
    
    # Initialize ensemble
    ensemble = EnsembleRAGSystem(
        pdf_folder, 
        models=models_to_use,
        weights={"llama2": 1.2, "mistral": 1.0}  # Give llama2 more weight
    )
    
    # Process PDFs
    ensemble.base_rag.process_pdfs()
    
    # Test question
    question = "Can two users have the same User Name in Rexx?"
    
    print(f"\n{'='*80}")
    print(f"Question: {question}")
    print(f"{'='*80}")
    
    # Strategy 1: Voting
    print("\n1️⃣  VOTING STRATEGY:")
    print("-" * 80)
    result1 = ensemble.query_voting(question)
    print(f"\nSelected Answer: {result1['answer'][:200]}...")
    
    # Strategy 2: Consensus
    print("\n2️⃣  CONSENSUS STRATEGY:")
    print("-" * 80)
    result2 = ensemble.query_consensus(question)
    print(f"\nConsensus Model: {result2['consensus_model']}")
    print(f"Consensus Score: {result2['consensus_score']:.3f}")
    print(f"Answer: {result2['answer'][:200]}...")
    
    # Strategy 3: Best-of-N
    print("\n3️⃣  BEST-OF-N STRATEGY:")
    print("-" * 80)
    result3 = ensemble.query_best_of_n(question, quality_metric="coherence")
    print(f"\nBest Model: {result3['best_model']}")
    print(f"Quality Score: {result3['quality_score']:.3f}")
    print(f"Answer: {result3['answer'][:200]}...")
    
    # Strategy 4: Weighted
    print("\n4️⃣  WEIGHTED STRATEGY:")
    print("-" * 80)
    result4 = ensemble.query_weighted(question)
    print(f"\nBest Model: {result4['best_model']}")
    print(f"Weighted Score: {result4['weighted_score']:.1f}")
    print(f"Answer: {result4['answer'][:200]}...")
    
    print("\n" + "="*80)
    print("ENSEMBLE COMPLETE")
    print("="*80)


if __name__ == "__main__":
    example_ensemble_usage()
