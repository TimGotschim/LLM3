"""
Multi-Model Comparison for Rexx RAG System
Compare different Ollama models to find the best one for your use case
"""

import os
from typing import List, Dict
import time
import pandas as pd
from rexx_rag_system import RexxRAGSystem
from rexx_evaluator import RAGEvaluator


class OllamaModelComparator:
    """Compare different Ollama models for RAG performance."""
    
    def __init__(self, pdf_folder: str):
        self.pdf_folder = pdf_folder
        self.results = []
    
    def get_available_ollama_models(self) -> List[str]:
        """Get list of locally available Ollama models."""
        import requests
        try:
            response = requests.get("http://localhost:11434/api/tags")
            if response.status_code == 200:
                models = response.json().get("models", [])
                return [model['name'] for model in models]
        except:
            print("⚠️  Could not connect to Ollama. Is it running?")
            return []
    
    def compare_models(self, 
                      models: List[str],
                      test_questions: List[Dict],
                      n_results: int = 5) -> pd.DataFrame:
        """
        Compare different Ollama models on the same test questions.
        
        Args:
            models: List of Ollama model names (e.g., ['llama2', 'mistral'])
            test_questions: Test questions with optional expected answers
            n_results: Number of chunks to retrieve
            
        Returns:
            DataFrame with comparison results
        """
        print("="*80)
        print("MULTI-MODEL COMPARISON")
        print("="*80)
        
        for model_name in models:
            print(f"\n{'='*80}")
            print(f"Testing Model: {model_name}")
            print(f"{'='*80}")
            
            # Initialize RAG system with this model
            rag_system = RexxRAGSystem(self.pdf_folder, model_name=model_name)
            
            # Process PDFs (only needed once, cached after first time)
            if not os.path.exists(os.path.join(self.pdf_folder, "../rexx_chroma_db")):
                rag_system.process_pdfs()
            
            # Evaluate this model
            evaluator = RAGEvaluator(rag_system)
            results = evaluator.evaluate_end_to_end(test_questions)
            
            # Store results
            model_results = {
                "model": model_name,
                "rag_score": results['overall'].get('rag_score', 0),
                "mrr": results['overall'].get('mrr', 0),
                "accuracy": results['overall'].get('accuracy', 'N/A'),
                "coherence": results['overall'].get('coherence', 0),
                "avg_response_time": results['overall'].get('avg_response_time', 0),
                "avg_response_length": results['overall'].get('avg_response_length', 0)
            }
            
            self.results.append(model_results)
            
            # Print summary
            print(f"\nResults for {model_name}:")
            print(f"  RAG Score: {model_results['rag_score']:.4f}")
            print(f"  MRR: {model_results['mrr']:.4f}")
            print(f"  Accuracy: {model_results['accuracy']}")
            print(f"  Coherence: {model_results['coherence']:.4f}")
            print(f"  Avg Response Time: {model_results['avg_response_time']:.2f}s")
        
        # Create comparison DataFrame
        df = pd.DataFrame(self.results)
        df = df.sort_values('rag_score', ascending=False)
        
        return df
    
    def print_comparison_report(self, df: pd.DataFrame):
        """Print a formatted comparison report."""
        print("\n" + "="*80)
        print("MODEL COMPARISON REPORT")
        print("="*80)
        
        print("\n📊 Overall Rankings (by RAG Score):")
        print("-" * 80)
        print(df.to_string(index=False))
        
        print("\n" + "="*80)
        print("🏆 Best Performers:")
        print("="*80)
        
        # Find best in each category
        best_rag = df.loc[df['rag_score'].idxmax()]
        best_mrr = df.loc[df['mrr'].idxmax()]
        best_coherence = df.loc[df['coherence'].idxmax()]
        fastest = df.loc[df['avg_response_time'].idxmin()]
        
        print(f"  Best RAG Score: {best_rag['model']} ({best_rag['rag_score']:.4f})")
        print(f"  Best MRR: {best_mrr['model']} ({best_mrr['mrr']:.4f})")
        print(f"  Best Coherence: {best_coherence['model']} ({best_coherence['coherence']:.4f})")
        print(f"  Fastest: {fastest['model']} ({fastest['avg_response_time']:.2f}s)")
        
        print("\n" + "="*80)
        print("💡 Recommendations:")
        print("="*80)
        
        # Give recommendation
        if best_rag['model'] == fastest['model']:
            print(f"  ✨ {best_rag['model']} is both the best performer AND fastest!")
            print(f"     Recommended for production use.")
        else:
            print(f"  🎯 For best quality: {best_rag['model']}")
            print(f"  ⚡ For fastest responses: {fastest['model']}")
            print(f"     Choose based on your priority: quality vs speed")
    
    def save_comparison(self, df: pd.DataFrame, filename: str = "model_comparison.csv"):
        """Save comparison results to CSV."""
        df.to_csv(filename, index=False)
        print(f"\n✓ Saved comparison results to {filename}")


def example_model_comparison():
    """Example: Compare different Ollama models."""
    
    # Setup
    script_dir = os.path.dirname(os.path.abspath(__file__))
    pdf_folder = os.path.join(script_dir, "rexx_pdfs")
    
    comparator = OllamaModelComparator(pdf_folder)
    
    # Check available models
    print("Checking available Ollama models...")
    available = comparator.get_available_ollama_models()
    
    if available:
        print(f"\n📦 Available models: {', '.join(available)}")
    else:
        print("\n⚠️  No models found or Ollama not running")
        print("   Please ensure Ollama is running: ollama serve")
        print("   And you have models installed: ollama pull llama2")
        return
    
    # Models to compare (adjust based on what you have installed)
    models_to_test = [
        "llama2",      # Default, good all-rounder
        "mistral",     # Fast and efficient
        "phi",         # Small and fast
        # "llama3",    # Latest version (if you have it)
        # "codellama", # Good for technical docs
    ]
    
    # Filter to only installed models
    models_to_test = [m for m in models_to_test if any(m in avail for avail in available)]
    
    if not models_to_test:
        print("\n❌ None of the recommended models are installed")
        print("   Install with: ollama pull llama2")
        return
    
    print(f"\n🧪 Testing models: {', '.join(models_to_test)}")
    
    # Test questions
    test_questions = [
        {"question": "What is Rexx Systems?", "expected_answer": "HR software"},
        {"question": "Can two users have the same User Name?", "expected_answer": "unique"},
        {"question": "How do I access the Jobportal Configurator?", "expected_answer": "permission"},
        {"question": "What is the organizational tree?", "expected_answer": "hierarchical"},
        {"question": "How are permission profiles evaluated?", "expected_answer": "order"},
        {"question": "What is FTE-Basis?", "expected_answer": "Full Time Equivalent"}
    ]
    
    # Run comparison
    df = comparator.compare_models(models_to_test, test_questions)
    
    # Show results
    comparator.print_comparison_report(df)
    
    # Save results
    comparator.save_comparison(df)
    
    return df


if __name__ == "__main__":
    example_model_comparison()
