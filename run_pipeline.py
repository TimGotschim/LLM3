#!/usr/bin/env python3
"""
Complete RAG Pipeline Demo for Rexx Systems Documentation
This script demonstrates the full workflow from PDF processing to evaluation
"""

import os
import sys
import json
from datetime import datetime

# Import our modules
from rexx_rag_system import RexxRAGSystem
from rexx_fine_tuner import RexxFineTuner
from rexx_evaluator import RAGEvaluator


def check_ollama():
    """Check if Ollama is running."""
    import requests
    try:
        response = requests.get("http://localhost:11434/api/tags")
        if response.status_code == 200:
            models = response.json().get("models", [])
            print(f"✓ Ollama is running with {len(models)} models")
            return True
    except:
        pass
    
    print("✗ Ollama is not running. Please start it with: ollama serve")
    return False


def main():
    """Run the complete RAG pipeline."""
    
    print("="*60)
    print("Rexx Systems RAG Pipeline Demo")
    print("="*60)
    
    # Check prerequisites
    if not check_ollama():
        sys.exit(1)
    
    # Configuration
    PDF_FOLDER = "../rexx_pdfs"
    MODEL_NAME = "llama2"  # Change this to your preferred model
    
    # Check if PDF folder exists
    if not os.path.exists(PDF_FOLDER):
        print(f"✗ PDF folder not found: {PDF_FOLDER}")
        print("Please update the PDF_FOLDER variable with your actual path")
        sys.exit(1)
    
    print(f"✓ PDF folder found: {PDF_FOLDER}")
    
    # Step 1: Initialize RAG System
    print("\n1. Initializing RAG System...")
    rag = RexxRAGSystem(PDF_FOLDER, model_name=MODEL_NAME)
    
    # Step 2: Process PDFs
    print("\n2. Processing PDF Documents...")
    rag.process_pdfs()
    
    # Show statistics
    stats = rag.get_stats()
    print(f"   - Processed {stats['total_documents']} documents")
    print(f"   - Created {stats['total_chunks']} chunks")
    
    # Step 3: Test Basic Queries
    print("\n3. Testing Basic Queries...")
    
    test_queries = [
        "What is Rexx Systems?",
        "What are the main features of the HR module?",
        "How do I configure user permissions?"
    ]
    
    for query in test_queries[:1]:  # Test just one query for demo
        print(f"\n   Query: {query}")
        response = rag.query(query, n_results=3)
        print(f"   Answer: {response['answer'][:200]}...")
        print(f"   Sources: {', '.join(set(response['sources']))}")
    
    # Step 4: Evaluate Baseline Performance
    print("\n4. Evaluating Baseline Performance...")
    
    evaluator = RAGEvaluator(rag)
    
    # Create test dataset
    test_dataset = [
        {"question": "What is Rexx Systems?", "expected_answer": "HR software"},
        {"question": "What modules are available?", "expected_answer": "HR payroll"},
        {"question": "How to add new employees?", "expected_answer": "employee management"},
        {"question": "What are the reporting features?", "expected_answer": "reports analytics"},
        {"question": "How to configure the system?", "expected_answer": "configuration settings"}
    ]
    
    # Run baseline evaluation
    baseline_results = evaluator.evaluate_end_to_end(test_dataset)
    
    print(f"   - RAG Score: {baseline_results['overall']['rag_score']:.3f}")
    print(f"   - MRR: {baseline_results['overall']['mrr']:.3f}")
    print(f"   - Coherence: {baseline_results['overall']['coherence']:.3f}")
    
    # Save baseline results
    with open("baseline_evaluation.json", "w") as f:
        json.dump(baseline_results, f, indent=2)
    
    # Step 5: Fine-tuning (Optional - requires GPU for best results)
    print("\n5. Fine-tuning Process...")
    print("   Note: Fine-tuning is computationally intensive.")
    print("   For demo purposes, we'll show the setup but skip actual training.")
    
    fine_tuner = RexxFineTuner()
    
    # Generate training data
    print("   - Generating training data from documents...")
    # Uncomment the following lines to actually run fine-tuning:
    # training_data = fine_tuner.create_training_data_from_chunks(rag, num_samples=20)
    # dataset = fine_tuner.prepare_dataset(training_data)
    # print(f"   - Created {len(dataset['train'])} training samples")
    
    # Step 6: Error Analysis
    print("\n6. Error Analysis (15Ps Framework)...")
    
    # Analyze errors from baseline evaluation
    errors = []
    for i, result in enumerate(baseline_results.get("detailed_results", [])):
        if result.get("expected_answer"):
            if result["expected_answer"].lower() not in result.get("generated_answer", "").lower():
                errors.append({
                    "question": result["question"],
                    "expected": result["expected_answer"],
                    "generated": result.get("generated_answer", ""),
                    "error_type": "missing_information"
                })
    
    print(f"   - Found {len(errors)} errors to analyze")
    
    # The 15Ps framework (example categories)
    error_categories = {
        "Purpose": "Understanding the intent of the question",
        "Problem": "Identifying what went wrong",
        "Plan": "Strategy for improvement",
        "Process": "Steps taken by the model",
        "Performance": "Metrics and measurements",
        # Add more P's as needed for your analysis
    }
    
    # Step 7: Generate Final Report
    print("\n7. Generating Final Report...")
    
    report = {
        "timestamp": datetime.now().isoformat(),
        "configuration": {
            "pdf_folder": PDF_FOLDER,
            "model": MODEL_NAME,
            "total_documents": stats['total_documents'],
            "total_chunks": stats['total_chunks']
        },
        "baseline_performance": {
            "rag_score": baseline_results['overall']['rag_score'],
            "mrr": baseline_results['overall']['mrr'],
            "accuracy": baseline_results['overall'].get('accuracy', 'N/A'),
            "avg_response_time": baseline_results['overall']['avg_response_time']
        },
        "error_analysis": {
            "total_errors": len(errors),
            "error_types": ["missing_information", "incorrect_context", "hallucination"]
        },
        "recommendations": [
            "Fine-tune model on domain-specific QA pairs",
            "Adjust chunk size for better context capture",
            "Implement query expansion for better retrieval",
            "Add more comprehensive test cases"
        ]
    }
    
    with open("final_report.json", "w") as f:
        json.dump(report, f, indent=2)
    
    print("\n" + "="*60)
    print("Pipeline Complete!")
    print("="*60)
    print("\nGenerated files:")
    print("- baseline_evaluation.json")
    print("- final_report.json")
    print("- rexx_chroma_db/ (vector database)")
    print("- rexx_rag_metadata.json")
    
    print("\nNext steps:")
    print("1. Review the evaluation results")
    print("2. Run fine-tuning if needed (requires more compute)")
    print("3. Implement improvements based on error analysis")
    print("4. Re-evaluate to show improvement")


if __name__ == "__main__":
    main()
