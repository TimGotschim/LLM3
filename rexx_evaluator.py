"""
Comprehensive evaluation script for Rexx RAG system
Implements various metrics for evaluating retrieval and generation quality
"""

import json
import time
from typing import List, Dict, Tuple, Optional
import numpy as np
from datetime import datetime
import pandas as pd
from collections import defaultdict

# Text similarity metrics
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.translate.bleu_score import sentence_bleu
from nltk.tokenize import word_tokenize

# For ROUGE scores
try:
    from rouge_score import rouge_scorer
    ROUGE_AVAILABLE = True
except ImportError:
    ROUGE_AVAILABLE = False
    print("Warning: rouge-score not installed. Install with: pip install rouge-score")

# For BERTScore
try:
    from bert_score import score as bert_score
    BERTSCORE_AVAILABLE = True
except ImportError:
    BERTSCORE_AVAILABLE = False
    print("Warning: bert-score not installed. Install with: pip install bert-score")


class RAGEvaluator:
    """
    Comprehensive evaluation system for RAG performance.
    """
    
    def __init__(self, rag_system):
        """
        Initialize evaluator.
        
        Args:
            rag_system: Initialized RexxRAGSystem instance
        """
        self.rag_system = rag_system
        
        # Download NLTK data if needed
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        
        # Initialize scorers
        if ROUGE_AVAILABLE:
            self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'])
        
        self.tfidf = TfidfVectorizer()
        
        # Results storage
        self.evaluation_results = {
            "timestamp": datetime.now().isoformat(),
            "metrics": {},
            "detailed_results": []
        }
    
    def create_test_dataset(self, manual_qa_pairs: Optional[List[Dict]] = None) -> List[Dict]:
        """
        Create or load test dataset.
        
        Args:
            manual_qa_pairs: Optional manual QA pairs
            
        Returns:
            List of test cases
        """
        test_dataset = []
        
        # Add manual QA pairs if provided
        if manual_qa_pairs:
            test_dataset.extend(manual_qa_pairs)
        
        # Generate automatic test cases based on document chunks
        auto_questions = [
            "What are the main features of the HR module?",
            "How does user authentication work?",
            "What are the system requirements?",
            "How to configure employee profiles?",
            "What reporting capabilities are available?",
            "How does the payroll module function?",
            "What are the data backup procedures?",
            "How to set up user permissions?",
            "What integrations are supported?",
            "How to generate compliance reports?"
        ]
        
        for q in auto_questions:
            test_dataset.append({"question": q})
        
        return test_dataset
    
    def evaluate_retrieval(self, test_questions: List[str], k_values: List[int] = [1, 3, 5, 10]) -> Dict:
        """
        Evaluate retrieval performance.
        
        Args:
            test_questions: List of test questions
            k_values: Different k values for retrieval evaluation
            
        Returns:
            Retrieval metrics
        """
        print("Evaluating retrieval performance...")
        
        retrieval_metrics = {
            "recall_at_k": {},
            "precision_at_k": {},
            "mrr": 0,  # Mean Reciprocal Rank
            "retrieval_times": []
        }
        
        reciprocal_ranks = []
        
        for question in test_questions:
            start_time = time.time()
            
            # Retrieve with maximum k
            max_k = max(k_values)
            retrieved_chunks = self.rag_system.retrieve(question, n_results=max_k)
            
            retrieval_time = time.time() - start_time
            retrieval_metrics["retrieval_times"].append(retrieval_time)
            
            # Calculate metrics for different k values
            for k in k_values:
                k_chunks = retrieved_chunks[:k]
                
                # Here we're using distance as a proxy for relevance
                # In practice, you might have ground truth relevance labels
                relevant_found = sum(1 for chunk in k_chunks if chunk['distance'] < 0.5)
                
                if question not in retrieval_metrics["recall_at_k"]:
                    retrieval_metrics["recall_at_k"][question] = {}
                    retrieval_metrics["precision_at_k"][question] = {}
                
                retrieval_metrics["recall_at_k"][question][k] = relevant_found / k
                retrieval_metrics["precision_at_k"][question][k] = relevant_found / k
            
            # Calculate reciprocal rank
            for i, chunk in enumerate(retrieved_chunks):
                if chunk['distance'] < 0.5:  # Relevant threshold
                    reciprocal_ranks.append(1 / (i + 1))
                    break
            else:
                reciprocal_ranks.append(0)
        
        # Calculate averages
        retrieval_metrics["mrr"] = np.mean(reciprocal_ranks)
        retrieval_metrics["avg_retrieval_time"] = np.mean(retrieval_metrics["retrieval_times"])
        
        # Calculate average precision/recall at k
        avg_recall_at_k = {}
        avg_precision_at_k = {}
        
        for k in k_values:
            recalls = [retrieval_metrics["recall_at_k"][q][k] for q in test_questions]
            precisions = [retrieval_metrics["precision_at_k"][q][k] for q in test_questions]
            
            avg_recall_at_k[f"R@{k}"] = np.mean(recalls)
            avg_precision_at_k[f"P@{k}"] = np.mean(precisions)
        
        retrieval_metrics["avg_recall_at_k"] = avg_recall_at_k
        retrieval_metrics["avg_precision_at_k"] = avg_precision_at_k
        
        return retrieval_metrics
    
    def evaluate_generation(self, test_cases: List[Dict]) -> Dict:
        """
        Evaluate generation quality.
        
        Args:
            test_cases: Test cases with questions and optionally expected answers
            
        Returns:
            Generation metrics
        """
        print("Evaluating generation quality...")
        
        generation_metrics = {
            "response_times": [],
            "response_lengths": [],
            "contains_answer": [],
            "rouge_scores": defaultdict(list),
            "bleu_scores": [],
            "bert_scores": defaultdict(list),
            "coherence_scores": []
        }
        
        for test_case in test_cases:
            question = test_case["question"]
            expected_answer = test_case.get("expected_answer", None)
            
            # Generate response
            start_time = time.time()
            response = self.rag_system.query(question, n_results=5)
            response_time = time.time() - start_time
            
            answer = response["answer"]
            
            # Basic metrics
            generation_metrics["response_times"].append(response_time)
            generation_metrics["response_lengths"].append(len(answer.split()))
            
            # If we have expected answer, calculate similarity metrics
            if expected_answer:
                # Check if answer contains expected content
                contains = expected_answer.lower() in answer.lower()
                generation_metrics["contains_answer"].append(contains)
                
                # BLEU score
                reference = word_tokenize(expected_answer.lower())
                hypothesis = word_tokenize(answer.lower())
                bleu = sentence_bleu([reference], hypothesis)
                generation_metrics["bleu_scores"].append(bleu)
                
                # ROUGE scores
                if ROUGE_AVAILABLE:
                    scores = self.rouge_scorer.score(expected_answer, answer)
                    for metric, score in scores.items():
                        generation_metrics["rouge_scores"][metric].append(score.fmeasure)
                
                # BERT Score
                if BERTSCORE_AVAILABLE:
                    P, R, F1 = bert_score(
                        [answer], [expected_answer], 
                        lang="en", verbose=False
                    )
                    generation_metrics["bert_scores"]["precision"].append(P.item())
                    generation_metrics["bert_scores"]["recall"].append(R.item())
                    generation_metrics["bert_scores"]["f1"].append(F1.item())
            
            # Coherence score (simplified - checks for complete sentences)
            coherence = self._calculate_coherence(answer)
            generation_metrics["coherence_scores"].append(coherence)
        
        # Calculate averages
        generation_metrics["avg_response_time"] = np.mean(generation_metrics["response_times"])
        generation_metrics["avg_response_length"] = np.mean(generation_metrics["response_lengths"])
        
        if generation_metrics["contains_answer"]:
            generation_metrics["accuracy"] = np.mean(generation_metrics["contains_answer"])
        
        if generation_metrics["bleu_scores"]:
            generation_metrics["avg_bleu"] = np.mean(generation_metrics["bleu_scores"])
        
        for metric in generation_metrics["rouge_scores"]:
            scores = generation_metrics["rouge_scores"][metric]
            generation_metrics[f"avg_{metric}"] = np.mean(scores) if scores else 0
        
        for metric in generation_metrics["bert_scores"]:
            scores = generation_metrics["bert_scores"][metric]
            generation_metrics[f"avg_bert_{metric}"] = np.mean(scores) if scores else 0
        
        generation_metrics["avg_coherence"] = np.mean(generation_metrics["coherence_scores"])
        
        return generation_metrics
    
    def _calculate_coherence(self, text: str) -> float:
        """
        Calculate coherence score for generated text.
        
        Args:
            text: Generated text
            
        Returns:
            Coherence score (0-1)
        """
        if not text:
            return 0
        
        sentences = text.split('.')
        complete_sentences = sum(
            1 for s in sentences 
            if len(s.strip()) > 10 and any(c.isalpha() for c in s)
        )
        
        # Check for proper structure
        has_proper_start = text[0].isupper() if text else False
        has_proper_end = text.rstrip()[-1] in '.!?' if text else False
        
        # Calculate score
        sentence_score = complete_sentences / max(len(sentences), 1)
        structure_score = (has_proper_start + has_proper_end) / 2
        
        return (sentence_score + structure_score) / 2
    
    def evaluate_end_to_end(self, test_dataset: List[Dict]) -> Dict:
        """
        Perform end-to-end evaluation.
        
        Args:
            test_dataset: Complete test dataset
            
        Returns:
            Comprehensive evaluation results
        """
        print("\nPerforming end-to-end RAG evaluation...")
        print(f"Test dataset size: {len(test_dataset)}")
        
        # Extract questions for retrieval evaluation
        questions = [tc["question"] for tc in test_dataset]
        
        # Evaluate retrieval
        retrieval_results = self.evaluate_retrieval(questions)
        
        # Evaluate generation
        generation_results = self.evaluate_generation(test_dataset)
        
        # Combine results
        self.evaluation_results["metrics"] = {
            "retrieval": retrieval_results,
            "generation": generation_results
        }
        
        # Calculate overall scores
        overall_metrics = self._calculate_overall_metrics()
        self.evaluation_results["overall"] = overall_metrics
        
        return self.evaluation_results
    
    def _calculate_overall_metrics(self) -> Dict:
        """Calculate overall performance metrics."""
        retrieval = self.evaluation_results["metrics"]["retrieval"]
        generation = self.evaluation_results["metrics"]["generation"]
        
        overall = {
            "mrr": retrieval["mrr"],
            "avg_retrieval_time": retrieval["avg_retrieval_time"],
            "avg_response_time": generation["avg_response_time"],
            "avg_response_length": generation["avg_response_length"],
            "coherence": generation["avg_coherence"]
        }
        
        # Add accuracy if available
        if "accuracy" in generation:
            overall["accuracy"] = generation["accuracy"]
        
        # Add ROUGE scores if available
        for metric in ["avg_rouge1", "avg_rouge2", "avg_rougeL"]:
            if metric in generation:
                overall[metric] = generation[metric]
        
        # Calculate F1 scores for retrieval
        for k in [1, 3, 5]:
            p_key = f"P@{k}"
            r_key = f"R@{k}"
            if p_key in retrieval["avg_precision_at_k"] and r_key in retrieval["avg_recall_at_k"]:
                p = retrieval["avg_precision_at_k"][p_key]
                r = retrieval["avg_recall_at_k"][r_key]
                f1 = 2 * (p * r) / (p + r) if (p + r) > 0 else 0
                overall[f"F1@{k}"] = f1
        
        # Overall RAG score (weighted combination)
        rag_score = (
            0.3 * overall.get("mrr", 0) +
            0.3 * overall.get("accuracy", overall.get("coherence", 0)) +
            0.2 * overall.get("F1@5", 0) +
            0.2 * (1 - min(overall.get("avg_response_time", 1), 1))  # Inverse of response time
        )
        overall["rag_score"] = rag_score
        
        return overall
    
    def generate_report(self, output_path: str = "evaluation_report.json"):
        """
        Generate evaluation report.
        
        Args:
            output_path: Path to save the report
        """
        # Save detailed report
        with open(output_path, 'w') as f:
            json.dump(self.evaluation_results, f, indent=2)
        
        # Generate summary
        print("\n" + "="*60)
        print("RAG EVALUATION REPORT")
        print("="*60)
        
        overall = self.evaluation_results.get("overall", {})
        
        print("\nOVERALL PERFORMANCE:")
        print(f"- RAG Score: {overall.get('rag_score', 0):.3f}")
        print(f"- MRR: {overall.get('mrr', 0):.3f}")
        print(f"- Accuracy: {overall.get('accuracy', 'N/A')}")
        print(f"- Coherence: {overall.get('coherence', 0):.3f}")
        
        print("\nRETRIEVAL METRICS:")
        retrieval = self.evaluation_results["metrics"]["retrieval"]
        for k in [1, 3, 5]:
            p = retrieval["avg_precision_at_k"].get(f"P@{k}", 0)
            r = retrieval["avg_recall_at_k"].get(f"R@{k}", 0)
            f1 = overall.get(f"F1@{k}", 0)
            print(f"- @{k}: P={p:.3f}, R={r:.3f}, F1={f1:.3f}")
        
        print(f"- Avg Retrieval Time: {retrieval['avg_retrieval_time']:.3f}s")
        
        print("\nGENERATION METRICS:")
        generation = self.evaluation_results["metrics"]["generation"]
        print(f"- Avg Response Time: {generation['avg_response_time']:.3f}s")
        print(f"- Avg Response Length: {generation['avg_response_length']:.0f} words")
        
        if "avg_bleu" in generation:
            print(f"- Avg BLEU Score: {generation['avg_bleu']:.3f}")
        
        if "avg_rouge1" in generation:
            print(f"- Avg ROUGE-1: {generation['avg_rouge1']:.3f}")
            print(f"- Avg ROUGE-2: {generation['avg_rouge2']:.3f}")
            print(f"- Avg ROUGE-L: {generation['avg_rougeL']:.3f}")
        
        print("\n" + "="*60)
        print(f"Report saved to: {output_path}")
    
    def compare_models(self, model_configs: List[Dict]) -> pd.DataFrame:
        """
        Compare multiple model configurations.
        
        Args:
            model_configs: List of model configurations to compare
            
        Returns:
            Comparison dataframe
        """
        comparison_results = []
        
        for config in model_configs:
            print(f"\nEvaluating configuration: {config['name']}")
            
            # Update RAG system with config
            self.rag_system.model_name = config.get("model_name", self.rag_system.model_name)
            
            # Run evaluation
            test_dataset = self.create_test_dataset()
            results = self.evaluate_end_to_end(test_dataset)
            
            # Extract key metrics
            metrics = {
                "Configuration": config['name'],
                "Model": config.get("model_name", "default"),
                "RAG Score": results["overall"]["rag_score"],
                "MRR": results["overall"]["mrr"],
                "F1@5": results["overall"].get("F1@5", 0),
                "Avg Response Time": results["overall"]["avg_response_time"],
                "Coherence": results["overall"]["coherence"]
            }
            
            comparison_results.append(metrics)
        
        # Create comparison dataframe
        df = pd.DataFrame(comparison_results)
        df = df.sort_values("RAG Score", ascending=False)
        
        return df


def main():
    """Example evaluation workflow."""

    import os
    from rexx_rag_system import RexxRAGSystem

    # Use relative path from script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    pdf_folder = os.path.join(script_dir, "rexx_pdfs")
    rag_system = RexxRAGSystem(pdf_folder)
    
    # Process PDFs if needed
    rag_system.process_pdfs()
    
    # Initialize evaluator
    evaluator = RAGEvaluator(rag_system)
    
    # Create test dataset with expected answers
    test_dataset = [
        {
            "question": "What is Rexx Systems?",
            "expected_answer": "HR software system"
        },
        {
            "question": "What are the main modules?",
            "expected_answer": "HR, payroll, recruiting"
        },
        {
            "question": "How to configure user permissions?",
            "expected_answer": "admin panel"
        },
        {
            "question": "What database does Rexx use?",
            "expected_answer": "SQL database"
        },
        {
            "question": "What are the system requirements?"
        },
        {
            "question": "How does the reporting module work?"
        },
        {
            "question": "What integrations are available?"
        }
    ]
    
    # Run evaluation
    results = evaluator.evaluate_end_to_end(test_dataset)
    
    # Generate report
    evaluator.generate_report("rexx_evaluation_report.json")
    
    # Compare different configurations
    print("\n\nComparing different model configurations...")
    
    model_configs = [
        {"name": "Baseline (Llama2)", "model_name": "llama2"},
        {"name": "Mistral", "model_name": "mistral"},
        {"name": "Neural-Chat", "model_name": "neural-chat"}
    ]
    
    # Note: This requires these models to be available in Ollama
    # comparison_df = evaluator.compare_models(model_configs)
    # print("\nModel Comparison:")
    # print(comparison_df.to_string(index=False))
    # comparison_df.to_csv("model_comparison.csv", index=False)


if __name__ == "__main__":
    main()
