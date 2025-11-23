"""
Baseline Evaluation Script for Rexx HR Q&A System
Evaluates llama2 model performance on test dataset
"""

import json
import requests
import time
from typing import List, Dict
from datetime import datetime
import os

# Metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re


class BaselineEvaluator:
    """Evaluate LLM baseline performance on Q&A dataset."""

    def __init__(self, model_name: str = "llama2", ollama_url: str = "http://localhost:11434"):
        self.model_name = model_name
        self.ollama_url = ollama_url
        self.results = []

    def check_ollama(self) -> bool:
        """Check if Ollama is running."""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False

    def query_model(self, question: str, use_context: bool = False, context: str = "") -> str:
        """Send question to Ollama model."""
        if use_context:
            prompt = f"""Du bist ein Experte für rexx HR Software. Beantworte die folgende Frage basierend auf dem Kontext.

Kontext:
{context}

Frage: {question}

Antwort:"""
        else:
            # Baseline without RAG context
            prompt = f"""Du bist ein Experte für rexx HR Software. Beantworte die folgende Frage kurz und präzise auf Deutsch.

Frage: {question}

Antwort:"""

        try:
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.3,
                        "num_predict": 256
                    }
                },
                timeout=120
            )

            if response.status_code == 200:
                return response.json().get('response', '').strip()
            else:
                return f"ERROR: Status {response.status_code}"

        except Exception as e:
            return f"ERROR: {str(e)}"

    def calculate_metrics(self, generated: str, expected: str) -> Dict:
        """Calculate evaluation metrics between generated and expected answer."""

        # Clean texts
        gen_clean = generated.lower().strip()
        exp_clean = expected.lower().strip()

        # 1. Exact Match
        exact_match = 1.0 if gen_clean == exp_clean else 0.0

        # 2. TF-IDF Cosine Similarity
        try:
            vectorizer = TfidfVectorizer()
            tfidf_matrix = vectorizer.fit_transform([gen_clean, exp_clean])
            cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        except:
            cosine_sim = 0.0

        # 3. Word Overlap (Jaccard-like)
        gen_words = set(re.findall(r'\w+', gen_clean))
        exp_words = set(re.findall(r'\w+', exp_clean))

        if len(exp_words) > 0:
            precision = len(gen_words & exp_words) / len(gen_words) if len(gen_words) > 0 else 0
            recall = len(gen_words & exp_words) / len(exp_words)
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        else:
            precision, recall, f1 = 0, 0, 0

        # 4. Contains key terms check
        key_terms_found = sum(1 for word in exp_words if word in gen_words and len(word) > 4)
        key_terms_ratio = key_terms_found / max(len([w for w in exp_words if len(w) > 4]), 1)

        # 5. Length ratio
        length_ratio = min(len(gen_clean), len(exp_clean)) / max(len(gen_clean), len(exp_clean), 1)

        return {
            "exact_match": exact_match,
            "cosine_similarity": round(cosine_sim, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4),
            "key_terms_ratio": round(key_terms_ratio, 4),
            "length_ratio": round(length_ratio, 4)
        }

    def evaluate_dataset(self, dataset_path: str, use_rag: bool = False, rag_system=None) -> Dict:
        """Run evaluation on test dataset."""

        # Load dataset
        with open(dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        test_questions = data['test']
        print(f"\nEvaluating {len(test_questions)} test questions...")
        print(f"Model: {self.model_name}")
        print(f"RAG enabled: {use_rag}")
        print("=" * 60)

        self.results = []
        start_time = time.time()

        for i, item in enumerate(test_questions):
            question = item['question']
            expected = item['expected_answer']

            print(f"\n[{i+1}/{len(test_questions)}] {question[:50]}...")

            # Get context from RAG if enabled
            context = ""
            if use_rag and rag_system:
                retrieved = rag_system.retrieve(question, n_results=3)
                context = "\n".join([chunk['text'] for chunk in retrieved])

            # Query model
            q_start = time.time()
            generated = self.query_model(question, use_context=use_rag, context=context)
            q_time = time.time() - q_start

            # Calculate metrics
            metrics = self.calculate_metrics(generated, expected)

            result = {
                "id": item.get('id', f'test_{i}'),
                "question": question,
                "expected_answer": expected,
                "generated_answer": generated,
                "category": item.get('category', 'unknown'),
                "difficulty": item.get('difficulty', 'unknown'),
                "metrics": metrics,
                "response_time": round(q_time, 2)
            }
            self.results.append(result)

            print(f"   Cosine Sim: {metrics['cosine_similarity']:.2f} | F1: {metrics['f1_score']:.2f} | Time: {q_time:.1f}s")

        total_time = time.time() - start_time

        # Calculate aggregate metrics
        aggregate = self.calculate_aggregate_metrics()
        aggregate['total_time_seconds'] = round(total_time, 2)
        aggregate['avg_response_time'] = round(total_time / len(test_questions), 2)

        return {
            "metadata": {
                "model": self.model_name,
                "rag_enabled": use_rag,
                "timestamp": datetime.now().isoformat(),
                "num_questions": len(test_questions)
            },
            "aggregate_metrics": aggregate,
            "results": self.results
        }

    def calculate_aggregate_metrics(self) -> Dict:
        """Calculate aggregate metrics across all results."""
        if not self.results:
            return {}

        metrics_keys = ['cosine_similarity', 'precision', 'recall', 'f1_score', 'key_terms_ratio']

        aggregate = {}
        for key in metrics_keys:
            values = [r['metrics'][key] for r in self.results]
            aggregate[f'avg_{key}'] = round(sum(values) / len(values), 4)

        # By category
        categories = {}
        for r in self.results:
            cat = r['category']
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(r['metrics']['f1_score'])

        aggregate['by_category'] = {
            cat: round(sum(scores)/len(scores), 4)
            for cat, scores in categories.items()
        }

        # By difficulty
        difficulties = {}
        for r in self.results:
            diff = r['difficulty']
            if diff not in difficulties:
                difficulties[diff] = []
            difficulties[diff].append(r['metrics']['f1_score'])

        aggregate['by_difficulty'] = {
            diff: round(sum(scores)/len(scores), 4)
            for diff, scores in difficulties.items()
        }

        return aggregate

    def save_results(self, output_path: str, evaluation_results: Dict):
        """Save evaluation results to JSON file."""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(evaluation_results, f, ensure_ascii=False, indent=2)
        print(f"\nResults saved to: {output_path}")

    def print_summary(self, evaluation_results: Dict):
        """Print evaluation summary."""
        agg = evaluation_results['aggregate_metrics']

        print("\n" + "=" * 60)
        print("EVALUATION SUMMARY")
        print("=" * 60)
        print(f"Model: {evaluation_results['metadata']['model']}")
        print(f"RAG Enabled: {evaluation_results['metadata']['rag_enabled']}")
        print(f"Questions Evaluated: {evaluation_results['metadata']['num_questions']}")
        print(f"Total Time: {agg.get('total_time_seconds', 0)}s")
        print(f"Avg Response Time: {agg.get('avg_response_time', 0)}s")

        print("\n--- Overall Metrics ---")
        print(f"Avg Cosine Similarity: {agg['avg_cosine_similarity']:.4f}")
        print(f"Avg Precision: {agg['avg_precision']:.4f}")
        print(f"Avg Recall: {agg['avg_recall']:.4f}")
        print(f"Avg F1 Score: {agg['avg_f1_score']:.4f}")
        print(f"Avg Key Terms Ratio: {agg['avg_key_terms_ratio']:.4f}")

        print("\n--- F1 Score by Category ---")
        for cat, score in agg.get('by_category', {}).items():
            print(f"  {cat}: {score:.4f}")

        print("\n--- F1 Score by Difficulty ---")
        for diff, score in agg.get('by_difficulty', {}).items():
            print(f"  {diff}: {score:.4f}")


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(script_dir, "rexx_qa_dataset_curated.json")

    # Initialize evaluator
    evaluator = BaselineEvaluator(model_name="llama2")

    # Check Ollama
    if not evaluator.check_ollama():
        print("ERROR: Ollama is not running!")
        print("Please start Ollama with: ollama serve")
        return

    print("Ollama is running. Starting baseline evaluation...")

    # Run baseline evaluation (without RAG)
    print("\n" + "=" * 60)
    print("BASELINE EVALUATION (without RAG)")
    print("=" * 60)

    results = evaluator.evaluate_dataset(dataset_path, use_rag=False)

    # Save results
    output_path = os.path.join(script_dir, "baseline_results.json")
    evaluator.save_results(output_path, results)

    # Print summary
    evaluator.print_summary(results)


if __name__ == "__main__":
    main()
