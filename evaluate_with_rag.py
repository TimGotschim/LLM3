"""
Evaluation Script with RAG (Retrieval Augmented Generation)
Uses the existing RAG system to provide context from PDFs
"""

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluate_baseline import BaselineEvaluator
from rexx_rag_system import RexxRAGSystem


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(script_dir, "rexx_qa_dataset_curated.json")
    pdf_folder = os.path.join(script_dir, "rexx_pdfs")

    # Initialize RAG system
    print("Initializing RAG system...")
    rag_system = RexxRAGSystem(pdf_folder, model_name="llama2")

    # Process PDFs (or load existing)
    print("Processing PDFs...")
    rag_system.process_pdfs()

    print(f"\nRAG Stats: {rag_system.get_stats()['total_chunks']} chunks indexed")

    # Load test dataset
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    test_questions = data['test']
    print(f"\nEvaluating {len(test_questions)} test questions with RAG...")
    print("=" * 60)

    results = []
    evaluator = BaselineEvaluator(model_name="llama2")

    start_time = time.time()

    for i, item in enumerate(test_questions):
        question = item['question']
        expected = item['expected_answer']

        print(f"\n[{i+1}/{len(test_questions)}] {question[:50]}...")

        # Query RAG system
        q_start = time.time()
        response = rag_system.query(question, n_results=5)
        q_time = time.time() - q_start

        generated = response['answer']

        # Calculate metrics
        metrics = evaluator.calculate_metrics(generated, expected)

        result = {
            "id": item.get('id', f'test_{i}'),
            "question": question,
            "expected_answer": expected,
            "generated_answer": generated,
            "category": item.get('category', 'unknown'),
            "difficulty": item.get('difficulty', 'unknown'),
            "sources_used": response.get('sources', []),
            "metrics": metrics,
            "response_time": round(q_time, 2)
        }
        results.append(result)

        print(f"   Cosine Sim: {metrics['cosine_similarity']:.2f} | F1: {metrics['f1_score']:.2f} | Time: {q_time:.1f}s")

    total_time = time.time() - start_time

    # Calculate aggregate metrics
    aggregate = {}
    metrics_keys = ['cosine_similarity', 'precision', 'recall', 'f1_score', 'key_terms_ratio']

    for key in metrics_keys:
        values = [r['metrics'][key] for r in results]
        aggregate[f'avg_{key}'] = round(sum(values) / len(values), 4)

    # By category
    categories = {}
    for r in results:
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
    for r in results:
        diff = r['difficulty']
        if diff not in difficulties:
            difficulties[diff] = []
        difficulties[diff].append(r['metrics']['f1_score'])

    aggregate['by_difficulty'] = {
        diff: round(sum(scores)/len(scores), 4)
        for diff, scores in difficulties.items()
    }

    aggregate['total_time_seconds'] = round(total_time, 2)
    aggregate['avg_response_time'] = round(total_time / len(test_questions), 2)

    # Save results
    output = {
        "metadata": {
            "model": "llama2",
            "rag_enabled": True,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "num_questions": len(test_questions)
        },
        "aggregate_metrics": aggregate,
        "results": results
    }

    output_path = os.path.join(script_dir, "rag_results.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to: {output_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("RAG EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Model: llama2 with RAG")
    print(f"Questions Evaluated: {len(test_questions)}")
    print(f"Total Time: {aggregate['total_time_seconds']}s")
    print(f"Avg Response Time: {aggregate['avg_response_time']}s")

    print("\n--- Overall Metrics ---")
    print(f"Avg Cosine Similarity: {aggregate['avg_cosine_similarity']:.4f}")
    print(f"Avg Precision: {aggregate['avg_precision']:.4f}")
    print(f"Avg Recall: {aggregate['avg_recall']:.4f}")
    print(f"Avg F1 Score: {aggregate['avg_f1_score']:.4f}")
    print(f"Avg Key Terms Ratio: {aggregate['avg_key_terms_ratio']:.4f}")

    print("\n--- F1 Score by Category ---")
    for cat, score in aggregate['by_category'].items():
        print(f"  {cat}: {score:.4f}")

    # Load baseline for comparison
    baseline_path = os.path.join(script_dir, "baseline_results.json")
    if os.path.exists(baseline_path):
        with open(baseline_path, 'r') as f:
            baseline = json.load(f)

        print("\n" + "=" * 60)
        print("COMPARISON: BASELINE vs RAG")
        print("=" * 60)

        baseline_agg = baseline['aggregate_metrics']

        metrics_compare = ['avg_cosine_similarity', 'avg_precision', 'avg_recall', 'avg_f1_score']

        print(f"\n{'Metric':<25} {'Baseline':>12} {'RAG':>12} {'Improvement':>15}")
        print("-" * 66)

        for metric in metrics_compare:
            base_val = baseline_agg.get(metric, 0)
            rag_val = aggregate.get(metric, 0)
            improvement = rag_val - base_val
            pct_improvement = (improvement / base_val * 100) if base_val > 0 else 0

            sign = "+" if improvement > 0 else ""
            print(f"{metric:<25} {base_val:>12.4f} {rag_val:>12.4f} {sign}{improvement:>10.4f} ({sign}{pct_improvement:.1f}%)")


if __name__ == "__main__":
    main()
