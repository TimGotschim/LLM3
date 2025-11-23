"""
Training Progress Tracker for Rexx RAG System
Tracks model improvements across different training iterations
and provides detailed analysis and visualization of results.
"""

import json
import os
from datetime import datetime
from typing import List, Dict, Optional
import numpy as np
from collections import defaultdict


class TrainingProgressTracker:
    """Track and analyze model improvements across training iterations."""
    
    def __init__(self, tracking_file: str = "training_progress.json"):
        self.tracking_file = tracking_file
        self.history = []
        self.load_history()
    
    def load_history(self):
        """Load existing training history."""
        if os.path.exists(self.tracking_file):
            with open(self.tracking_file, 'r') as f:
                self.history = json.load(f)
            print(f"Loaded {len(self.history)} training iterations")
        else:
            print("No training history found. Starting fresh.")
    
    def save_history(self):
        """Save training history."""
        with open(self.tracking_file, 'w') as f:
            json.dump(self.history, f, indent=2)
        print(f"✓ Saved training history to {self.tracking_file}")
    
    def record_iteration(self,
                        iteration_name: str,
                        training_data_info: Dict,
                        evaluation_results: Dict,
                        model_config: Dict,
                        notes: str = ""):
        """
        Record a training iteration with its results.
        
        Args:
            iteration_name: Name/ID for this iteration (e.g., "baseline", "iter_1")
            training_data_info: Info about training data used
            evaluation_results: Results from evaluation
            model_config: Model configuration details
            notes: Additional notes about this iteration
        """
        iteration = {
            "iteration_name": iteration_name,
            "timestamp": datetime.now().isoformat(),
            "training_data": training_data_info,
            "evaluation": {
                "overall": evaluation_results.get("overall", {}),
                "retrieval": {
                    "mrr": evaluation_results.get("metrics", {}).get("retrieval", {}).get("mrr", 0),
                    "avg_retrieval_time": evaluation_results.get("metrics", {}).get("retrieval", {}).get("avg_retrieval_time", 0),
                },
                "generation": {
                    "accuracy": evaluation_results.get("metrics", {}).get("generation", {}).get("accuracy", None),
                    "avg_response_time": evaluation_results.get("metrics", {}).get("generation", {}).get("avg_response_time", 0),
                    "coherence": evaluation_results.get("metrics", {}).get("generation", {}).get("avg_coherence", 0),
                    "avg_bleu": evaluation_results.get("metrics", {}).get("generation", {}).get("avg_bleu", None),
                }
            },
            "model_config": model_config,
            "notes": notes
        }
        
        self.history.append(iteration)
        self.save_history()
        
        print(f"✓ Recorded iteration: {iteration_name}")
        return iteration
    
    def compare_iterations(self, iteration1: str, iteration2: str) -> Dict:
        """
        Compare two training iterations.
        
        Args:
            iteration1: First iteration name
            iteration2: Second iteration name
            
        Returns:
            Comparison dictionary with improvements/regressions
        """
        iter1 = None
        iter2 = None
        
        for it in self.history:
            if it['iteration_name'] == iteration1:
                iter1 = it
            if it['iteration_name'] == iteration2:
                iter2 = it
        
        if not iter1 or not iter2:
            return {"error": "One or both iterations not found"}
        
        comparison = {
            "iteration1": iteration1,
            "iteration2": iteration2,
            "improvements": {},
            "regressions": {},
            "summary": {}
        }
        
        # Compare key metrics
        metrics_to_compare = [
            ("RAG Score", "evaluation.overall.rag_score"),
            ("MRR", "evaluation.retrieval.mrr"),
            ("Accuracy", "evaluation.generation.accuracy"),
            ("Coherence", "evaluation.generation.coherence"),
            ("BLEU Score", "evaluation.generation.avg_bleu"),
            ("Response Time", "evaluation.generation.avg_response_time"),
        ]
        
        for metric_name, metric_path in metrics_to_compare:
            val1 = self._get_nested_value(iter1, metric_path)
            val2 = self._get_nested_value(iter2, metric_path)
            
            if val1 is not None and val2 is not None:
                change = val2 - val1
                percent_change = (change / val1 * 100) if val1 != 0 else 0
                
                result = {
                    "old_value": round(val1, 4),
                    "new_value": round(val2, 4),
                    "absolute_change": round(change, 4),
                    "percent_change": round(percent_change, 2)
                }
                
                # Response time is better when lower
                if "time" in metric_name.lower():
                    if change < 0:
                        comparison["improvements"][metric_name] = result
                    elif change > 0:
                        comparison["regressions"][metric_name] = result
                else:
                    if change > 0:
                        comparison["improvements"][metric_name] = result
                    elif change < 0:
                        comparison["regressions"][metric_name] = result
        
        # Summary statistics
        comparison["summary"] = {
            "total_improvements": len(comparison["improvements"]),
            "total_regressions": len(comparison["regressions"]),
            "net_improvement": len(comparison["improvements"]) - len(comparison["regressions"])
        }
        
        return comparison
    
    def _get_nested_value(self, d: Dict, path: str):
        """Get nested dictionary value using dot notation."""
        keys = path.split('.')
        value = d
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return None
        return value
    
    def get_best_iteration(self, metric: str = "rag_score") -> Optional[Dict]:
        """
        Find the best performing iteration based on a metric.
        
        Args:
            metric: Metric to optimize (rag_score, mrr, accuracy, coherence)
        """
        if not self.history:
            return None
        
        metric_paths = {
            "rag_score": "evaluation.overall.rag_score",
            "mrr": "evaluation.retrieval.mrr",
            "accuracy": "evaluation.generation.accuracy",
            "coherence": "evaluation.generation.coherence",
            "bleu": "evaluation.generation.avg_bleu"
        }
        
        metric_path = metric_paths.get(metric, metric_paths["rag_score"])
        
        best_iter = None
        best_value = -float('inf')
        
        for iteration in self.history:
            value = self._get_nested_value(iteration, metric_path)
            if value is not None and value > best_value:
                best_value = value
                best_iter = iteration
        
        return best_iter
    
    def print_progress_report(self):
        """Print a comprehensive progress report."""
        if not self.history:
            print("No training history available")
            return
        
        print("\n" + "="*80)
        print("TRAINING PROGRESS REPORT")
        print("="*80)
        print(f"Total Iterations: {len(self.history)}")
        
        # Show all iterations chronologically
        print("\nIteration History:")
        print("-" * 80)
        
        for i, iteration in enumerate(self.history):
            print(f"\n{i+1}. {iteration['iteration_name']} ({iteration['timestamp'][:10]})")
            print(f"   Training Data: {iteration['training_data'].get('total_samples', 'N/A')} samples")
            
            eval_data = iteration['evaluation']
            overall = eval_data.get('overall', {})
            
            print(f"   RAG Score: {overall.get('rag_score', 'N/A'):.4f}")
            print(f"   MRR: {overall.get('mrr', 'N/A'):.4f}")
            print(f"   Accuracy: {overall.get('accuracy', 'N/A')}")
            print(f"   Coherence: {overall.get('coherence', 'N/A'):.4f}")
            
            if iteration['notes']:
                print(f"   Notes: {iteration['notes']}")
        
        # Best performers
        print("\n" + "-"*80)
        print("Best Performers:")
        
        metrics = ["rag_score", "mrr", "accuracy", "coherence"]
        for metric in metrics:
            best = self.get_best_iteration(metric)
            if best:
                value = self._get_nested_value(best, f"evaluation.overall.{metric}" if metric == "rag_score" else 
                                               f"evaluation.retrieval.{metric}" if metric == "mrr" else
                                               f"evaluation.generation.{metric}")
                if value is not None:
                    print(f"   {metric.upper()}: {best['iteration_name']} ({value:.4f})")
        
        # Latest vs Baseline comparison
        if len(self.history) >= 2:
            print("\n" + "-"*80)
            print("Latest vs Baseline Comparison:")
            
            baseline = self.history[0]
            latest = self.history[-1]
            
            comparison = self.compare_iterations(
                baseline['iteration_name'],
                latest['iteration_name']
            )
            
            if 'error' not in comparison:
                print(f"\nImprovements ({comparison['summary']['total_improvements']}):")
                for metric, data in comparison['improvements'].items():
                    print(f"   ✓ {metric}: {data['old_value']:.4f} → {data['new_value']:.4f} "
                          f"({data['percent_change']:+.2f}%)")
                
                if comparison['regressions']:
                    print(f"\nRegressions ({comparison['summary']['total_regressions']}):")
                    for metric, data in comparison['regressions'].items():
                        print(f"   ✗ {metric}: {data['old_value']:.4f} → {data['new_value']:.4f} "
                              f"({data['percent_change']:+.2f}%)")
        
        print("="*80)
    
    def export_for_visualization(self, output_file: str = "training_progress_viz.json"):
        """Export data in format suitable for visualization."""
        viz_data = {
            "iterations": [],
            "metrics_over_time": defaultdict(list)
        }
        
        for iteration in self.history:
            iteration_info = {
                "name": iteration['iteration_name'],
                "timestamp": iteration['timestamp'],
                "training_samples": iteration['training_data'].get('total_samples', 0)
            }
            
            # Extract metrics
            metrics = {
                "rag_score": self._get_nested_value(iteration, "evaluation.overall.rag_score"),
                "mrr": self._get_nested_value(iteration, "evaluation.retrieval.mrr"),
                "accuracy": self._get_nested_value(iteration, "evaluation.generation.accuracy"),
                "coherence": self._get_nested_value(iteration, "evaluation.generation.coherence"),
                "response_time": self._get_nested_value(iteration, "evaluation.generation.avg_response_time")
            }
            
            iteration_info["metrics"] = metrics
            viz_data["iterations"].append(iteration_info)
            
            # For time series
            for metric_name, metric_value in metrics.items():
                if metric_value is not None:
                    viz_data["metrics_over_time"][metric_name].append({
                        "iteration": iteration['iteration_name'],
                        "value": metric_value
                    })
        
        with open(output_file, 'w') as f:
            json.dump(dict(viz_data), f, indent=2)
        
        print(f"✓ Exported visualization data to {output_file}")


# ============================================================================
# INTEGRATION FUNCTIONS
# ============================================================================

def evaluate_and_track(rag_system, 
                       iteration_name: str,
                       training_data_file: Optional[str] = None,
                       test_dataset: Optional[List[Dict]] = None,
                       notes: str = ""):
    """
    Convenience function to evaluate and track in one step.
    
    Args:
        rag_system: Initialized RexxRAGSystem
        iteration_name: Name for this iteration
        training_data_file: Path to training data used
        test_dataset: Test dataset for evaluation
        notes: Additional notes
    """
    from rexx_evaluator import RAGEvaluator
    
    # Run evaluation
    evaluator = RAGEvaluator(rag_system)
    
    if test_dataset is None:
        # Use default test dataset
        test_dataset = [
            {"question": "What is Rexx Systems?", "expected_answer": "HR software"},
            {"question": "What modules are available?", "expected_answer": "HR payroll"},
            {"question": "How to add new employees?", "expected_answer": "employee management"},
            {"question": "What are the reporting features?", "expected_answer": "reports analytics"},
            {"question": "How to configure the system?", "expected_answer": "configuration settings"},
            {"question": "Can two users have the same User Name?", "expected_answer": "unique User Name"},
            {"question": "What is the organizational tree?", "expected_answer": "hierarchical structure"},
            {"question": "How does user authentication work?", "expected_answer": "login credentials"}
        ]
    
    results = evaluator.evaluate_end_to_end(test_dataset)
    
    # Get training data info
    training_info = {"total_samples": 0, "source": "none"}
    if training_data_file and os.path.exists(training_data_file):
        with open(training_data_file, 'r') as f:
            training_data = json.load(f)
            training_info = {
                "total_samples": len(training_data),
                "source": training_data_file,
                "manual_samples": sum(1 for ex in training_data if ex.get('metadata', {}).get('manual', False)),
                "automated_samples": sum(1 for ex in training_data if not ex.get('metadata', {}).get('manual', True))
            }
    
    # Track progress
    tracker = TrainingProgressTracker()
    tracker.record_iteration(
        iteration_name=iteration_name,
        training_data_info=training_info,
        evaluation_results=results,
        model_config={
            "model_name": rag_system.model_name,
            "embedding_model": "all-MiniLM-L6-v2",
            "chunk_size": 1000,
            "chunk_overlap": 200
        },
        notes=notes
    )
    
    return results, tracker


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

def example_tracking_workflow():
    """Example showing how to track improvements across iterations."""
    from rexx_rag_system import RexxRAGSystem
    import os
    
    print("Example: Tracking Model Improvements")
    print("="*60)
    
    # Initialize RAG system
    script_dir = os.path.dirname(os.path.abspath(__file__))
    pdf_folder = os.path.join(script_dir, "rexx_pdfs")
    rag_system = RexxRAGSystem(pdf_folder)
    rag_system.process_pdfs()
    
    # 1. Baseline evaluation (no training)
    print("\n1. Running baseline evaluation...")
    results1, tracker = evaluate_and_track(
        rag_system,
        iteration_name="baseline",
        notes="Initial system with no fine-tuning"
    )
    
    # 2. After automated training data
    print("\n2. Evaluation after automated training...")
    # (In practice, you would fine-tune here)
    results2, _ = evaluate_and_track(
        rag_system,
        iteration_name="automated_training",
        training_data_file="rexx_training_data.json",
        notes="Using 50 automated training examples"
    )
    
    # 3. After adding manual training data
    print("\n3. Evaluation after manual training...")
    results3, _ = evaluate_and_track(
        rag_system,
        iteration_name="manual_training",
        training_data_file="combined_training_data.json",
        notes="Added 10 high-quality manual examples"
    )
    
    # Show progress report
    tracker.print_progress_report()
    
    # Export for visualization
    tracker.export_for_visualization()


if __name__ == "__main__":
    example_tracking_workflow()
