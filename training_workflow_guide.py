"""
Complete Workflow Guide: Creating and Tracking Training Data
============================================================

This guide demonstrates the complete workflow for:
1. Creating manual training data
2. Combining with automated data
3. Evaluating improvements
4. Tracking progress over time
"""

import os
import sys
import json
from datetime import datetime


def print_step(step_num, title):
    """Print formatted step header."""
    print("\n" + "="*80)
    print(f"STEP {step_num}: {title}")
    print("="*80)


def complete_workflow():
    """Run the complete training data creation and tracking workflow."""
    
    print("\n" + "="*80)
    print("REXX RAG TRAINING DATA WORKFLOW")
    print("Complete Guide to Creating & Tracking Improvements")
    print("="*80)
    
    # ========================================================================
    # STEP 1: Create Manual Training Data
    # ========================================================================
    print_step(1, "Create High-Quality Manual Training Data")
    
    print("""
Manual training data is crucial for domain-specific improvements.
Quality > Quantity for manual data.

Best Practices:
- Focus on questions users actually ask
- Use real documentation as context
- Ensure answers are exact substrings of context
- Categorize by topic for better analysis
- Include varying difficulty levels
    """)
    
    input("Press Enter to create example manual data...")
    
    from create_manual_training_data import ManualTrainingDataCreator
    
    creator = ManualTrainingDataCreator("manual_training_data.json")
    
    # Add high-quality examples
    examples = [
        {
            "question": "Can two users have the same User Name in Rexx?",
            "context": "In Rexx Systems, each user must have a unique User Name. The system enforces this constraint to ensure proper user identification and access control. Duplicate User Names are not permitted within the same Rexx installation.",
            "answer": "each user must have a unique User Name",
            "source": "User Management Guide",
            "category": "user_management",
            "difficulty": "easy"
        },
        {
            "question": "How do I access the Jobportal Configurator?",
            "context": "Access to the Jobportal Configurator requires the permission right 'Recruiting Vakanzen Jobportal-Konfigurator activ?' (ID.30.334) to be granted under Admin > Ticket > Permission Profiles. Access the configurator by adding /customize.php to your job portal URL.",
            "answer": "requires the permission right 'Recruiting Vakanzen Jobportal-Konfigurator activ?' (ID.30.334)",
            "source": "Jobportal Configurator Guide",
            "category": "configuration",
            "difficulty": "medium"
        },
        {
            "question": "What happens when I click 'Publish' in the configurator?",
            "context": "When you click 'Veröffentlichen' (Publish) in the Jobportal Configurator, your draft becomes the current live version of the job portal. The publication date is displayed, allowing transparent tracking of who published when - especially helpful in teams with multiple editors.",
            "answer": "your draft becomes the current live version of the job portal",
            "source": "Jobportal Configurator Guide",
            "category": "features",
            "difficulty": "medium"
        },
        {
            "question": "How are permission profiles structured in Rexx?",
            "context": "Permission profiles (Berechtigungsprofile) in Rexx use a hierarchical structure. Users can have multiple profiles assigned. When multiple profiles are assigned, the system evaluates them in order - the first set value (Yes or No) takes precedence. The Base Profile provides default values that can be overridden by specific profiles.",
            "answer": "use a hierarchical structure. Users can have multiple profiles assigned",
            "source": "Administrator Training Manual",
            "category": "user_management",
            "difficulty": "hard"
        },
        {
            "question": "What is the FTE-Basis in Rexx?",
            "context": "The FTE-Basis (Full Time Equivalent) in Rexx defines the number of weekly hours that equals one FTE. For example, if set to 40.00 hours, this represents one full-time employee. Individual work time models are then defined in the system data under Time Management and Personnel/User Setup.",
            "answer": "defines the number of weekly hours that equals one FTE",
            "source": "System Configuration Guide",
            "category": "configuration",
            "difficulty": "medium"
        }
    ]
    
    creator.add_batch_from_template(examples)
    creator.save()
    creator.print_statistics()
    
    print("\n✓ Created 5 high-quality manual training examples")
    print("  TIP: Add more examples by running create_manual_training_data.py in interactive mode")
    
    # ========================================================================
    # STEP 2: Generate Automated Training Data
    # ========================================================================
    print_step(2, "Generate Automated Training Data")
    
    print("""
Automated data provides volume, manual data provides quality.
Best practice: Combine both approaches.
    """)
    
    input("Press Enter to generate automated training data...")
    
    from rexx_rag_system import RexxRAGSystem
    from rexx_fine_tuner import RexxFineTuner
    
    # Initialize systems
    script_dir = os.path.dirname(os.path.abspath(__file__))
    pdf_folder = os.path.join(script_dir, "rexx_pdfs")
    
    print("\nInitializing RAG system...")
    rag_system = RexxRAGSystem(pdf_folder)
    rag_system.process_pdfs()
    
    print("\nGenerating automated training data...")
    fine_tuner = RexxFineTuner()
    automated_data = fine_tuner.create_training_data_from_chunks(rag_system, num_samples=20)
    
    print(f"\n✓ Generated {len(automated_data)} automated training examples")
    
    # ========================================================================
    # STEP 3: Merge Training Data
    # ========================================================================
    print_step(3, "Merge Manual and Automated Data")
    
    print("""
Combining manual and automated data gives you:
- Volume from automated generation
- Quality from manual curation
- Balanced training dataset
    """)
    
    input("Press Enter to merge training data...")
    
    creator.merge_with_automated_data(
        automated_file="rexx_training_data.json",
        output_file="combined_training_data.json"
    )
    
    # ========================================================================
    # STEP 4: Baseline Evaluation
    # ========================================================================
    print_step(4, "Baseline Evaluation (Before Training)")
    
    print("""
Establishing baseline performance is crucial for measuring improvement.
We'll evaluate the system BEFORE any fine-tuning.
    """)
    
    input("Press Enter to run baseline evaluation...")
    
    from track_training_progress import evaluate_and_track
    
    # Define comprehensive test dataset
    test_dataset = [
        {"question": "What is Rexx Systems?", "expected_answer": "HR software"},
        {"question": "Can two users have the same User Name?", "expected_answer": "unique User Name"},
        {"question": "How do I access the Jobportal Configurator?", "expected_answer": "permission"},
        {"question": "What is the organizational tree?", "expected_answer": "hierarchical structure"},
        {"question": "How do permission profiles work?", "expected_answer": "profiles"},
        {"question": "What is FTE-Basis?", "expected_answer": "Full Time Equivalent"},
        {"question": "How to configure user permissions?", "expected_answer": "admin"},
        {"question": "What modules are in Rexx HR?", "expected_answer": "Personnel Management"}
    ]
    
    print("\nRunning baseline evaluation with 8 test questions...")
    baseline_results, tracker = evaluate_and_track(
        rag_system,
        iteration_name="baseline",
        training_data_file=None,
        test_dataset=test_dataset,
        notes="Initial system with no fine-tuning"
    )
    
    print("\nBaseline Results:")
    print(f"  RAG Score: {baseline_results['overall']['rag_score']:.4f}")
    print(f"  MRR: {baseline_results['overall']['mrr']:.4f}")
    print(f"  Accuracy: {baseline_results['overall'].get('accuracy', 'N/A')}")
    print(f"  Coherence: {baseline_results['overall']['coherence']:.4f}")
    
    # ========================================================================
    # STEP 5: Fine-Tuning (Simulated)
    # ========================================================================
    print_step(5, "Fine-Tuning the Model")
    
    print("""
NOTE: Actual fine-tuning requires significant compute resources.
For this demo, we'll simulate the improvement by showing the workflow.

In production, you would:
1. Load combined training data
2. Prepare dataset for fine-tuning
3. Train for 3-5 epochs
4. Save fine-tuned model
5. Load fine-tuned model for inference
    """)
    
    print("\nFine-tuning workflow (simulated):")
    print("  1. Loading combined_training_data.json")
    print("  2. Preparing dataset splits (80/20 train/val)")
    print("  3. Training for 3 epochs...")
    print("  4. Saving to ./rexx_fine_tuned_model/")
    print("  ✓ Fine-tuning complete (simulated)")
    
    # ========================================================================
    # STEP 6: Post-Training Evaluation
    # ========================================================================
    print_step(6, "Post-Training Evaluation")
    
    print("""
Now we evaluate the system after training to measure improvement.
In a real scenario, you would load the fine-tuned model here.
    """)
    
    input("Press Enter to run post-training evaluation...")
    
    # Simulate improved performance (in reality, load fine-tuned model)
    print("\nRunning post-training evaluation...")
    post_training_results, _ = evaluate_and_track(
        rag_system,
        iteration_name="after_combined_training",
        training_data_file="combined_training_data.json",
        test_dataset=test_dataset,
        notes="After fine-tuning with 25 total examples (5 manual + 20 automated)"
    )
    
    print("\nPost-Training Results:")
    print(f"  RAG Score: {post_training_results['overall']['rag_score']:.4f}")
    print(f"  MRR: {post_training_results['overall']['mrr']:.4f}")
    print(f"  Accuracy: {post_training_results['overall'].get('accuracy', 'N/A')}")
    print(f"  Coherence: {post_training_results['overall']['coherence']:.4f}")
    
    # ========================================================================
    # STEP 7: Compare and Analyze
    # ========================================================================
    print_step(7, "Compare Iterations and Analyze Improvements")
    
    print("\nGenerating comprehensive progress report...")
    
    tracker.print_progress_report()
    
    # Show specific comparison
    print("\n" + "-"*80)
    print("DETAILED COMPARISON: Baseline vs After Training")
    print("-"*80)
    
    comparison = tracker.compare_iterations("baseline", "after_combined_training")
    
    if 'error' not in comparison:
        print(f"\n✓ Total Improvements: {comparison['summary']['total_improvements']}")
        print(f"✗ Total Regressions: {comparison['summary']['total_regressions']}")
        print(f"Net Improvement: {comparison['summary']['net_improvement']}")
        
        if comparison['improvements']:
            print("\nImproved Metrics:")
            for metric, data in comparison['improvements'].items():
                print(f"  ✓ {metric}:")
                print(f"      {data['old_value']:.4f} → {data['new_value']:.4f}")
                print(f"      Change: {data['absolute_change']:+.4f} ({data['percent_change']:+.2f}%)")
    
    # ========================================================================
    # STEP 8: Export for Further Analysis
    # ========================================================================
    print_step(8, "Export Results for Visualization and Reporting")
    
    print("\nExporting data for visualization...")
    tracker.export_for_visualization("training_progress_viz.json")
    
    print("\n✓ All files generated:")
    files = [
        "manual_training_data.json - Your manual training examples",
        "rexx_training_data.json - Automated training examples",
        "combined_training_data.json - Merged dataset for training",
        "training_progress.json - Complete training history",
        "training_progress_viz.json - Data formatted for charts/graphs"
    ]
    for f in files:
        print(f"  - {f}")
    
    # ========================================================================
    # FINAL RECOMMENDATIONS
    # ========================================================================
    print("\n" + "="*80)
    print("RECOMMENDATIONS FOR CONTINUED IMPROVEMENT")
    print("="*80)
    
    recommendations = [
        "1. Add More Manual Examples:",
        "   - Focus on questions where the system performed poorly",
        "   - Add examples for underrepresented categories",
        "   - Aim for 50-100 high-quality manual examples",
        "",
        "2. Iterate on Training:",
        "   - Run fine-tuning with combined dataset",
        "   - Evaluate after each iteration",
        "   - Compare results to identify what works",
        "",
        "3. Test on Real User Questions:",
        "   - Collect actual questions from users",
        "   - Add them to your test dataset",
        "   - Measure improvement on real-world queries",
        "",
        "4. Balance Your Dataset:",
        "   - Ensure even distribution across categories",
        "   - Include various difficulty levels",
        "   - Mix factual and reasoning questions",
        "",
        "5. Track Progress Over Time:",
        "   - Run evaluate_and_track() after each change",
        "   - Monitor trends in key metrics",
        "   - Document what changes led to improvements"
    ]
    
    for rec in recommendations:
        print(rec)
    
    print("\n" + "="*80)
    print("WORKFLOW COMPLETE!")
    print("="*80)
    print("\nNext steps:")
    print("1. Review generated training data files")
    print("2. Add more manual examples using create_manual_training_data.py")
    print("3. Run actual fine-tuning with rexx_fine_tuner.py")
    print("4. Continue tracking with track_training_progress.py")


if __name__ == "__main__":
    print("\nREXX RAG TRAINING DATA WORKFLOW")
    print("="*80)
    print("This workflow will guide you through:")
    print("  1. Creating manual training data")
    print("  2. Generating automated training data")
    print("  3. Merging datasets")
    print("  4. Evaluating baseline performance")
    print("  5. Tracking improvements over time")
    print("\nEstimated time: 5-10 minutes")
    print("="*80)
    
    proceed = input("\nProceed with workflow? (y/n): ").strip().lower()
    
    if proceed == 'y':
        complete_workflow()
    else:
        print("\nWorkflow cancelled. Run this script again when ready.")
