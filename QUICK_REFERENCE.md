# Training Data Quick Reference

## 🎯 Yes, You Can Create More Training Data!

### Three Ways to Expand Training Data:

```
┌─────────────────────────────────────────────────────────────┐
│  1. AUTOMATED GENERATION (Volume)                           │
├─────────────────────────────────────────────────────────────┤
│  • Generates 50-200+ examples automatically                 │
│  • Based on existing document chunks                        │
│  • Fast but generic questions                               │
│  • Command: python rexx_fine_tuner.py                       │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│  2. MANUAL CREATION (Quality)                               │
├─────────────────────────────────────────────────────────────┤
│  • High-quality, domain-specific examples                   │
│  • Based on real user questions                             │
│  • Time-consuming but highly effective                      │
│  • Command: python create_manual_training_data.py           │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│  3. COMBINED APPROACH (Best Results)                        │
├─────────────────────────────────────────────────────────────┤
│  • 80% automated for volume                                 │
│  • 20% manual for quality                                   │
│  • Balanced and comprehensive                               │
│  • Command: Merge function in script                        │
└─────────────────────────────────────────────────────────────┘
```

---

## 📊 Tracking Improvements - The Complete Picture

```
WORKFLOW VISUALIZATION:
═══════════════════════════════════════════════════════════════

Step 1: BASELINE                    │ Metrics Tracked:
├─ Process PDFs                      │ • RAG Score
├─ Run evaluation                    │ • MRR (Retrieval)
└─ Record results                    │ • Accuracy
   ↓                                 │ • Coherence
   RAG Score: 0.45                   │ • Response Time
                                     │
Step 2: ADD AUTOMATED DATA           │
├─ Generate 50 examples              │
├─ Fine-tune model (optional)        │
├─ Run evaluation                    │
└─ Record results                    │
   ↓                                 │
   RAG Score: 0.49 (+8.9%)          │
                                     │
Step 3: ADD MANUAL DATA              │
├─ Create 10 quality examples        │
├─ Merge with automated              │
├─ Fine-tune model                   │
├─ Run evaluation                    │
└─ Record results                    │
   ↓                                 │
   RAG Score: 0.52 (+15.6%)         │
                                     │
Step 4: ANALYZE & ITERATE            │
├─ Compare all iterations            │
├─ Identify what worked              │
├─ Add targeted examples             │
└─ Continue improving                │
   ↓                                 │
   RAG Score: 0.58 (+28.9%)         │

═══════════════════════════════════════════════════════════════
```

---

## 🔢 Sample Metrics Tracking

```
ITERATION COMPARISON TABLE:
┌────────────────┬──────────┬─────┬──────────┬──────────┐
│ Iteration      │ RAG Score│ MRR │ Accuracy │ Samples  │
├────────────────┼──────────┼─────┼──────────┼──────────┤
│ Baseline       │  0.4523  │ 0.60│  37.5%   │    0     │
│ Automated      │  0.4891  │ 0.65│  50.0%   │   50     │
│ +Manual        │  0.5234  │ 0.70│  62.5%   │   60     │
│ +More Manual   │  0.5678  │ 0.75│  75.0%   │   80     │
└────────────────┴──────────┴─────┴──────────┴──────────┘

IMPROVEMENT BREAKDOWN:
                  Baseline → Final
RAG Score:        +25.5% ↑
MRR:             +25.0% ↑
Accuracy:        +100.0% ↑ (37.5% → 75%)
```

---

## 💻 Code Examples

### Creating Manual Training Data

```python
from create_manual_training_data import ManualTrainingDataCreator

# Initialize
creator = ManualTrainingDataCreator()

# Add a single example
creator.add_training_example(
    question="Can users share User Names?",
    context="Each user must have a unique User Name in Rexx.",
    answer="unique User Name",
    category="user_management",
    difficulty="easy"
)

# Add multiple examples at once
examples = [
    {
        "question": "How to access configurator?",
        "context": "Access via URL/customize.php...",
        "answer": "URL/customize.php",
        "category": "configuration",
        "difficulty": "medium"
    },
    # ... more examples
]

creator.add_batch_from_template(examples)
creator.save()
```

### Tracking Progress

```python
from track_training_progress import evaluate_and_track
from rexx_rag_system import RexxRAGSystem

# Setup
rag = RexxRAGSystem("rexx_pdfs")
rag.process_pdfs()

# Evaluate and track automatically
results, tracker = evaluate_and_track(
    rag,
    iteration_name="my_iteration_1",
    training_data_file="combined_training_data.json",
    notes="Added 10 manual examples focusing on permissions"
)

# View progress
tracker.print_progress_report()

# Compare iterations
comparison = tracker.compare_iterations("baseline", "my_iteration_1")
```

---

## 📈 What Gets Tracked Automatically

```
FOR EACH ITERATION:
├─ Timestamp
├─ Training Data Info
│  ├─ Total samples
│  ├─ Manual vs automated breakdown
│  └─ Source file
├─ Model Configuration
│  ├─ Model name
│  ├─ Embedding model
│  └─ Hyperparameters
├─ Evaluation Metrics
│  ├─ RAG Score (overall quality)
│  ├─ MRR (retrieval quality)
│  ├─ Accuracy (answer correctness)
│  ├─ Coherence (answer quality)
│  ├─ Response time
│  └─ Individual test results
└─ Notes (your observations)

ALL SAVED TO: training_progress.json
```

---

## ⚡ Quick Commands

```bash
# 1. Complete guided workflow
python training_workflow_guide.py

# 2. Create manual data (interactive)
python create_manual_training_data.py
# → Select option 2

# 3. View statistics
python create_manual_training_data.py
# → Select option 3

# 4. Generate automated data
python rexx_fine_tuner.py

# 5. Merge datasets
python create_manual_training_data.py
# → Select option 4

# 6. View training progress
python -c "from track_training_progress import TrainingProgressTracker; TrainingProgressTracker().print_progress_report()"
```

---

## 🎯 Best Practices Summary

```
DO:
✅ Focus on quality for manual examples
✅ Use real user questions
✅ Track every iteration
✅ Document what works
✅ Balance categories
✅ Mix difficulty levels

DON'T:
❌ Create generic questions
❌ Skip baseline evaluation
❌ Train without tracking
❌ Add examples randomly
❌ Ignore failed test cases
❌ Forget to merge datasets
```

---

## 📊 Expected Improvements

```
TYPICAL IMPROVEMENT TRAJECTORY:

Baseline (no training)
├─ RAG Score: 0.40-0.50
├─ Accuracy: 30-40%
└─ Purpose: Establish starting point

After Automated Data (50 examples)
├─ RAG Score: +5-10%
├─ Accuracy: +10-15%
└─ Purpose: General improvement

After Manual Data (10-20 examples)
├─ RAG Score: +10-20%
├─ Accuracy: +20-40%
└─ Purpose: Domain-specific gains

After Combined + Iteration (80-100 examples)
├─ RAG Score: +25-35%
├─ Accuracy: +50-100%
└─ Purpose: Optimized performance

Note: Actual improvements vary by:
- Quality of training data
- Relevance to test questions
- Model architecture
- Fine-tuning parameters
```

---

## 🎓 For Academic Assignment

```
YOUR DELIVERABLES:

1. Training Data Files
   ├─ manual_training_data.json
   ├─ rexx_training_data.json
   └─ combined_training_data.json

2. Progress Tracking
   ├─ training_progress.json
   ├─ baseline_evaluation.json
   └─ final_report.json

3. Analysis
   ├─ Progress reports
   ├─ Iteration comparisons
   └─ Improvement percentages

4. Documentation
   ├─ Methodology explanation
   ├─ Error analysis (15Ps)
   └─ Recommendations
```

---

## 🚀 Get Started Now!

```bash
# Fastest way to see everything:
python training_workflow_guide.py

# This will:
# ✓ Create example training data
# ✓ Run baseline evaluation
# ✓ Track improvements
# ✓ Generate all files
# ✓ Show you exactly how it works

Time: ~5-10 minutes
```

---

## 📞 Need Help?

Check `TRAINING_DATA_GUIDE.md` for:
- Detailed explanations
- Troubleshooting tips
- Advanced techniques
- Code examples
- Best practices

---

**Remember**: The key to improvement is **systematic tracking**. 
Every change should be evaluated and recorded. This creates a 
clear narrative of what works and drives continuous improvement.
