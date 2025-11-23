# Training Data Creation & Progress Tracking Guide

## Overview

This guide provides three powerful tools for **creating high-quality training data** and **tracking model improvements** over time for your Rexx RAG system.

---

## 📁 New Files

### 1. `create_manual_training_data.py`
**Purpose**: Create and manage high-quality manual training data

**Features**:
- ✅ Interactive mode for adding training examples
- ✅ Template-based batch creation
- ✅ Automatic categorization and difficulty tracking
- ✅ Merge with automated training data
- ✅ Statistics and quality analysis

### 2. `track_training_progress.py`
**Purpose**: Track model performance across training iterations

**Features**:
- ✅ Record evaluation results for each iteration
- ✅ Compare different training approaches
- ✅ Identify best-performing configurations
- ✅ Generate progress reports
- ✅ Export data for visualization

### 3. `training_workflow_guide.py`
**Purpose**: Complete end-to-end workflow demonstration

**Features**:
- ✅ Step-by-step guided workflow
- ✅ Combines all tools into one process
- ✅ Generates example datasets
- ✅ Shows before/after comparisons
- ✅ Provides recommendations

---

## 🚀 Quick Start

### Option 1: Complete Guided Workflow (Recommended for first time)

```bash
python training_workflow_guide.py
```

This interactive guide walks you through the entire process:
1. Creating manual training data
2. Generating automated data
3. Merging datasets
4. Baseline evaluation
5. Training and re-evaluation
6. Tracking improvements

**Time**: ~5-10 minutes  
**Output**: All training data files + progress tracking

---

### Option 2: Create Manual Training Data Only

#### Interactive Mode (Easiest)
```bash
python create_manual_training_data.py
# Select option 2 for interactive mode
```

#### Programmatic Creation
```python
from create_manual_training_data import ManualTrainingDataCreator

creator = ManualTrainingDataCreator("manual_training_data.json")

# Add training examples
creator.add_training_example(
    question="Can two users have the same User Name in Rexx?",
    context="In Rexx Systems, each user must have a unique User Name...",
    answer="each user must have a unique User Name",
    source="User Management Guide",
    category="user_management",
    difficulty="easy"
)

creator.save()
```

---

### Option 3: Track Progress Only

```python
from track_training_progress import evaluate_and_track
from rexx_rag_system import RexxRAGSystem

# Initialize your RAG system
rag_system = RexxRAGSystem("rexx_pdfs")
rag_system.process_pdfs()

# Evaluate and track
results, tracker = evaluate_and_track(
    rag_system,
    iteration_name="my_experiment_1",
    training_data_file="combined_training_data.json",
    notes="Testing with 50 training examples"
)

# View progress
tracker.print_progress_report()
```

---

## 📊 How to Track Improvements

### The Complete Tracking Workflow

```python
from rexx_rag_system import RexxRAGSystem
from track_training_progress import evaluate_and_track, TrainingProgressTracker

# Initialize system
rag_system = RexxRAGSystem("rexx_pdfs")
rag_system.process_pdfs()

# 1. BASELINE - Before any training
results_baseline, tracker = evaluate_and_track(
    rag_system,
    iteration_name="baseline",
    notes="No training yet"
)

# 2. ITERATION 1 - After automated training
# (Generate automated data first with rexx_fine_tuner.py)
results_iter1, _ = evaluate_and_track(
    rag_system,
    iteration_name="automated_only",
    training_data_file="rexx_training_data.json",
    notes="Used 50 automated examples"
)

# 3. ITERATION 2 - After adding manual training
# (Create manual data first with create_manual_training_data.py)
results_iter2, _ = evaluate_and_track(
    rag_system,
    iteration_name="with_manual_data",
    training_data_file="combined_training_data.json",
    notes="Added 10 high-quality manual examples"
)

# 4. VIEW PROGRESS REPORT
tracker.print_progress_report()

# 5. COMPARE SPECIFIC ITERATIONS
comparison = tracker.compare_iterations("baseline", "with_manual_data")
print(f"Improvements: {comparison['summary']['total_improvements']}")

# 6. FIND BEST PERFORMING ITERATION
best = tracker.get_best_iteration("rag_score")
print(f"Best: {best['iteration_name']} with score {best['evaluation']['overall']['rag_score']:.4f}")
```

### Output Example:

```
================================================================================
TRAINING PROGRESS REPORT
================================================================================
Total Iterations: 3

Iteration History:
--------------------------------------------------------------------------------

1. baseline (2024-11-23)
   Training Data: 0 samples
   RAG Score: 0.4523
   MRR: 0.6000
   Accuracy: 0.375
   Coherence: 0.8200

2. automated_only (2024-11-23)
   Training Data: 50 samples
   RAG Score: 0.4891
   MRR: 0.6500
   Accuracy: 0.500
   Coherence: 0.8400

3. with_manual_data (2024-11-23)
   Training Data: 60 samples (10 manual + 50 automated)
   RAG Score: 0.5234
   MRR: 0.7000
   Accuracy: 0.625
   Coherence: 0.8750

--------------------------------------------------------------------------------
Latest vs Baseline Comparison:

Improvements (5):
   ✓ RAG Score: 0.4523 → 0.5234 (+15.72%)
   ✓ MRR: 0.6000 → 0.7000 (+16.67%)
   ✓ Accuracy: 0.3750 → 0.6250 (+66.67%)
   ✓ Coherence: 0.8200 → 0.8750 (+6.71%)
================================================================================
```

---

## 📝 Manual Training Data Best Practices

### What Makes Good Training Data?

1. **Specific Questions**
   ```python
   ✅ GOOD: "Can two users have the same User Name in Rexx?"
   ❌ BAD:  "What is described in this section?"
   ```

2. **Clear Context**
   ```python
   ✅ GOOD: Full paragraph with relevant information
   ❌ BAD:  Fragmented or incomplete sentences
   ```

3. **Exact Answer Match**
   ```python
   # Answer MUST be a substring of context
   context = "User Names must be unique in Rexx Systems."
   answer = "User Names must be unique"  # ✅ Found in context
   answer = "Usernames are unique"       # ❌ Not exact match
   ```

4. **Categorization**
   - `user_management`: Users, permissions, authentication
   - `features`: System capabilities and modules
   - `configuration`: Setup and settings
   - `troubleshooting`: Problem-solving

5. **Difficulty Levels**
   - `easy`: Direct factual questions
   - `medium`: Requires understanding relationships
   - `hard`: Multi-step reasoning or complex concepts

---

## 🎯 Strategies for Increasing Training Data

### Method 1: Systematic Document Coverage

```python
# Create examples for each major section of your documentation
categories = [
    "User Management",
    "Organizational Structure", 
    "Permission Profiles",
    "Workflows",
    "Job Portal Configuration",
    "System Administration"
]

for category in categories:
    # Create 5-10 examples per category
    # Focus on common user questions
```

### Method 2: Real User Questions

```python
# Collect actual questions users ask
# Add them as training examples with correct answers

real_questions = [
    "How do I reset my password?",
    "Where do I find the job portal configurator?",
    "Why can't I see certain menu items?"
]

# Convert each to training example
```

### Method 3: Error-Driven Creation

```python
# After evaluation, identify questions the system gets wrong
# Create targeted training examples for those topics

# Example: If system struggles with permissions
creator.add_training_example(
    question="How are permission profiles evaluated when a user has multiple profiles?",
    context="When multiple profiles are assigned, the system evaluates them in order...",
    answer="system evaluates them in order",
    category="user_management",
    difficulty="hard"
)
```

### Method 4: Difficulty Progression

```python
# Create progressively harder questions on same topic

# Easy
"What is a permission profile?"

# Medium  
"How do I create a new permission profile?"

# Hard
"How does the system resolve conflicts when a user has multiple permission profiles?"
```

---

## 📈 Metrics Explained

### RAG Score (0-1)
- **Weighted combination** of all metrics
- **Higher is better**
- Formula: `0.3*MRR + 0.3*Accuracy + 0.2*F1@5 + 0.2*(1-ResponseTime)`

### MRR (Mean Reciprocal Rank)
- **Measures retrieval quality**
- How quickly relevant documents are found
- **Range**: 0-1, higher is better

### Accuracy
- **Percentage of correct answers**
- Requires expected answers in test data
- **Range**: 0-1, higher is better

### Coherence (0-1)
- **Measures answer quality**
- Checks for complete sentences and structure
- **Higher is better**

### BLEU & ROUGE
- **Similarity to expected answers**
- Used when expected answers provided
- **Range**: 0-1, higher is better

---

## 🔄 Recommended Workflow

### Initial Setup (Once)
```bash
# 1. Process your PDFs
python rexx_rag_system.py

# 2. Run baseline evaluation
python -c "from track_training_progress import evaluate_and_track; from rexx_rag_system import RexxRAGSystem; rag = RexxRAGSystem('rexx_pdfs'); rag.process_pdfs(); evaluate_and_track(rag, 'baseline', notes='Initial system')"
```

### Iterative Improvement (Repeat)
```bash
# 1. Create/add manual training data
python create_manual_training_data.py

# 2. Merge with automated data
# (Script will prompt you)

# 3. Fine-tune model (if needed)
python rexx_fine_tuner.py

# 4. Evaluate improvements
python -c "from track_training_progress import evaluate_and_track; from rexx_rag_system import RexxRAGSystem; rag = RexxRAGSystem('rexx_pdfs'); rag.process_pdfs(); evaluate_and_track(rag, 'iteration_X', training_data_file='combined_training_data.json')"

# 5. Review progress
python -c "from track_training_progress import TrainingProgressTracker; t = TrainingProgressTracker(); t.print_progress_report()"
```

---

## 📁 Generated Files

| File | Description | Created By |
|------|-------------|------------|
| `manual_training_data.json` | Your manual training examples | create_manual_training_data.py |
| `rexx_training_data.json` | Automated examples | rexx_fine_tuner.py |
| `combined_training_data.json` | Merged dataset | create_manual_training_data.py |
| `training_progress.json` | Complete training history | track_training_progress.py |
| `training_progress_viz.json` | Data for charts/graphs | track_training_progress.py |

---

## 💡 Tips for Success

### 1. **Quality Over Quantity**
- 10 excellent manual examples > 100 mediocre automated examples
- Focus on questions users actually ask

### 2. **Balanced Dataset**
- Mix easy, medium, and hard questions
- Cover all major categories evenly
- Include both factual and reasoning questions

### 3. **Iterative Approach**
- Add 5-10 examples at a time
- Evaluate after each addition
- Focus on areas where system struggles

### 4. **Track Everything**
- Use `evaluate_and_track()` after every change
- Compare iterations regularly
- Document what works and what doesn't

### 5. **Real-World Testing**
- Test with actual user questions
- Ask colleagues to try the system
- Collect feedback for new training examples

---

## 🆘 Troubleshooting

### "Answer not found in context" Warning
```python
# Make sure answer is EXACT substring of context
context = "The system uses unique identifiers."
answer = "unique identifiers"  # ✅ Works
answer = "unique identifier"   # ❌ Won't work (missing 's')
```

### Low Accuracy Scores
- Add more examples for that category
- Ensure test questions match training data topics
- Check if answers are too generic

### No Improvement After Training
- Verify training data quality
- Increase training epochs
- Add more diverse examples
- Check if test dataset is representative

---

## 📞 Next Steps

1. **Start Simple**: Run `training_workflow_guide.py` for complete walkthrough
2. **Create Data**: Use `create_manual_training_data.py` to add your examples
3. **Track Progress**: Use `track_training_progress.py` after each iteration
4. **Iterate**: Keep improving based on metrics and real usage

---

## 🎓 For Your University Assignment

This workflow perfectly demonstrates:
- ✅ **Systematic evaluation** (baseline → training → improvement)
- ✅ **Quantitative metrics** (RAG score, MRR, accuracy)
- ✅ **Error analysis** (15Ps framework integration)
- ✅ **Iterative refinement** (tracked improvements)
- ✅ **Documentation** (all results saved and tracked)

**Key Deliverables**:
- `training_progress.json` - Shows all iterations
- Progress reports - Demonstrate improvements
- Comparison charts - Visualize gains
- Training data files - Show your methodology

Good luck with your project! 🚀
