# Rexx HR Q&A System

LLM-based Q&A system for Rexx HR Software documentation (German language).

## Overview

This project evaluates different approaches for a domain-specific Q&A system:
- **Baseline**: LLM without domain knowledge (Qwen2.5-1.5B-Instruct)
- **Fine-Tuning**: LoRA fine-tuning on Rexx-specific data
- **RAG**: Retrieval-Augmented Generation with PDF documents

We compared multiple models on the same test dataset to evaluate their performance:
- **Generators**: tinyllama (1.1B), llama2 (7B), gemma2:2b (2B)
- **Embeddings**: MiniLM-L6-v2 (English), Multilingual-MiniLM-L12-v2 (German support)

## Setup

```bash
pip install -r requirements.txt
```

For RAG, Ollama is required:
```bash
# Install Ollama: https://ollama.ai
ollama pull llama2
ollama pull tinyllama
ollama pull gemma2:2b
```

## Scripts

### Evaluation
| Script | Description |
|--------|-------------|
| `evaluate_baseline.py` | Baseline evaluation (Qwen2.5-1.5B) |
| `finetune_model.py` | Fine-tuning with LoRA |
| `evaluate_finetuned.py` | Fine-tuned model evaluation |

### RAG System
| Script | Description |
|--------|-------------|
| `rag_system.py` | Build RAG index (PDFs → ChromaDB) |
| `evaluate_rag.py` | RAG evaluation |
| `app.py` | Flask Web UI (Port 5001) |

### Comparisons
| Script | Description |
|--------|-------------|
| `compare_rag_models.py` | Generator comparison (tinyllama, llama2, gemma2:2b) |
| `compare_embeddings.py` | Embedding comparison (MiniLM vs Multilingual) |

## Workflow

```bash
# 1. Evaluate baseline
python evaluate_baseline.py

# 2. Fine-tuning
python finetune_model.py
python evaluate_finetuned.py

# 3. Build and evaluate RAG
python rag_system.py          # Build index
python evaluate_rag.py        # Evaluate

# 4. Model comparisons
python compare_rag_models.py  # Generator comparison
python compare_embeddings.py  # Embedding comparison

# 5. Start Web UI
python app.py
```

## Results

All results are saved as JSON:
- `qwen25_baseline_results.json`
- `qwen25_finetuned_results.json`
- `rag_results.json`
- `rag_model_comparison.json`
- `embedding_comparison.json`

## Metrics

- **Cosine Similarity**: TF-IDF based similarity
- **Precision/Recall/F1**: Word-overlap metrics
- **Key Terms Ratio**: Important terms found

## Data

- `rexx_qa_dataset_curated.json`: Train/Test Q&A pairs
- `rexx_pdfs/`: PDF documentation (69 files)
