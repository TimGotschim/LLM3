# Rexx Systems RAG Implementation Guide

## Overview

This is a complete Retrieval-Augmented Generation (RAG) system for processing and querying Rexx Systems HR software documentation. The system includes:

1. **PDF Processing**: Extracts text from PDF documentation
2. **Vector Storage**: Uses ChromaDB for efficient similarity search
3. **Retrieval**: Finds relevant documentation chunks for queries
4. **Generation**: Uses Ollama to generate answers based on retrieved context
5. **Fine-tuning**: Improves model performance on domain-specific questions
6. **Evaluation**: Comprehensive metrics for measuring system performance

## Installation

### 1. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 2. Install and Setup Ollama

1. Download Ollama from https://ollama.com/download/mac
2. Install the application
3. Start Ollama service:
   ```bash
   ollama serve
   ```
4. Pull a model (e.g., Llama2):
   ```bash
   ollama pull llama2
   ```

### 3. (Optional) Install Additional Evaluation Metrics

For advanced evaluation metrics:
```bash
pip install rouge-score bert-score
```

## Project Structure

```
├── rexx_rag_system.py      # Main RAG system implementation
├── rexx_fine_tuner.py      # Fine-tuning module
├── rexx_evaluator.py       # Evaluation metrics and reporting
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── rexx_chroma_db/        # Vector database storage (created automatically)
├── rexx_rag_metadata.json # System metadata (created automatically)
└── rexx_training_data.json # Training data for fine-tuning (created when fine-tuning)
```

## Usage

### 1. Basic RAG Usage

```python
from rexx_rag_system import RexxRAGSystem

# Initialize the system with your PDF folder
pdf_folder = "/Users/timgotschim/Documents/LLM/Large-Language-Models/rexx_pdfs"
rag = RexxRAGSystem(pdf_folder, model_name="llama2")

# Process PDFs (only needed once, unless PDFs change)
rag.process_pdfs()

# Query the system
response = rag.query("What are the main features of Rexx Systems?")
print(f"Answer: {response['answer']}")
print(f"Sources: {response['sources']}")
```

### 2. Running the Complete Pipeline

```bash
# Process PDFs and test the system
python rexx_rag_system.py

# Fine-tune the model
python rexx_fine_tuner.py

# Evaluate performance
python rexx_evaluator.py
```

### 3. Evaluation Workflow

The evaluation system measures:
- **Retrieval metrics**: Precision@k, Recall@k, MRR
- **Generation metrics**: BLEU, ROUGE, BERTScore, coherence
- **End-to-end metrics**: Response time, accuracy, RAG score

```python
from rexx_evaluator import RAGEvaluator

# Create test dataset
test_dataset = [
    {"question": "What is Rexx Systems?", "expected_answer": "HR software"},
    {"question": "How to configure users?", "expected_answer": "admin panel"}
]

# Evaluate
evaluator = RAGEvaluator(rag)
results = evaluator.evaluate_end_to_end(test_dataset)
evaluator.generate_report("evaluation_report.json")
```

### 4. Fine-tuning Process

Fine-tuning improves model performance on your specific documentation:

```python
from rexx_fine_tuner import RexxFineTuner

# Initialize fine-tuner
fine_tuner = RexxFineTuner(base_model="distilbert-base-uncased-distilled-squad")

# Generate training data from your documents
training_data = fine_tuner.create_training_data_from_chunks(rag, num_samples=100)

# Fine-tune
dataset = fine_tuner.prepare_dataset(training_data)
trainer = fine_tuner.fine_tune(dataset, epochs=3)

# Evaluate improvement
results, metrics = fine_tuner.evaluate_improvement(test_questions, rag)
```

## Configuration Options

### RAG System Parameters

- `model_name`: Ollama model to use (default: "llama2")
- `embedding_model`: Sentence transformer model (default: "all-MiniLM-L6-v2")
- `chunk_size`: Size of text chunks (default: 1000)
- `chunk_overlap`: Overlap between chunks (default: 200)

### Available Ollama Models

Popular models for RAG:
- `llama2`: Good balance of performance and speed
- `mistral`: Fast and efficient
- `neural-chat`: Optimized for conversational responses
- `codellama`: If documentation includes code

Pull models with:
```bash
ollama pull <model_name>
```

## Evaluation Metrics Explained

### Retrieval Metrics
- **MRR (Mean Reciprocal Rank)**: Average of reciprocal ranks of first relevant result
- **Precision@k**: Fraction of retrieved documents that are relevant
- **Recall@k**: Fraction of relevant documents that are retrieved
- **F1@k**: Harmonic mean of precision and recall

### Generation Metrics
- **BLEU**: Measures n-gram overlap with expected answers
- **ROUGE**: Measures overlap of sequences with expected answers
- **BERTScore**: Semantic similarity using BERT embeddings
- **Coherence**: Checks for well-formed, complete responses

### Overall Metrics
- **RAG Score**: Weighted combination of all metrics (0-1)
- **Response Time**: Average time to generate answer
- **Accuracy**: Percentage of answers containing expected information

## Best Practices

1. **PDF Quality**: Ensure PDFs have extractable text (not scanned images)
2. **Chunk Size**: Adjust based on your content (smaller for precise info, larger for context)
3. **Model Selection**: Test different Ollama models for your use case
4. **Evaluation**: Create domain-specific test questions with expected answers
5. **Fine-tuning**: Use when base model performance is insufficient

## Troubleshooting

### Common Issues

1. **"Cannot connect to Ollama"**
   - Ensure Ollama is running: `ollama serve`
   - Check if model is pulled: `ollama list`

2. **"No text extracted from PDF"**
   - PDF might be scanned/image-based
   - Try OCR with pytesseract (already in requirements)

3. **"Out of memory"**
   - Reduce batch size in fine-tuning
   - Use smaller embedding model
   - Process PDFs in batches

4. **Poor retrieval quality**
   - Adjust chunk size and overlap
   - Try different embedding models
   - Check if PDFs are being processed correctly

## Performance Tips

1. **First-time Processing**: PDF processing and embedding generation is slow initially but cached for subsequent runs
2. **GPU Acceleration**: Use GPU for embeddings and fine-tuning if available
3. **Batch Processing**: Process queries in batches for better throughput
4. **Model Caching**: Ollama caches models, so subsequent calls are faster

## Extension Ideas

1. **Web Interface**: Add Streamlit or Gradio UI
2. **API Server**: Wrap in FastAPI for production deployment
3. **Multi-language**: Add support for other languages
4. **Document Updates**: Implement incremental updates for new PDFs
5. **Query Expansion**: Use synonyms and related terms for better retrieval

## Assignment Deliverables

For your university assignment, make sure to include:

1. **Code**: All Python files with proper documentation
2. **Evaluation Report**: Generated by `rexx_evaluator.py`
3. **Model Comparison**: Results from baseline vs fine-tuned model
4. **Error Analysis**: The "15Ps" mentioned in your requirements
5. **Improvement Metrics**: Show that fine-tuning improves performance

## References

- [Ollama Documentation](https://github.com/jmorganca/ollama)
- [ChromaDB Documentation](https://docs.trychroma.com/)
- [Sentence Transformers](https://www.sbert.net/)
- [Hugging Face Fine-tuning Guide](https://huggingface.co/docs/transformers/training)
