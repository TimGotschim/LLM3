"""
Fine-tuning script for Rexx Systems RAG model
This script handles the fine-tuning process for improving model performance
on domain-specific questions.
"""

import json
import os
from typing import List, Dict, Tuple
import numpy as np
from datetime import datetime
import requests
from tqdm import tqdm

# For dataset preparation
import pandas as pd
from sklearn.model_selection import train_test_split

# For fine-tuning with Hugging Face
from transformers import (
    AutoTokenizer, 
    AutoModelForQuestionAnswering,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
from datasets import Dataset, DatasetDict
import torch
import evaluate


class RexxFineTuner:
    """
    Fine-tuning system for Rexx documentation QA model.
    """
    
    def __init__(self, base_model: str = "distilbert-base-uncased-distilled-squad"):
        """
        Initialize fine-tuner.
        
        Args:
            base_model: Pretrained model to fine-tune
        """
        self.base_model = base_model
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        self.model = AutoModelForQuestionAnswering.from_pretrained(base_model)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Move model to device
        self.model.to(self.device)
        
        # Metrics
        self.metric = evaluate.load("squad")
        
        # Data paths
        self.training_data_path = "rexx_training_data.json"
        self.fine_tuned_model_path = "./rexx_fine_tuned_model"
        
    def create_training_data_from_chunks(self, rag_system, num_samples: int = 100):
        """
        Generate training data from RAG system chunks.
        
        Args:
            rag_system: Initialized RAG system with processed documents
            num_samples: Number of training samples to generate
        """
        training_data = []
        
        # Get all chunks from the collection
        all_results = rag_system.collection.get(limit=num_samples * 2)
        
        print(f"Generating {num_samples} training samples...")
        
        for i in tqdm(range(min(num_samples, len(all_results['documents'])))):
            context = all_results['documents'][i]
            source = all_results['metadatas'][i]['source']
            
            # Generate questions based on context (simplified approach)
            # In practice, you might want to use a question generation model
            questions = self._generate_questions_from_context(context)
            
            for question in questions:
                # Find answer span in context
                answer_start, answer_text = self._extract_answer_from_context(
                    question, context
                )
                
                if answer_text:
                    training_data.append({
                        "context": context,
                        "question": question,
                        "answer": {
                            "text": answer_text,
                            "answer_start": answer_start
                        },
                        "source": source,
                        "id": f"rexx_{len(training_data)}"
                    })
        
        # Save training data
        with open(self.training_data_path, 'w') as f:
            json.dump(training_data, f, indent=2)
        
        print(f"Generated {len(training_data)} training examples")
        return training_data
    
    def _generate_questions_from_context(self, context: str) -> List[str]:
        """
        Generate questions from context (simplified version).
        In practice, use a question generation model.
        """
        questions = []
        
        # Extract key phrases and create questions
        if "feature" in context.lower():
            questions.append("What features are mentioned?")
        if "configuration" in context.lower():
            questions.append("How is the configuration done?")
        if "process" in context.lower():
            questions.append("What is the process described?")
        if "requirement" in context.lower():
            questions.append("What are the requirements?")
        
        # Add generic questions
        questions.extend([
            "What is described in this section?",
            "What are the main points covered?"
        ])
        
        return questions[:2]  # Limit questions per context
    
    def _extract_answer_from_context(self, question: str, context: str) -> Tuple[int, str]:
        """
        Extract answer from context (simplified version).
        Returns answer start position and answer text.
        """
        # This is a simplified approach - in practice, use more sophisticated methods
        sentences = context.split('. ')
        
        # Find most relevant sentence as answer
        for i, sentence in enumerate(sentences):
            if any(keyword in sentence.lower() for keyword in question.lower().split()):
                answer_start = context.find(sentence)
                return answer_start, sentence
        
        # If no match, return first sentence
        if sentences:
            return 0, sentences[0]
        
        return -1, ""
    
    def prepare_dataset(self, training_data: List[Dict]) -> DatasetDict:
        """
        Prepare dataset for fine-tuning.
        
        Args:
            training_data: List of training examples
            
        Returns:
            DatasetDict with train and validation splits
        """
        # Split data
        train_data, val_data = train_test_split(
            training_data, test_size=0.2, random_state=42
        )
        
        # Create datasets
        train_dataset = Dataset.from_list(train_data)
        val_dataset = Dataset.from_list(val_data)
        
        # Tokenize datasets
        train_dataset = train_dataset.map(
            self._preprocess_function,
            batched=True,
            remove_columns=train_dataset.column_names
        )
        
        val_dataset = val_dataset.map(
            self._preprocess_function,
            batched=True,
            remove_columns=val_dataset.column_names
        )
        
        return DatasetDict({
            'train': train_dataset,
            'validation': val_dataset
        })
    
    def _preprocess_function(self, examples):
        """Tokenize examples for question answering."""
        questions = [q.strip() for q in examples["question"]]
        contexts = examples["context"]
        
        # Tokenize
        inputs = self.tokenizer(
            questions,
            contexts,
            truncation=True,
            padding="max_length",
            max_length=384,
            stride=128,
            return_overflowing_tokens=True,
            return_offsets_mapping=True
        )
        
        # Process answers
        answers = examples["answer"]
        start_positions = []
        end_positions = []
        
        for i, offset_mapping in enumerate(inputs["offset_mapping"]):
            answer = answers[i]
            start_char = answer["answer_start"]
            end_char = start_char + len(answer["text"])
            
            # Find token positions
            sequence_ids = inputs.sequence_ids(i)
            context_start = sequence_ids.index(1)
            context_end = len(sequence_ids) - 1 - sequence_ids[::-1].index(1)
            
            # Default to CLS token if answer not in span
            if offset_mapping[context_start][0] > start_char or offset_mapping[context_end][1] < end_char:
                start_positions.append(0)
                end_positions.append(0)
            else:
                # Find start and end tokens
                token_start = context_start
                while token_start <= context_end and offset_mapping[token_start][0] <= start_char:
                    token_start += 1
                start_positions.append(token_start - 1)
                
                token_end = context_end
                while token_end >= context_start and offset_mapping[token_end][1] >= end_char:
                    token_end -= 1
                end_positions.append(token_end + 1)
        
        inputs["start_positions"] = start_positions
        inputs["end_positions"] = end_positions
        
        return inputs
    
    def fine_tune(self, dataset: DatasetDict, epochs: int = 3):
        """
        Fine-tune the model.
        
        Args:
            dataset: Prepared dataset
            epochs: Number of training epochs
        """
        # Training arguments
        training_args = TrainingArguments(
            output_dir=self.fine_tuned_model_path,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            num_train_epochs=epochs,
            weight_decay=0.01,
            logging_dir='./logs',
            logging_steps=10,
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            report_to="none"  # Disable wandb/tensorboard
        )
        
        # Data collator
        data_collator = DataCollatorWithPadding(self.tokenizer)
        
        # Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset['train'],
            eval_dataset=dataset['validation'],
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=self._compute_metrics
        )
        
        # Train
        print("Starting fine-tuning...")
        trainer.train()
        
        # Save model
        trainer.save_model()
        self.tokenizer.save_pretrained(self.fine_tuned_model_path)
        
        print(f"Model saved to {self.fine_tuned_model_path}")
        
        return trainer
    
    def _compute_metrics(self, eval_pred):
        """Compute metrics for evaluation."""
        predictions, labels = eval_pred
        start_predictions = np.argmax(predictions[0], axis=1)
        end_predictions = np.argmax(predictions[1], axis=1)
        
        # Simple accuracy
        start_accuracy = np.mean(start_predictions == labels[0])
        end_accuracy = np.mean(end_predictions == labels[1])
        
        return {
            "start_accuracy": start_accuracy,
            "end_accuracy": end_accuracy,
            "overall_accuracy": (start_accuracy + end_accuracy) / 2
        }
    
    def evaluate_improvement(self, test_questions: List[Dict], rag_system):
        """
        Evaluate improvement by comparing base and fine-tuned models.
        
        Args:
            test_questions: Test questions with expected answers
            rag_system: RAG system for retrieval
        """
        results = {
            "base_model": [],
            "fine_tuned": []
        }
        
        # Load fine-tuned model
        fine_tuned_model = AutoModelForQuestionAnswering.from_pretrained(
            self.fine_tuned_model_path
        )
        fine_tuned_model.to(self.device)
        
        print("Evaluating models...")
        
        for test_case in tqdm(test_questions):
            question = test_case['question']
            
            # Get context from RAG
            retrieved = rag_system.retrieve(question, n_results=1)
            if retrieved:
                context = retrieved[0]['text']
                
                # Evaluate base model
                base_answer = self._get_answer(
                    self.model, question, context
                )
                results["base_model"].append({
                    "question": question,
                    "answer": base_answer,
                    "expected": test_case.get('expected_answer', '')
                })
                
                # Evaluate fine-tuned model
                ft_answer = self._get_answer(
                    fine_tuned_model, question, context
                )
                results["fine_tuned"].append({
                    "question": question,
                    "answer": ft_answer,
                    "expected": test_case.get('expected_answer', '')
                })
        
        # Calculate metrics
        metrics = self._calculate_comparison_metrics(results)
        
        return results, metrics
    
    def _get_answer(self, model, question: str, context: str) -> str:
        """Get answer from model."""
        inputs = self.tokenizer(
            question, context, 
            return_tensors="pt",
            truncation=True,
            max_length=384
        ).to(self.device)
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        start_scores = outputs.start_logits
        end_scores = outputs.end_logits
        
        # Get most likely answer
        start_idx = torch.argmax(start_scores)
        end_idx = torch.argmax(end_scores)
        
        answer_tokens = inputs['input_ids'][0][start_idx:end_idx+1]
        answer = self.tokenizer.decode(answer_tokens, skip_special_tokens=True)
        
        return answer
    
    def _calculate_comparison_metrics(self, results: Dict) -> Dict:
        """Calculate comparison metrics between base and fine-tuned models."""
        metrics = {}
        
        for model_name in ["base_model", "fine_tuned"]:
            model_results = results[model_name]
            
            # Calculate accuracy (if expected answers provided)
            if any('expected' in r for r in model_results):
                correct = sum(
                    1 for r in model_results 
                    if r['expected'] and r['expected'].lower() in r['answer'].lower()
                )
                accuracy = correct / len(model_results)
            else:
                accuracy = None
            
            # Calculate answer length stats
            answer_lengths = [len(r['answer']) for r in model_results]
            
            metrics[model_name] = {
                "accuracy": accuracy,
                "avg_answer_length": np.mean(answer_lengths),
                "empty_answers": sum(1 for length in answer_lengths if length == 0)
            }
        
        # Calculate improvement
        if metrics["base_model"]["accuracy"] and metrics["fine_tuned"]["accuracy"]:
            metrics["improvement"] = {
                "accuracy_gain": metrics["fine_tuned"]["accuracy"] - metrics["base_model"]["accuracy"],
                "relative_improvement": (
                    (metrics["fine_tuned"]["accuracy"] - metrics["base_model"]["accuracy"]) 
                    / metrics["base_model"]["accuracy"] * 100
                )
            }
        
        return metrics


def main():
    """Example fine-tuning workflow."""
    
    # Import the RAG system
    from rexx_rag_system import RexxRAGSystem
    
    # Initialize systems
    pdf_folder = "/Users/timgotschim/Documents/LLM/Large-Language-Models/rexx_pdfs"
    rag_system = RexxRAGSystem(pdf_folder)
    
    # Ensure PDFs are processed
    rag_system.process_pdfs()
    
    # Initialize fine-tuner
    fine_tuner = RexxFineTuner()
    
    # Generate training data
    training_data = fine_tuner.create_training_data_from_chunks(
        rag_system, num_samples=50
    )
    
    # Prepare dataset
    dataset = fine_tuner.prepare_dataset(training_data)
    
    print(f"\nDataset sizes:")
    print(f"- Training: {len(dataset['train'])}")
    print(f"- Validation: {len(dataset['validation'])}")
    
    # Fine-tune model
    trainer = fine_tuner.fine_tune(dataset, epochs=3)
    
    # Evaluate improvement
    test_questions = [
        {"question": "What is Rexx Systems?", "expected_answer": "HR software"},
        {"question": "What modules are available?"},
        {"question": "How to configure the system?"},
        {"question": "What are the main features?"},
        {"question": "What database does it use?"}
    ]
    
    results, metrics = fine_tuner.evaluate_improvement(test_questions, rag_system)
    
    print("\n" + "="*50)
    print("Evaluation Results:")
    print(f"\nBase Model Performance:")
    print(f"- Accuracy: {metrics['base_model']['accuracy']:.2%}" if metrics['base_model']['accuracy'] else "- Accuracy: N/A")
    print(f"- Avg Answer Length: {metrics['base_model']['avg_answer_length']:.1f}")
    print(f"- Empty Answers: {metrics['base_model']['empty_answers']}")
    
    print(f"\nFine-tuned Model Performance:")
    print(f"- Accuracy: {metrics['fine_tuned']['accuracy']:.2%}" if metrics['fine_tuned']['accuracy'] else "- Accuracy: N/A")
    print(f"- Avg Answer Length: {metrics['fine_tuned']['avg_answer_length']:.1f}")
    print(f"- Empty Answers: {metrics['fine_tuned']['empty_answers']}")
    
    if 'improvement' in metrics:
        print(f"\nImprovement:")
        print(f"- Accuracy Gain: {metrics['improvement']['accuracy_gain']:.2%}")
        print(f"- Relative Improvement: {metrics['improvement']['relative_improvement']:.1f}%")


if __name__ == "__main__":
    main()
