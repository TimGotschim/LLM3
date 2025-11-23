"""
Multi-Model Fine-Tuning System for Rexx RAG
Fine-tune and compare multiple HuggingFace models simultaneously
"""

import os
import json
from typing import List, Dict, Optional
import numpy as np
import pandas as pd
from datetime import datetime
from tqdm import tqdm

import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForQuestionAnswering,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
from datasets import Dataset
import evaluate


class MultiModelFineTuner:
    """
    Fine-tune multiple HuggingFace models and compare their performance.
    
    Supported strategies:
    1. Train multiple models separately and compare
    2. Ensemble predictions from multiple models
    3. Model stacking (use one model's output as input to another)
    """
    
    def __init__(self, output_dir: str = "./fine_tuned_models"):
        """
        Initialize multi-model fine-tuner.
        
        Args:
            output_dir: Base directory for saving fine-tuned models
        """
        self.output_dir = output_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.metric = evaluate.load("squad")
        
        # Track all trained models
        self.trained_models = {}
        self.training_history = []
        
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"Device: {self.device}")
        print(f"Output directory: {output_dir}")
    
    def get_recommended_models(self) -> Dict[str, Dict]:
        """
        Get list of recommended HuggingFace models for QA fine-tuning.
        
        Returns:
            Dictionary of model configurations
        """
        return {
            "distilbert": {
                "name": "distilbert-base-uncased-distilled-squad",
                "description": "Fast, efficient, good baseline",
                "size": "Small (66M params)",
                "speed": "Fast",
                "quality": "Good"
            },
            "bert-base": {
                "name": "bert-base-uncased",
                "description": "Standard BERT, reliable",
                "size": "Medium (110M params)",
                "speed": "Medium",
                "quality": "Very Good"
            },
            "roberta": {
                "name": "deepset/roberta-base-squad2",
                "description": "RoBERTa optimized for QA",
                "size": "Medium (125M params)",
                "speed": "Medium",
                "quality": "Excellent"
            },
            "albert": {
                "name": "twmkn9/albert-base-v2-squad2",
                "description": "Lightweight alternative to BERT",
                "size": "Small (12M params)",
                "speed": "Very Fast",
                "quality": "Good"
            },
            "electra": {
                "name": "google/electra-base-discriminator",
                "description": "Efficient pre-training method",
                "size": "Medium (110M params)",
                "speed": "Fast",
                "quality": "Very Good"
            },
            "deberta": {
                "name": "microsoft/deberta-v3-base",
                "description": "State-of-the-art performance",
                "size": "Large (184M params)",
                "speed": "Slow",
                "quality": "Excellent"
            }
        }
    
    def print_model_recommendations(self):
        """Print formatted model recommendations."""
        models = self.get_recommended_models()
        
        print("\n" + "="*80)
        print("RECOMMENDED MODELS FOR QA FINE-TUNING")
        print("="*80)
        
        for model_id, info in models.items():
            print(f"\n{model_id.upper()}")
            print(f"  Model: {info['name']}")
            print(f"  Description: {info['description']}")
            print(f"  Size: {info['size']}")
            print(f"  Speed: {info['speed']}")
            print(f"  Quality: {info['quality']}")
        
        print("\n" + "="*80)
        print("RECOMMENDATIONS:")
        print("  • Start with: distilbert (fast baseline)")
        print("  • Best quality: roberta or deberta")
        print("  • Resource-limited: albert")
        print("  • Production: roberta (good balance)")
        print("="*80)
    
    def fine_tune_single_model(self,
                               model_name: str,
                               model_id: str,
                               training_data: List[Dict],
                               epochs: int = 3,
                               batch_size: int = 16,
                               learning_rate: float = 2e-5) -> Dict:
        """
        Fine-tune a single model.
        
        Args:
            model_name: Identifier for this model (e.g., "distilbert")
            model_id: HuggingFace model ID
            training_data: Training data
            epochs: Number of training epochs
            batch_size: Training batch size
            learning_rate: Learning rate
            
        Returns:
            Training results
        """
        print(f"\n{'='*80}")
        print(f"FINE-TUNING: {model_name}")
        print(f"Model: {model_id}")
        print(f"{'='*80}")
        
        # Load model and tokenizer
        print("Loading model and tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForQuestionAnswering.from_pretrained(model_id)
        model.to(self.device)
        
        # Prepare dataset
        print("Preparing dataset...")
        from sklearn.model_selection import train_test_split
        train_data, val_data = train_test_split(training_data, test_size=0.2, random_state=42)
        
        train_dataset = Dataset.from_list(train_data)
        val_dataset = Dataset.from_list(val_data)
        
        # Tokenize
        train_dataset = train_dataset.map(
            lambda x: self._preprocess_function(x, tokenizer),
            batched=True,
            remove_columns=train_dataset.column_names
        )
        
        val_dataset = val_dataset.map(
            lambda x: self._preprocess_function(x, tokenizer),
            batched=True,
            remove_columns=val_dataset.column_names
        )
        
        # Training arguments
        model_output_dir = os.path.join(self.output_dir, model_name)
        training_args = TrainingArguments(
            output_dir=model_output_dir,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            learning_rate=learning_rate,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=epochs,
            weight_decay=0.01,
            logging_dir=f'./logs/{model_name}',
            logging_steps=10,
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            report_to="none"
        )
        
        # Data collator
        data_collator = DataCollatorWithPadding(tokenizer)
        
        # Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=self._compute_metrics
        )
        
        # Train
        print(f"Training {model_name}...")
        start_time = datetime.now()
        train_result = trainer.train()
        training_time = (datetime.now() - start_time).total_seconds()
        
        # Save model
        trainer.save_model(model_output_dir)
        tokenizer.save_pretrained(model_output_dir)
        
        # Store results
        results = {
            "model_name": model_name,
            "model_id": model_id,
            "training_time": training_time,
            "final_loss": train_result.training_loss,
            "epochs": epochs,
            "train_samples": len(train_data),
            "val_samples": len(val_data),
            "saved_to": model_output_dir
        }
        
        self.trained_models[model_name] = {
            "model": model,
            "tokenizer": tokenizer,
            "results": results
        }
        
        self.training_history.append(results)
        
        print(f"✓ {model_name} trained in {training_time:.1f}s")
        print(f"✓ Final loss: {train_result.training_loss:.4f}")
        print(f"✓ Saved to: {model_output_dir}")
        
        return results
    
    def fine_tune_multiple_models(self,
                                  model_configs: List[Dict],
                                  training_data: List[Dict],
                                  epochs: int = 3) -> pd.DataFrame:
        """
        Fine-tune multiple models and compare results.
        
        Args:
            model_configs: List of model configurations
                Example: [
                    {"name": "distilbert", "model_id": "distilbert-base-uncased-distilled-squad"},
                    {"name": "roberta", "model_id": "deepset/roberta-base-squad2"}
                ]
            training_data: Training data
            epochs: Number of training epochs
            
        Returns:
            Comparison dataframe
        """
        print("\n" + "="*80)
        print("MULTI-MODEL FINE-TUNING")
        print(f"Models: {len(model_configs)}")
        print(f"Training samples: {len(training_data)}")
        print(f"Epochs: {epochs}")
        print("="*80)
        
        results = []
        
        for config in model_configs:
            try:
                result = self.fine_tune_single_model(
                    model_name=config["name"],
                    model_id=config["model_id"],
                    training_data=training_data,
                    epochs=epochs,
                    batch_size=config.get("batch_size", 16),
                    learning_rate=config.get("learning_rate", 2e-5)
                )
                results.append(result)
            except Exception as e:
                print(f"✗ Error training {config['name']}: {e}")
                continue
        
        # Create comparison dataframe
        df = pd.DataFrame(results)
        
        return df
    
    def evaluate_all_models(self, 
                           test_questions: List[Dict],
                           rag_system) -> pd.DataFrame:
        """
        Evaluate all fine-tuned models.
        
        Args:
            test_questions: Test questions with expected answers
            rag_system: RAG system for retrieval
            
        Returns:
            Evaluation results dataframe
        """
        print("\n" + "="*80)
        print("EVALUATING ALL FINE-TUNED MODELS")
        print("="*80)
        
        results = []
        
        for model_name, model_info in self.trained_models.items():
            print(f"\nEvaluating {model_name}...")
            
            model = model_info["model"]
            tokenizer = model_info["tokenizer"]
            
            correct = 0
            total = 0
            response_times = []
            
            for test_case in tqdm(test_questions, desc=model_name):
                question = test_case['question']
                expected = test_case.get('expected_answer', '')
                
                # Get context from RAG
                retrieved = rag_system.retrieve(question, n_results=1)
                if not retrieved:
                    continue
                
                context = retrieved[0]['text']
                
                # Get answer from model
                start_time = datetime.now()
                answer = self._get_answer(model, tokenizer, question, context)
                response_time = (datetime.now() - start_time).total_seconds()
                response_times.append(response_time)
                
                # Check correctness
                if expected and expected.lower() in answer.lower():
                    correct += 1
                total += 1
            
            # Calculate metrics
            accuracy = correct / total if total > 0 else 0
            avg_response_time = np.mean(response_times) if response_times else 0
            
            results.append({
                "model": model_name,
                "accuracy": accuracy,
                "correct": correct,
                "total": total,
                "avg_response_time": avg_response_time
            })
            
            print(f"  Accuracy: {accuracy:.2%}")
            print(f"  Avg Response Time: {avg_response_time:.3f}s")
        
        df = pd.DataFrame(results)
        df = df.sort_values("accuracy", ascending=False)
        
        return df
    
    def create_ensemble(self, 
                       models_to_ensemble: List[str],
                       strategy: str = "voting") -> 'ModelEnsemble':
        """
        Create an ensemble from multiple fine-tuned models.
        
        Args:
            models_to_ensemble: List of model names to include
            strategy: Ensemble strategy ('voting', 'averaging', 'stacking')
            
        Returns:
            ModelEnsemble object
        """
        ensemble_models = {}
        
        for model_name in models_to_ensemble:
            if model_name in self.trained_models:
                ensemble_models[model_name] = self.trained_models[model_name]
            else:
                print(f"⚠️  Model {model_name} not found in trained models")
        
        if not ensemble_models:
            raise ValueError("No valid models found for ensemble")
        
        print(f"✓ Created ensemble with {len(ensemble_models)} models")
        print(f"  Strategy: {strategy}")
        print(f"  Models: {', '.join(ensemble_models.keys())}")
        
        return ModelEnsemble(ensemble_models, strategy)
    
    def _preprocess_function(self, examples, tokenizer):
        """Tokenize examples for question answering."""
        questions = [q.strip() for q in examples["question"]]
        contexts = examples["context"]
        
        inputs = tokenizer(
            questions,
            contexts,
            truncation=True,
            padding="max_length",
            max_length=384,
            stride=128,
            return_overflowing_tokens=True,
            return_offsets_mapping=True
        )
        
        answers = examples["answer"]
        start_positions = []
        end_positions = []
        
        for i, offset_mapping in enumerate(inputs["offset_mapping"]):
            answer = answers[i]
            start_char = answer["answer_start"]
            end_char = start_char + len(answer["text"])
            
            sequence_ids = inputs.sequence_ids(i)
            context_start = sequence_ids.index(1)
            context_end = len(sequence_ids) - 1 - sequence_ids[::-1].index(1)
            
            if offset_mapping[context_start][0] > start_char or offset_mapping[context_end][1] < end_char:
                start_positions.append(0)
                end_positions.append(0)
            else:
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
    
    def _compute_metrics(self, eval_pred):
        """Compute metrics for evaluation."""
        predictions, labels = eval_pred
        start_predictions = np.argmax(predictions[0], axis=1)
        end_predictions = np.argmax(predictions[1], axis=1)
        
        start_accuracy = np.mean(start_predictions == labels[0])
        end_accuracy = np.mean(end_predictions == labels[1])
        
        return {
            "start_accuracy": start_accuracy,
            "end_accuracy": end_accuracy,
            "overall_accuracy": (start_accuracy + end_accuracy) / 2
        }
    
    def _get_answer(self, model, tokenizer, question: str, context: str) -> str:
        """Get answer from model."""
        inputs = tokenizer(
            question, context,
            return_tensors="pt",
            truncation=True,
            max_length=384
        ).to(self.device)
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        start_scores = outputs.start_logits
        end_scores = outputs.end_logits
        
        start_idx = torch.argmax(start_scores)
        end_idx = torch.argmax(end_scores)
        
        answer_tokens = inputs['input_ids'][0][start_idx:end_idx+1]
        answer = tokenizer.decode(answer_tokens, skip_special_tokens=True)
        
        return answer
    
    def save_training_summary(self, filename: str = "multi_model_training_summary.json"):
        """Save training summary to JSON."""
        summary = {
            "timestamp": datetime.now().isoformat(),
            "models_trained": len(self.trained_models),
            "training_history": self.training_history
        }
        
        with open(filename, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"✓ Training summary saved to {filename}")


class ModelEnsemble:
    """Ensemble multiple fine-tuned models."""
    
    def __init__(self, models: Dict, strategy: str = "voting"):
        self.models = models
        self.strategy = strategy
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def predict(self, question: str, context: str) -> Dict:
        """
        Get prediction from ensemble.
        
        Args:
            question: Question
            context: Context
            
        Returns:
            Ensemble prediction
        """
        predictions = {}
        
        # Get prediction from each model
        for model_name, model_info in self.models.items():
            model = model_info["model"]
            tokenizer = model_info["tokenizer"]
            
            inputs = tokenizer(
                question, context,
                return_tensors="pt",
                truncation=True,
                max_length=384
            ).to(self.device)
            
            with torch.no_grad():
                outputs = model(**inputs)
            
            start_idx = torch.argmax(outputs.start_logits)
            end_idx = torch.argmax(outputs.end_logits)
            
            answer_tokens = inputs['input_ids'][0][start_idx:end_idx+1]
            answer = tokenizer.decode(answer_tokens, skip_special_tokens=True)
            
            predictions[model_name] = answer
        
        # Apply ensemble strategy
        if self.strategy == "voting":
            # Most common answer
            from collections import Counter
            answer_counts = Counter(predictions.values())
            ensemble_answer = answer_counts.most_common(1)[0][0]
        elif self.strategy == "longest":
            # Longest answer (often more detailed)
            ensemble_answer = max(predictions.values(), key=len)
        else:
            # Default: first model
            ensemble_answer = list(predictions.values())[0]
        
        return {
            "ensemble_answer": ensemble_answer,
            "individual_answers": predictions,
            "strategy": self.strategy
        }


def example_multi_model_training():
    """Example: Train and compare multiple HuggingFace models."""
    
    print("="*80)
    print("MULTI-MODEL FINE-TUNING EXAMPLE")
    print("="*80)
    
    # Initialize
    fine_tuner = MultiModelFineTuner()
    
    # Show recommendations
    fine_tuner.print_model_recommendations()
    
    # Load training data
    training_data_file = "rexx_training_data.json"
    if not os.path.exists(training_data_file):
        print(f"\n❌ Training data not found: {training_data_file}")
        print("   Generate it first with: python rexx_fine_tuner.py")
        return
    
    with open(training_data_file, 'r') as f:
        training_data = json.load(f)
    
    print(f"\n✓ Loaded {len(training_data)} training examples")
    
    # Define models to train
    model_configs = [
        {
            "name": "distilbert",
            "model_id": "distilbert-base-uncased-distilled-squad",
            "batch_size": 16
        },
        {
            "name": "roberta",
            "model_id": "deepset/roberta-base-squad2",
            "batch_size": 8  # Larger model, smaller batch
        },
        # Add more models as needed
    ]
    
    # Train all models
    print("\nStarting multi-model training...")
    training_results = fine_tuner.fine_tune_multiple_models(
        model_configs,
        training_data[:50],  # Use subset for demo
        epochs=2  # Quick training for demo
    )
    
    print("\n" + "="*80)
    print("TRAINING RESULTS")
    print("="*80)
    print(training_results.to_string(index=False))
    
    # Save summary
    fine_tuner.save_training_summary()
    
    print("\n✓ Multi-model training complete!")
    print(f"  Models trained: {len(fine_tuner.trained_models)}")
    print(f"  Models saved to: {fine_tuner.output_dir}")


if __name__ == "__main__":
    example_multi_model_training()
