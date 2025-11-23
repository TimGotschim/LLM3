"""
Fine-tuning Script for Rexx HR Q&A
Uses HuggingFace Transformers with LoRA for efficient fine-tuning
"""

import json
import os
import sys
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)

# Check for MPS (Apple Silicon) or CUDA
if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

print(f"Using device: {DEVICE}", flush=True)


def load_training_data(dataset_path: str) -> Dataset:
    """Load and format training data for fine-tuning."""
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    train_data = data['train']

    # Format as instruction-following format
    formatted_data = []
    for item in train_data:
        text = f"""### Instruction:
Beantworte die folgende Frage über rexx HR Software präzise und auf Deutsch.

### Question:
{item['question']}

### Answer:
{item['expected_answer']}"""
        formatted_data.append({"text": text})

    return Dataset.from_list(formatted_data)


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(script_dir, "rexx_qa_dataset_curated.json")
    output_dir = os.path.join(script_dir, "rexx_finetuned_model")

    # Model selection - use a smaller model for faster training
    # Options: "TinyLlama/TinyLlama-1.1B-Chat-v1.0", "microsoft/phi-2", "google/gemma-2b"
    MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

    print(f"Loading model: {MODEL_NAME}")
    print(f"Device: {DEVICE}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Load model - CPU/MPS compatible
    print("Loading model weights...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32,
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )
    print("Model loaded!", flush=True)

    # Configure LoRA
    lora_config = LoraConfig(
        r=16,  # LoRA rank
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Load and prepare dataset
    print("\nLoading training data...")
    train_dataset = load_training_data(dataset_path)
    print(f"Training examples: {len(train_dataset)}")

    # Tokenize dataset
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=512,
            padding="max_length"
        )

    tokenized_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=["text"])

    # Training arguments - optimized for CPU/MPS
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        weight_decay=0.01,
        logging_steps=5,
        save_steps=50,
        save_total_limit=2,
        fp16=False,
        bf16=False,
        push_to_hub=False,
        report_to="none",
        remove_unused_columns=False,
        use_cpu=True,  # Force CPU for stability
        dataloader_pin_memory=False,
    )

    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )

    # Train
    print("\n" + "=" * 60)
    print("STARTING FINE-TUNING")
    print("=" * 60)

    trainer.train()

    # Save model
    print("\nSaving fine-tuned model...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    print(f"\nModel saved to: {output_dir}")
    print("\nFine-tuning complete!")


if __name__ == "__main__":
    main()
