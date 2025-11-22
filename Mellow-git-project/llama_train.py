#!/usr/bin/env python3
"""
LLaMA Fine-Tuning Script with PEFT/LoRA.
Supports multiple model sizes (7B, 13B, 70B) with efficient training.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional

import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
)
from datasets import load_dataset
import wandb


def load_model_and_tokenizer(
    model_name: str,
    use_4bit: bool = True,
    use_8bit: bool = False,
) -> tuple:
    """Load LLaMA model and tokenizer with quantization."""
    print(f"[MODEL] Loading {model_name}...")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Configure quantization
    quantization_config = None
    if use_4bit:
        from transformers import BitsAndBytesConfig
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    elif use_8bit:
        from transformers import BitsAndBytesConfig
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16 if use_4bit or use_8bit else torch.float32,
    )
    
    # Prepare for k-bit training if quantized
    if use_4bit or use_8bit:
        model = prepare_model_for_kbit_training(model)
    
    print(f"[MODEL] Loaded model ({model.num_parameters():,} parameters)")
    return model, tokenizer


def setup_lora(
    model,
    r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    target_modules: Optional[list] = None,
) -> tuple:
    """Setup LoRA configuration and apply to model."""
    if target_modules is None:
        # Default target modules for LLaMA
        target_modules = ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    
    lora_config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    return model, lora_config


class SFTDataCollator:
    """Data collator for SFT format with proper loss masking."""
    
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    
    def __call__(self, features):
        # Stack inputs
        input_ids = torch.stack([f['input_ids'] for f in features])
        attention_mask = torch.stack([f['attention_mask'] for f in features])
        labels = torch.stack([f['labels'] for f in features])
        
        # Mask prompt tokens in labels (set to -100 to ignore in loss)
        prompt_lengths = [f['prompt_length'] for f in features]
        for i, prompt_len in enumerate(prompt_lengths):
            labels[i, :prompt_len] = -100
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
        }


def compute_metrics(eval_pred, tokenizer):
    """Compute perplexity and other metrics."""
    predictions, labels = eval_pred
    # Shift so that tokens < n predict n
    shift_logits = predictions[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    
    # Flatten the tokens
    shift_logits = shift_logits.view(-1, shift_logits.size(-1))
    shift_labels = shift_labels.view(-1)
    
    # Calculate perplexity
    loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
    loss = loss_fct(shift_logits, shift_labels)
    perplexity = torch.exp(loss)
    
    return {
        'perplexity': perplexity.item(),
        'loss': loss.item(),
    }


def main():
    parser = argparse.ArgumentParser(
        description='Fine-tune LLaMA with PEFT/LoRA',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Model configuration
    parser.add_argument('--model-name', type=str,
                       default='meta-llama/Llama-2-7b-hf',
                       help='Hugging Face model identifier')
    parser.add_argument('--model-size', type=str, choices=['7b', '13b', '70b'],
                       default='7b',
                       help='Model size (7b, 13b, 70b)')
    parser.add_argument('--use-4bit', action='store_true',
                       help='Use 4-bit quantization')
    parser.add_argument('--use-8bit', action='store_true',
                       help='Use 8-bit quantization')
    
    # Data configuration
    parser.add_argument('--dataset', required=True, type=Path,
                       help='Path to SFT JSONL dataset')
    parser.add_argument('--golden-set', type=Path,
                       help='Path to golden set JSONL for evaluation')
    parser.add_argument('--max-length', type=int, default=2048,
                       help='Maximum sequence length')
    
    # LoRA configuration
    parser.add_argument('--lora-r', type=int, default=16,
                       help='LoRA rank')
    parser.add_argument('--lora-alpha', type=int, default=32,
                       help='LoRA alpha')
    parser.add_argument('--lora-dropout', type=float, default=0.05,
                       help='LoRA dropout')
    
    # Training configuration
    parser.add_argument('--output-dir', required=True, type=Path,
                       help='Output directory for checkpoints')
    parser.add_argument('--config', type=Path,
                       help='Training config JSON file')
    parser.add_argument('--batch-size', type=int,
                       help='Training batch size (overrides config)')
    parser.add_argument('--gradient-accumulation-steps', type=int,
                       help='Gradient accumulation steps (overrides config)')
    parser.add_argument('--learning-rate', type=float,
                       help='Learning rate (overrides config)')
    parser.add_argument('--num-epochs', type=int,
                       help='Number of epochs (overrides config)')
    parser.add_argument('--eval-steps', type=int, default=100,
                       help='Evaluation steps interval')
    parser.add_argument('--save-steps', type=int, default=500,
                       help='Save checkpoint steps interval')
    parser.add_argument('--logging-steps', type=int, default=10,
                       help='Logging steps interval')
    
    # Other options
    parser.add_argument('--early-stopping-patience', type=int, default=3,
                       help='Early stopping patience')
    parser.add_argument('--use-wandb', action='store_true',
                       help='Use Weights & Biases for logging')
    parser.add_argument('--wandb-project', type=str, default='llama-sft',
                       help='W&B project name')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Load training config if provided
    training_config = {}
    if args.config and args.config.exists():
        with open(args.config, 'r') as f:
            training_config = json.load(f)
        print(f"[CONFIG] Loaded training config from {args.config}")
    
    # Override with command-line args
    batch_size = args.batch_size or training_config.get('training', {}).get('recommended_batch_size', 4)
    gradient_accumulation = args.gradient_accumulation_steps or training_config.get('training', {}).get('recommended_gradient_accumulation_steps', 4)
    learning_rate = args.learning_rate or training_config.get('training', {}).get('recommended_learning_rate', 2e-5)
    num_epochs = args.num_epochs or training_config.get('training', {}).get('recommended_num_epochs', 3)
    max_length = args.max_length or training_config.get('training', {}).get('recommended_max_seq_length', 2048)
    
    print(f"[TRAINING] Configuration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Gradient accumulation: {gradient_accumulation}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Epochs: {num_epochs}")
    print(f"  Max length: {max_length}")
    
    # Initialize W&B if requested
    if args.use_wandb:
        wandb.init(
            project=args.wandb_project,
            config={
                'model_name': args.model_name,
                'model_size': args.model_size,
                'batch_size': batch_size,
                'learning_rate': learning_rate,
                'num_epochs': num_epochs,
                'lora_r': args.lora_r,
                'lora_alpha': args.lora_alpha,
            }
        )
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(
        args.model_name,
        use_4bit=args.use_4bit,
        use_8bit=args.use_8bit,
    )
    
    # Setup LoRA
    model, lora_config = setup_lora(
        model,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )
    
    # Load datasets
    sys.path.insert(0, str(Path(__file__).parent))
    from llama_dataset import LLaMASFTDataset
    
    train_dataset = LLaMASFTDataset(
        args.dataset,
        tokenizer,
        max_length=max_length,
        split='train',
    )
    
    eval_dataset = None
    if args.golden_set and args.golden_set.exists():
        eval_dataset = LLaMASFTDataset(
            args.golden_set,
            tokenizer,
            max_length=max_length,
        )
        print(f"[EVAL] Using golden set for evaluation: {len(eval_dataset)} examples")
    else:
        # Use validation split from main dataset
        eval_dataset = LLaMASFTDataset(
            args.dataset,
            tokenizer,
            max_length=max_length,
            split='val',
        )
        print(f"[EVAL] Using validation split: {len(eval_dataset)} examples")
    
    # Data collator
    data_collator = SFTDataCollator(tokenizer)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation,
        learning_rate=learning_rate,
        fp16=True,
        logging_steps=args.logging_steps,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        evaluation_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,
        report_to="wandb" if args.use_wandb else None,
        seed=args.seed,
        warmup_steps=100,
        max_steps=-1,
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    
    # Train
    print("[TRAINING] Starting training...")
    trainer.train()
    
    # Save final model
    final_model_dir = args.output_dir / "final_model"
    trainer.save_model(str(final_model_dir))
    tokenizer.save_pretrained(str(final_model_dir))
    
    print(f"[SUCCESS] Training complete. Model saved to {final_model_dir}")
    
    # Final evaluation
    print("[EVAL] Running final evaluation...")
    eval_results = trainer.evaluate()
    print(f"[EVAL] Final results: {eval_results}")
    
    if args.use_wandb:
        wandb.finish()


if __name__ == '__main__':
    main()

