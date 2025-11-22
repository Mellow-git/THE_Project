#!/usr/bin/env python3
"""
LLaMA Dataset Loader for SFT format.
Handles tokenization, truncation, and padding for LLaMA 2 models.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class LLaMASFTDataset(Dataset):
    """
    Dataset loader for SFT format compatible with LLaMA tokenizer.
    """
    
    def __init__(
        self,
        jsonl_path: Path,
        tokenizer: AutoTokenizer,
        max_length: int = 2048,
        split: Optional[str] = None,
    ):
        """
        Initialize dataset.
        
        Args:
            jsonl_path: Path to SFT JSONL file
            tokenizer: LLaMA tokenizer instance
            max_length: Maximum sequence length
            split: Filter by split_hint ('train', 'val', 'test') or None for all
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []
        
        # Load examples
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    example = json.loads(line)
                    # Filter by split if specified
                    if split and example.get('meta', {}).get('split_hint') != split:
                        continue
                    self.examples.append(example)
                except json.JSONDecodeError:
                    continue
        
        print(f"[DATASET] Loaded {len(self.examples)} examples" + 
              (f" (split: {split})" if split else ""))
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        
        # Build prompt: instruction + context
        instruction = example.get('instruction', '')
        context = example.get('context', '')
        response = example.get('response', '')
        
        # Format as instruction-following prompt
        # LLaMA 2 chat format: [INST] instruction [/INST] response
        prompt = f"[INST] {instruction}\n\nContext: {context} [/INST]"
        full_text = f"{prompt} {response}"
        
        # Tokenize
        encoding = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt',
        )
        
        # Also tokenize prompt separately for loss calculation
        prompt_encoding = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt',
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': encoding['input_ids'].squeeze().clone(),
            'prompt_length': prompt_encoding['input_ids'].squeeze().sum().item(),
            'meta': example.get('meta', {}),
        }
    
    def get_example_text(self, idx: int) -> Dict[str, str]:
        """Get raw text of example for inspection."""
        example = self.examples[idx]
        return {
            'instruction': example.get('instruction', ''),
            'context': example.get('context', ''),
            'response': example.get('response', ''),
        }


def load_tokenizer(model_name: str = "meta-llama/Llama-2-7b-hf") -> AutoTokenizer:
    """
    Load LLaMA tokenizer.
    
    Args:
        model_name: Hugging Face model identifier
    
    Returns:
        Tokenizer instance
    """
    print(f"[TOKENIZER] Loading tokenizer for {model_name}...")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
        )
        
        # Set pad token if not present
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        
        # Set padding side
        tokenizer.padding_side = "right"
        
        print(f"[TOKENIZER] Loaded tokenizer (vocab_size: {len(tokenizer)})")
        return tokenizer
    
    except Exception as e:
        print(f"[ERROR] Failed to load tokenizer: {e}", file=__import__('sys').stderr)
        raise


def get_dataset_stats(dataset: LLaMASFTDataset) -> Dict:
    """Get statistics about the dataset."""
    total = len(dataset)
    if total == 0:
        return {}
    
    # Sample some examples to get average lengths
    sample_size = min(100, total)
    total_length = 0
    
    for i in range(sample_size):
        item = dataset[i]
        total_length += item['input_ids'].sum().item()
    
    avg_length = total_length / sample_size if sample_size > 0 else 0
    
    return {
        'total_examples': total,
        'average_sequence_length': avg_length,
        'max_length': dataset.max_length,
    }

