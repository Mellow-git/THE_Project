#!/usr/bin/env python3
"""
Evaluation script for fine-tuned LLaMA models.
Evaluates on validation and golden sets, generates sample outputs.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from llama_dataset import LLaMASFTDataset


def load_model_for_inference(
    base_model_name: str,
    adapter_path: Path,
    use_4bit: bool = True,
) -> tuple:
    """Load base model and LoRA adapter for inference."""
    print(f"[MODEL] Loading base model: {base_model_name}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_name,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load base model with quantization
    if use_4bit:
        from transformers import BitsAndBytesConfig
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16,
        )
    
    # Load LoRA adapter
    if adapter_path.exists():
        print(f"[MODEL] Loading LoRA adapter from {adapter_path}")
        model = PeftModel.from_pretrained(model, str(adapter_path))
        model.eval()
    else:
        print(f"[WARN] Adapter path not found: {adapter_path}", file=sys.stderr)
    
    return model, tokenizer


def generate_response(
    model,
    tokenizer,
    instruction: str,
    context: str,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> str:
    """Generate response for given instruction and context."""
    prompt = f"[INST] {instruction}\n\nContext: {context} [/INST]"
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    # Decode only the generated part
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract response (after [/INST])
    if "[/INST]" in generated_text:
        response = generated_text.split("[/INST]")[-1].strip()
    else:
        response = generated_text[len(prompt):].strip()
    
    return response


def evaluate_dataset(
    model,
    tokenizer,
    dataset: LLaMASFTDataset,
    num_samples: int = 50,
    max_new_tokens: int = 256,
) -> Dict:
    """Evaluate model on dataset and generate samples."""
    results = {
        'total_examples': len(dataset),
        'evaluated': min(num_samples, len(dataset)),
        'samples': [],
    }
    
    print(f"[EVAL] Evaluating on {results['evaluated']} examples...")
    
    for i in tqdm(range(results['evaluated'])):
        example = dataset.examples[i]
        instruction = example.get('instruction', '')
        context = example.get('context', '')
        expected_response = example.get('response', '')
        
        # Generate response
        generated_response = generate_response(
            model,
            tokenizer,
            instruction,
            context,
            max_new_tokens=max_new_tokens,
        )
        
        results['samples'].append({
            'instruction': instruction,
            'context': context[:200] + '...' if len(context) > 200 else context,
            'expected_response': expected_response[:200] + '...' if len(expected_response) > 200 else expected_response,
            'generated_response': generated_response,
            'meta': example.get('meta', {}),
        })
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate fine-tuned LLaMA model',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--base-model', required=True, type=str,
                       help='Base model name (e.g., meta-llama/Llama-2-7b-hf)')
    parser.add_argument('--adapter-path', required=True, type=Path,
                       help='Path to LoRA adapter')
    parser.add_argument('--dataset', required=True, type=Path,
                       help='Evaluation dataset JSONL')
    parser.add_argument('--output', required=True, type=Path,
                       help='Output JSON file for results')
    parser.add_argument('--num-samples', type=int, default=50,
                       help='Number of samples to evaluate')
    parser.add_argument('--max-new-tokens', type=int, default=256,
                       help='Maximum tokens to generate')
    parser.add_argument('--max-length', type=int, default=2048,
                       help='Maximum input sequence length')
    parser.add_argument('--use-4bit', action='store_true',
                       help='Use 4-bit quantization')
    
    args = parser.parse_args()
    
    # Load model
    model, tokenizer = load_model_for_inference(
        args.base_model,
        args.adapter_path,
        use_4bit=args.use_4bit,
    )
    
    # Load dataset
    dataset = LLaMASFTDataset(
        args.dataset,
        tokenizer,
        max_length=args.max_length,
    )
    
    # Evaluate
    results = evaluate_dataset(
        model,
        tokenizer,
        dataset,
        num_samples=args.num_samples,
        max_new_tokens=args.max_new_tokens,
    )
    
    # Save results
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n[SUCCESS] Evaluation complete. Results saved to {args.output}")
    print(f"[SUMMARY] Evaluated {results['evaluated']}/{results['total_examples']} examples")
    
    # Print sample outputs
    print("\n[SAMPLE OUTPUTS]")
    for i, sample in enumerate(results['samples'][:5], 1):
        print(f"\n--- Sample {i} ---")
        print(f"Instruction: {sample['instruction']}")
        print(f"Context: {sample['context'][:100]}...")
        print(f"Generated: {sample['generated_response'][:200]}...")


if __name__ == '__main__':
    main()

