#!/usr/bin/env python3
"""
Training preparation script.
Configures training environment, validates data, and prepares for model training.
"""

import argparse
import json
import sys
from pathlib import Path
from collections import Counter
from typing import Dict, Any


def analyze_dataset_statistics(sft_file: Path) -> Dict[str, Any]:
    """Analyze SFT dataset statistics."""
    examples = []
    
    try:
        with open(sft_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    example = json.loads(line)
                    examples.append(example)
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        print(f"[ERROR] Failed to read SFT file: {e}", file=sys.stderr)
        return {}
    
    if not examples:
        return {}
    
    # Compute statistics
    context_lengths = [len(e.get('context', '')) for e in examples]
    instruction_lengths = [len(e.get('instruction', '')) for e in examples]
    topics = [e.get('meta', {}).get('topic', 'unknown') for e in examples]
    splits = [e.get('meta', {}).get('split_hint', 'train') for e in examples]
    domains = [e.get('meta', {}).get('domain', 'unknown') for e in examples]
    
    return {
        'total_examples': len(examples),
        'context_length': {
            'min': min(context_lengths),
            'max': max(context_lengths),
            'mean': sum(context_lengths) / len(context_lengths),
            'median': sorted(context_lengths)[len(context_lengths) // 2],
        },
        'instruction_length': {
            'min': min(instruction_lengths),
            'max': max(instruction_lengths),
            'mean': sum(instruction_lengths) / len(instruction_lengths),
        },
        'topic_distribution': dict(Counter(topics)),
        'split_distribution': dict(Counter(splits)),
        'domain_distribution': dict(Counter(domains)),
    }


def generate_training_config(stats: Dict[str, Any], output_path: Path) -> Dict[str, Any]:
    """Generate training configuration based on dataset statistics."""
    total = stats.get('total_examples', 0)
    mean_context = stats.get('context_length', {}).get('mean', 500)
    
    # Estimate tokens (rough: 1 token ≈ 4 characters)
    estimated_tokens_per_example = mean_context / 4
    total_tokens = total * estimated_tokens_per_example
    
    # Recommend batch size based on context length
    if mean_context < 500:
        batch_size = 8
        gradient_accumulation = 2
    elif mean_context < 2000:
        batch_size = 4
        gradient_accumulation = 4
    else:
        batch_size = 2
        gradient_accumulation = 8
    
    config = {
        'dataset': {
            'total_examples': total,
            'estimated_tokens': int(total_tokens),
            'mean_context_length': int(mean_context),
        },
        'training': {
            'recommended_batch_size': batch_size,
            'recommended_gradient_accumulation_steps': gradient_accumulation,
            'recommended_max_seq_length': int(mean_context * 1.5),
            'recommended_learning_rate': 2e-5,
            'recommended_num_epochs': 3,
        },
        'model': {
            'recommended_base': 'gpt2',  # User should replace with actual model
            'note': 'Replace with your target model architecture',
        },
        'evaluation': {
            'validation_split': 'val',
            'test_split': 'test',
            'metrics': ['loss', 'perplexity'],
        },
    }
    
    return config


def main():
    parser = argparse.ArgumentParser(
        description='Prepare training environment and configuration',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--sft-dataset', required=True, type=Path,
                       help='Input SFT JSONL dataset file')
    parser.add_argument('--output-config', type=Path,
                       help='Output training configuration JSON file')
    parser.add_argument('--output-stats', type=Path,
                       help='Output dataset statistics JSON file')
    
    args = parser.parse_args()
    
    if not args.sft_dataset.exists():
        print(f"[ERROR] SFT dataset file not found: {args.sft_dataset}", file=sys.stderr)
        sys.exit(1)
    
    print(f"[INFO] Analyzing SFT dataset: {args.sft_dataset}")
    stats = analyze_dataset_statistics(args.sft_dataset)
    
    if not stats:
        print("[ERROR] Failed to analyze dataset", file=sys.stderr)
        sys.exit(1)
    
    print("\n[DATASET STATISTICS]")
    print(f"  Total examples: {stats['total_examples']}")
    print(f"\n  Context length:")
    ctx = stats['context_length']
    print(f"    Min: {ctx['min']} chars")
    print(f"    Max: {ctx['max']} chars")
    print(f"    Mean: {ctx['mean']:.1f} chars")
    print(f"    Median: {ctx['median']} chars")
    
    print(f"\n  Instruction length:")
    inst = stats['instruction_length']
    print(f"    Min: {inst['min']} chars")
    print(f"    Max: {inst['max']} chars")
    print(f"    Mean: {inst['mean']:.1f} chars")
    
    print(f"\n  Topic distribution:")
    for topic, count in stats['topic_distribution'].items():
        print(f"    {topic}: {count} ({count/stats['total_examples']*100:.1f}%)")
    
    print(f"\n  Split distribution:")
    for split, count in stats['split_distribution'].items():
        print(f"    {split}: {count} ({count/stats['total_examples']*100:.1f}%)")
    
    # Generate training config
    config = generate_training_config(stats, args.output_config or Path('training_config.json'))
    
    print("\n[TRAINING RECOMMENDATIONS]")
    print(f"  Batch size: {config['training']['recommended_batch_size']}")
    print(f"  Gradient accumulation: {config['training']['recommended_gradient_accumulation_steps']}")
    print(f"  Max sequence length: {config['training']['recommended_max_seq_length']}")
    print(f"  Learning rate: {config['training']['recommended_learning_rate']}")
    print(f"  Epochs: {config['training']['recommended_num_epochs']}")
    print(f"  Estimated tokens: {config['dataset']['estimated_tokens']:,}")
    
    # Write outputs
    if args.output_stats:
        print(f"\n[INFO] Writing statistics to {args.output_stats}...")
        with open(args.output_stats, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
    
    if args.output_config:
        print(f"[INFO] Writing training config to {args.output_config}...")
        with open(args.output_config, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        print(f"[SUCCESS] Training configuration saved to {args.output_config}")
        print("[NOTE] Review and adjust configuration before training")
    
    sys.exit(0)


if __name__ == '__main__':
    main()

