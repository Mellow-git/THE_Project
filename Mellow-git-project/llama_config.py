#!/usr/bin/env python3
"""
Model configuration and hardware requirements for different LLaMA sizes.
"""

from pathlib import Path

MODEL_CONFIGS = {
    '7b': {
        'model_name': 'meta-llama/Llama-2-7b-hf',
        'parameters': 7_000_000_000,
        'memory_gb': {
            'fp32': 28,
            'fp16': 14,
            'int8': 7,
            'int4': 4,
        },
        'recommended_gpu': 'NVIDIA RTX 3090 (24GB) or better',
        'min_gpu_memory': 16,
        'lora_r': 16,
        'lora_alpha': 32,
        'batch_size': 4,
        'gradient_accumulation': 4,
    },
    '13b': {
        'model_name': 'meta-llama/Llama-2-13b-hf',
        'parameters': 13_000_000_000,
        'memory_gb': {
            'fp32': 52,
            'fp16': 26,
            'int8': 13,
            'int4': 7,
        },
        'recommended_gpu': 'NVIDIA A100 (40GB) or better',
        'min_gpu_memory': 24,
        'lora_r': 32,
        'lora_alpha': 64,
        'batch_size': 2,
        'gradient_accumulation': 8,
    },
    '70b': {
        'model_name': 'meta-llama/Llama-2-70b-hf',
        'parameters': 70_000_000_000,
        'memory_gb': {
            'fp32': 280,
            'fp16': 140,
            'int8': 70,
            'int4': 35,
        },
        'recommended_gpu': 'Multiple A100 (80GB) or H100',
        'min_gpu_memory': 80,
        'lora_r': 64,
        'lora_alpha': 128,
        'batch_size': 1,
        'gradient_accumulation': 16,
    },
}


def get_model_config(model_size: str) -> dict:
    """Get configuration for model size."""
    if model_size not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model size: {model_size}. Choose from {list(MODEL_CONFIGS.keys())}")
    return MODEL_CONFIGS[model_size]


def print_model_info(model_size: str):
    """Print model information and requirements."""
    config = get_model_config(model_size)
    
    print(f"\n[LLaMA {model_size.upper()} Configuration]")
    print(f"  Model: {config['model_name']}")
    print(f"  Parameters: {config['parameters']:,}")
    print(f"\n  Memory Requirements:")
    for precision, memory in config['memory_gb'].items():
        print(f"    {precision}: {memory} GB")
    print(f"\n  Recommended GPU: {config['recommended_gpu']}")
    print(f"  Minimum GPU Memory: {config['min_gpu_memory']} GB")
    print(f"\n  LoRA Configuration:")
    print(f"    r: {config['lora_r']}")
    print(f"    alpha: {config['lora_alpha']}")
    print(f"\n  Training Configuration:")
    print(f"    Batch size: {config['batch_size']}")
    print(f"    Gradient accumulation: {config['gradient_accumulation']}")


def generate_training_command(model_size: str, dataset_path: Path, output_dir: Path, **kwargs) -> str:
    """Generate training command for model size."""
    config = get_model_config(model_size)
    
    cmd_parts = [
        'python3', 'llama_train.py',
        '--model-name', config['model_name'],
        '--model-size', model_size,
        '--dataset', str(dataset_path),
        '--output-dir', str(output_dir),
        '--lora-r', str(config['lora_r']),
        '--lora-alpha', str(config['lora_alpha']),
        '--batch-size', str(config['batch_size']),
        '--gradient-accumulation-steps', str(config['gradient_accumulation']),
        '--use-4bit',  # Use 4-bit by default for efficiency
    ]
    
    # Add optional arguments
    if 'golden_set' in kwargs:
        cmd_parts.extend(['--golden-set', str(kwargs['golden_set'])])
    if 'config' in kwargs:
        cmd_parts.extend(['--config', str(kwargs['config'])])
    if 'use_wandb' in kwargs and kwargs['use_wandb']:
        cmd_parts.append('--use-wandb')
    
    return ' '.join(cmd_parts)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='LLaMA model configuration')
    parser.add_argument('--model-size', choices=['7b', '13b', '70b'],
                       help='Model size to show info for')
    parser.add_argument('--list', action='store_true',
                       help='List all available model sizes')
    
    args = parser.parse_args()
    
    if args.list:
        print("Available model sizes:")
        for size in MODEL_CONFIGS.keys():
            print(f"  - {size}")
    elif args.model_size:
        print_model_info(args.model_size)
    else:
        print("Use --model-size to see configuration or --list to see available sizes")

