#!/usr/bin/env python3
"""
Pilot training script for LLaMA 7B on golden set.
Quick validation of the training pipeline before scaling.
"""

import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description='Run pilot training on golden set with LLaMA 7B',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--golden-set', required=True, type=Path,
                       help='Path to golden set JSONL')
    parser.add_argument('--output-dir', required=True, type=Path,
                       help='Output directory for checkpoints')
    parser.add_argument('--config', type=Path,
                       help='Training config JSON (optional)')
    parser.add_argument('--use-wandb', action='store_true',
                       help='Use Weights & Biases for logging')
    parser.add_argument('--max-steps', type=int, default=100,
                       help='Maximum training steps (default: 100)')
    parser.add_argument('--eval-steps', type=int, default=25,
                       help='Evaluation steps interval (default: 25)')
    
    args = parser.parse_args()
    
    if not args.golden_set.exists():
        print(f"[ERROR] Golden set not found: {args.golden_set}", file=sys.stderr)
        sys.exit(1)
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    print("[PILOT] Starting pilot training on golden set")
    print(f"[PILOT] Golden set: {args.golden_set}")
    print(f"[PILOT] Output directory: {args.output_dir}")
    print(f"[PILOT] Max steps: {args.max_steps}")
    
    # Build training command
    cmd = [
        sys.executable, 'llama_train.py',
        '--model-name', 'meta-llama/Llama-2-7b-hf',
        '--model-size', '7b',
        '--dataset', str(args.golden_set),
        '--golden-set', str(args.golden_set),  # Use as eval set too
        '--output-dir', str(args.output_dir),
        '--use-4bit',
        '--batch-size', '2',  # Smaller for pilot
        '--gradient-accumulation-steps', '2',
        '--learning-rate', '2e-5',
        '--num-epochs', '1',  # Single epoch for pilot
        '--eval-steps', str(args.eval_steps),
        '--save-steps', str(args.eval_steps),
        '--logging-steps', '5',
        '--max-length', '1024',  # Shorter for pilot
    ]
    
    if args.config and args.config.exists():
        cmd.extend(['--config', str(args.config)])
    
    if args.use_wandb:
        cmd.append('--use-wandb')
        cmd.extend(['--wandb-project', 'llama-sft-pilot'])
    
    # Add max steps override (hack: use num_epochs=1 and control via steps)
    # Actually, we'll just use a small dataset and single epoch
    
    print(f"\n[PILOT] Running command:")
    print(f"  {' '.join(cmd)}\n")
    
    # Run training
    try:
        result = subprocess.run(cmd, check=True)
        print("\n[PILOT] Training completed successfully!")
        print(f"[PILOT] Checkpoints saved to: {args.output_dir}")
        
        # Run quick evaluation
        print("\n[PILOT] Running evaluation...")
        eval_cmd = [
            sys.executable, 'llama_evaluate.py',
            '--base-model', 'meta-llama/Llama-2-7b-hf',
            '--adapter-path', str(args.output_dir / 'final_model'),
            '--dataset', str(args.golden_set),
            '--output', str(args.output_dir / 'pilot_evaluation.json'),
            '--num-samples', '10',
            '--use-4bit',
        ]
        
        subprocess.run(eval_cmd, check=True)
        print(f"[PILOT] Evaluation results saved to: {args.output_dir / 'pilot_evaluation.json'}")
        
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Training failed with exit code {e.returncode}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n[PILOT] Training interrupted by user")
        sys.exit(1)


if __name__ == '__main__':
    main()

