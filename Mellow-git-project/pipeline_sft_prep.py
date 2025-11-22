#!/usr/bin/env python3
"""
Master pipeline script for SFT data preparation.
Orchestrates the full pipeline from enriched data to training-ready format.
"""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Optional


def run_command(cmd: list, description: str) -> bool:
    """Run a command and return success status."""
    print(f"\n{'='*60}")
    print(f"[PIPELINE] {description}")
    print(f"{'='*60}")
    print(f"[CMD] {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Command failed with exit code {e.returncode}", file=sys.stderr)
        return False
    except FileNotFoundError:
        print(f"[ERROR] Command not found: {cmd[0]}", file=sys.stderr)
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Master pipeline for SFT data preparation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This pipeline runs:
1. Validation of enriched dataset
2. Golden set extraction
3. SFT transformation
4. Training preparation

Example:
  %(prog)s --enriched enriched_events.jsonl --output-dir ./sft_output
        """
    )
    parser.add_argument('--enriched', required=True, type=Path,
                       help='Input enriched JSONL file')
    parser.add_argument('--output-dir', required=True, type=Path,
                       help='Output directory for all generated files')
    parser.add_argument('--golden-size', type=int, default=150,
                       help='Size of golden set (default: 150)')
    parser.add_argument('--skip-validation', action='store_true',
                       help='Skip validation step')
    parser.add_argument('--skip-golden', action='store_true',
                       help='Skip golden set extraction')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility (default: 42)')
    
    args = parser.parse_args()
    
    if not args.enriched.exists():
        print(f"[ERROR] Enriched dataset not found: {args.enriched}", file=sys.stderr)
        sys.exit(1)
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"[INFO] Starting SFT preparation pipeline")
    print(f"[INFO] Input: {args.enriched}")
    print(f"[INFO] Output directory: {args.output_dir}")
    
    success = True
    
    # Step 1: Validation
    if not args.skip_validation:
        validation_report = args.output_dir / 'validation_report.json'
        cmd = [
            sys.executable, 'validate_enriched_dataset.py',
            '--input', str(args.enriched),
            '--output-report', str(validation_report),
            '--verbose',
        ]
        if not run_command(cmd, "Step 1: Validating enriched dataset"):
            print("[WARN] Validation found issues, but continuing...", file=sys.stderr)
            # Don't fail on validation errors, just warn
    
    # Step 2: Golden set extraction
    if not args.skip_golden:
        golden_set = args.output_dir / 'golden_set.jsonl'
        golden_csv = args.output_dir / 'golden_set.csv'
        cmd = [
            sys.executable, 'extract_golden_set.py',
            '--input', str(args.enriched),
            '--output', str(golden_set),
            '--size', str(args.golden_size),
            '--seed', str(args.seed),
            '--export-csv', str(golden_csv),
        ]
        if not run_command(cmd, "Step 2: Extracting golden set"):
            print("[ERROR] Golden set extraction failed", file=sys.stderr)
            success = False
    
    # Step 3: SFT transformation
    sft_dataset = args.output_dir / 'sft_dataset.jsonl'
    schema_path = Path(__file__).parent / 'schemas' / 'dataset.schema.json'
    cmd = [
        sys.executable, 'sft_transform.py',
        '--input', str(args.enriched),
        '--output', str(sft_dataset),
        '--seed', str(args.seed),
        '--validate',
    ]
    if schema_path.exists():
        cmd.extend(['--schema', str(schema_path)])
    
    if not run_command(cmd, "Step 3: Transforming to SFT format"):
        print("[ERROR] SFT transformation failed", file=sys.stderr)
        success = False
    
    # Step 4: Training preparation
    training_config = args.output_dir / 'training_config.json'
    training_stats = args.output_dir / 'dataset_statistics.json'
    cmd = [
        sys.executable, 'prepare_training.py',
        '--sft-dataset', str(sft_dataset),
        '--output-config', str(training_config),
        '--output-stats', str(training_stats),
    ]
    if not run_command(cmd, "Step 4: Preparing training configuration"):
        print("[ERROR] Training preparation failed", file=sys.stderr)
        success = False
    
    # Summary
    print(f"\n{'='*60}")
    print("[PIPELINE SUMMARY]")
    print(f"{'='*60}")
    
    if success:
        print("[SUCCESS] Pipeline completed successfully!")
        print(f"\nGenerated files in {args.output_dir}:")
        if not args.skip_validation:
            print(f"  - validation_report.json")
        if not args.skip_golden:
            print(f"  - golden_set.jsonl")
            print(f"  - golden_set.csv")
        print(f"  - sft_dataset.jsonl")
        print(f"  - training_config.json")
        print(f"  - dataset_statistics.json")
        print("\nNext steps:")
        print("  1. Review golden_set.csv for manual annotation")
        print("  2. Review training_config.json and adjust as needed")
        print("  3. Begin training with sft_dataset.jsonl")
    else:
        print("[ERROR] Pipeline completed with errors", file=sys.stderr)
        sys.exit(1)
    
    sys.exit(0)


if __name__ == '__main__':
    main()

