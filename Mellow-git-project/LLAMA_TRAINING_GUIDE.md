# LLaMA Fine-Tuning Guide

Complete guide for fine-tuning LLaMA models with PEFT/LoRA on the prepared SFT dataset.

## Overview

This pipeline enables efficient fine-tuning of LLaMA models (7B, 13B, 70B) using:
- **PEFT/LoRA**: Parameter-efficient fine-tuning with low-rank adaptation
- **4-bit Quantization**: Reduced memory footprint
- **Hugging Face Transformers**: Standard model interface
- **Custom SFT Dataset**: Instruction-following format

## Prerequisites

### Hardware Requirements

| Model Size | Minimum GPU Memory | Recommended GPU |
|------------|-------------------|-----------------|
| 7B         | 16 GB             | RTX 3090 (24GB) or better |
| 13B        | 24 GB             | A100 (40GB) or better |
| 70B        | 80 GB             | Multiple A100 (80GB) or H100 |

### Software Requirements

```bash
# Install dependencies
pip install -r requirements_training.txt

# Verify CUDA (if using GPU)
python -c "import torch; print(torch.cuda.is_available())"
```

### Model Access

You need access to LLaMA 2 models from Meta. Request access at:
- https://huggingface.co/meta-llama/Llama-2-7b-hf
- https://huggingface.co/meta-llama/Llama-2-13b-hf
- https://huggingface.co/meta-llama/Llama-2-70b-hf

After approval, authenticate:
```bash
huggingface-cli login
```

## Quick Start

### 1. Pilot Training (Recommended First Step)

Run a quick pilot on the golden set to verify the pipeline:

```bash
python3 llama_pilot.py \
  --golden-set ./sft_output/golden_set.jsonl \
  --output-dir ./pilot_output \
  --max-steps 100 \
  --use-wandb
```

This will:
- Train on golden set (150 examples)
- Use 4-bit quantization
- Run for 100 steps
- Evaluate and generate sample outputs

### 2. Full Training

Once pilot is successful, run full training:

```bash
python3 llama_train.py \
  --model-size 7b \
  --dataset ./sft_output/sft_dataset.jsonl \
  --golden-set ./sft_output/golden_set.jsonl \
  --output-dir ./training_output \
  --config ./sft_output/training_config.json \
  --use-4bit \
  --use-wandb
```

## Components

### 1. Dataset Loader (`llama_dataset.py`)

Loads SFT JSONL format and tokenizes for LLaMA:
- Handles instruction/context/response format
- Applies truncation and padding
- Supports train/val/test splits
- Compatible with LLaMA 2 tokenizer

### 2. Training Script (`llama_train.py`)

Main training script with:
- PEFT/LoRA integration
- 4-bit/8-bit quantization support
- Multi-model size support
- Early stopping
- W&B logging
- Checkpointing

### 3. Evaluation Script (`llama_evaluate.py`)

Evaluates fine-tuned models:
- Generates responses on test set
- Computes metrics
- Produces sample outputs
- Saves results to JSON

### 4. Model Configuration (`llama_config.py`)

Model-specific configurations:
- Hardware requirements
- Recommended hyperparameters
- LoRA settings per model size

### 5. Model Packaging (`llama_package.py`)

Packages models for distribution:
- Creates model archives
- Includes metadata
- Generates usage instructions
- Creates requirements file

## Training Configuration

### Model Sizes

```bash
# 7B model (recommended for most users)
python3 llama_train.py --model-size 7b ...

# 13B model (better quality, more memory)
python3 llama_train.py --model-size 13b ...

# 70B model (best quality, requires multiple GPUs)
python3 llama_train.py --model-size 70b ...
```

### LoRA Configuration

Default LoRA settings per model:

| Model | r | alpha | dropout |
|-------|---|-------|--------|
| 7B    | 16| 32    | 0.05   |
| 13B   | 32| 64    | 0.05   |
| 70B   | 64| 128   | 0.05   |

Override with:
```bash
--lora-r 32 --lora-alpha 64 --lora-dropout 0.1
```

### Training Hyperparameters

Default settings (can be overridden):
- **Batch size**: 4 (7B), 2 (13B), 1 (70B)
- **Gradient accumulation**: 4 (7B), 8 (13B), 16 (70B)
- **Learning rate**: 2e-5
- **Epochs**: 3
- **Max sequence length**: 2048

Override with:
```bash
--batch-size 8 --gradient-accumulation-steps 2 --learning-rate 1e-5
```

## Evaluation

### During Training

Evaluation runs automatically:
- Every `--eval-steps` (default: 100)
- On validation set or golden set
- Logs perplexity and loss

### After Training

Run evaluation script:

```bash
python3 llama_evaluate.py \
  --base-model meta-llama/Llama-2-7b-hf \
  --adapter-path ./training_output/final_model \
  --dataset ./sft_output/golden_set.jsonl \
  --output ./evaluation_results.json \
  --num-samples 50 \
  --use-4bit
```

## Model Packaging

Package trained model for distribution:

```bash
python3 llama_package.py \
  --adapter-path ./training_output/final_model \
  --base-model meta-llama/Llama-2-7b-hf \
  --model-size 7b \
  --output-dir ./packages
```

Creates:
- Model archive (`.tar.gz`)
- Metadata JSON
- README with usage instructions
- Requirements file

## Monitoring Training

### Weights & Biases

Enable W&B logging:
```bash
--use-wandb --wandb-project llama-sft
```

Tracks:
- Training/validation loss
- Perplexity
- Learning rate
- GPU utilization

### Local Logging

Training logs include:
- Step-by-step progress
- Evaluation metrics
- Checkpoint saves
- Sample generations

## Troubleshooting

### Out of Memory

1. Reduce batch size: `--batch-size 1`
2. Increase gradient accumulation: `--gradient-accumulation-steps 8`
3. Use 4-bit quantization: `--use-4bit`
4. Reduce sequence length: `--max-length 1024`

### Slow Training

1. Use mixed precision: `fp16=True` (default)
2. Enable gradient checkpointing (if supported)
3. Use multiple GPUs with `accelerate`
4. Reduce evaluation frequency: `--eval-steps 500`

### Model Not Loading

1. Verify Hugging Face authentication: `huggingface-cli login`
2. Check model name matches your access
3. Ensure sufficient disk space for model download
4. Try loading base model separately first

### Poor Quality Outputs

1. Increase training epochs: `--num-epochs 5`
2. Adjust learning rate: `--learning-rate 1e-5`
3. Increase LoRA rank: `--lora-r 32`
4. Train on larger dataset
5. Check data quality with validation script

## Best Practices

1. **Start with Pilot**: Always run pilot training first
2. **Monitor Metrics**: Watch validation loss for overfitting
3. **Save Checkpoints**: Regular checkpoints enable recovery
4. **Evaluate Regularly**: Check sample outputs during training
5. **Use Golden Set**: Evaluate on curated golden set for quality
6. **Version Control**: Track hyperparameters and configs
7. **Resource Planning**: Ensure sufficient GPU memory

## Example Workflow

```bash
# 1. Prepare data (already done)
# python3 pipeline_sft_prep.py --enriched ... --output-dir ./sft_output

# 2. Pilot training
python3 llama_pilot.py \
  --golden-set ./sft_output/golden_set.jsonl \
  --output-dir ./pilot_output

# 3. Review pilot results
cat ./pilot_output/pilot_evaluation.json

# 4. Full training
python3 llama_train.py \
  --model-size 7b \
  --dataset ./sft_output/sft_dataset.jsonl \
  --golden-set ./sft_output/golden_set.jsonl \
  --output-dir ./training_output \
  --config ./sft_output/training_config.json \
  --use-4bit \
  --use-wandb

# 5. Evaluate
python3 llama_evaluate.py \
  --base-model meta-llama/Llama-2-7b-hf \
  --adapter-path ./training_output/final_model \
  --dataset ./sft_output/golden_set.jsonl \
  --output ./final_evaluation.json

# 6. Package
python3 llama_package.py \
  --adapter-path ./training_output/final_model \
  --base-model meta-llama/Llama-2-7b-hf \
  --model-size 7b \
  --output-dir ./packages
```

## Next Steps

After training:

1. **Evaluate Quality**: Review evaluation results and sample outputs
2. **Iterate**: Adjust hyperparameters based on results
3. **Scale**: Train on full dataset or larger models
4. **Deploy**: Package and distribute fine-tuned models
5. **Monitor**: Track model performance in production

## Support

For issues:
1. Check logs for error messages
2. Verify hardware requirements
3. Review model access permissions
4. Consult Hugging Face documentation
5. Check PEFT/LoRA documentation

