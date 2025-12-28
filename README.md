# Tiny Personalization Pipeline

**Turn your browsing history into a tiny, specialized AI model that runs on Raspberry Pi‑class devices.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svghttps://img.shields.io/badge/Raspberry%20Pi-4%2F5%20(4GB%2B.svgIt Does

This pipeline transforms your personal browsing data into a **tiny, privacy‑preserving language model** that understands your interests, habits, and learning patterns—without ever leaving your device.

### Core Workflow

1. **Collect & Enrich**: Extracts browsing history from Safari/Chrome, enriches YouTube videos with metadata, and joins behavioral sessions from macOS knowledgeC.db.[1][2]
2. **Teacher Distillation**: Uses a large teacher model (e.g., Qwen3‑Coder‑32B) to generate high‑quality training signals from your enriched data.[3][4]
3. **Tiny Model Fine‑tuning**: Trains a **1–3B parameter student model** (e.g., Qwen2.5‑Coder‑3B) with LoRA/PEFT for efficient adaptation.[5][6]
4. **Pi Deployment**: Converts the fine‑tuned model to **GGUF format** via llama.cpp for inference on Raspberry Pi‑class devices.[7][8][9]

### Key Features

- **Privacy‑First**: All data stays local by default; no cloud uploads or telemetry.
- **Modular Architecture**: Each stage is independent—collect, enrich, distill, train, package—no cross‑file fixes.
- **Hardware‑Aware**: Estimates RAM usage and validates device compatibility before deployment.
- **Reproducible**: Deterministic seeds, schema validation, and versioned configs ensure consistent results.

***

## Quick Start

### 1. Install Dependencies

```bash
git clone https://github.com/yourusername/tiny-personalization.git
cd tiny-personalization
pip install -r requirements.txt
```

### 2. Collect & Enrich Your Data

On macOS, grant Full Disk Access to Terminal (System Settings → Privacy & Security → Full Disk Access), then:[10]

```bash
# Collect browsing history (Safari + Chrome)
python3 collect_history.py --browser=both --out enriched.jsonl

# Validate enrichment quality
python3 pipeline/validate_tiny_pipeline.py --enriched enriched.jsonl
```

### 3. Run Full Pipeline

```bash
# Distill → Fine‑tune → Package (takes ~30–60 min depending on dataset size)
python3 pipeline/pipeline_tiny_personalization.py \
  --enriched enriched.jsonl \
  --output-dir ./my_model
```

### 4. Deploy to Raspberry Pi

```bash
# Copy package to Pi
scp -r ./my_model/pi_package/* pi@<your-pi-ip>:~/models/

# On Pi: run setup and start model
ssh pi@<your-pi-ip> "cd ~/models && ./setup_pi.sh"
ssh pi@<your-pi-ip> "cd ~/llama.cpp && ./main -m ~/models/pi_model.gguf -n 256 --temp 0.1"
```

***

## Detailed Setup

### macOS Data Collection

1. **Grant Full Disk Access**: Open System Settings → Privacy & Security → Full Disk Access, add Terminal (or your IDE).[11][10]
2. **Run Collectors**: The pipeline reads Safari/Chrome history from `~/Library/` and knowledgeC.db from `/private/var/db/CoreDuet/Knowledge/`.[2][1]

### Teacher Model Setup

- **Local**: If you have a powerful GPU, run Qwen3‑Coder‑32B locally via Hugging Face Transformers.
- **API**: Set `api_base` and `api_key` in `configs/teacher_config.json` for remote endpoints.

### Training Hardware

- **Minimum**: 16 GB RAM, modern GPU (RTX 3060+ recommended).
- **Tiny Model**: Qwen2.5‑Coder‑3B fits in ~6 GB VRAM; LoRA reduces memory further.

### Raspberry Pi Requirements

- **Model**: Pi 4 or 5 with 4 GB RAM minimum.
- **OS**: Raspberry Pi OS (64‑bit).
- **Storage**: 4 GB free for model + llama.cpp.

***

## Usage Examples

### Individual Pipeline Steps

```bash
# 1. Distill teacher knowledge
make distill

# 2. Fine‑tune tiny model
make train

# 3. Package for Pi
make package

# 4. Full pipeline
make pipeline

# 5. Run tests
make test
```

### Configuration

Edit `configs/tiny_model_spec.json` to customize:

```json
{
  "base_model_name": "Qwen/Qwen2.5-Coder-3B",
  "max_context_length": 2048,
  "target_quantization": "Q4_K_M",
  "lora_config": {
    "r": 16,
    "alpha": 32,
    "dropout": 0.05
  }
}
```

### Pi Inference

```bash
# Direct inference
cd ~/llama.cpp
./main -m ~/models/pi_model.gguf -n 256 --temp 0.1

# Server mode for API access
./server -m ~/models/pi_model.gguf --host 0.0.0.0 --port 8080
```

***

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Data Collection Layer                         │
│  Safari/Chrome History → Enrichment (YouTube, knowledgeC.db) → JSONL│
└───────────────────────────────┬─────────────────────────────────────┘
                                │
┌───────────────────────────────▼─────────────────────────────────────┐
│                      Teacher Distillation Layer                      │
│  Large Teacher Model (Qwen3‑Coder‑32B) → Student SFT Dataset        │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
┌───────────────────────────────▼─────────────────────────────────────┐
│                    Tiny Model Fine‑tuning Layer                      │
│  Qwen2.5‑Coder‑3B + LoRA → Fine‑tuned Model (HF format)             │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
┌───────────────────────────────▼─────────────────────────────────────┐
│                      Packaging & Deployment Layer                    │
│  HF → GGUF (llama.cpp) → Pi Package (model + config + setup)        │
└─────────────────────────────────────────────────────────────────────┘
```

Each layer is **independent** and **testable**—no cross‑file modifications allowed.

***

## Privacy & Security

- **Local‑First**: All browsing data stays on your Mac; teacher inference can run locally or with explicit opt‑in.[10]
- **No Telemetry**: The pipeline does not send any data to external servers.
- **User Control**: You control which data is collected, which teacher model is used, and when to delete artifacts.

***

## Troubleshooting

### Permission Denied on macOS

```bash
# Add Terminal to Full Disk Access
# System Settings → Privacy & Security → Full Disk Access → + → Terminal
```

### Out‑of‑Memory During Training

- Reduce `batch_size` in `src/train_tiny_lora.py`
- Use a smaller base model (e.g., Qwen2.5‑Coder‑1.5B)
- Enable gradient checkpointing in the training script

### Model Won’t Load on Pi

```bash
# Check RAM
free -h

# Verify GGUF file
cd ~/llama.cpp && ./main -m ~/models/pi_model.gguf --check-tensors

# Re‑package with smaller quantization
# Edit configs/tiny_model_spec.json → "target_quantization": "Q3_K_M"
```

### Validation Failures

```bash
# Run comprehensive validation
python3 pipeline/validate_tiny_pipeline.py --enriched enriched.jsonl --verbose
```

***

## Contributing

We welcome contributions! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

**Guidelines**:
- Keep scripts modular and self‑contained
- Add tests for new functionality
- Update documentation
- Follow the existing code style

***

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

***

## Acknowledgments

- **Qwen Team** for excellent open‑source coding models[4][3]
- **llama.cpp** community for enabling LLM inference on edge devices[8][7]
- **Hugging Face** for transformers and PEFT libraries[6][5]

***

**Ready to build your own tiny AI?** Start with `make pipeline` and watch your browsing habits become a personalized model that fits in your pocket!

