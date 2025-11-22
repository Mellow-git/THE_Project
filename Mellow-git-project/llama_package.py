#!/usr/bin/env python3
"""
Package fine-tuned LLaMA models for distribution.
Creates model packages with metadata and usage instructions.
"""

import argparse
import json
import shutil
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict


def create_model_package(
    adapter_path: Path,
    base_model_name: str,
    model_size: str,
    output_dir: Path,
    metadata: Dict = None,
) -> Path:
    """
    Create a distributable model package.
    
    Args:
        adapter_path: Path to LoRA adapter directory
        base_model_name: Base model identifier
        model_size: Model size (7b, 13b, 70b)
        output_dir: Output directory for package
        metadata: Additional metadata dict
    
    Returns:
        Path to created package directory
    """
    if not adapter_path.exists():
        raise FileNotFoundError(f"Adapter path not found: {adapter_path}")
    
    # Create package directory
    package_name = f"llama-{model_size}-sft-{datetime.now().strftime('%Y%m%d')}"
    package_dir = output_dir / package_name
    package_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"[PACKAGE] Creating package: {package_name}")
    
    # Copy adapter files
    adapter_dest = package_dir / "adapter"
    shutil.copytree(adapter_path, adapter_dest, dirs_exist_ok=True)
    print(f"[PACKAGE] Copied adapter to {adapter_dest}")
    
    # Create metadata
    package_metadata = {
        'base_model': base_model_name,
        'model_size': model_size,
        'adapter_type': 'LoRA',
        'created_at': datetime.now().isoformat(),
        'package_version': '1.0',
    }
    
    if metadata:
        package_metadata.update(metadata)
    
    metadata_file = package_dir / "metadata.json"
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(package_metadata, f, indent=2, ensure_ascii=False)
    print(f"[PACKAGE] Created metadata: {metadata_file}")
    
    # Create README
    readme_content = f"""# LLaMA {model_size.upper()} Fine-Tuned Model

## Model Information

- **Base Model**: {base_model_name}
- **Model Size**: {model_size.upper()}
- **Adapter Type**: LoRA (Low-Rank Adaptation)
- **Created**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Usage

### Loading the Model

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    "{base_model_name}",
    device_map="auto",
    torch_dtype=torch.float16,
)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("{base_model_name}")

# Load LoRA adapter
model = PeftModel.from_pretrained(base_model, "./adapter")
model.eval()
```

### Inference Example

```python
def generate_response(model, tokenizer, instruction, context):
    prompt = f"[INST] {{instruction}}\\n\\nContext: {{context}} [/INST]"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.7,
            top_p=0.9,
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response.split("[/INST]")[-1].strip()
```

## Hardware Requirements

- **Minimum GPU Memory**: {get_memory_requirement(model_size)} GB
- **Recommended**: Use 4-bit quantization for inference

## Files

- `adapter/`: LoRA adapter weights
- `metadata.json`: Model metadata
- `README.md`: This file

## License

Please refer to the base model's license ({base_model_name}).
"""
    
    readme_file = package_dir / "README.md"
    with open(readme_file, 'w', encoding='utf-8') as f:
        f.write(readme_content)
    print(f"[PACKAGE] Created README: {readme_file}")
    
    # Create requirements.txt
    requirements = """torch>=2.0.0
transformers>=4.30.0
peft>=0.4.0
accelerate>=0.20.0
bitsandbytes>=0.39.0
"""
    
    requirements_file = package_dir / "requirements.txt"
    with open(requirements_file, 'w') as f:
        f.write(requirements)
    print(f"[PACKAGE] Created requirements: {requirements_file}")
    
    # Create archive
    archive_path = output_dir / f"{package_name}.tar.gz"
    shutil.make_archive(
        str(archive_path).replace('.tar.gz', ''),
        'gztar',
        str(package_dir),
    )
    print(f"[PACKAGE] Created archive: {archive_path}")
    
    return package_dir


def get_memory_requirement(model_size: str) -> int:
    """Get memory requirement for model size."""
    sys.path.insert(0, str(Path(__file__).parent))
    from llama_config import get_model_config
    config = get_model_config(model_size)
    return config['memory_gb']['int4']


def main():
    parser = argparse.ArgumentParser(
        description='Package fine-tuned LLaMA model for distribution',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--adapter-path', required=True, type=Path,
                       help='Path to LoRA adapter directory')
    parser.add_argument('--base-model', required=True, type=str,
                       help='Base model name')
    parser.add_argument('--model-size', required=True, type=str,
                       choices=['7b', '13b', '70b'],
                       help='Model size')
    parser.add_argument('--output-dir', required=True, type=Path,
                       help='Output directory for package')
    parser.add_argument('--metadata', type=Path,
                       help='Additional metadata JSON file')
    
    args = parser.parse_args()
    
    # Load additional metadata if provided
    metadata = {}
    if args.metadata and args.metadata.exists():
        with open(args.metadata, 'r') as f:
            metadata = json.load(f)
    
    # Create package
    package_dir = create_model_package(
        args.adapter_path,
        args.base_model,
        args.model_size,
        args.output_dir,
        metadata=metadata,
    )
    
    print(f"\n[SUCCESS] Model package created: {package_dir}")
    print(f"[INFO] Package includes:")
    print(f"  - LoRA adapter weights")
    print(f"  - Metadata and documentation")
    print(f"  - Usage instructions")
    print(f"  - Compressed archive")


if __name__ == '__main__':
    main()

