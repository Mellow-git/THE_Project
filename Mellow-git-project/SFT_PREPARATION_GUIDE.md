# Supervised Fine-Tuning Data Preparation Guide

This guide documents the complete pipeline for preparing enriched browser history data for supervised fine-tuning.

## Overview

The SFT preparation pipeline consists of several modular components:

1. **Enrichment Modules** (already created):
   - `enrich_youtube.py` - YouTube metadata extraction from Chrome cache
   - `join_knowledgec.py` - Behavioral session data joining

2. **Validation & Quality Assurance**:
   - `validate_enriched_dataset.py` - Comprehensive dataset validation
   - `extract_golden_set.py` - Curated sample extraction for manual review

3. **Transformation**:
   - `sft_transform.py` - Convert enriched events to SFT format

4. **Training Preparation**:
   - `prepare_training.py` - Generate training configuration and statistics

5. **Pipeline Orchestration**:
   - `pipeline_sft_prep.py` - Master script to run full pipeline

## Step-by-Step Workflow

### Step 1: Create Enriched Dataset

First, enrich your raw browser history with YouTube metadata and session data:

```bash
# Enrich with YouTube metadata
python3 enrich_youtube.py \
  --input safari.jsonl \
  --output enriched_safari.jsonl \
  --consent-captions

# Join with knowledgeC.db behavioral data
python3 join_knowledgec.py \
  --input enriched_safari.jsonl \
  --output fully_enriched.jsonl \
  --slop-seconds 600
```

### Step 2: Validate Enriched Dataset

Validate the enriched dataset for completeness and quality:

```bash
python3 validate_enriched_dataset.py \
  --input fully_enriched.jsonl \
  --output-report validation_report.json \
  --verbose \
  --spot-check 20
```

This will:
- Check schema completeness
- Validate YouTube metadata coherence
- Verify session join consistency
- Check label coherence
- Compute coverage metrics
- Generate detailed validation report

### Step 3: Extract Golden Set

Extract a curated sample (100-200 events) for manual annotation:

```bash
python3 extract_golden_set.py \
  --input fully_enriched.jsonl \
  --output golden_set.jsonl \
  --size 150 \
  --seed 42 \
  --export-csv golden_set.csv
```

The golden set is stratified by:
- Label reasons
- Enrichment status (enriched vs not)
- YouTube vs non-YouTube URLs
- Session joins vs no session

### Step 4: Transform to SFT Format

Convert enriched events to canonical SFT format:

```bash
python3 sft_transform.py \
  --input fully_enriched.jsonl \
  --output sft_dataset.jsonl \
  --schema schemas/dataset.schema.json \
  --validate \
  --seed 42 \
  --min-context-length 10
```

This creates:
- Instruction/context/response structure
- Deterministic train/val/test splits
- Normalized metadata
- Schema-validated examples

### Step 5: Prepare Training Configuration

Generate training configuration based on dataset statistics:

```bash
python3 prepare_training.py \
  --sft-dataset sft_dataset.jsonl \
  --output-config training_config.json \
  --output-stats dataset_statistics.json
```

This provides:
- Recommended batch sizes
- Gradient accumulation steps
- Learning rate suggestions
- Sequence length recommendations
- Dataset statistics

### Step 6: Run Full Pipeline (Automated)

Or run the complete pipeline in one command:

```bash
python3 pipeline_sft_prep.py \
  --enriched fully_enriched.jsonl \
  --output-dir ./sft_output \
  --golden-size 150 \
  --seed 42
```

## Output Files

After running the pipeline, you'll have:

```
sft_output/
├── validation_report.json      # Validation results and metrics
├── golden_set.jsonl            # Curated sample for annotation
├── golden_set.csv              # CSV format for easy annotation
├── sft_dataset.jsonl           # Training-ready SFT format
├── training_config.json        # Training configuration
└── dataset_statistics.json     # Dataset analysis
```

## Quality Metrics

The validation script reports:

- **Cache Hit Rate**: Percentage of YouTube URLs with cached metadata
- **Session Join Rate**: Percentage of events matched to knowledgeC sessions
- **YouTube Enrichment Rate**: YouTube URLs successfully enriched
- **Timestamp Validity**: Percentage of valid timestamps
- **Label Coverage**: Percentage of events with labels

## Schema Validation

The SFT format follows the schema defined in `schemas/dataset.schema.json`:

```json
{
  "instruction": "string",
  "context": "string (minLength: 1)",
  "response": "string",
  "meta": {
    "url": "uri",
    "title": "string|null",
    "ts": "date-time|null",
    "domain": "string",
    "topic": "string",
    "source_id": "string",
    "chunk_id": "string",
    "generated": "boolean",
    "length_chars": "integer",
    "split_hint": "train|val|test"
  }
}
```

## Training Recommendations

Based on dataset analysis, the preparation script recommends:

- **Batch Size**: Based on average context length
- **Gradient Accumulation**: To maintain effective batch size
- **Max Sequence Length**: 1.5x average context length
- **Learning Rate**: 2e-5 (standard for fine-tuning)
- **Epochs**: 3 (typical for instruction tuning)

## Reproducibility

All scripts use deterministic seeds (default: 42) for:
- Random sampling
- Split assignment
- Stratified golden set extraction

This ensures reproducible results across runs.

## Error Handling

All scripts:
- Handle missing files gracefully
- Provide clear error messages
- Use proper exit codes (0 = success, 1 = failure)
- Log warnings without stopping execution where appropriate

## Next Steps

After data preparation:

1. **Manual Annotation**: Review and annotate `golden_set.csv`
2. **Model Selection**: Choose base model architecture
3. **Training Setup**: Configure training environment
4. **Pilot Training**: Run initial training on golden set
5. **Evaluation**: Assess model outputs and iterate

## Troubleshooting

### Low Cache Hit Rate
- Ensure Chrome cache directories are accessible
- Check Full Disk Access permissions
- Verify YouTube URLs were visited in Chrome

### Low Session Join Rate
- Verify knowledgeC.db files exist and are accessible
- Check Full Disk Access permissions
- Adjust `--slop-seconds` parameter

### Schema Validation Errors
- Check that enriched dataset has all required fields
- Verify timestamp formats are ISO 8601
- Ensure labels structure is correct

## Modularity

All components are fully modular:
- No modifications to existing collector/snapshot code
- Each script can run independently
- Clear input/output contracts
- No cross-file dependencies

