#!/usr/bin/env python3
"""
Prepare Supervised Fine-Tuning Dataset

Transform enriched events into canonical SFT JSONL format with:
- instruction, context, response, meta fields
- Schema validation
- Deterministic splits (train/val/test)
- Normalized data across browsers and domains
"""

import argparse
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
from urllib.parse import urlparse


RANDOM_SEED = 42


def get_domain(url: str) -> str:
    """Extract eTLD+1 domain from URL."""
    try:
        parsed = urlparse(url)
        netloc = parsed.netloc.lower()
        # Remove port if present
        if ':' in netloc:
            netloc = netloc.split(':')[0]
        # Simple domain extraction (for full PSL support, use tldextract)
        parts = netloc.split('.')
        if len(parts) >= 2:
            return '.'.join(parts[-2:])
        return netloc
    except Exception:
        return "unknown"


def build_instruction(meta: Dict[str, Any], content_meta: Dict[str, Any]) -> str:
    """Build instruction based on content type and metadata."""
    reason = meta.get('reason', 'general')
    is_shorts = content_meta.get('is_shorts', False)
    duration = content_meta.get('duration')
    
    if is_shorts or (duration and duration < 60):
        return 'Summarize in one sentence.'
    
    if reason == 'tutorial/learning':
        return 'Explain step by step how to follow this tutorial.'
    
    if reason == 'music/ambient':
        return 'Describe the style and mood of this music/audio.'
    
    if reason == 'news/information':
        return 'Summarize the key facts or event.'
    
    if reason == 'deep-dive/research':
        return 'Write a detailed summary highlighting new knowledge or findings.'
    
    return 'Summarize the main idea of this content.'


def build_context(event: Dict[str, Any]) -> str:
    """Build context string from event data."""
    parts = []
    
    # Prefer transcript if present
    transcript = event.get('transcript_excerpt')
    if transcript:
        return transcript
    
    # Build from content_meta
    content_meta = event.get('content_meta') or {}
    
    title = content_meta.get('title') or event.get('title', '')
    if title:
        parts.append(title)
    
    channel = content_meta.get('channel', '')
    if channel:
        parts.append(f"Channel: {channel}")
    
    publish_date = content_meta.get('publishDate', '')
    if publish_date:
        parts.append(f"Published: {publish_date}")
    
    keywords = content_meta.get('keywords', [])
    if keywords:
        parts.append(f"Keywords: {', '.join(keywords[:10])}")  # Limit keywords
    
    duration = content_meta.get('duration')
    if duration:
        parts.append(f"Duration: {duration}s")
    
    views = content_meta.get('views')
    if views:
        parts.append(f"Views: {views:,}")
    
    # Fallback to URL if no content
    if not parts:
        url = event.get('url', '')
        if url:
            parts.append(f"URL: {url}")
    
    return ' | '.join(parts) if parts else "No context available"


def generate_source_id(event: Dict[str, Any]) -> str:
    """Generate deterministic source ID from event."""
    url = event.get('url', '')
    timestamp = event.get('visited_at_iso', '')
    browser = event.get('browser', '')
    profile = event.get('profile', '')
    
    # Create hash from key fields
    key = f"{url}|{timestamp}|{browser}|{profile}"
    hash_obj = hashlib.sha256(key.encode('utf-8'))
    return hash_obj.hexdigest()[:16]


def generate_chunk_id(source_id: str, chunk_idx: int) -> str:
    """Generate chunk ID from source ID and index."""
    return f"{source_id}#c{chunk_idx:03d}"


def assign_split(domain: str, seed: int = RANDOM_SEED) -> str:
    """Deterministically assign split based on domain."""
    import random
    random.seed(seed)
    
    # Hash domain for deterministic assignment
    domain_hash = int(hashlib.md5(domain.encode()).hexdigest(), 16)
    random.seed(domain_hash)
    rand_val = random.random()
    
    if rand_val < 0.8:
        return 'train'
    elif rand_val < 0.9:
        return 'val'
    else:
        return 'test'


def transform_event_to_sft(event: Dict[str, Any], chunk_idx: int = 0) -> Optional[Dict[str, Any]]:
    """Transform enriched event to SFT format."""
    # Validate required fields
    url = event.get('url')
    if not url:
        return None
    
    visited_at_iso = event.get('visited_at_iso', '')
    domain = get_domain(url)
    title = event.get('title') or ''
    
    # Build instruction and context
    content_meta = event.get('content_meta') or {}
    labels = event.get('labels', {})
    reason = labels.get('reason', 'general')
    topics = labels.get('topics', [])
    
    meta_for_instruction = {
        'reason': reason,
        **content_meta
    }
    
    instruction = build_instruction(meta_for_instruction, content_meta)
    context = build_context(event)
    
    # Validate context is not empty
    if not context or len(context.strip()) < 1:
        return None
    
    # Generate IDs
    source_id = generate_source_id(event)
    chunk_id = generate_chunk_id(source_id, chunk_idx)
    
    # Build meta field matching schema
    sft_meta = {
        'url': url,
        'title': title if title else None,
        'ts': visited_at_iso if visited_at_iso else None,
        'domain': domain,
        'topic': reason,  # Use reason as topic
        'source_id': source_id,
        'chunk_id': chunk_id,
        'generated': False,
        'length_chars': len(context),
        'split_hint': assign_split(domain),
    }
    
    # Build SFT record
    sft_record = {
        'instruction': instruction,
        'context': context,
        'response': '',  # Empty response for now (to be filled by annotation)
        'meta': sft_meta,
    }
    
    return sft_record


def main():
    parser = argparse.ArgumentParser(
        description='Prepare supervised fine-tuning dataset from enriched events',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--input', required=True, type=Path,
                       help='Input enriched_event.jsonl file')
    parser.add_argument('--output', required=True, type=Path,
                       help='Output SFT dataset JSONL file')
    parser.add_argument('--output-dir', type=Path,
                       help='Output directory for split files (train/val/test)')
    parser.add_argument('--schema', type=Path,
                       default=Path(__file__).parent / 'schemas' / 'dataset.schema.json',
                       help='JSON schema file for validation')
    parser.add_argument('--seed', type=int, default=RANDOM_SEED,
                       help='Random seed for deterministic splits (default: 42)')
    parser.add_argument('--min-context-length', type=int, default=10,
                       help='Minimum context length in characters (default: 10)')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be done without writing output')
    
    args = parser.parse_args()
    
    if not args.input.exists():
        print(f"[ERROR] Input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)
    
    print("[INFO] Preparing SFT dataset...")
    print(f"[INFO] Input: {args.input}")
    print(f"[INFO] Seed: {args.seed}")
    
    # Load events
    events = []
    parse_errors = 0
    
    try:
        with open(args.input, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                    events.append(event)
                except json.JSONDecodeError as e:
                    parse_errors += 1
                    if parse_errors <= 5:
                        print(f"[WARN] Line {idx}: Invalid JSON: {e}", file=sys.stderr)
                    continue
    except Exception as e:
        print(f"[ERROR] Failed to read input: {e}", file=sys.stderr)
        sys.exit(1)
    
    total_events = len(events)
    print(f"[INFO] Loaded {total_events} events (parse errors: {parse_errors})")
    
    if total_events == 0:
        print("[ERROR] No valid events found", file=sys.stderr)
        sys.exit(1)
    
    # Transform events
    print("[INFO] Transforming events to SFT format...")
    sft_records = []
    dropped = 0
    
    for event in events:
        sft_record = transform_event_to_sft(event)
        if sft_record:
            # Check minimum context length
            if len(sft_record['context']) >= args.min_context_length:
                sft_records.append(sft_record)
            else:
                dropped += 1
        else:
            dropped += 1
    
    print(f"[INFO] Transformed {len(sft_records)} events (dropped: {dropped})")
    
    # Split by split_hint
    splits = {'train': [], 'val': [], 'test': []}
    for record in sft_records:
        split = record['meta']['split_hint']
        if split in splits:
            splits[split].append(record)
    
    print("\n[INFO] Split distribution:")
    for split_name, split_records in splits.items():
        print(f"  {split_name}: {len(split_records)} ({len(split_records)/len(sft_records)*100:.1f}%)")
    
    # Label distribution
    label_counter = Counter()
    for record in sft_records:
        topic = record['meta'].get('topic', 'unknown')
        label_counter[topic] += 1
    
    print("\n[INFO] Topic distribution:")
    for topic, count in label_counter.most_common():
        pct = (count / len(sft_records) * 100) if sft_records else 0
        print(f"  {topic}: {count} ({pct:.1f}%)")
    
    if args.dry_run:
        print("\n[DRY RUN] Would write output files:")
        print(f"  Main: {args.output}")
        if args.output_dir:
            for split_name in splits:
                print(f"  {split_name}: {args.output_dir}/dataset_{split_name}.jsonl")
        sys.exit(0)
    
    # Write main output
    print(f"\n[INFO] Writing SFT dataset to {args.output}...")
    try:
        with open(args.output, 'w', encoding='utf-8') as f:
            for record in sft_records:
                json.dump(record, f, ensure_ascii=False)
                f.write('\n')
        print(f"[INFO] Wrote {len(sft_records)} records to {args.output}")
    except Exception as e:
        print(f"[ERROR] Failed to write output: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Write split files if output_dir specified
    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n[INFO] Writing split files to {args.output_dir}...")
        for split_name, split_records in splits.items():
            split_path = args.output_dir / f"dataset_{split_name}.jsonl"
            try:
                with open(split_path, 'w', encoding='utf-8') as f:
                    for record in split_records:
                        json.dump(record, f, ensure_ascii=False)
                        f.write('\n')
                print(f"[INFO] Wrote {len(split_records)} records to {split_path}")
            except Exception as e:
                print(f"[ERROR] Failed to write {split_path}: {e}", file=sys.stderr)
    
    print("\n[SUCCESS] SFT dataset preparation complete!")
    sys.exit(0)


if __name__ == '__main__':
    main()

