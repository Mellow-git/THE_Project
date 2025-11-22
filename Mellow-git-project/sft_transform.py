#!/usr/bin/env python3
"""
Transform enriched events into canonical SFT format.
Normalizes data across browsers and domains for schema consistency.
"""

import argparse
import json
import sys
import hashlib
import random
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from datetime import datetime
from urllib.parse import urlparse


def normalize_domain(url: str) -> str:
    """Extract and normalize domain from URL."""
    try:
        parsed = urlparse(url)
        domain = parsed.netloc.lower()
        # Remove www. prefix
        if domain.startswith('www.'):
            domain = domain[4:]
        return domain
    except Exception:
        return 'unknown'


def generate_source_id(event: Dict[str, Any]) -> str:
    """Generate deterministic source ID from event."""
    url = event.get('url', '')
    browser = event.get('browser', '')
    profile = event.get('profile', '')
    timestamp = event.get('visited_at_iso', '')
    
    # Create hash from key fields
    key = f"{browser}:{profile}:{url}:{timestamp}"
    hash_obj = hashlib.sha256(key.encode('utf-8'))
    return hash_obj.hexdigest()[:16]


def generate_chunk_id(source_id: str, chunk_idx: int) -> str:
    """Generate chunk ID from source ID and index."""
    return f"{source_id}#c{chunk_idx:03d}"


def extract_topic(event: Dict[str, Any]) -> str:
    """Extract topic from event labels or infer from content."""
    labels = event.get('labels', {})
    reason = labels.get('reason', 'general')
    
    # Map reason to topic
    topic_map = {
        'shorts/entertainment': 'entertainment',
        'tutorial/learning': 'education',
        'news/information': 'news',
        'music/ambient': 'music',
        'deep-dive/research': 'research',
        'night_browsing': 'general',
        'general': 'general',
    }
    
    return topic_map.get(reason, 'general')


def build_instruction(event: Dict[str, Any], content_meta: Dict[str, Any]) -> str:
    """Build instruction based on event characteristics."""
    reason = event.get('labels', {}).get('reason', 'general')
    is_shorts = content_meta.get('is_shorts', False)
    duration = content_meta.get('duration')
    
    if is_shorts or (duration and duration < 60):
        return 'Summarize this short video in one sentence.'
    
    if reason == 'tutorial/learning':
        return 'Explain step by step how to follow this tutorial or learning content.'
    
    if reason == 'music/ambient':
        return 'Describe the style, mood, and characteristics of this music or audio content.'
    
    if reason == 'news/information':
        return 'Summarize the key facts, events, or information presented.'
    
    if reason == 'deep-dive/research':
        return 'Write a detailed summary highlighting new knowledge, findings, or insights.'
    
    return 'Summarize the main idea and key points of this content.'


def build_context(event: Dict[str, Any], content_meta: Dict[str, Any]) -> str:
    """Build context string from event and metadata."""
    parts = []
    
    # Prefer transcript if available
    transcript = event.get('transcript_excerpt')
    if transcript:
        return transcript
    
    # Build from content metadata
    title = content_meta.get('title') or event.get('title', '')
    if title:
        parts.append(f"Title: {title}")
    
    channel = content_meta.get('channel')
    if channel:
        parts.append(f"Channel: {channel}")
    
    publish_date = content_meta.get('publishDate')
    if publish_date:
        parts.append(f"Published: {publish_date}")
    
    duration = content_meta.get('duration')
    if duration:
        minutes = duration // 60
        seconds = duration % 60
        parts.append(f"Duration: {minutes}m {seconds}s")
    
    views = content_meta.get('views')
    if views:
        parts.append(f"Views: {views:,}")
    
    keywords = content_meta.get('keywords', [])
    if keywords:
        parts.append(f"Keywords: {', '.join(keywords[:5])}")
    
    # Fallback to URL if no metadata
    if not parts:
        url = event.get('url', '')
        parts.append(f"URL: {url}")
    
    return ' | '.join(parts) if parts else 'No context available'


def normalize_timestamp(ts_str: Optional[str]) -> Optional[str]:
    """Normalize timestamp to ISO 8601 format."""
    if not ts_str:
        return None
    
    try:
        # Remove 'Z' and parse
        ts_clean = ts_str.replace('Z', '+00:00')
        dt = datetime.fromisoformat(ts_clean)
        # Return in ISO format
        return dt.isoformat()
    except Exception:
        return None


def assign_split_hint(source_id: str, domain: str, seed: int = 42) -> str:
    """Deterministically assign train/val/test split."""
    random.seed(seed)
    # Use hash of source_id for deterministic assignment
    hash_val = int(hashlib.md5(source_id.encode()).hexdigest(), 16)
    
    # 80% train, 10% val, 10% test
    mod = hash_val % 100
    if mod < 80:
        return 'train'
    elif mod < 90:
        return 'val'
    else:
        return 'test'


def transform_event(event: Dict[str, Any], chunk_idx: int = 0) -> Optional[Dict[str, Any]]:
    """Transform enriched event to SFT format."""
    # Validate required fields
    url = event.get('url')
    if not url:
        return None
    
    # Generate IDs
    source_id = generate_source_id(event)
    chunk_id = generate_chunk_id(source_id, chunk_idx)
    
    # Extract metadata
    content_meta = event.get('content_meta') or {}
    domain = normalize_domain(url)
    topic = extract_topic(event)
    timestamp = normalize_timestamp(event.get('visited_at_iso'))
    
    # Build instruction and context
    instruction = build_instruction(event, content_meta)
    context = build_context(event, content_meta)
    
    # Validate context length
    if len(context) < 1:
        return None
    
    # Build meta object matching schema
    meta = {
        'url': url,
        'title': event.get('title') or content_meta.get('title'),
        'ts': timestamp,
        'domain': domain,
        'topic': topic,
        'source_id': source_id,
        'chunk_id': chunk_id,
        'generated': False,
        'length_chars': len(context),
        'split_hint': assign_split_hint(source_id, domain),
    }
    
    # Build SFT example
    sft_example = {
        'instruction': instruction,
        'context': context,
        'response': '',  # Empty response for now (to be filled by annotation)
        'meta': meta,
    }
    
    return sft_example


def validate_sft_example(example: Dict[str, Any], schema: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """Validate SFT example against schema."""
    # Check required fields
    required = schema.get('required', [])
    for field in required:
        if field not in example:
            return False, f"Missing required field: {field}"
    
    # Check meta fields
    meta = example.get('meta', {})
    meta_required = schema.get('properties', {}).get('meta', {}).get('required', [])
    for field in meta_required:
        if field not in meta:
            return False, f"Missing required meta field: {field}"
    
    # Check context length
    context = example.get('context', '')
    min_length = schema.get('properties', {}).get('context', {}).get('minLength', 1)
    if len(context) < min_length:
        return False, f"Context too short: {len(context)} < {min_length}"
    
    # Check split_hint enum
    split_hint = meta.get('split_hint')
    valid_splits = schema.get('properties', {}).get('meta', {}).get('properties', {}).get('split_hint', {}).get('enum', [])
    if split_hint and split_hint not in valid_splits:
        return False, f"Invalid split_hint: {split_hint}"
    
    return True, None


def main():
    parser = argparse.ArgumentParser(
        description='Transform enriched events to SFT format',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--input', required=True, type=Path,
                       help='Input enriched JSONL file')
    parser.add_argument('--output', required=True, type=Path,
                       help='Output SFT JSONL file')
    parser.add_argument('--schema', type=Path,
                       default=Path(__file__).parent / 'schemas' / 'dataset.schema.json',
                       help='JSON schema file for validation')
    parser.add_argument('--validate', action='store_true',
                       help='Validate output against schema')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for split assignment (default: 42)')
    parser.add_argument('--min-context-length', type=int, default=10,
                       help='Minimum context length in characters (default: 10)')
    
    args = parser.parse_args()
    
    if not args.input.exists():
        print(f"[ERROR] Input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)
    
    # Load schema if validation requested
    schema = None
    if args.validate:
        if not args.schema.exists():
            print(f"[WARN] Schema file not found: {args.schema}", file=sys.stderr)
            print("[WARN] Continuing without validation", file=sys.stderr)
        else:
            try:
                with open(args.schema, 'r', encoding='utf-8') as f:
                    schema = json.load(f)
                print(f"[INFO] Loaded schema from {args.schema}")
            except Exception as e:
                print(f"[WARN] Failed to load schema: {e}", file=sys.stderr)
                print("[WARN] Continuing without validation", file=sys.stderr)
    
    print(f"[INFO] Loading enriched events from {args.input}...")
    events = []
    
    try:
        with open(args.input, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                    events.append(event)
                except json.JSONDecodeError as e:
                    print(f"[WARN] Line {line_num}: Invalid JSON, skipping: {e}", file=sys.stderr)
                    continue
    except Exception as e:
        print(f"[ERROR] Failed to read input: {e}", file=sys.stderr)
        sys.exit(1)
    
    print(f"[INFO] Loaded {len(events)} events")
    
    # Transform events
    print("[INFO] Transforming events to SFT format...")
    sft_examples = []
    skipped = 0
    validation_errors = []
    
    for event in events:
        sft_example = transform_event(event, chunk_idx=0)
        if not sft_example:
            skipped += 1
            continue
        
        # Check minimum context length
        if len(sft_example['context']) < args.min_context_length:
            skipped += 1
            continue
        
        # Validate if requested
        if schema:
            is_valid, error = validate_sft_example(sft_example, schema)
            if not is_valid:
                validation_errors.append((event.get('url', 'unknown'), error))
                if len(validation_errors) <= 10:  # Only report first 10
                    print(f"[WARN] Validation error: {error}", file=sys.stderr)
                continue
        
        sft_examples.append(sft_example)
    
    print(f"[INFO] Transformed {len(sft_examples)} events (skipped {skipped})")
    
    if validation_errors:
        print(f"[WARN] Found {len(validation_errors)} validation errors", file=sys.stderr)
    
    # Compute split distribution
    splits = {'train': 0, 'val': 0, 'test': 0}
    for example in sft_examples:
        split = example['meta'].get('split_hint', 'train')
        splits[split] = splits.get(split, 0) + 1
    
    print("\n[SPLIT DISTRIBUTION]")
    for split, count in splits.items():
        print(f"  {split}: {count} ({count/len(sft_examples)*100:.1f}%)")
    
    # Write output
    print(f"\n[INFO] Writing SFT examples to {args.output}...")
    try:
        with open(args.output, 'w', encoding='utf-8') as f:
            for example in sft_examples:
                json.dump(example, f, ensure_ascii=False)
                f.write('\n')
        print(f"[SUCCESS] Wrote {len(sft_examples)} SFT examples to {args.output}")
    except Exception as e:
        print(f"[ERROR] Failed to write output: {e}", file=sys.stderr)
        sys.exit(1)
    
    sys.exit(0)


if __name__ == '__main__':
    main()

