#!/usr/bin/env python3
"""
Extract curated golden set for manual annotation review.
Samples enriched events with stratification by labels and enrichment status.
"""

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Any


def stratify_sample(events: List[Dict[str, Any]], target_size: int = 150) -> List[Dict[str, Any]]:
    """
    Stratified sampling to ensure representation across:
    - Label reasons
    - Enrichment status (enriched vs not)
    - YouTube vs non-YouTube
    - Session joins vs no session
    """
    # Group events by characteristics
    by_reason = defaultdict(list)
    enriched_youtube = []
    non_enriched_youtube = []
    with_session = []
    without_session = []
    
    for event in events:
        reason = event.get('labels', {}).get('reason', 'NONE')
        by_reason[reason].append(event)
        
        url = event.get('url', '')
        is_youtube = 'youtube.com' in url or 'youtu.be' in url
        has_meta = bool(event.get('content_meta'))
        has_session = bool(event.get('session_id'))
        
        if is_youtube:
            if has_meta:
                enriched_youtube.append(event)
            else:
                non_enriched_youtube.append(event)
        
        if has_session:
            with_session.append(event)
        else:
            without_session.append(event)
    
    # Calculate sample sizes per stratum
    total = len(events)
    samples = []
    
    # Sample by reason (proportional but with minimums)
    reason_counts = Counter(e.get('labels', {}).get('reason', 'NONE') for e in events)
    per_reason_target = max(5, target_size // len(by_reason)) if by_reason else 0
    
    for reason, event_list in by_reason.items():
        sample_size = min(per_reason_target, len(event_list))
        if sample_size > 0:
            samples.extend(random.sample(event_list, sample_size))
    
    # Ensure YouTube representation
    youtube_target = min(30, target_size // 5)
    if enriched_youtube and len(samples) < target_size:
        needed = min(youtube_target, len(enriched_youtube), target_size - len(samples))
        new_samples = random.sample(enriched_youtube, needed)
        # Avoid duplicates
        existing_urls = {e.get('url') for e in samples}
        samples.extend(e for e in new_samples if e.get('url') not in existing_urls)
    
    if non_enriched_youtube and len(samples) < target_size:
        needed = min(10, len(non_enriched_youtube), target_size - len(samples))
        new_samples = random.sample(non_enriched_youtube, needed)
        existing_urls = {e.get('url') for e in samples}
        samples.extend(e for e in new_samples if e.get('url') not in existing_urls)
    
    # Ensure session representation
    if with_session and len(samples) < target_size:
        needed = min(20, len(with_session), target_size - len(samples))
        new_samples = random.sample(with_session, needed)
        existing_urls = {e.get('url') for e in samples}
        samples.extend(e for e in new_samples if e.get('url') not in existing_urls)
    
    # Fill remaining with random sample
    if len(samples) < target_size:
        remaining = [e for e in events if e.get('url') not in {s.get('url') for s in samples}]
        needed = min(target_size - len(samples), len(remaining))
        if needed > 0:
            samples.extend(random.sample(remaining, needed))
    
    # Shuffle to avoid ordering bias
    random.shuffle(samples)
    
    return samples[:target_size]


def compute_golden_metrics(events: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute metrics for golden set."""
    total = len(events)
    if total == 0:
        return {}
    
    enriched = sum(1 for e in events if e.get('content_meta'))
    youtube = sum(1 for e in events if 'youtube.com' in e.get('url', '') or 'youtu.be' in e.get('url', ''))
    youtube_enriched = sum(1 for e in events if ('youtube.com' in e.get('url', '') or 'youtu.be' in e.get('url', '')) and e.get('content_meta'))
    with_session = sum(1 for e in events if e.get('session_id'))
    with_labels = sum(1 for e in events if e.get('labels', {}).get('reason'))
    
    reasons = Counter(e.get('labels', {}).get('reason', 'NONE') for e in events)
    
    return {
        'total': total,
        'enriched': enriched,
        'enrichment_rate': enriched / total if total > 0 else 0,
        'youtube_urls': youtube,
        'youtube_enriched': youtube_enriched,
        'youtube_enrichment_rate': youtube_enriched / youtube if youtube > 0 else 0,
        'with_session': with_session,
        'session_join_rate': with_session / total if total > 0 else 0,
        'with_labels': with_labels,
        'label_coverage': with_labels / total if total > 0 else 0,
        'label_distribution': dict(reasons),
    }


def main():
    parser = argparse.ArgumentParser(
        description='Extract curated golden set for manual annotation',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--input', required=True, type=Path,
                       help='Input enriched JSONL file')
    parser.add_argument('--output', required=True, type=Path,
                       help='Output golden set JSONL file')
    parser.add_argument('--size', type=int, default=150,
                       help='Target size for golden set (default: 150)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--export-csv', type=Path,
                       help='Also export as CSV for annotation')
    
    args = parser.parse_args()
    
    if not args.input.exists():
        print(f"[ERROR] Input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)
    
    # Set random seed for reproducibility
    random.seed(args.seed)
    
    print(f"[INFO] Loading enriched dataset from {args.input}...")
    events = []
    
    try:
        with open(args.input, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                    events.append(event)
                except json.JSONDecodeError as e:
                    print(f"[WARN] Skipping invalid JSON line: {e}", file=sys.stderr)
                    continue
    except Exception as e:
        print(f"[ERROR] Failed to read input: {e}", file=sys.stderr)
        sys.exit(1)
    
    print(f"[INFO] Loaded {len(events)} events")
    
    if len(events) == 0:
        print("[ERROR] No events found in input file", file=sys.stderr)
        sys.exit(1)
    
    # Extract stratified sample
    print(f"[INFO] Extracting stratified sample of {args.size} events...")
    golden_set = stratify_sample(events, args.size)
    
    print(f"[INFO] Selected {len(golden_set)} events for golden set")
    
    # Compute metrics
    metrics = compute_golden_metrics(golden_set)
    print("\n[GOLDEN SET METRICS]")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2%}")
        elif isinstance(value, dict):
            print(f"  {key}:")
            for k, v in value.items():
                print(f"    {k}: {v}")
        else:
            print(f"  {key}: {value}")
    
    # Write golden set
    print(f"\n[INFO] Writing golden set to {args.output}...")
    try:
        with open(args.output, 'w', encoding='utf-8') as f:
            for event in golden_set:
                json.dump(event, f, ensure_ascii=False)
                f.write('\n')
        print(f"[SUCCESS] Wrote {len(golden_set)} events to {args.output}")
    except Exception as e:
        print(f"[ERROR] Failed to write output: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Export CSV if requested
    if args.export_csv:
        print(f"[INFO] Exporting CSV to {args.export_csv}...")
        try:
            import csv
            
            # Flatten nested structures for CSV
            fieldnames = [
                'url', 'title', 'browser', 'profile', 'domain',
                'visited_at_iso', 'time_of_day', 'day_of_week',
                'session_id', 'duration_hint',
                'label_reason', 'label_topics',
                'content_meta_videoId', 'content_meta_title', 'content_meta_channel',
                'content_meta_duration', 'content_meta_views', 'content_meta_is_shorts',
            ]
            
            with open(args.export_csv, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                
                for event in golden_set:
                    row = {
                        'url': event.get('url', ''),
                        'title': event.get('title', ''),
                        'browser': event.get('browser', ''),
                        'profile': event.get('profile', ''),
                        'domain': event.get('domain', ''),
                        'visited_at_iso': event.get('visited_at_iso', ''),
                        'time_of_day': event.get('time_of_day', ''),
                        'day_of_week': event.get('day_of_week', ''),
                        'session_id': event.get('session_id', ''),
                        'duration_hint': event.get('duration_hint', ''),
                        'label_reason': event.get('labels', {}).get('reason', ''),
                        'label_topics': ', '.join(event.get('labels', {}).get('topics', [])),
                    }
                    
                    # Flatten content_meta
                    cm = event.get('content_meta') or {}
                    row['content_meta_videoId'] = cm.get('videoId', '')
                    row['content_meta_title'] = cm.get('title', '')
                    row['content_meta_channel'] = cm.get('channel', '')
                    row['content_meta_duration'] = cm.get('duration', '')
                    row['content_meta_views'] = cm.get('views', '')
                    row['content_meta_is_shorts'] = cm.get('is_shorts', '')
                    
                    writer.writerow(row)
            
            print(f"[SUCCESS] Exported CSV to {args.export_csv}")
        except Exception as e:
            print(f"[ERROR] Failed to export CSV: {e}", file=sys.stderr)
            sys.exit(1)
    
    sys.exit(0)


if __name__ == '__main__':
    main()
