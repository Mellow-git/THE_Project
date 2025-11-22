#!/usr/bin/env python3
"""
Comprehensive validation script for enriched dataset.
Checks completeness, quality, and coherence of enriched events.
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import re


REQUIRED_FIELDS = [
    'browser', 'profile', 'url', 'domain', 'title', 'visited_at_iso',
    'content_meta', 'transcript_excerpt', 'session_id', 'time_of_day',
    'day_of_week', 'duration_hint', 'labels'
]

OPTIONAL_FIELDS = ['transcript_excerpt']


def parse_iso_timestamp(ts_str: str) -> Tuple[bool, Optional[datetime]]:
    """Parse ISO timestamp, return (is_valid, datetime_obj)."""
    if not ts_str:
        return False, None
    try:
        ts_clean = ts_str.replace('Z', '+00:00')
        dt = datetime.fromisoformat(ts_clean)
        # Check if timestamp is reasonable (within last 5 years)
        now = datetime.now(dt.tzinfo) if dt.tzinfo else datetime.now()
        years_diff = abs((now - dt).days / 365.25)
        return years_diff <= 5, dt
    except Exception:
        return False, None


def validate_schema(event: Dict[str, Any], line_num: int) -> List[str]:
    """Validate event schema, return list of errors."""
    errors = []
    
    # Check required fields
    for field in REQUIRED_FIELDS:
        if field not in event:
            errors.append(f"Missing required field: {field}")
    
    # Check field types
    if 'url' in event and not isinstance(event['url'], str):
        errors.append("Field 'url' must be string")
    
    if 'visited_at_iso' in event:
        is_valid, dt = parse_iso_timestamp(event['visited_at_iso'])
        if not is_valid:
            errors.append(f"Invalid timestamp: {event['visited_at_iso']}")
    
    if 'content_meta' in event and event['content_meta'] is not None:
        if not isinstance(event['content_meta'], dict):
            errors.append("Field 'content_meta' must be dict or null")
    
    if 'labels' in event:
        if not isinstance(event['labels'], dict):
            errors.append("Field 'labels' must be dict")
        elif 'reason' not in event['labels']:
            errors.append("Field 'labels' missing 'reason' key")
    
    return errors


def validate_youtube_metadata(content_meta: Dict[str, Any], url: str) -> List[str]:
    """Validate YouTube metadata coherence."""
    errors = []
    
    if 'youtube.com' in url or 'youtu.be' in url:
        if not content_meta:
            errors.append("YouTube URL missing content_meta")
            return errors
        
        # Check for videoId
        if 'videoId' not in content_meta:
            errors.append("YouTube metadata missing videoId")
        
        # Validate videoId matches URL
        if 'videoId' in content_meta:
            video_id = content_meta['videoId']
            if video_id not in url:
                errors.append(f"videoId {video_id} not found in URL {url}")
        
        # Check duration consistency
        if 'duration' in content_meta and content_meta['duration']:
            duration = content_meta['duration']
            if duration < 0 or duration > 86400:  # More than 24 hours
                errors.append(f"Suspicious duration: {duration}s")
        
        # Check is_shorts consistency
        if content_meta.get('is_shorts') and content_meta.get('duration', 0) > 300:
            errors.append("Shorts marked but duration > 5 minutes")
    
    return errors


def validate_session_join(event: Dict[str, Any]) -> List[str]:
    """Validate session join coherence."""
    errors = []
    
    session_id = event.get('session_id')
    duration_hint = event.get('duration_hint')
    time_of_day = event.get('time_of_day')
    day_of_week = event.get('day_of_week')
    visited_at_iso = event.get('visited_at_iso')
    
    # If session_id exists, other fields should exist
    if session_id:
        if not duration_hint:
            errors.append("session_id present but duration_hint missing")
        if not time_of_day:
            errors.append("session_id present but time_of_day missing")
        if not day_of_week:
            errors.append("session_id present but day_of_week missing")
    
    # Validate time_of_day format
    if time_of_day:
        if not re.match(r'^\d{2}:\d{2}$', time_of_day):
            errors.append(f"Invalid time_of_day format: {time_of_day}")
    
    # Validate day_of_week
    valid_days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    if day_of_week and day_of_week not in valid_days:
        errors.append(f"Invalid day_of_week: {day_of_week}")
    
    # Cross-validate timestamp with time_of_day
    if visited_at_iso and time_of_day:
        is_valid, dt = parse_iso_timestamp(visited_at_iso)
        if is_valid and dt:
            expected_time = dt.strftime('%H:%M')
            if expected_time != time_of_day:
                errors.append(f"time_of_day mismatch: {time_of_day} vs {expected_time}")
    
    return errors


def validate_label_coherence(event: Dict[str, Any]) -> List[str]:
    """Validate label coherence with content."""
    errors = []
    
    labels = event.get('labels', {})
    reason = labels.get('reason')
    topics = labels.get('topics', [])
    content_meta = event.get('content_meta') or {}
    
    # Check reason is valid
    valid_reasons = [
        'general', 'shorts/entertainment', 'tutorial/learning',
        'news/information', 'music/ambient', 'deep-dive/research', 'night_browsing'
    ]
    if reason and reason not in valid_reasons:
        errors.append(f"Unknown reason: {reason}")
    
    # Check shorts label consistency
    if reason == 'shorts/entertainment':
        if not content_meta.get('is_shorts') and content_meta.get('duration', 0) >= 60:
            errors.append("Labeled as shorts but metadata suggests long-form")
    
    # Check topics is list
    if not isinstance(topics, list):
        errors.append("Field 'topics' must be a list")
    
    return errors


def compute_coverage_metrics(events: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute enrichment coverage metrics."""
    total = len(events)
    if total == 0:
        return {}
    
    cache_hits = sum(1 for e in events if e.get('content_meta'))
    session_joins = sum(1 for e in events if e.get('session_id'))
    youtube_urls = sum(1 for e in events if 'youtube.com' in e.get('url', '') or 'youtu.be' in e.get('url', ''))
    youtube_enriched = sum(1 for e in events if ('youtube.com' in e.get('url', '') or 'youtu.be' in e.get('url', '')) and e.get('content_meta'))
    
    valid_timestamps = sum(1 for e in events if parse_iso_timestamp(e.get('visited_at_iso', ''))[0])
    has_labels = sum(1 for e in events if e.get('labels', {}).get('reason'))
    
    return {
        'total_events': total,
        'cache_hits': cache_hits,
        'cache_hit_rate': cache_hits / total if total > 0 else 0,
        'session_joins': session_joins,
        'session_join_rate': session_joins / total if total > 0 else 0,
        'youtube_urls': youtube_urls,
        'youtube_enriched': youtube_enriched,
        'youtube_enrichment_rate': youtube_enriched / youtube_urls if youtube_urls > 0 else 0,
        'valid_timestamps': valid_timestamps,
        'timestamp_validity_rate': valid_timestamps / total if total > 0 else 0,
        'labeled_events': has_labels,
        'label_coverage_rate': has_labels / total if total > 0 else 0,
    }


def main():
    parser = argparse.ArgumentParser(
        description='Validate enriched dataset completeness and quality',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--input', required=True, type=Path,
                       help='Input enriched JSONL file')
    parser.add_argument('--output-report', type=Path,
                       help='Output validation report JSON file')
    parser.add_argument('--verbose', action='store_true',
                       help='Show detailed validation errors')
    parser.add_argument('--spot-check', type=int, default=10,
                       help='Number of events to spot-check in detail')
    
    args = parser.parse_args()
    
    if not args.input.exists():
        print(f"[ERROR] Input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)
    
    print(f"[INFO] Loading enriched dataset from {args.input}...")
    events = []
    schema_errors = []
    youtube_errors = []
    session_errors = []
    label_errors = []
    
    try:
        with open(args.input, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    event = json.loads(line)
                    events.append(event)
                    
                    # Schema validation
                    schema_errs = validate_schema(event, line_num)
                    if schema_errs:
                        schema_errors.append((line_num, event.get('url', 'unknown'), schema_errs))
                    
                    # YouTube metadata validation
                    if event.get('content_meta'):
                        yt_errs = validate_youtube_metadata(event.get('content_meta', {}), event.get('url', ''))
                        if yt_errs:
                            youtube_errors.append((line_num, event.get('url', 'unknown'), yt_errs))
                    
                    # Session join validation
                    sess_errs = validate_session_join(event)
                    if sess_errs:
                        session_errors.append((line_num, event.get('url', 'unknown'), sess_errs))
                    
                    # Label coherence validation
                    label_errs = validate_label_coherence(event)
                    if label_errs:
                        label_errors.append((line_num, event.get('url', 'unknown'), label_errs))
                
                except json.JSONDecodeError as e:
                    print(f"[ERROR] Line {line_num}: Invalid JSON: {e}", file=sys.stderr)
                    continue
    
    except Exception as e:
        print(f"[ERROR] Failed to read input file: {e}", file=sys.stderr)
        sys.exit(1)
    
    print(f"[INFO] Loaded {len(events)} events")
    
    # Compute coverage metrics
    print("\n[COVERAGE METRICS]")
    metrics = compute_coverage_metrics(events)
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2%}")
        else:
            print(f"  {key}: {value}")
    
    # Label distribution
    print("\n[LABEL DISTRIBUTION]")
    reasons = [e.get('labels', {}).get('reason', 'NONE') for e in events]
    reason_counts = Counter(reasons)
    for reason, count in reason_counts.most_common():
        print(f"  {reason}: {count} ({count/len(events)*100:.1f}%)")
    
    # Error summary
    print("\n[VALIDATION SUMMARY]")
    print(f"  Schema errors: {len(schema_errors)}")
    print(f"  YouTube metadata errors: {len(youtube_errors)}")
    print(f"  Session join errors: {len(session_errors)}")
    print(f"  Label coherence errors: {len(label_errors)}")
    
    total_errors = len(schema_errors) + len(youtube_errors) + len(session_errors) + len(label_errors)
    if total_errors > 0:
        print(f"\n[WARNING] Found {total_errors} validation errors")
        if args.verbose:
            print("\n[SCHEMA ERRORS]")
            for line_num, url, errs in schema_errors[:args.spot_check]:
                print(f"  Line {line_num} ({url[:50]}...):")
                for err in errs:
                    print(f"    - {err}")
            
            print("\n[YOUTUBE METADATA ERRORS]")
            for line_num, url, errs in youtube_errors[:args.spot_check]:
                print(f"  Line {line_num} ({url[:50]}...):")
                for err in errs:
                    print(f"    - {err}")
            
            print("\n[SESSION JOIN ERRORS]")
            for line_num, url, errs in session_errors[:args.spot_check]:
                print(f"  Line {line_num} ({url[:50]}...):")
                for err in errs:
                    print(f"    - {err}")
            
            print("\n[LABEL COHERENCE ERRORS]")
            for line_num, url, errs in label_errors[:args.spot_check]:
                print(f"  Line {line_num} ({url[:50]}...):")
                for err in errs:
                    print(f"    - {err}")
    else:
        print("[SUCCESS] No validation errors found!")
    
    # Spot-check sample events
    if events and args.spot_check > 0:
        print(f"\n[SPOT-CHECK] Sample of {min(args.spot_check, len(events))} events:")
        import random
        random.seed(42)  # Deterministic
        sample_events = random.sample(events, min(args.spot_check, len(events)))
        for i, event in enumerate(sample_events, 1):
            print(f"\n  Event {i}:")
            print(f"    URL: {event.get('url', 'N/A')[:60]}...")
            print(f"    Browser: {event.get('browser', 'N/A')}")
            print(f"    Timestamp: {event.get('visited_at_iso', 'N/A')}")
            print(f"    Has content_meta: {bool(event.get('content_meta'))}")
            print(f"    Has session_id: {bool(event.get('session_id'))}")
            print(f"    Label reason: {event.get('labels', {}).get('reason', 'N/A')}")
            if event.get('content_meta'):
                cm = event['content_meta']
                print(f"    Video ID: {cm.get('videoId', 'N/A')}")
                print(f"    Duration: {cm.get('duration', 'N/A')}s")
                print(f"    Is Shorts: {cm.get('is_shorts', False)}")
    
    # Generate report
    if args.output_report:
        report = {
            'total_events': len(events),
            'metrics': metrics,
            'label_distribution': dict(reason_counts),
            'error_counts': {
                'schema_errors': len(schema_errors),
                'youtube_errors': len(youtube_errors),
                'session_errors': len(session_errors),
                'label_errors': len(label_errors),
            },
            'sample_errors': {
                'schema_errors': schema_errors[:10],
                'youtube_errors': youtube_errors[:10],
                'session_errors': session_errors[:10],
                'label_errors': label_errors[:10],
            }
        }
        
        with open(args.output_report, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)
        print(f"\n[INFO] Validation report written to {args.output_report}")
    
    # Exit code based on errors
    sys.exit(0 if total_errors == 0 else 1)


if __name__ == '__main__':
    main()

