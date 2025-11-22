#!/usr/bin/env python3
"""
Validate Enriched Dataset Completeness and Quality

This script performs comprehensive validation of enriched_event.jsonl including:
- Event coverage and schema validation
- Null-safe field checks
- Timestamp conversion validation
- Session join validation
- Metadata coherence checks
- Label distribution analysis
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Tuple
from urllib.parse import urlparse


def is_valid_iso_timestamp(ts_str: Optional[str]) -> bool:
    """Validate ISO 8601 timestamp format."""
    if not ts_str:
        return False
    try:
        ts_clean = ts_str.replace('Z', '+00:00')
        datetime.fromisoformat(ts_clean)
        return True
    except (ValueError, AttributeError):
        return False


def validate_schema(event: Dict[str, Any], idx: int) -> Tuple[bool, List[str]]:
    """Validate event schema completeness."""
    required_fields = [
        'browser', 'profile', 'url', 'domain', 'title', 'visited_at_iso',
        'content_meta', 'session_id', 'time_of_day', 'day_of_week',
        'duration_hint', 'labels'
    ]
    
    missing = []
    for field in required_fields:
        if field not in event:
            missing.append(field)
    
    return len(missing) == 0, missing


def validate_timestamp(event: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """Validate timestamp format and coherence."""
    ts = event.get('visited_at_iso')
    if not ts:
        return False, "Missing visited_at_iso"
    
    if not is_valid_iso_timestamp(ts):
        return False, f"Invalid ISO format: {ts}"
    
    # Check time_of_day and day_of_week coherence
    try:
        ts_clean = ts.replace('Z', '+00:00')
        dt = datetime.fromisoformat(ts_clean)
        
        expected_tod = dt.strftime('%H:%M')
        expected_dow = dt.strftime('%A')
        
        actual_tod = event.get('time_of_day')
        actual_dow = event.get('day_of_week')
        
        if actual_tod and actual_tod != expected_tod:
            return False, f"time_of_day mismatch: {actual_tod} != {expected_tod}"
        
        if actual_dow and actual_dow != expected_dow:
            return False, f"day_of_week mismatch: {actual_dow} != {expected_dow}"
        
        return True, None
    except Exception as e:
        return False, f"Timestamp parsing error: {e}"


def validate_youtube_metadata(event: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """Validate YouTube metadata coherence."""
    url = event.get('url', '')
    content_meta = event.get('content_meta')
    
    is_youtube = 'youtube.com' in url or 'youtu.be' in url
    
    if is_youtube:
        if content_meta is None:
            return True, None  # Missing cache is acceptable
        
        if not isinstance(content_meta, dict):
            return False, "content_meta should be dict or None"
        
        # Check videoId matches URL
        video_id_meta = content_meta.get('videoId')
        if video_id_meta:
            if 'youtube.com/watch' in url:
                if f'v={video_id_meta}' not in url and f'watch?v={video_id_meta}' not in url:
                    return False, f"videoId mismatch: {video_id_meta} not in URL"
            elif 'youtu.be/' in url:
                if video_id_meta not in url:
                    return False, f"videoId mismatch: {video_id_meta} not in URL"
        
        # Validate metadata fields
        if 'duration' in content_meta and content_meta['duration'] is not None:
            if not isinstance(content_meta['duration'], int) or content_meta['duration'] < 0:
                return False, f"Invalid duration: {content_meta['duration']}"
        
        if 'views' in content_meta and content_meta['views'] is not None:
            if not isinstance(content_meta['views'], int) or content_meta['views'] < 0:
                return False, f"Invalid views: {content_meta['views']}"
    
    return True, None


def validate_session_join(event: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """Validate session join coherence."""
    session_id = event.get('session_id')
    duration_hint = event.get('duration_hint')
    
    # If session_id exists, duration_hint should also exist
    if session_id and duration_hint is None:
        return False, "session_id present but duration_hint missing"
    
    # If duration_hint exists, it should be positive
    if duration_hint is not None:
        if not isinstance(duration_hint, int) or duration_hint < 0:
            return False, f"Invalid duration_hint: {duration_hint}"
    
    return True, None


def validate_labels(event: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """Validate label structure and coherence."""
    labels = event.get('labels')
    
    if labels is None:
        return False, "Missing labels field"
    
    if not isinstance(labels, dict):
        return False, "labels should be a dict"
    
    if 'reason' not in labels:
        return False, "labels missing 'reason' field"
    
    reason = labels.get('reason')
    if not isinstance(reason, str) or not reason:
        return False, f"Invalid reason: {reason}"
    
    topics = labels.get('topics', [])
    if not isinstance(topics, list):
        return False, "topics should be a list"
    
    return True, None


def get_domain(url: str) -> str:
    """Extract domain from URL."""
    try:
        parsed = urlparse(url)
        return parsed.netloc.lower()
    except Exception:
        return ""


def main():
    parser = argparse.ArgumentParser(
        description='Validate enriched dataset completeness and quality',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--input', required=True, type=Path,
                       help='Input enriched_event.jsonl file')
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
    
    print("[INFO] Validating enriched dataset...")
    print(f"[INFO] Input: {args.input}")
    
    # Load events
    events = []
    parse_errors = []
    
    try:
        with open(args.input, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                    events.append((idx, event))
                except json.JSONDecodeError as e:
                    parse_errors.append((idx, str(e)))
    except Exception as e:
        print(f"[ERROR] Failed to read input: {e}", file=sys.stderr)
        sys.exit(1)
    
    total_events = len(events)
    print(f"[INFO] Loaded {total_events} events")
    
    if parse_errors:
        print(f"[WARN] {len(parse_errors)} JSON parse errors")
        if args.verbose:
            for idx, err in parse_errors[:10]:
                print(f"  Line {idx}: {err}")
    
    # Validation results
    schema_errors = []
    timestamp_errors = []
    youtube_errors = []
    session_errors = []
    label_errors = []
    
    # Coverage metrics
    cache_hits = 0
    session_joins = 0
    youtube_events = 0
    youtube_enriched = 0
    
    # Label distribution
    label_counter = Counter()
    domain_counter = Counter()
    browser_counter = Counter()
    
    # Spot-check detailed validation
    spot_check_events = []
    if args.spot_check > 0:
        import random
        random.seed(42)  # Deterministic
        spot_indices = random.sample(range(len(events)), min(args.spot_check, len(events)))
        spot_check_events = [events[i] for i in spot_indices]
    
    # Validate each event
    for idx, event in events:
        # Schema validation
        schema_ok, missing = validate_schema(event, idx)
        if not schema_ok:
            schema_errors.append((idx, missing))
        
        # Timestamp validation
        ts_ok, ts_err = validate_timestamp(event)
        if not ts_ok:
            timestamp_errors.append((idx, ts_err))
        
        # YouTube metadata validation
        yt_ok, yt_err = validate_youtube_metadata(event)
        if not yt_ok:
            youtube_errors.append((idx, yt_err))
        
        # Session join validation
        sess_ok, sess_err = validate_session_join(event)
        if not sess_ok:
            session_errors.append((idx, sess_err))
        
        # Label validation
        label_ok, label_err = validate_labels(event)
        if not label_ok:
            label_errors.append((idx, label_err))
        
        # Coverage metrics
        if event.get('content_meta'):
            cache_hits += 1
        
        if event.get('session_id'):
            session_joins += 1
        
        url = event.get('url', '')
        if 'youtube.com' in url or 'youtu.be' in url:
            youtube_events += 1
            if event.get('content_meta'):
                youtube_enriched += 1
        
        # Distribution tracking
        labels = event.get('labels', {})
        reason = labels.get('reason', 'unknown')
        label_counter[reason] += 1
        
        domain = get_domain(url)
        if domain:
            domain_counter[domain] += 1
        
        browser = event.get('browser', 'unknown')
        browser_counter[browser] += 1
    
    # Print summary
    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)
    
    print(f"\n[SCHEMA] Schema errors: {len(schema_errors)}/{total_events}")
    if schema_errors and args.verbose:
        for idx, missing in schema_errors[:10]:
            print(f"  Line {idx}: Missing fields: {', '.join(missing)}")
    
    print(f"\n[TIMESTAMP] Timestamp errors: {len(timestamp_errors)}/{total_events}")
    if timestamp_errors and args.verbose:
        for idx, err in timestamp_errors[:10]:
            print(f"  Line {idx}: {err}")
    
    print(f"\n[YOUTUBE] YouTube metadata errors: {len(youtube_errors)}/{total_events}")
    if youtube_errors and args.verbose:
        for idx, err in youtube_errors[:10]:
            print(f"  Line {idx}: {err}")
    
    print(f"\n[SESSION] Session join errors: {len(session_errors)}/{total_events}")
    if session_errors and args.verbose:
        for idx, err in session_errors[:10]:
            print(f"  Line {idx}: {err}")
    
    print(f"\n[LABELS] Label errors: {len(label_errors)}/{total_events}")
    if label_errors and args.verbose:
        for idx, err in label_errors[:10]:
            print(f"  Line {idx}: {err}")
    
    # Coverage metrics
    print("\n" + "="*60)
    print("COVERAGE METRICS")
    print("="*60)
    
    cache_coverage = (cache_hits / total_events * 100) if total_events > 0 else 0
    session_coverage = (session_joins / total_events * 100) if total_events > 0 else 0
    youtube_coverage = (youtube_enriched / youtube_events * 100) if youtube_events > 0 else 0
    
    print(f"\n[CACHE] Cache-based metadata: {cache_hits}/{total_events} ({cache_coverage:.1f}%)")
    print(f"[SESSION] Session joins: {session_joins}/{total_events} ({session_coverage:.1f}%)")
    print(f"[YOUTUBE] YouTube enrichment: {youtube_enriched}/{youtube_events} ({youtube_coverage:.1f}%)")
    
    # Label distribution
    print("\n" + "="*60)
    print("LABEL DISTRIBUTION")
    print("="*60)
    for reason, count in label_counter.most_common():
        pct = (count / total_events * 100) if total_events > 0 else 0
        print(f"  {reason}: {count} ({pct:.1f}%)")
    
    # Top domains
    print("\n" + "="*60)
    print("TOP DOMAINS (top 10)")
    print("="*60)
    for domain, count in domain_counter.most_common(10):
        pct = (count / total_events * 100) if total_events > 0 else 0
        print(f"  {domain}: {count} ({pct:.1f}%)")
    
    # Browser distribution
    print("\n" + "="*60)
    print("BROWSER DISTRIBUTION")
    print("="*60)
    for browser, count in browser_counter.most_common():
        pct = (count / total_events * 100) if total_events > 0 else 0
        print(f"  {browser}: {count} ({pct:.1f}%)")
    
    # Spot-check details
    if spot_check_events:
        print("\n" + "="*60)
        print(f"SPOT-CHECK DETAILS ({len(spot_check_events)} events)")
        print("="*60)
        for idx, event in spot_check_events[:5]:
            print(f"\n[Event {idx}]")
            print(f"  URL: {event.get('url', 'N/A')[:80]}")
            print(f"  Browser: {event.get('browser', 'N/A')}")
            print(f"  Timestamp: {event.get('visited_at_iso', 'N/A')}")
            print(f"  Session ID: {event.get('session_id', 'N/A')}")
            print(f"  Content Meta: {'Present' if event.get('content_meta') else 'Missing'}")
            print(f"  Labels: {event.get('labels', {})}")
    
    # Generate report
    report = {
        'total_events': total_events,
        'parse_errors': len(parse_errors),
        'validation_errors': {
            'schema': len(schema_errors),
            'timestamp': len(timestamp_errors),
            'youtube': len(youtube_errors),
            'session': len(session_errors),
            'labels': len(label_errors),
        },
        'coverage': {
            'cache_hits': cache_hits,
            'cache_coverage_pct': cache_coverage,
            'session_joins': session_joins,
            'session_coverage_pct': session_coverage,
            'youtube_events': youtube_events,
            'youtube_enriched': youtube_enriched,
            'youtube_coverage_pct': youtube_coverage,
        },
        'distributions': {
            'labels': dict(label_counter),
            'browsers': dict(browser_counter),
            'top_domains': dict(domain_counter.most_common(20)),
        },
        'validation_passed': (
            len(schema_errors) == 0 and
            len(timestamp_errors) == 0 and
            len(youtube_errors) == 0 and
            len(session_errors) == 0 and
            len(label_errors) == 0
        ),
    }
    
    if args.output_report:
        with open(args.output_report, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"\n[INFO] Validation report written to {args.output_report}")
    
    # Exit code
    if report['validation_passed']:
        print("\n[SUCCESS] All validations passed!")
        sys.exit(0)
    else:
        print("\n[WARNING] Some validations failed. Review errors above.")
        sys.exit(1)


if __name__ == '__main__':
    from typing import Optional
    main()

