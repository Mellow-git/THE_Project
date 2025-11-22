#!/usr/bin/env python3
"""
knowledgeC.db Behavioral Joiner Module - Standalone enrichment component.

This module joins browser history events with behavioral data from macOS
knowledgeC.db files. It operates independently without modifying existing
collector or snapshot code.

Input: JSONL with browser history events (enriched or raw)
Output: JSONL with joined behavioral metadata (session_id, duration_hint, etc.)
"""

import argparse
import json
import sqlite3
import sys
import tempfile
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any, Tuple


# macOS Core Data epoch offset (seconds since 2001-01-01 00:00:00 UTC)
MACOS_EPOCH_OFFSET = 978307200


def snapshot_sqlite(src_path: Path, appname: str = "join_knowledgec") -> Optional[Path]:
    """
    Create a WAL-safe snapshot of SQLite database.
    Returns path to snapshot or None on failure.
    """
    if not src_path.exists():
        return None
    
    cache_dir = Path(tempfile.gettempdir()) / f"{appname}_snapshots"
    cache_dir.mkdir(parents=True, exist_ok=True)
    snapshot = cache_dir / f"{src_path.name}.snapshot"
    
    try:
        # Use URI mode for read-only access
        src = sqlite3.connect(f"file:{src_path}?mode=ro", uri=True)
        dst = sqlite3.connect(snapshot)
        src.backup(dst, pages=1000)
        dst.commit()
        dst.close()
        src.close()
        return snapshot
    except sqlite3.OperationalError as e:
        print(f"[WARN] Failed to snapshot {src_path}: {e}", file=sys.stderr)
        return None
    except Exception as e:
        print(f"[WARN] Unexpected error snapshotting {src_path}: {e}", file=sys.stderr)
        return None


def load_knowledgec_sessions(snapshot_path: Optional[Path]) -> List[Dict[str, Any]]:
    """
    Load session/activity intervals from knowledgeC.db snapshot.
    Returns list of session dicts with stream, value, start, end.
    """
    sessions = []
    
    if not snapshot_path or not snapshot_path.exists():
        return sessions
    
    try:
        conn = sqlite3.connect(snapshot_path)
        cursor = conn.cursor()
        
        # Query ZOBJECT table for browser/activity streams
        # Common stream names for browser activity
        stream_names = [
            'com.apple.corespotlight.domain',
            'com.apple.browser',
            'com.apple.Safari',
            'com.apple.Chrome',
            'com.apple.webkit',
        ]
        
        placeholders = ','.join(['?'] * len(stream_names))
        query = f"""
            SELECT ZSTREAMNAME, ZVALUESTRING, ZSTARTDATE, ZENDDATE
            FROM ZOBJECT
            WHERE ZSTREAMNAME IN ({placeholders})
            AND ZSTARTDATE IS NOT NULL
            AND ZENDDATE IS NOT NULL
            ORDER BY ZSTARTDATE DESC
        """
        
        cursor.execute(query, stream_names)
        
        for stream_name, value_str, zstart, zend in cursor.fetchall():
            try:
                # Convert macOS timestamp to UTC datetime
                start_timestamp = zstart + MACOS_EPOCH_OFFSET
                end_timestamp = zend + MACOS_EPOCH_OFFSET
                
                start_dt = datetime.utcfromtimestamp(start_timestamp)
                end_dt = datetime.utcfromtimestamp(end_timestamp)
                
                sessions.append({
                    'stream': stream_name,
                    'value': value_str or '',
                    'start': start_dt,
                    'end': end_dt,
                })
            except (ValueError, OSError, OverflowError) as e:
                # Skip invalid timestamps
                continue
        
        conn.close()
        
    except sqlite3.Error as e:
        print(f"[WARN] SQLite error loading sessions: {e}", file=sys.stderr)
    except Exception as e:
        print(f"[WARN] Unexpected error loading sessions: {e}", file=sys.stderr)
    
    return sessions


def find_matching_session(event_time: datetime, sessions: List[Dict[str, Any]], 
                         slop_seconds: int = 600) -> Optional[Dict[str, Any]]:
    """
    Find session that matches event timestamp.
    Returns matching session dict or None.
    
    Args:
        event_time: Event timestamp
        sessions: List of session dicts
        slop_seconds: Allowable time difference in seconds (default 10 minutes)
    """
    if not event_time:
        return None
    
    best_match = None
    best_overlap = None
    
    for session in sessions:
        start = session['start']
        end = session['end']
        
        # Exact overlap
        if start <= event_time <= end:
            overlap = (end - start).total_seconds()
            if best_overlap is None or overlap > best_overlap:
                best_match = session
                best_overlap = overlap
        
        # Within slop window before session start
        elif event_time < start:
            diff = (start - event_time).total_seconds()
            if diff <= slop_seconds:
                overlap = (end - start).total_seconds()
                if best_overlap is None or overlap > best_overlap:
                    best_match = session
                    best_overlap = overlap
        
        # Within slop window after session end
        elif event_time > end:
            diff = (event_time - end).total_seconds()
            if diff <= slop_seconds:
                overlap = (end - start).total_seconds()
                if best_overlap is None or overlap > best_overlap:
                    best_match = session
                    best_overlap = overlap
    
    return best_match


def parse_iso_timestamp(ts_str: Optional[str]) -> Optional[datetime]:
    """Parse ISO 8601 timestamp string to datetime."""
    if not ts_str:
        return None
    
    try:
        # Remove 'Z' suffix and parse
        ts_clean = ts_str.replace('Z', '+00:00')
        return datetime.fromisoformat(ts_clean)
    except (ValueError, AttributeError):
        return None


def join_event_with_sessions(event: Dict[str, Any], 
                            user_sessions: List[Dict[str, Any]],
                            system_sessions: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Join event with matching session data.
    Returns event with added fields: session_id, duration_hint, time_of_day, day_of_week
    """
    # Parse event timestamp
    visited_at_iso = event.get('visited_at_iso')
    event_time = parse_iso_timestamp(visited_at_iso)
    
    # Find matching session (prefer user over system)
    best_session = None
    if event_time:
        best_session = find_matching_session(event_time, user_sessions)
        if not best_session:
            best_session = find_matching_session(event_time, system_sessions)
    
    # Calculate duration hint
    duration_hint = None
    if best_session:
        duration_seconds = int((best_session['end'] - best_session['start']).total_seconds())
        duration_hint = duration_seconds
    
    # Generate session ID
    session_id = None
    if best_session:
        session_id = f"{best_session['stream']}:{best_session['start'].isoformat()}"
    
    # Extract time of day and day of week
    time_of_day = None
    day_of_week = None
    if event_time:
        time_of_day = event_time.strftime('%H:%M')
        day_of_week = event_time.strftime('%A')
    
    # Add fields to event (don't overwrite existing fields)
    enriched_event = event.copy()
    
    # Only add if not already present (preserve existing values)
    if 'session_id' not in enriched_event:
        enriched_event['session_id'] = session_id
    if 'duration_hint' not in enriched_event:
        enriched_event['duration_hint'] = duration_hint
    if 'time_of_day' not in enriched_event:
        enriched_event['time_of_day'] = time_of_day
    if 'day_of_week' not in enriched_event:
        enriched_event['day_of_week'] = day_of_week
    
    return enriched_event


def parse_input_jsonl(input_path: Path) -> Tuple[List[Dict[str, Any]], int]:
    """
    Parse input JSONL file with error handling.
    Returns (events, warning_count)
    """
    events = []
    warn_count = 0
    
    if not input_path.exists():
        print(f"[ERROR] Input file not found: {input_path}", file=sys.stderr)
        return events, warn_count
    
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    event = json.loads(line)
                    if not isinstance(event, dict):
                        print(f"[WARN] Line {idx}: Not a JSON object, skipping", file=sys.stderr)
                        warn_count += 1
                        continue
                    
                    # Validate required fields
                    if 'visited_at_iso' not in event:
                        print(f"[WARN] Line {idx}: Missing visited_at_iso, skipping", file=sys.stderr)
                        warn_count += 1
                        continue
                    
                    events.append(event)
                except json.JSONDecodeError as e:
                    print(f"[WARN] Line {idx}: Invalid JSON ({e}), skipping", file=sys.stderr)
                    warn_count += 1
                    continue
    except Exception as e:
        print(f"[ERROR] Failed to read input file: {e}", file=sys.stderr)
        return events, warn_count
    
    return events, warn_count


def main():
    parser = argparse.ArgumentParser(
        description='knowledgeC.db Behavioral Joiner - Join events with session data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --input history.jsonl --output joined.jsonl
  %(prog)s --input enriched.jsonl --output joined.jsonl --knowledgec-user ~/Library/Application\\ Support/Knowledge/knowledgeC.db
  %(prog)s --input history.jsonl --output joined.jsonl --slop-seconds 300 --dry-run
        """
    )
    parser.add_argument('--input', required=True, type=Path,
                       help='Input JSONL file with browser history events')
    parser.add_argument('--output', required=True, type=Path,
                       help='Output JSONL file with joined behavioral metadata')
    parser.add_argument('--knowledgec-user', type=Path,
                       default=Path.home() / 'Library' / 'Application Support' / 'Knowledge' / 'knowledgeC.db',
                       help='User knowledgeC.db path (default: ~/Library/Application Support/Knowledge/knowledgeC.db)')
    parser.add_argument('--knowledgec-system', type=Path,
                       default=Path('/private/var/db/CoreDuet/Knowledge/knowledgeC.db'),
                       help='System knowledgeC.db path (default: /private/var/db/CoreDuet/Knowledge/knowledgeC.db)')
    parser.add_argument('--slop-seconds', type=int, default=600,
                       help='Time window in seconds for matching events to sessions (default: 600)')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be done without writing output')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Validate input
    if not args.input.exists():
        print(f"[ERROR] Input file does not exist: {args.input}", file=sys.stderr)
        sys.exit(1)
    
    # Check permissions
    print("[INFO] knowledgeC.db Behavioral Joiner - Privacy-first, on-device processing")
    print("[INFO] Requires Full Disk Access in macOS System Settings > Privacy & Security")
    
    # Parse input
    print(f"[INFO] Reading input from {args.input}...")
    events, warn_count = parse_input_jsonl(args.input)
    
    if not events:
        print("[ERROR] No valid events found in input file", file=sys.stderr)
        sys.exit(1)
    
    print(f"[INFO] Loaded {len(events)} events (warnings: {warn_count})")
    
    # Snapshot and load knowledgeC.db files
    user_sessions = []
    system_sessions = []
    
    if args.knowledgec_user.exists():
        print(f"[INFO] Snapshotting user knowledgeC.db: {args.knowledgec_user}...")
        user_snapshot = snapshot_sqlite(args.knowledgec_user)
        if user_snapshot:
            print("[INFO] Loading user sessions...")
            user_sessions = load_knowledgec_sessions(user_snapshot)
            print(f"[INFO] Loaded {len(user_sessions)} user sessions")
        else:
            print("[WARN] Failed to snapshot user knowledgeC.db", file=sys.stderr)
    else:
        print(f"[WARN] User knowledgeC.db not found: {args.knowledgec_user}", file=sys.stderr)
        print("[WARN] Grant Full Disk Access if recently denied", file=sys.stderr)
    
    # Check system knowledgeC.db (may require special permissions)
    try:
        if args.knowledgec_system.exists():
            print(f"[INFO] Snapshotting system knowledgeC.db: {args.knowledgec_system}...")
            system_snapshot = snapshot_sqlite(args.knowledgec_system)
            if system_snapshot:
                print("[INFO] Loading system sessions...")
                system_sessions = load_knowledgec_sessions(system_snapshot)
                print(f"[INFO] Loaded {len(system_sessions)} system sessions")
            else:
                print("[WARN] Failed to snapshot system knowledgeC.db", file=sys.stderr)
        else:
            print(f"[WARN] System knowledgeC.db not found: {args.knowledgec_system}", file=sys.stderr)
            print("[WARN] Grant Full Disk Access if recently denied", file=sys.stderr)
    except (PermissionError, OSError) as e:
        print(f"[WARN] Cannot access system knowledgeC.db: {e}", file=sys.stderr)
        print("[WARN] Grant Full Disk Access in System Settings > Privacy & Security", file=sys.stderr)
    
    total_sessions = len(user_sessions) + len(system_sessions)
    if total_sessions == 0:
        print("[WARN] No sessions loaded from knowledgeC.db files", file=sys.stderr)
        print("[WARN] Continuing with empty session data", file=sys.stderr)
    
    # Join events with sessions
    print(f"[INFO] Joining events with sessions (slop: {args.slop_seconds}s)...")
    joined_count = 0
    
    joined_events = []
    for event in events:
        joined_event = join_event_with_sessions(event, user_sessions, system_sessions)
        if joined_event.get('session_id'):
            joined_count += 1
        joined_events.append(joined_event)
    
    print(f"[INFO] Joined {joined_count}/{len(events)} events with sessions ({joined_count/len(events)*100:.1f}%)")
    
    if args.dry_run:
        print("[DRY RUN] Would write output to:", args.output)
        print(f"[DRY RUN] Total events: {len(joined_events)}")
        print(f"[DRY RUN] Joined: {joined_count}/{len(events)} events")
        sys.exit(0)
    
    # Write output
    print(f"[INFO] Writing output to {args.output}...")
    try:
        with open(args.output, 'w', encoding='utf-8') as f:
            for event in joined_events:
                json.dump(event, f, ensure_ascii=False)
                f.write('\n')
        print(f"[INFO] Successfully wrote {len(joined_events)} events to {args.output}")
    except Exception as e:
        print(f"[ERROR] Failed to write output: {e}", file=sys.stderr)
        sys.exit(1)
    
    sys.exit(0)


if __name__ == '__main__':
    main()

