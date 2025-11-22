#!/usr/bin/env python3
"""
YouTube Enricher Module - Standalone enrichment component.

This module enriches browser history events with YouTube metadata extracted
from Chrome cache directories. It operates independently without modifying
existing collector or snapshot code.

Input: JSONL/CSV browser history with url, browser, profile, visited_at_iso
Output: JSONL with enriched content_meta field for YouTube URLs
"""

import argparse
import csv
import json
import re
import sys
import tempfile
import shutil
from pathlib import Path
from typing import Optional, Dict, List, Any
from urllib.parse import urlparse, parse_qs
import sqlite3


def extract_video_id(url: str) -> Optional[str]:
    """Extract YouTube video ID from URL."""
    if 'youtube.com/watch' in url:
        parsed = urlparse(url)
        params = parse_qs(parsed.query)
        if 'v' in params:
            return params['v'][0]
    elif 'youtu.be/' in url:
        parts = url.split('youtu.be/')
        if len(parts) > 1:
            video_id = parts[1].split('?')[0].split('&')[0]
            return video_id
    return None


def parse_yt_player_response(html: str) -> Optional[Dict[str, Any]]:
    """
    Parse ytInitialPlayerResponse JSON from HTML.
    Returns metadata dict or None on failure.
    """
    # Try multiple patterns for ytInitialPlayerResponse
    patterns = [
        r'ytInitialPlayerResponse\s*=\s*({.*?});',
        r'"ytInitialPlayerResponse":\s*({.*?}),\s*"',
        r'var\s+ytInitialPlayerResponse\s*=\s*({.*?});',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, html, re.DOTALL)
        if match:
    try:
                resp = json.loads(match.group(1))
        vinfo = resp.get('videoDetails') or {}
                
                # Extract publish date from microformat
                publish_date = None
                try:
                    microformat = resp.get('microformat', {})
                    player_microformat = microformat.get('playerMicroformatRenderer', {})
                    publish_date = player_microformat.get('publishDate')
                except Exception:
                    pass
                
        meta = {
                    'videoId': vinfo.get('videoId'),
            'title': vinfo.get('title'),
            'channel': vinfo.get('author'),
            'channelId': vinfo.get('channelId'),
                    'duration': None,
                    'views': None,
            'keywords': vinfo.get('keywords', []),
                    'publishDate': publish_date,
                    'is_shorts': False,
                }
                
                # Parse duration
                length_seconds = vinfo.get('lengthSeconds')
                if length_seconds:
                    try:
                        meta['duration'] = int(length_seconds)
                    except (ValueError, TypeError):
                        pass
                
                # Parse views
                view_count = vinfo.get('viewCount')
                if view_count:
                    try:
                        meta['views'] = int(view_count)
                    except (ValueError, TypeError):
                        pass
                
                # Detect shorts
                short_desc = (vinfo.get('shortDescription') or '').lower()
                if 'shorts' in short_desc or meta.get('duration', 0) < 60:
                    meta['is_shorts'] = True
                
                # Only return if we have at least videoId
                if meta.get('videoId'):
        return meta
            except (json.JSONDecodeError, KeyError, AttributeError) as e:
                continue
    
        return None


def find_chrome_cache_directories(profiles_root: Path, cache_root: Path) -> Dict[str, Path]:
    """
    Map Chrome profile names to their cache directories.
    Returns dict: {profile_name: cache_path}
    """
    mapping = {}
    if not profiles_root.exists() or not cache_root.exists():
        return mapping
    
    # Find all Chrome profiles
    for profile_dir in profiles_root.iterdir():
        if profile_dir.is_dir() and (profile_dir.name == 'Default' or profile_dir.name.startswith('Profile ')):
            # Chrome cache structure: Cache/Cache_Data or just Cache
            cache_dir = cache_root / profile_dir.name / 'Cache' / 'Cache_Data'
            if not cache_dir.exists():
                cache_dir = cache_root / profile_dir.name / 'Cache'
            if cache_dir.exists():
                mapping[profile_dir.name] = cache_dir
    
    return mapping


def search_cache_for_video(cache_dir: Path, video_id: str, max_files: int = 1000) -> Optional[str]:
    """
    Search cache directory for YouTube watch page containing video_id.
    Returns HTML content or None.
    """
    if not cache_dir.exists():
        return None
    
    files_checked = 0
    # Search recursively but limit depth and file count
    try:
        for cache_file in cache_dir.rglob('*'):
            if files_checked >= max_files:
                break
            
            if not cache_file.is_file():
                continue
            
            # Skip very large files (>10MB) - unlikely to be watch pages
            try:
                if cache_file.stat().st_size > 10 * 1024 * 1024:
                    continue
            except OSError:
                continue
            
            files_checked += 1
            
            try:
                # Try reading as text
                content = cache_file.read_text(errors='ignore')
                if video_id in content and 'ytInitialPlayerResponse' in content:
                    return content
            except (UnicodeDecodeError, IOError, PermissionError):
                # Try binary search for video ID
                try:
                    content_bytes = cache_file.read_bytes()
                    if video_id.encode() in content_bytes and b'ytInitialPlayerResponse' in content_bytes:
                        # Try to decode as UTF-8
                        content = content_bytes.decode('utf-8', errors='ignore')
                        return content
                except (IOError, PermissionError):
                    continue
    except (PermissionError, OSError) as e:
        # Cache directory may be locked or inaccessible
        return None
    
    return None


def extract_captions(html: str, max_length: int = 500) -> Optional[str]:
    """
    Extract caption/transcript excerpt from HTML if present.
    Returns excerpt string or None.
    """
    # Look for caption tracks in ytInitialPlayerResponse
    try:
        match = re.search(r'ytInitialPlayerResponse\s*=\s*({.*?});', html, re.DOTALL)
        if match:
            resp = json.loads(match.group(1))
            caption_tracks = resp.get('captions', {}).get('playerCaptionsTracklistRenderer', {}).get('captionTracks', [])
            if caption_tracks:
                # Return first caption track language code as indicator
                # Full transcript extraction would require network calls
                return caption_tracks[0].get('languageCode')
    except Exception:
        pass
    
    return None


def parse_input_file(input_path: Path) -> tuple[List[Dict[str, Any]], int]:
    """
    Parse input file (JSONL or CSV) with error handling.
    Returns (rows, warning_count)
    """
    rows = []
    warn_count = 0
    
    if not input_path.exists():
        print(f"[ERROR] Input file not found: {input_path}", file=sys.stderr)
        return rows, warn_count
    
    if str(input_path).lower().endswith('.csv'):
        try:
            with open(input_path, 'r', encoding='utf-8', newline='') as f:
            reader = csv.DictReader(f)
                for idx, row in enumerate(reader, 1):
                    if not row.get('url'):
                        print(f"[WARN] Row {idx}: Missing url field, skipping", file=sys.stderr)
                    warn_count += 1
                    continue
                    rows.append(row)
        except Exception as e:
            print(f"[ERROR] Failed to parse CSV: {e}", file=sys.stderr)
            return rows, warn_count
    else:
        # Assume JSONL
        try:
            with open(input_path, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                try:
                        obj = json.loads(line)
                        if not obj.get('url'):
                            print(f"[WARN] Line {idx}: Missing url field, skipping", file=sys.stderr)
                            warn_count += 1
                            continue
                        rows.append(obj)
                    except json.JSONDecodeError as e:
                        print(f"[WARN] Line {idx}: Invalid JSON ({e}), skipping", file=sys.stderr)
                        warn_count += 1
                        continue
                except Exception as e:
            print(f"[ERROR] Failed to read JSONL: {e}", file=sys.stderr)
            return rows, warn_count
    
    return rows, warn_count


def enrich_youtube_event(event: Dict[str, Any], cache_mapping: Dict[str, Path], 
                        consent_captions: bool = False) -> Dict[str, Any]:
    """
    Enrich a single event with YouTube metadata if applicable.
    Returns event with content_meta field added.
    """
    url = event.get('url', '')
    if not url:
        return event
    
    # Check if this is a YouTube URL
    video_id = extract_video_id(url)
    if not video_id:
        # Not a YouTube URL, return event unchanged
        event['content_meta'] = None
        return event
    
    # Get profile from event
    profile = event.get('profile')
    if not profile:
        # Try to infer from browser
        browser = event.get('browser', '').lower()
        if browser == 'chrome':
            profile = 'Default'  # Fallback
        else:
            event['content_meta'] = None
            return event
    
    # Find cache directory for this profile
    cache_dir = cache_mapping.get(profile)
    if not cache_dir:
        event['content_meta'] = None
        return event
    
    # Search cache for this video
    html = search_cache_for_video(cache_dir, video_id)
    if not html:
        event['content_meta'] = None
        return event
    
    # Parse metadata
    meta = parse_yt_player_response(html)
    if not meta:
        event['content_meta'] = None
        return event
    
    # Optionally extract captions
    if consent_captions:
        caption_info = extract_captions(html)
        if caption_info:
            meta['caption_language'] = caption_info
    
    event['content_meta'] = meta
    return event


def main():
    parser = argparse.ArgumentParser(
        description='YouTube Enricher - Extract metadata from Chrome cache for YouTube URLs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --input history.jsonl --output enriched.jsonl
  %(prog)s --input history.csv --output enriched.jsonl --consent-captions
  %(prog)s --input history.jsonl --output enriched.jsonl --cache-root ~/Library/Caches/Google/Chrome --dry-run
        """
    )
    parser.add_argument('--input', required=True, type=Path,
                       help='Input JSONL or CSV file with browser history')
    parser.add_argument('--output', required=True, type=Path,
                       help='Output JSONL file with enriched events')
    parser.add_argument('--profiles-root', type=Path,
                       default=Path.home() / 'Library' / 'Application Support' / 'Google' / 'Chrome',
                       help='Chrome profiles root directory (default: ~/Library/Application Support/Google/Chrome)')
    parser.add_argument('--cache-root', type=Path,
                       default=Path.home() / 'Library' / 'Caches' / 'Google' / 'Chrome',
                       help='Chrome cache root directory (default: ~/Library/Caches/Google/Chrome)')
    parser.add_argument('--consent-captions', action='store_true',
                       help='Extract caption/transcript information if present (requires user consent)')
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
    print("[INFO] YouTube Enricher - Privacy-first, on-device processing")
    print("[INFO] Requires Full Disk Access in macOS System Settings > Privacy & Security")
    
    if not args.profiles_root.exists():
        print(f"[WARN] Profiles root not found: {args.profiles_root}", file=sys.stderr)
        print("[WARN] Grant Full Disk Access if recently denied", file=sys.stderr)
    
    if not args.cache_root.exists():
        print(f"[WARN] Cache root not found: {args.cache_root}", file=sys.stderr)
        print("[WARN] Grant Full Disk Access if recently denied", file=sys.stderr)
    
    # Parse input
    print(f"[INFO] Reading input from {args.input}...")
    events, warn_count = parse_input_file(args.input)
    
    if not events:
        print("[ERROR] No valid events found in input file", file=sys.stderr)
        sys.exit(1)
    
    print(f"[INFO] Loaded {len(events)} events (warnings: {warn_count})")
    
    # Find cache directories
    print("[INFO] Discovering Chrome cache directories...")
    cache_mapping = find_chrome_cache_directories(args.profiles_root, args.cache_root)
    
    if not cache_mapping:
        print("[WARN] No Chrome cache directories found", file=sys.stderr)
        print("[WARN] Continuing with empty cache mapping", file=sys.stderr)
    else:
        print(f"[INFO] Found {len(cache_mapping)} Chrome profile cache(s)")
        if args.verbose:
            for profile, cache_path in cache_mapping.items():
                print(f"  {profile}: {cache_path}")
    
    # Enrich events
    print("[INFO] Enriching YouTube events...")
    enriched_count = 0
    youtube_count = 0
    
    enriched_events = []
    for event in events:
        url = event.get('url', '')
        if 'youtube.com' in url or 'youtu.be' in url:
            youtube_count += 1
            enriched_event = enrich_youtube_event(event, cache_mapping, args.consent_captions)
            if enriched_event.get('content_meta'):
                enriched_count += 1
            enriched_events.append(enriched_event)
        else:
            # Non-YouTube event - add null content_meta
            event['content_meta'] = None
            enriched_events.append(event)
    
    print(f"[INFO] Processed {youtube_count} YouTube URLs, enriched {enriched_count} with metadata")
    
    if args.dry_run:
        print("[DRY RUN] Would write output to:", args.output)
        print(f"[DRY RUN] Total events: {len(enriched_events)}")
        print(f"[DRY RUN] Enriched: {enriched_count}/{youtube_count} YouTube URLs")
        sys.exit(0)
    
    # Write output
    print(f"[INFO] Writing output to {args.output}...")
    try:
    with open(args.output, 'w', encoding='utf-8') as f:
            for event in enriched_events:
                json.dump(event, f, ensure_ascii=False)
            f.write('\n')
        print(f"[INFO] Successfully wrote {len(enriched_events)} events to {args.output}")
        print(f"[INFO] Enrichment rate: {enriched_count}/{youtube_count} ({enriched_count/youtube_count*100:.1f}%)" if youtube_count > 0 else "[INFO] No YouTube URLs found")
    except Exception as e:
        print(f"[ERROR] Failed to write output: {e}", file=sys.stderr)
        sys.exit(1)
    
    sys.exit(0)


if __name__ == '__main__':
    main()
