#!/usr/bin/env python3

import argparse, sqlite3, tempfile, os, sys, shutil, csv, json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional

def print_permissions_note():
    print("""\n[INFO] This tool reads browser history from ~/Library. \nFor macOS 10.15+ you must add this Terminal (or your app) to\nSystem Settings > Privacy & Security > Full Disk Access.\n""")

def find_safari_history(safari_path: Optional[str]) -> Optional[Path]:
    candidates = [Path(safari_path)] if safari_path else [Path.home() / 'Library' / 'Safari' / 'History.db']
    for p in candidates:
        if p.exists():
            return p
    return None

def find_chrome_profiles(chrome_base: Optional[str]) -> list:
    profiles = []
    default_base = Path(chrome_base) if chrome_base else Path.home() / 'Library' / 'Application Support' / 'Google' / 'Chrome'
    if not default_base.exists():
        return []
    for d in default_base.iterdir():
        if d.is_dir() and (d.name == "Default" or d.name.startswith("Profile ")):
            db_path = d / 'History'
            if db_path.exists():
                profiles.append((d.name, db_path))
    return profiles

def snapshot_sqlite(src_path: Path, appname: str = "collect_history") -> Path:
    cache_dir = Path(tempfile.gettempdir()) / f"{appname}_snapshots"
    cache_dir.mkdir(parents=True, exist_ok=True)
    snapshot = cache_dir / f"{src_path.name}.snapshot"
    try:
        src = sqlite3.connect(f"file:{src_path}?mode=ro", uri=True)
        dst = sqlite3.connect(snapshot)
        src.backup(dst, pages=1000)
        dst.commit(); dst.close(); src.close()
        return snapshot
    except sqlite3.OperationalError as e:
        raise RuntimeError(f"WAL-safe snapshot failed for {src_path}: {e}")

def chrome_rows(snapshot: Path, limit=1000):
    sql = f"SELECT url, title, last_visit_time, visit_count FROM urls ORDER BY last_visit_time DESC LIMIT ?"
    con = sqlite3.connect(snapshot)
    rows = []
    for url, title, t_micro, count in con.execute(sql, (limit,)):
        unix_secs = t_micro / 1e6 - 11644473600
        visited_at_iso = None
        try:
            visited_at_iso = datetime.utcfromtimestamp(unix_secs).isoformat(timespec="seconds") + "Z"
        except: pass
        rows.append(dict(browser="chrome", profile=None, url=url, title=title, visited_at_iso=visited_at_iso,
                        visit_count=count, source_path=str(snapshot), snapshot_path=str(snapshot)))
    con.close()
    return rows

def safari_rows(snapshot: Path, limit=1000):
    sql = ("SELECT hi.url AS url, COALESCE(hv.title, '') AS title, hv.visit_time "
           "FROM history_visits hv "
           "JOIN history_items hi ON hi.id = hv.history_item "
           "ORDER BY hv.visit_time DESC LIMIT ?")
    con = sqlite3.connect(snapshot)
    rows = []
    for url, title, visit_time in con.execute(sql, (limit,)):
        try:
            # Safari: visit_time is seconds since 2001-01-01, convert to UNIX epoch
            unix_time = visit_time + 978307200
            visited = datetime.utcfromtimestamp(unix_time)
            visited_at_iso = visited.isoformat(timespec="seconds")+"Z"
        except: 
            visited_at_iso = None
        rows.append(dict(browser="safari", profile="Default", url=url, title=title or '',
                         visited_at_iso=visited_at_iso, visit_count=None, 
                         source_path=str(snapshot), snapshot_path=str(snapshot)))
    con.close()
    return rows

def write_jsonl(rows, out=sys.stdout):
    for r in rows:
        print(json.dumps(r, ensure_ascii=False), file=out)

def write_csv(rows, out=sys.stdout):
    fieldnames = ["browser", "profile", "url", "title", "visited_at_iso", "visit_count", "source_path", "snapshot_path"]
    writer = csv.DictWriter(out, fieldnames=fieldnames)
    writer.writeheader()
    for r in rows:
        writer.writerow(r)

def main():
    parser = argparse.ArgumentParser(description="WAL-safe browser history collector (macOS only)")
    parser.add_argument("--browser", default="both", choices=["safari", "chrome", "both"])
    parser.add_argument("--out", default="jsonl", choices=["jsonl", "csv"])
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--profiles", help="Override Chrome profiles base directory")
    parser.add_argument("--safari-path", help="Override Safari History.db path")
    parser.add_argument("--run-tests", action="store_true")
    args = parser.parse_args()

    if args.run_tests:
        run_tests()
        sys.exit(0)

    print_permissions_note()

    work_done = False
    all_rows = []

    if args.browser in ("safari", "both"):
        safari_db = find_safari_history(args.safari_path)
        if not safari_db:
            print("Safari History.db not found or access denied.")
        else:
            try:
                snap = snapshot_sqlite(safari_db)
                work_done = True
                con = sqlite3.connect(snap)
                count = con.execute("SELECT COUNT(*) FROM history_visits").fetchone()[0]
                print(f"[Safari] Found {count} history rows in snapshot {snap}")
                if not args.dry_run:
                    all_rows.extend(safari_rows(snap, args.limit))
                con.close()
            except Exception as e:
                print(f"[Safari] ERROR: {e}")
    if args.browser in ("chrome", "both"):
        chrome_profiles = find_chrome_profiles(args.profiles)
        if not chrome_profiles:
            print("No Chrome profiles with History found.")
        for prof_name, prof_db in chrome_profiles:
            try:
                snap = snapshot_sqlite(prof_db)
                work_done = True
                con = sqlite3.connect(snap)
                count = con.execute("SELECT COUNT(*) FROM urls").fetchone()[0]
                print(f"[Chrome:{prof_name}] Found {count} url rows in snapshot {snap}")
                if not args.dry_run:
                    r = chrome_rows(snap, args.limit)
                    for row in r:
                        row['profile'] = prof_name
                    all_rows.extend(r)
                con.close()
            except Exception as e:
                print(f"[Chrome:{prof_name}] ERROR: {e}")
    if args.dry_run:
        sys.exit(0 if work_done else 1)
    # Write output
    if args.out == "jsonl":
        write_jsonl(all_rows)
    else:
        write_csv(all_rows)
    print(f"[INFO] Exported {len(all_rows)} rows.")

def run_tests():
    print("Running smoke/unit tests (skipping if no source files)...")
    # Test timestamp conversion (Chrome)
    t_micro = 13330835056789012
    unix_secs = t_micro / 1e6 - 11644473600
    iso = datetime.utcfromtimestamp(unix_secs).isoformat(timespec="seconds")+"Z"
    assert iso.startswith("2022-") or iso.startswith("2023-") or iso.startswith("2024-"), f"Chrome timestamp test failed: {iso}"
    # Test timestamp conversion (Safari)
    visit_time = 750664177  # seconds since 2001-01-01T00:00:00
    origin = datetime.utcfromtimestamp(978307200)
    iso2 = (origin + timedelta(seconds=visit_time)).isoformat(timespec="seconds")+"Z"
    assert iso2.startswith("2024-") or iso2.startswith("2023-") or iso2.startswith("2022-"), f"Safari timestamp test failed: {iso2}"
    # Test CSV header
    import io
    rows = [dict(browser='chrome', profile='Profile 1', url='u', title='t', visited_at_iso='iso', visit_count=1, source_path='s', snapshot_path='s')]
    f = io.StringIO()
    write_csv(rows, out=f)
    assert 'browser,profile,url,title,visited_at_iso,visit_count,source_path,snapshot_path' in f.getvalue(), "CSV header test failed"
    print("All unit tests passed.")

if __name__ == "__main__":
    main()
