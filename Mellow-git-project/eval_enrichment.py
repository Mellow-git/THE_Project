#!/usr/bin/env python3
import argparse, json, random, csv
from datetime import datetime
from collections import Counter, defaultdict

def is_sane_iso(ts):
    try:
        dt = datetime.fromisoformat(ts.replace('Z',''))
        this_year = datetime.now().year
        return abs(dt.year - this_year) <= 2
    except Exception:
        return False

def main():
    parser = argparse.ArgumentParser(description='Evaluate enriched_event.jsonl labels and schema.')
    parser.add_argument('--in', dest='input', required=True)
    parser.add_argument('--sample', type=int, default=200)
    parser.add_argument('--export-csv', dest='csv_out', default=None)
    args = parser.parse_args()
    with open(args.input, encoding='utf-8') as f:
        events = [json.loads(line) for line in f]
    print(f"[INFO] Loaded {len(events)} events from {args.input}")
    # Schema sanity, per-label stats
    labels, cache_hit, session_join = [], 0, 0
    bad_ts, total = [], 0
    label_to_events = defaultdict(list)
    for e in events:
        total += 1
        # Schema: must have all fields (browser, profile, url, domain, title, visited_at_iso, content_meta, transcript_excerpt, session_id, time_of_day, day_of_week, duration_hint, labels)
        if any(k not in e for k in ['browser','profile','url','domain','title','visited_at_iso','content_meta','transcript_excerpt','session_id','time_of_day','day_of_week','duration_hint','labels']):
            print(f"[WARNING] Schema miss in line {total} id={e.get('url')}")
        if e['content_meta']:
            cache_hit += 1
        if e['session_id']:
            session_join += 1
        if not e['visited_at_iso'] or not is_sane_iso(e['visited_at_iso']):
            bad_ts.append((total, e.get('visited_at_iso')))
        label = e['labels']['reason'] if e['labels'] else 'NONE'
        labels.append(label)
        label_to_events[label].append(e)
    cnt = Counter(labels)
    print(f"[CACHE] cache-based metadata coverage: {cache_hit}/{total} ({cache_hit/total*100:.1f}%)")
    print(f"[SESSION] knowledgeC join rate: {session_join}/{total} ({session_join/total*100:.1f}%)")
    print(f"[TS] Bad/missing timestamps: {bad_ts[:5]}")
    print("[LABEL COUNTS]", dict(cnt))
    # Label sampling and golden CSV
    sample_n = min(len(events), args.sample)
    events_sample = random.sample(events, sample_n)
    print("\nSampled events per label:")
    label_sam = defaultdict(list)
    for e in events_sample:
        label_sam[e['labels']['reason']].append(e)
    for label, elist in label_sam.items():
        print(f"[SAMPLE:{label}] ----")
        for e in elist[:3]:
            print(json.dumps({'url':e['url'],'title':e['title'],'reason':label,'duration':e['content_meta'].get('duration',None),'time_of_day':e['time_of_day']}, ensure_ascii=False))
        print()
    if args.csv_out:
        keys = list(events_sample[0].keys())
        with open(args.csv_out, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            for e in events_sample:
                writer.writerow(e)
        print(f"[EXPORT] wrote golden CSV for annotation: {args.csv_out}")

if __name__ == '__main__':
    main()
