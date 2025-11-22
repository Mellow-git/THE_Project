#!/usr/bin/env python3
import argparse, json

def build_instruction(meta):
    if meta.get('is_shorts') or (meta.get('duration') and meta['duration'] < 60):
        return 'Summarize in one sentence.'
    if meta.get('reason') == 'tutorial/learning':
        return 'Explain step by step how to follow this tutorial.'
    if meta.get('reason') == 'music/ambient':
        return 'Describe the style and mood of this music/audio.'
    if meta.get('reason') == 'news/information':
        return 'Summarize the key facts or event.'
    if meta.get('reason') == 'deep-dive/research':
        return 'Write a detailed summary highlighting new knowledge or findings.'
    return 'Summarize the main idea of this video.'

def build_context(e):
    # Prefer transcript if present, else summary from meta
    tx = e.get('transcript_excerpt')
    cm = e.get('content_meta') or {}
    if tx: return tx
    s = [cm.get('title',''), cm.get('channel',''), f"Published: {cm.get('publishDate','')}" if cm.get('publishDate') else '']
    if cm.get('keywords'): s.append('Keywords: '+', '.join(cm['keywords']))
    if cm.get('duration'): s.append(f'Duration: {cm["duration"]}s')
    if cm.get('views'): s.append(f'Views: {cm["views"]}')
    return ' | '.join([x for x in s if x])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--in', dest='input', required=True)
    parser.add_argument('--out', dest='output', required=True)
    parser.add_argument('--sample', type=int, default=None)
    args = parser.parse_args()
    with open(args.input, encoding='utf-8') as f:
        events = [json.loads(line) for line in f]
    out = []
    n = args.sample or len(events)
    for e in events[:n]:
        cm = e.get('content_meta') or {}
        meta = {
            'browser': e['browser'],
            'profile': e['profile'],
            'url': e['url'],
            'domain': e['domain'],
            'title': e['title'],
            'visited_at_iso': e['visited_at_iso'],
            'reason': e['labels']['reason'] if e.get('labels') else None,
            'topics': e['labels']['topics'] if e.get('labels') else [],
            'duration_hint': e.get('duration_hint')
        }
        row = {
            'instruction': build_instruction({**cm, **meta}),
            'context': build_context(e),
            'response': '',
            'meta': meta
        }
        out.append(row)
    with open(args.output, 'w', encoding='utf-8') as f:
        for r in out:
            json.dump(r, f, ensure_ascii=False)
            f.write('\n')
    print(f"[SFT] Wrote {len(out)} SFT pack rows to {args.output}")
    # Label dist
    from collections import Counter
    rc = Counter(r['meta']['reason'] for r in out)
    print('[SFT] Reason distribution:', dict(rc))

if __name__ == '__main__':
    main()
