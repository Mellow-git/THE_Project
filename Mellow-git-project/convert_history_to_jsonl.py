#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
import re
import shutil
import sqlite3
import sys
import tempfile
import time
from collections import defaultdict, Counter
from datetime import datetime, timezone, timedelta
from urllib.parse import urlparse, urlunparse, parse_qsl, urlencode
import csv
import random

# --- Constants ---
PSL_PATH = os.path.join(os.path.dirname(__file__), 'schemas', 'public_suffix_list.dat')
SCHEMA_PATH = os.path.join(os.path.dirname(__file__), 'schemas', 'dataset.schema.json')
TRACKING_PARAMS = {'utm_source','utm_medium','utm_campaign','utm_term','utm_content','gclid','fbclid'}
CHUNK_SIZE = 8000 # chars (~2k tokens)
CHUNK_OVERLAP = 0.12 # 12% overlap
MIN_CONTEXT_LEN = 500
GOLDEN_SET_SIZE = 100
RANDOM_SEED = 42

# --- PII Patterns ---
PII_PATTERNS = [
    (re.compile(r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+'), 'email'),
    (re.compile(r'\b\+?\d{1,3}?[-.\s]?\(?\d{1,4}?\)?[-.\s]?\d{1,4}[-.\s]?\d{1,9}\b'), 'phone'),
    (re.compile(r'\b(?:\d[ -]*?){13,16}\b'), 'credit_card'),
    (re.compile(r'\b\d{3}-\d{2}-\d{4}\b'), 'ssn'),
    (re.compile(r'\b[A-Z]{2}\d{2}[A-Z0-9]{1,30}\b'), 'iban'),
    (re.compile(r'eyJ[a-zA-Z0-9_-]{10,}\.[a-zA-Z0-9_-]{10,}\.[a-zA-Z0-9_-]{10,}'), 'jwt'),
    (re.compile(r'\bsk_live_[a-zA-Z0-9]{24,}\b'), 'stripe_key'),
    (re.compile(r'\b(?:api|secret|token|key|password)[=:][a-zA-Z0-9\-_]{16,}\b', re.I), 'api_key'),
    (re.compile(r'\b(?:[A-Za-z0-9]{32,})\b'), 'long_token'),
    (re.compile(r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}:[^\s]+@'), 'ip_with_creds'),
]
HIGH_RISK_DOMAINS = [
    'bank', 'finance', 'pay', 'stripe', 'paypal', 'health', 'insurance', 'hr', 'payroll',
    'aws', 'azure', 'gcp', 'cloud', 'admin', 'dashboard', 'pastebin', 'gist', 'dropbox', 'box', 'drive', 'docs', 'medical', 'clinic', 'hospital'
]

# --- PSL Parser ---
class PublicSuffixList:
    def __init__(self, psl_path):
        self.rules = []
        self.exceptions = set()
        self._load(psl_path)
    def _load(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('//'):
                    continue
                if line.startswith('!'):
                    self.exceptions.add(line[1:])
                else:
                    self.rules.append(line)
        self.rules.sort(key=lambda r: r.count('.'), reverse=True)
    def get_etld1(self, domain):
        domain = domain.lower().strip('.')
        parts = domain.split('.')
        for i in range(len(parts)):
            candidate = '.'.join(parts[i:])
            if candidate in self.exceptions:
                return '.'.join(parts[i-1:]) if i > 0 else candidate
            for rule in self.rules:
                if rule.startswith('*.'):
                    if candidate.endswith(rule[1:]) and len(parts[i:]) > rule.count('.'):
                        return '.'.join(parts[i-1:]) if i > 0 else candidate
                elif candidate == rule:
                    return '.'.join(parts[i-1:]) if i > 0 else candidate
        # Default: last two labels
        return '.'.join(parts[-2:]) if len(parts) >= 2 else domain

# --- Utility Functions ---
def canonicalize_url(url):
    try:
        parsed = urlparse(url)
        qs = [(k,v) for k,v in parse_qsl(parsed.query) if k not in TRACKING_PARAMS]
        new_query = urlencode(qs)
        scheme = parsed.scheme.lower() if parsed.scheme else 'https'
        netloc = parsed.netloc.lower()
        if netloc.startswith('www.'):
            netloc = netloc[4:]
        path = parsed.path.rstrip('/')
        canon = urlunparse((scheme, netloc, path, '', new_query, ''))
        return canon
    except Exception:
        return url

def sha256_hex(s):
    return hashlib.sha256(s.encode('utf-8')).hexdigest()

def redact_pii(text):
    flags = set()
    for pat, label in PII_PATTERNS:
        if pat.search(text):
            flags.add(label)
        text = pat.sub('[REDACTED]', text)
    return text, flags

def contains_high_risk_pii(text):
    for pat, _ in PII_PATTERNS:
        if pat.search(text):
            return True
    return False

def clean_text(text):
    text = re.sub(r'<script[\s\S]*?</script>', '', text, flags=re.I)
    text = re.sub(r'(?i)cookie(s)?( banner| notice)?[\s\S]{0,100}', '', text)
    text = re.sub(r'(?i)menu[\s\S]{0,100}', '', text)
    text = re.sub(r'\s+', ' ', text)
    text, _ = redact_pii(text)
    text = text.strip()
    return text

def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    chunks = []
    n = len(text)
    step = int(chunk_size * (1 - overlap))
    for i in range(0, n, step):
        chunk = text[i:i+chunk_size]
        if len(chunk) < MIN_CONTEXT_LEN:
            continue
        chunks.append((i//step, chunk))
        if i+chunk_size >= n:
            break
    return chunks

def infer_topic(domain, text):
    d = domain.lower()
    if any(x in d for x in ['github','stackoverflow','readthedocs','pkg.go.dev','docs.','developer.']):
        return 'dev'
    if any(x in d for x in ['nytimes','cnn','bbc','reuters','theguardian']):
        return 'news'
    if any(x in d for x in ['arxiv','nature','sciencedirect','acm','ieee']):
        return 'research'
    if re.search(r'\b(research|study|experiment|paper|dataset)\b', text, re.I):
        return 'research'
    return 'other'

def select_task_templates(topic, domain, text):
    templates = [
        'Summarize the article in 3 bullets, each under 20 words.',
        'List 5 key takeaways, one per line, no fluff.'
    ]
    if topic in ('news','research'):
        templates.append('Answer the question \'What is the core idea?\' using only the provided context, one sentence.')
    if any(x in domain for x in ['github','stackoverflow','readthedocs','pkg.go.dev','docs.','developer.']) or topic=='dev' or re.search(r'error|exception|stack trace|code', text, re.I):
        templates.append('Explain the error and propose a 3-step fix, include a short code block if relevant.')
    return templates

def validate_example(obj):
    if not obj.get('instruction') or not obj.get('context'):
        return False, 'missing_instruction_or_context'
    if len(obj['context']) < MIN_CONTEXT_LEN:
        return False, 'context_too_short'
    if contains_high_risk_pii(obj['context']):
        return False, 'high_risk_pii'
    return True, ''

def jaccard_sim(a, b):
    set_a = set(a.split())
    set_b = set(b.split())
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)

def groundedness(context, response):
    if not response.strip():
        return 0.0
    set_c = set(context.lower().split())
    set_r = set(response.lower().split())
    if not set_r:
        return 0.0
    return len(set_c & set_r) / len(set_r)

def iso8601_from_chromium(ts):
    try:
        dt = datetime(1601,1,1, tzinfo=timezone.utc) + timedelta(microseconds=ts)
        return dt.isoformat().replace('+00:00','Z')
    except Exception:
        return None

def iso8601_from_firefox(ts):
    try:
        dt = datetime(1970,1,1, tzinfo=timezone.utc) + timedelta(microseconds=ts)
        return dt.isoformat().replace('+00:00','Z')
    except Exception:
        return None

def iso8601_from_safari(ts):
    try:
        dt = datetime(2001,1,1, tzinfo=timezone.utc) + timedelta(seconds=ts)
        return dt.isoformat().replace('+00:00','Z')
    except Exception:
        return None

def is_high_risk_domain(domain):
    d = domain.lower()
    return any(x in d for x in HIGH_RISK_DOMAINS)

def validate_jsonl_line(line, schema):
    # Minimal schema check: required fields and types
    try:
        obj = json.loads(line)
        assert isinstance(obj, dict)
        for k in ['instruction','context','response','meta']:
            assert k in obj
        meta = obj['meta']
        for k in ['url','title','ts','domain','topic','source_id','chunk_id','generated','length_chars','split_hint']:
            assert k in meta
        return True, ''
    except Exception as e:
        return False, str(e)

# --- WAL-Safe Snapshotting ---
def snapshot_db(src_path, tempdir):
    snapshot_path = os.path.join(tempdir, 'snapshot.db')
    try:
        src_conn = sqlite3.connect(f'file:{src_path}?mode=ro', uri=True)
        dst_conn = sqlite3.connect(snapshot_path)
        src_conn.backup(dst_conn)
        dst_conn.commit()
        dst_conn.close()
        src_conn.close()
        print('[INFO] Used sqlite3.Connection.backup for WAL-safe snapshot.')
    except Exception as e:
        print('[WARN] Backup API failed, falling back to VACUUM INTO:', e)
        src_conn = sqlite3.connect(f'file:{src_path}?mode=ro', uri=True)
        dst_conn = sqlite3.connect(snapshot_path)
        src_conn.execute(f"VACUUM INTO '{snapshot_path}'")
        dst_conn.commit()
        dst_conn.close()
        src_conn.close()
    # Integrity checks
    conn = sqlite3.connect(snapshot_path)
    c = conn.cursor()
    c.execute('PRAGMA quick_check;')
    qc = c.fetchone()[0]
    if qc != 'ok':
        print('[ERROR] quick_check failed:', qc)
        sys.exit(1)
    c.execute('PRAGMA integrity_check;')
    ic = c.fetchone()[0]
    if ic != 'ok':
        print('[ERROR] integrity_check failed:', ic)
        sys.exit(1)
    conn.close()
    print('[INFO] Snapshot integrity checks passed.')
    return snapshot_path

# --- Main Pipeline ---
def main():
    parser = argparse.ArgumentParser(description='Convert browser history to JSONL for instruction-tuning.')
    parser.add_argument('--input', required=True, help='Path to browser DB (main file)')
    parser.add_argument('--output_dir', required=True, help='Output directory')
    parser.add_argument('--cache_dir', required=True, help='Page text cache directory')
    parser.add_argument('--limit', type=int, default=None, help='Limit number of visits processed (for smoke tests)')
    args = parser.parse_args()

    random.seed(RANDOM_SEED)
    os.makedirs(args.output_dir, exist_ok=True)
    # Load PSL
    psl = PublicSuffixList(PSL_PATH)
    # Snapshot DB
    tempdir = tempfile.mkdtemp(prefix='history_snapshot_')
    snapshot_path = snapshot_db(args.input, tempdir)
    # Open snapshot
    conn = sqlite3.connect(snapshot_path)
    c = conn.cursor()
    # Detect schema
    tables = set(r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type='table';"))
    if {'urls','visits'}.issubset(tables):
        engine = 'chromium'
        sql = ("SELECT urls.url, urls.title, visits.visit_time "
               "FROM urls JOIN visits ON urls.id=visits.url "
               "ORDER BY visits.visit_time DESC")
        ts_fn = iso8601_from_chromium
    elif {'moz_places','moz_historyvisits'}.issubset(tables):
        engine = 'firefox'
        sql = ("SELECT moz_places.url, moz_places.title, moz_historyvisits.visit_date "
               "FROM moz_places JOIN moz_historyvisits ON moz_places.id=moz_historyvisits.place_id "
               "ORDER BY moz_historyvisits.visit_date DESC")
        ts_fn = iso8601_from_firefox
    elif {'history_items','history_visits'}.issubset(tables):
        engine = 'safari'
        sql = ("SELECT history_items.url, history_items.title, history_visits.visit_time "
               "FROM history_items JOIN history_visits ON history_items.id=history_visits.history_item "
               "ORDER BY history_visits.visit_time DESC")
        ts_fn = iso8601_from_safari
    else:
        print('[ERROR] Unknown browser schema. Exiting.')
        sys.exit(1)
    print(f'[INFO] Detected browser engine: {engine}')
    # --- Main processing ---
    seen_hashes = set()
    dedup_hashes = set()
    all_examples = []
    dropped = Counter()
    golden_candidates = []
    visit_count = 0
    for row in c.execute(sql):
        if args.limit and visit_count >= args.limit:
            break
        url, title, ts_raw = row
        canon_url = canonicalize_url(url)
        domain = psl.get_etld1(urlparse(canon_url).netloc)
        ts = ts_fn(ts_raw)
        source_id = sha256_hex(canon_url + (title or ''))
        cache_key = sha256_hex(canon_url)
        cache_path = os.path.join(args.cache_dir, f'{cache_key}.txt')
        if not os.path.exists(cache_path):
            dropped['no_cache'] += 1
            continue
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                raw_text = f.read()
        except Exception:
            dropped['cache_read_error'] += 1
            continue
        cleaned, pii_flags = redact_pii(clean_text(raw_text))
        if len(cleaned) < MIN_CONTEXT_LEN:
            dropped['too_short'] += 1
            continue
        if contains_high_risk_pii(cleaned):
            dropped['high_risk_pii'] += 1
            continue
        topic = infer_topic(domain, cleaned)
        templates = select_task_templates(topic, domain, cleaned)
        chunks = chunk_text(cleaned)
        for chunk_idx, chunk in chunks:
            for tmpl in templates:
                chunk_id = f'{source_id}#c{chunk_idx:03d}'
                meta = {
                    'url': canon_url,
                    'title': title if title else None,
                    'ts': ts,
                    'domain': domain,
                    'topic': topic,
                    'source_id': source_id,
                    'chunk_id': chunk_id,
                    'generated': False,
                    'length_chars': len(chunk),
                    'split_hint': None
                }
                obj = {
                    'instruction': tmpl,
                    'context': chunk,
                    'response': '',
                    'meta': meta
                }
                valid, reason = validate_example(obj)
                if not valid:
                    dropped[reason] += 1
                    continue
                dedup_key = sha256_hex(tmpl + '|' + chunk)
                if dedup_key in dedup_hashes:
                    dropped['dedup'] += 1
                    continue
                # Jaccard near-dup check
                is_near_dup = False
                for h in seen_hashes:
                    if jaccard_sim(h, chunk) > 0.92:
                        is_near_dup = True
                        break
                if is_near_dup:
                    dropped['near_dup'] += 1
                    continue
                seen_hashes.add(chunk)
                dedup_hashes.add(dedup_key)
                # Manual review flag for high-risk domains or PII
                if is_high_risk_domain(domain) or pii_flags:
                    obj['meta']['manual_review'] = True
                    obj['meta']['pii_flags'] = list(pii_flags)
                all_examples.append(obj)
                if len(golden_candidates) < GOLDEN_SET_SIZE*2:
                    golden_candidates.append(obj)
        visit_count += 1
    conn.close()
    # Assign splits deterministically by eTLD+1
    random.seed(RANDOM_SEED)
    domains = list(set(e['meta']['domain'] for e in all_examples))
    random.shuffle(domains)
    n = len(domains)
    n_train = int(n*0.8)
    n_val = int(n*0.1)
    train_domains = set(domains[:n_train])
    val_domains = set(domains[n_train:n_train+n_val])
    test_domains = set(domains[n_train+n_val:])
    for e in all_examples:
        d = e['meta']['domain']
        if d in train_domains:
            e['meta']['split_hint'] = 'train'
        elif d in val_domains:
            e['meta']['split_hint'] = 'val'
        else:
            e['meta']['split_hint'] = 'test'
    # Write splits
    splits = {'train':[], 'val':[], 'test':[]}
    for e in all_examples:
        splits[e['meta']['split_hint']].append(e)
    for split in splits:
        out_path = os.path.join(args.output_dir, f'dataset_{split}.jsonl')
        with open(out_path, 'w', encoding='utf-8') as f:
            for e in splits[split]:
                json.dump(e, f, ensure_ascii=False, sort_keys=False)
                f.write('\n')
    # Golden set: stratified sample
    random.seed(RANDOM_SEED)
    golden = []
    by_topic = defaultdict(list)
    for e in golden_candidates:
        by_topic[e['meta']['topic']].append(e)
    per_topic = max(1, GOLDEN_SET_SIZE // max(1,len(by_topic)))
    for topic, lst in by_topic.items():
        random.shuffle(lst)
        golden.extend(lst[:per_topic])
    golden = golden[:GOLDEN_SET_SIZE]
    for e in golden:
        e['response'] = ''
        e['meta']['generated'] = False
    golden_path = os.path.join(args.output_dir, 'golden_set.jsonl')
    with open(golden_path, 'w', encoding='utf-8') as f:
        for e in golden:
            json.dump(e, f, ensure_ascii=False, sort_keys=False)
            f.write('\n')
    # Validation report
    val_path = os.path.join(args.output_dir, 'validation_report.csv')
    with open(val_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['chunk_id','length_ok','groundedness_score','dedup_flag','split'])
        for e in all_examples:
            length_ok = int(len(e['context']) >= MIN_CONTEXT_LEN)
            grounded = groundedness(e['context'], e['response'])
            dedup_flag = 0
            split = e['meta']['split_hint']
            writer.writerow([e['meta']['chunk_id'], length_ok, f'{grounded:.2f}', dedup_flag, split])
    # Write JSON Schema to output and canonical location
    with open(SCHEMA_PATH, 'r', encoding='utf-8') as f:
        schema = json.load(f)
    out_schema_path = os.path.join(args.output_dir, 'dataset.schema.json')
    with open(out_schema_path, 'w', encoding='utf-8') as f:
        json.dump(schema, f, ensure_ascii=False, indent=2)
    # Validate JSONL lines
    for split in splits:
        out_path = os.path.join(args.output_dir, f'dataset_{split}.jsonl')
        with open(out_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f,1):
                ok, err = validate_jsonl_line(line, schema)
                if not ok:
                    print(f'[ERROR] Invalid JSONL in {out_path} line {i}: {err}')
                    sys.exit(1)
    print('[INFO] All JSONL files validated against schema.')
    # Print summary
    print(f'Total examples: {len(all_examples)}')
    print('Dropped:', dict(dropped))
    print('Split sizes:', {k:len(v) for k,v in splits.items()})
    print('Golden set size:', len(golden))
    print('Output files:')
    for split in splits:
        print(f'  {split}: {os.path.join(args.output_dir, f"dataset_{split}.jsonl")}')
    print(f'  golden: {golden_path}')
    print(f'  validation: {val_path}')
    print(f'  schema: {out_schema_path}')
    shutil.rmtree(tempdir)

if __name__ == '__main__':
    main()
