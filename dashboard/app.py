"""Prophet Mind: read-only, local visualizations of a Rhizome ledger."""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
from collections import Counter, OrderedDict
from datetime import datetime, timedelta, timezone
from functools import lru_cache
import os
from pathlib import Path
import re
import sqlite3
import threading
import time

import numpy as np
from flask import Flask, jsonify, request, send_from_directory, render_template
from jinja2 import ChoiceLoader, FileSystemLoader
import requests
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
from gensim.models import Word2Vec

MODEL_LOCK = threading.Lock()
MODEL_DIR = Path(os.environ.get("MIND_MODEL_DIR", "~/.cache/prophet-mind/models")).expanduser()
TOKEN = re.compile(r"(?u)\b[^\W\d_][^\W\d_\-]{1,}\b")

STATIC = Path(__file__).parent / 'static'
STOP = sorted(set(ENGLISH_STOP_WORDS) | {'https', 'http', 'www', 'com', 'org', 've', 'll', 'don', 'doesn', 'isn', 'aren', 'wasn', 'wouldn', 'didn', 'couldn', 'shouldn', 're', 'amp'})
LINK = re.compile(r'https?://\S+')


def day_of(value):
    parsed = datetime.fromisoformat(value.replace('Z', '+00:00'))
    return (parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed.astimezone(timezone.utc)).date()


def integer(value, default, low, high):
    try:
        result = int(value) if value is not None else default
    except (TypeError, ValueError):
        raise ValueError('Expected a whole number') from None
    if not low <= result <= high:
        raise ValueError(f'Expected a number from {low} to {high}')
    return result


class Corpus:
    def __init__(self, rows):
        self.rows = rows
        self.vectorizer = CountVectorizer(stop_words=STOP, max_features=5000, min_df=2 if len(rows) > 10 else 1,
            token_pattern=r'(?u)\b[^\W\d_][^\W\d_\-]{2,}\b', dtype=np.float32)
        try:
            self.counts = self.vectorizer.fit_transform([LINK.sub(' ', r['thought']) for r in rows])
            self.terms = self.vectorizer.get_feature_names_out().tolist()
        except ValueError:
            from scipy.sparse import csr_matrix
            self.counts = csr_matrix((len(rows), 0), dtype=np.float32)
            self.terms = []
        self.lookup = {word: index for index, word in enumerate(self.terms)}
        self.totals = np.asarray(self.counts.sum(axis=0)).ravel()
        self.presence = self.counts.copy()
        self.presence.data[:] = 1
        self.model = None
        self.model_info = None
        self.neighbors = lru_cache(maxsize=128)(self.neighbors)
        self.days = [day_of(r['timestamp']) for r in rows]
        self.top = sorted(range(len(self.terms)), key=lambda i: (-float(self.totals[i]), self.terms[i]))

    def words(self, limit=60):
        return [{'term': self.terms[i], 'count': int(self.totals[i])} for i in self.top[:limit]]

    def ensure_model(self):
        if self.model is not None:
            return
        with MODEL_LOCK:
            if self.model is not None:
                return
            fingerprint = hashlib.sha256(b'word2vec-sg-128-window6-min3-epochs20-v1')
            for row in self.rows:
                fingerprint.update(json.dumps([row['id'], row['thought']], ensure_ascii=False).encode())
            cache = MODEL_DIR / (fingerprint.hexdigest() + '.model')
            MODEL_DIR.mkdir(parents=True, exist_ok=True, mode=0o700)
            started = time.perf_counter()
            if cache.exists():
                model = Word2Vec.load(str(cache))
            else:
                sentences = [TOKEN.findall(LINK.sub(' ', r['thought']).lower()) for r in self.rows]
                sentences = [s for s in sentences if len(s) >= 2]
                if not sentences:
                    raise ValueError('This selection has too little text to train Word2Vec.')
                model = Word2Vec(vector_size=128, window=6,
                    min_count=3 if len(sentences) >= 100 else 1, workers=4,
                    sg=1, negative=10, epochs=20, seed=42)
                model.build_vocab(sentences)
                if not len(model.wv):
                    raise ValueError('This selection has too little repeated language for Word2Vec.')
                model.train(sentences, total_examples=model.corpus_count, epochs=model.epochs)
                model.save(str(cache))
                cache.chmod(0o600)
            if not np.isfinite(model.wv.vectors).all():
                raise ValueError('Word2Vec returned non-finite vectors; model is unavailable.')
            self.model = model
            self.common_words = set(self.terms)
            eligible = [word for word in model.wv.index_to_key if len(word) >= 3 and word not in STOP and TOKEN.fullmatch(word)]
            self.vector_terms = eligible
            self.vector_matrix = np.stack([model.wv[word] for word in eligible]) if eligible else np.zeros((0,128))
            self.vector_matrix /= np.maximum(np.linalg.norm(self.vector_matrix, axis=1, keepdims=True), 1e-12)
            self.model_info = {'algorithm': 'Word2Vec skip-gram', 'dimensions': 128, 'epochs': 20,
                'workers': 4, 'device': 'CPU', 'vocabulary': len(model.wv),
                'thoughts': len(self.rows), 'load_or_train_ms': round((time.perf_counter()-started)*1000,1)}

    def neighbors(self, term):
        self.ensure_model()
        if term not in self.model.wv:
            return []
        vector = self.model.wv[term]
        vector = vector / max(np.linalg.norm(vector), 1e-12)
        scores = self.vector_matrix @ vector
        return [{'term': self.vector_terms[i], 'score': round(float(scores[i]), 4),
                 'count': int(self.model.wv.get_vecattr(self.vector_terms[i], 'count')),
                 'fruit': self.vector_terms[i] not in self.common_words}
                for i in np.argsort(-scores, kind='stable')[:100]
                if scores[i] > 0 and self.vector_terms[i] != term][:80]

    def tree(self, seed, depth=5, width=3):
        seed = seed.strip().lower()
        self.ensure_model()
        if seed not in self.model.wv:
            return {'root': None, 'words': [], 'suggestions': self.words(12),
                    'message': f'“{seed}” has no trained vector in this selection. Try a suggested concept.'}
        used = {seed}
        words = [{'term': seed, 'level': 0, 'count': int(self.model.wv.get_vecattr(seed,'count')), 'fruit': False, 'score': 1.0}]
        def branch(term, level, relation=None):
            node = {'term': term, 'count': int(self.model.wv.get_vecattr(term, 'count')), 'children': [], **(relation or {})}
            if level < depth and not node.get('fruit') and len(words) < 130:
                neighbors = self.neighbors(term)
                familiar = [r for r in neighbors if not r['fruit'] and r['term'] not in used][:width]
                novel = next((r for r in neighbors if r['fruit'] and r['term'] not in used), None)
                choices = familiar
                # The original fruit is the closest out-of-lexicon word when
                # it outranks the closest familiar word; it is not expanded.
                if novel and (not familiar or novel['score'] > familiar[0]['score']):
                    choices = [novel, *familiar]
                for related in choices:
                    if related['term'] in used or len(words) >= 130:
                        continue
                    used.add(related['term'])
                    words.append({**related, 'level': level+1})
                    node['children'].append(branch(related['term'], level + 1, related))
            return node
        root = branch(seed, 0)
        return {'root': root, 'words': words, 'concepts': len(used), 'model': self.model_info,
                'fruits': sum(w['fruit'] for w in words),
                'method': 'Word2Vec semantic tree rendered as the original recenterable cloud. Blue fruits are outside the selected record’s frequent-word lexicon.'}

    def stream(self, selected):
        if not self.rows:
            return {'buckets': [], 'terms': [], 'unit': 'day', 'top_terms': [], 'total': 0}
        duration = (max(self.days) - min(self.days)).days
        unit = 'day' if duration <= 120 else 'week' if duration <= 900 else 'month'
        def floor(d):
            return d.replace(day=1) if unit == 'month' else d - timedelta(days=d.weekday()) if unit == 'week' else d
        def advance(d):
            return (d.replace(year=d.year + 1, month=1) if d.month == 12 else d.replace(month=d.month + 1)) if unit == 'month' else d + timedelta(days=7 if unit == 'week' else 1)
        keys, cursor, last = [], floor(min(self.days)), floor(max(self.days))
        while cursor <= last:
            keys.append(cursor.isoformat()); cursor = advance(cursor)
        slots = {value: i for i, value in enumerate(keys)}
        indices = [self.lookup[w] for w in selected if w in self.lookup][:18]
        if not indices:
            indices = self.top[:12]
        buckets = np.zeros((len(keys), len(indices)), dtype=np.float32)
        totals = np.zeros(len(keys), dtype=np.int64)
        matrix = self.counts[:, indices].tocsr()
        # O(number of selected nonzero word counts), not days × all words × thoughts.
        positions = np.array([slots[floor(d).isoformat()] for d in self.days])
        np.add.at(totals, positions, 1)
        coo = matrix.tocoo()
        np.add.at(buckets, (positions[coo.row], coo.col), coo.data)
        return {'buckets': [{'date': key, 'values': [int(x) for x in buckets[i]], 'thoughts': int(totals[i])} for i, key in enumerate(keys)],
                'terms': [self.terms[i] for i in indices], 'unit': unit, 'top_terms': self.words(), 'total': len(self.rows), 'vocabulary': len(self.terms), 'start': min(self.days).isoformat(), 'end': max(self.days).isoformat()}

    def summary(self, start='', end='', selected=()):
        left = day_of(start) if start else min(self.days, default=None)
        right = day_of(end) if end else max(self.days, default=None)
        if left and right and left > right:
            raise ValueError('Start date must precede end date')
        indices = [i for i,d in enumerate(self.days) if (not left or d >= left) and (not right or d <= right)]
        counts = np.asarray(self.counts[indices].sum(axis=0)).ravel()
        ranked = sorted((i for i,v in enumerate(counts) if v > 0), key=lambda i:(-float(counts[i]),self.terms[i]))[:60]
        for term in selected:
            index = self.lookup.get(term)
            if index is not None and index not in ranked:
                ranked.append(index)
        return {'total':len(indices),'vocabulary':int(np.count_nonzero(counts)),
                'top_terms':[{'term':self.terms[i],'count':int(counts[i])} for i in ranked],
                'start':left.isoformat() if left else None,'end':right.isoformat() if right else None}

    def thoughts(self, term, related, offset):
        mask = np.ones(len(self.rows), dtype=bool)
        for word in [term, related]:
            if word:
                index = self.lookup.get(word.lower())
                if index is None:
                    mask &= np.fromiter((word.lower() in TOKEN.findall(LINK.sub(' ', r['thought']).lower()) for r in self.rows), dtype=bool, count=len(self.rows))
                else:
                    mask &= self.presence[:, index].toarray().ravel() > 0
        indices = np.flatnonzero(mask)[::-1]
        result = [self.rows[i] for i in indices[offset:offset + 30]]
        return {'thoughts': result, 'total': len(indices), 'more': offset + 30 < len(indices)}


class Ledger:
    def __init__(self, path):
        self.path = Path(path).expanduser().resolve()
        self.lock = threading.RLock()
        self.stamp = None
        self.checked = 0
        self.rows = []
        self.cache = OrderedDict()
        self.revision = 0

    def signature(self):
        result = []
        for path in [self.path, Path(str(self.path) + '-wal')]:
            try:
                stat = path.stat(); result.append((stat.st_mtime_ns, stat.st_size))
            except FileNotFoundError:
                result.append(None)
        if result[0] is None:
            raise ValueError('Ledger not found. Set RHIZOME_FOURTHOUGHT_DB to an existing SQLite ledger.')
        return tuple(result)

    def refresh(self):
        with self.lock:
            if time.monotonic() - self.checked < 1:
                return
            signature = self.signature()
            self.checked = time.monotonic()
            if signature == self.stamp:
                return
            conn = sqlite3.connect(self.path.as_uri() + '?mode=ro', uri=True)
            conn.row_factory = sqlite3.Row
            try:
                rows = [dict(r) for r in conn.execute("SELECT id,speaker_id,username,thought,timestamp,type,domain,verity,valence,tier,reply_to,references_json FROM fourthought_logs WHERE tier IN ('local','global') ORDER BY timestamp,id")]
                # Vote entries repeat the parent's text/type; keep actual authored replies.
                parents = {r['id']: (r['thought'], r['type']) for r in rows}
                self.rows = [r for r in rows if parents.get(r['reply_to']) != (r['thought'], r['type'])]
            finally:
                conn.close()
            self.stamp = signature
            self.cache.clear()
            self.revision += 1

    def overview(self):
        self.refresh()
        with self.lock:
            speakers = {}
            for row in self.rows:
                sid = row['speaker_id']
                if sid not in speakers:
                    speakers[sid] = {'id': sid, 'name': row['username'], 'count': 0}
                speakers[sid]['count'] += 1
            ordered = sorted(speakers.values(), key=lambda s: -s['count'])
            days = [day_of(r['timestamp']) for r in self.rows]
            return {'speakers': ordered, 'domains': sorted({r['domain'] for r in self.rows}),
                    'start': min(days).isoformat() if days else None, 'end': max(days).isoformat() if days else None,
                    'total': len(self.rows), 'revision': self.revision,
                    'scope': 'Local and global authored thoughts. Private thoughts and duplicate votes are excluded. Dates use UTC.'}

    def corpus(self, args):
        self.refresh()
        key = tuple((args.get(k) or '').strip() for k in ['speaker', 'domain', 'start', 'end', 'q'])
        speaker, domain, start, end, query = key
        if start: day_of(start)
        if end: day_of(end)
        if start and end and start > end:
            raise ValueError('Start date must precede end date')
        with self.lock:
            if speaker and not any(r['speaker_id'] == speaker for r in self.rows):
                raise ValueError('Unknown speaker ID')
            if key in self.cache:
                self.cache.move_to_end(key)
                return self.cache[key]
            rows = [r for r in self.rows if (not speaker or r['speaker_id'] == speaker)
                and (not domain or r['domain'] == domain) and (not start or day_of(r['timestamp']).isoformat() >= start)
                and (not end or day_of(r['timestamp']).isoformat() <= end) and (not query or query.lower() in r['thought'].lower())]
            result = Corpus(rows)
            self.cache[key] = result
            while len(self.cache) > 4:
                self.cache.popitem(last=False)
            return result


def create_app(db, node_url=None):
    app = Flask(__name__, static_folder=str(STATIC), static_url_path='/static')
    ledger = Ledger(db)
    app.config['LEDGER'] = ledger
    from rhizome.blueprint import rhizome_static_blueprint
    app.register_blueprint(rhizome_static_blueprint())
    import rhizome
    app.jinja_loader = ChoiceLoader([FileSystemLoader(str(STATIC)), FileSystemLoader(str(Path(rhizome.__file__).parent / 'templates'))])
    app.config['RHIZOME_NODE_URL'] = node_url

    @app.before_request
    def local_only():
        from urllib.parse import urlsplit
        if request.host.split(':')[0] not in {'127.0.0.1', 'localhost'}:
            return jsonify(error='This dashboard is local only'), 403
        origin = request.headers.get('Origin')
        if origin and urlsplit(origin).netloc != request.host:
            return jsonify(error='Cross-origin access is not permitted'), 403
        if request.headers.get('Sec-Fetch-Site') == 'cross-site':
            return jsonify(error='Cross-site access is not permitted'), 403

    @app.after_request
    def protect(response):
        response.headers['Cache-Control'] = 'no-store'
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['Referrer-Policy'] = 'no-referrer'
        hashes = []
        if response.mimetype == 'text/html':
            for script in re.findall(r'<script>(.*?)</script>', response.get_data(as_text=True), re.S):
                hashes.append("'sha256-"+base64.b64encode(hashlib.sha256(script.encode()).digest()).decode()+"'")
        response.headers['Content-Security-Policy'] = "default-src 'self'; script-src 'self' "+' '.join(hashes)+"; style-src 'self' 'unsafe-inline'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'"
        return response

    @app.errorhandler(ValueError)
    def invalid(exc):
        return jsonify(error=str(exc)), 400

    @app.get('/')
    def index():
        context = {'node_id':'','iris_name':'Iris','iris_speaker_id':'','accounts_enabled':False,'account':None,'csrf_token':''}
        if app.config['RHIZOME_NODE_URL']:
            try:
                headers = {k:request.headers[k] for k in ('Cookie','Authorization') if k in request.headers}
                source = app.config['RHIZOME_NODE_URL'].rstrip('/')
                manifest = requests.get(source+'/.well-known/rhizome',timeout=3).json()
                account = requests.get(source+'/api/account',headers=headers,timeout=3).json()
                health = requests.get(source+'/api/health',timeout=3).json()
                context.update(node_id=manifest.get('node_id',''),iris_name=manifest.get('iris_name','Iris'),
                    accounts_enabled=health.get('accounts_enabled',False),account=account if account.get('authenticated') else None,
                    csrf_token=account.get('csrf_token',''))
            except (requests.RequestException,ValueError):
                pass
        return render_template('index.html',fourthought_context=context)

    def forward_read(path):
        source = app.config['RHIZOME_NODE_URL']
        if not source:
            return jsonify(error='The shared FourThought view requires the local Rhizome node.'),503
        headers = {k:request.headers[k] for k in ('Cookie','Authorization') if k in request.headers}
        try:
            upstream = requests.get(source.rstrip('/')+path,params=request.args,headers=headers,timeout=10,allow_redirects=False)
            return app.response_class(upstream.content,status=upstream.status_code,content_type=upstream.headers.get('Content-Type','application/json'))
        except requests.RequestException:
            return jsonify(error='The local Rhizome node is unavailable.'),503

    @app.get('/api/fourthought/<resource>')
    def shared_fourthought(resource):
        if resource not in {'list','range','speakers','thread','exchanges'}:
            return jsonify(error='Read operation unavailable'),404
        return forward_read('/api/fourthought/'+resource)

    @app.get('/api/account/speaker-groups')
    def shared_groups():
        return forward_read('/api/account/speaker-groups')

    @app.get('/api/overview')
    def overview():
        return jsonify(ledger.overview())

    @app.get('/api/stream')
    def stream():
        started = time.perf_counter()
        result = ledger.corpus(request.args).stream([v.strip().lower() for v in request.args.get('terms', '').split(',') if v.strip()])
        return jsonify(**result, elapsed_ms=round((time.perf_counter() - started)*1000, 1))

    @app.get('/api/summary')
    def summary():
        return jsonify(ledger.corpus(request.args).summary(request.args.get('focus_start',''), request.args.get('focus_end',''), request.args.get('terms','').split(',')))

    @app.get('/api/tree')
    def tree():
        started = time.perf_counter()
        corpus = ledger.corpus(request.args)
        default = corpus.words(1)
        seed = request.args.get('seed', default[0]['term'] if default else 'mind')
        result = corpus.tree(seed, integer(request.args.get('depth'), 5, 1, 5), integer(request.args.get('width'), 3, 2, 4))
        return jsonify(**result, elapsed_ms=round((time.perf_counter() - started)*1000, 1))

    @app.get('/api/thoughts')
    def thoughts():
        return jsonify(ledger.corpus(request.args).thoughts(request.args.get('term', ''), request.args.get('related', ''), integer(request.args.get('offset'), 0, 0, 1000000)))

    return app


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--db', default=os.getenv('RHIZOME_FOURTHOUGHT_DB'))
    parser.add_argument('--port', type=int, default=7788)
    parser.add_argument('--node-url', default=os.getenv('RHIZOME_NODE_URL','http://127.0.0.1:4444'))
    args = parser.parse_args()
    if not args.db:
        parser.error('Pass --db or set RHIZOME_FOURTHOUGHT_DB')
    create_app(args.db, args.node_url).run(host='127.0.0.1', port=args.port, threaded=True, use_reloader=False)
