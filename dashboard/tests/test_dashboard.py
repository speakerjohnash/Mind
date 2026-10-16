import json
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import app as dashboard


@pytest.fixture
def application(tmp_path, monkeypatch):
    monkeypatch.setattr(dashboard, 'MODEL_DIR', tmp_path / 'models')
    db = tmp_path / 'fictional.db'
    conn = sqlite3.connect(db)
    conn.execute('CREATE TABLE fourthought_logs (id INTEGER PRIMARY KEY, speaker_id TEXT, username TEXT, thought TEXT, timestamp TEXT, type TEXT, domain TEXT, verity REAL, valence REAL, tier TEXT, reply_to INTEGER, references_json TEXT)')
    rows = [(i, 'voice-a', 'Example', 'forest trees roots grow together' if i % 2 else 'music rhythm melodies sound together',
             f'2020-{1+i%2:02d}-01T00:00:00Z', 'statement', 'general', .8, .2, 'local', None, '[]') for i in range(1,65)]
    rows += [(65, 'voice-b', 'Example', 'ocean currents tide waves', '2021-01-01T00:00:00Z', 'statement', 'general', .7, .1, 'global', None, '[]'),
             (66, 'voice-a', 'Example', 'secret hidden private material', '2020-01-01T00:00:00Z', 'statement', 'general', .9, 0, 'private', None, '[]'),
             (67, 'voice-b', 'Example', rows[0][3], rows[0][4], 'statement', 'general', .6, .1, 'local', 1, '[]')]
    conn.executemany('INSERT INTO fourthought_logs VALUES (?,?,?,?,?,?,?,?,?,?,?,?)', rows)
    conn.commit(); conn.close()
    return dashboard.create_app(db)


def test_privacy_vote_exclusion_and_identity(application):
    client = application.test_client()
    overview = client.get('/api/overview').get_json()
    assert overview['total'] == 65
    assert {s['id'] for s in overview['speakers']} == {'voice-a', 'voice-b'}
    assert client.get('/api/stream?speaker=Example').status_code == 400
    data = client.get('/api/thoughts?speaker=voice-a').get_json()
    assert data['total'] == 64
    assert all(r['tier'] != 'private' and r['id'] != 67 for r in data['thoughts'])
    assert client.get('/api/stream?speaker=voice-b').get_json()['total'] == 1


def test_stream_counts_missing_months_filters_and_pagination(application):
    client = application.test_client()
    data = client.get('/api/stream?speaker=voice-a&terms=forest,music').get_json()
    assert data['terms'] == ['forest','music']
    assert sum(sum(b['values']) for b in data['buckets']) == 64
    assert sum(b['thoughts'] for b in data['buckets']) == 64
    assert client.get('/api/thoughts?term=forest').get_json()['total'] == 32
    second = client.get('/api/thoughts?term=forest&offset=30').get_json()
    assert len(second['thoughts']) == 2 and not second['more']
    assert client.get('/api/stream?speaker=voice-a&start=2020-02-01').get_json()['total'] == 32
    assert client.get('/api/stream?q=absent').get_json()['buckets'] == []
    assert client.get('/api/stream?start=2022-01-01&end=2020-01-01').status_code == 400


def test_actual_word2vec_vectors_and_cached_model(application):
    client = application.test_client()
    data = client.get('/api/tree?speaker=voice-a&seed=forest&depth=2').get_json()
    assert data['root']['term'] == 'forest'
    assert data['model']['algorithm'] == 'Word2Vec skip-gram'
    corpus = application.config['LEDGER'].corpus({'speaker':'voice-a'})
    assert isinstance(corpus.model, dashboard.Word2Vec)
    neighbor = corpus.neighbors('forest')[0]
    assert neighbor['score'] == pytest.approx(float(corpus.model.wv.similarity('forest', neighbor['term'])), abs=.0001)
    assert np.isfinite(corpus.model.wv.vectors).all()
    reloaded = dashboard.Corpus(corpus.rows)
    reloaded.ensure_model()
    assert np.array_equal(reloaded.model.wv['forest'], corpus.model.wv['forest'])
    assert client.get('/api/tree?depth=10').status_code == 400
    assert client.get('/api/tree?seed=absent').get_json()['root'] is None


def test_changes_invalidate_cached_corpus(application):
    ledger = application.config['LEDGER']
    first = ledger.corpus({'speaker':'voice-a'})
    assert ledger.corpus({'speaker':'voice-a'}) is first
    conn = sqlite3.connect(ledger.path)
    conn.execute("UPDATE fourthought_logs SET thought='new concepts change meaning' WHERE id IN (2,4)")
    conn.commit(); conn.close()
    ledger.checked = 0
    changed = ledger.corpus({'speaker':'voice-a'})
    assert changed is not first
    assert 'concepts' in changed.lookup


def test_browser_boundaries(application):
    client = application.test_client()
    assert client.get('/api/overview',headers={'Host':'attacker.invalid'}).status_code == 403
    assert client.get('/api/overview',headers={'Origin':'https://attacker.invalid'}).status_code == 403
    assert client.get('/api/overview',headers={'Sec-Fetch-Site':'cross-site'}).status_code == 403
    response = client.get('/')
    assert response.status_code == 200
    assert response.headers['Cache-Control'] == 'no-store'
    assert 'frame-ancestors' in response.headers['Content-Security-Policy']


def test_blue_fruits_include_outside_frequent_vocabulary(application):
    corpus = application.config['LEDGER'].corpus({'speaker':'voice-a'})
    corpus.ensure_model()
    corpus.common_words = {'forest'}
    result = corpus.tree('forest', depth=2)
    assert result['words'][0]['term'] == 'forest'
    assert result['fruits'] > 0
    assert all(w['fruit'] for w in result['words'][1:])
    assert all(w['term'] not in corpus.common_words for w in result['words'][1:])
    # Rare words remain searchable even if the stream vocabulary omits them.
    corpus.lookup.pop('roots')
    assert corpus.thoughts('roots', '', 0)['total'] == 32


def test_focus_counts_and_rankings_follow_the_selected_period(application):
    client = application.test_client()
    jan = client.get('/api/summary?speaker=voice-a&focus_start=2020-01-01&focus_end=2020-01-31&terms=forest,music').get_json()
    feb = client.get('/api/summary?speaker=voice-a&focus_start=2020-02-01&focus_end=2020-02-29&terms=forest,music').get_json()
    assert jan['total'] == feb['total'] == 32
    assert {w['term']:w['count'] for w in jan['top_terms']}['music'] == 32
    assert {w['term']:w['count'] for w in jan['top_terms']}['forest'] == 0
    assert {w['term']:w['count'] for w in feb['top_terms']}['forest'] == 32
    assert client.get('/api/summary?focus_start=2022-01-01&focus_end=2020-01-01').status_code == 400
    assert client.get('/api/summary?focus_start=2010-01-01&focus_end=2011-01-01').get_json()['total'] == 0
    page = client.get('/').get_data(as_text=True)
    assert 'id="ft-list"' in page and 'id="ft-search"' in page
    assert 'data-read-only="true"' in page
    assert 'id="thought-list"' not in page
