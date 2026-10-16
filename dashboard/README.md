# Prophet Mind dashboard

A local dashboard reconnecting the original Prophet Mind visualizations to a current Rhizome SQLite ledger:

- **Stream of Consciousness:** word-frequency ribbons, a draggable full-history context timeline, speaker/domain/date/text filters, selectable concepts, and source-thought inspection.
- **Tree of Knowledge:** expandable and collapsible concept branches; double-click a concept to reroot. Relationships use a real Gensim Word2Vec skip-gram model trained on the selected ledger corpus on CPU (128 dimensions, window 6, 20 epochs, four workers). The complete model is cached so it can be loaded and trained further. No GPU or external inference service is used.

The historical UI remains in `UI/thoughtStream`, `UI/knowledgeTree`, and `UI/treeCloud`.

## Run

From this directory:

```sh
uv sync
uv run python app.py --db /absolute/path/to/fourthought.db
```

Open http://127.0.0.1:7788. `RHIZOME_FOURTHOUGHT_DB` can supply the database path instead; `--port` selects a different local port.

The ledger is opened in SQLite read-only mode. Local and global authored thoughts are included; private thoughts and duplicate vote entries are excluded. Speaker filters use stable speaker IDs; displayed names come from the ledger. The app binds only to loopback and rejects cross-origin browser requests.

No ledger content or models are written into the repository. Models are cached under `~/.cache/prophet-mind/models` (override with `MIND_MODEL_DIR`). A model fingerprint includes the selected source text and training configuration. Changed source data invalidates the bounded in-memory corpus cache. A new filter selection trains its own model the first time its tree is opened; subsequent tree queries reuse the model. Click **Refresh record** to load newly written thoughts.

Stream width measures word occurrences, not confidence, value, or objective importance. Buckets adapt between days, weeks, and months; missing periods are retained as zeroes. Dates use UTC, with legacy timezone-free timestamps treated as UTC. Tree links measure learned vector similarity, not truth, causality, agreement, or explicit references. Small corpora provide weaker evidence for semantic neighborhoods. The vocabulary selector displays frequent non-stopword concepts, while Word2Vec training retains surrounding stopwords for linguistic context.

## Verify

```sh
uv run pytest -q tests
node --check static/app.js
```

Tests use fictional temporary ledgers and cover source isolation, word counts, stable identity, pagination, refresh invalidation, real Word2Vec training/cache round trips, and browser access boundaries.
