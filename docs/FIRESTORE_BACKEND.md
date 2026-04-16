# Firestore Storage Backend

Drop-in replacement for MemPalace's default ChromaDB backend, using Google Cloud Firestore for document storage and vector search.

## Quick Start

```bash
pip install google-cloud-firestore sentence-transformers
export MEMPALACE_BACKEND=firestore
```

Or programmatically:

```python
from google.cloud import firestore
from mempalace.backends.firestore import FirestoreBackend
from mempalace.palace import set_backend

db = firestore.Client()
set_backend(FirestoreBackend(db))
```

## Architecture

### Path-Agnostic Scoping

The backend treats `palace_path` as an opaque Firestore path prefix. Collections are created at `{palace_path}/{collection_name}`. The caller decides the scoping scheme:

```python
backend = FirestoreBackend(db)

# Per-user
col = backend.get_collection("users/abc123", "mempalace_drawers")

# Per-project
col = backend.get_collection("projects/myapp", "mempalace_drawers")

# Single-tenant
col = backend.get_collection("palace", "mempalace_drawers")
```

### Document Schema

Each document in Firestore:

```
{
  "document": "the text content",
  "embedding": VectorValue (384-dim),
  "meta": {
    "wing": "...",
    "room": "...",
    ...any metadata fields...
  }
}
```

Metadata is nested under `"meta"` to avoid collisions with the top-level `document` and `embedding` fields. The ChromaDB API passes metadata as flat dicts — the backend nests/unnests transparently.

### Embedding Model

Uses `sentence-transformers` with `all-MiniLM-L6-v2` (384 dimensions) — the same model ChromaDB uses internally. This ensures identical search quality. The model is lazy-loaded on first use.

You can provide a custom embedding function:

```python
def my_embed_fn(texts: list[str]) -> list[list[float]]:
    return [[0.1, 0.2, ...] for _ in texts]

backend = FirestoreBackend(db, embed_fn=my_embed_fn)
```

## ChromaDB Compatibility

All behavior has been verified against ChromaDB's actual semantics.

### Operations

| Operation | Behavior | Notes |
|---|---|---|
| `add` | Skip duplicate IDs silently | Checks existence before write |
| `upsert` | Merge metadata (old keys preserved) | Uses Firestore `set(merge=True)` |
| `update` | Merge metadata, silent no-op for missing IDs | Checks existence, skips missing |
| `query` | Vector similarity via `find_nearest` | Supports multiple `query_texts` (nested results) |
| `get` | Batch fetch via `get_all` | Single round-trip for multiple IDs |
| `delete` | By IDs, metadata filter, or document content | `where_document` via client-side filtering |
| `count` | Aggregation query | Efficient, no full scan |

### Include Filtering

Non-included fields are returned as `None` (not omitted from the response), matching ChromaDB:

```python
# Only include documents
result = col.get(ids=["a"], include=["documents"])
# result["documents"] = ["hello"]
# result["metadatas"] = None  (not absent — explicitly None)
```

### Where Filters

All ChromaDB operators are supported:

| Operator | Example | Firestore Implementation |
|---|---|---|
| Equality | `{"field": "value"}` | `FieldFilter("meta.field", "==", value)` |
| `$eq` | `{"field": {"$eq": "value"}}` | `FieldFilter("meta.field", "==", value)` |
| `$ne` | `{"field": {"$ne": "value"}}` | `FieldFilter("meta.field", "!=", value)` |
| `$gt` | `{"field": {"$gt": 5}}` | `FieldFilter("meta.field", ">", 5)` |
| `$gte` | `{"field": {"$gte": 5}}` | `FieldFilter("meta.field", ">=", 5)` |
| `$lt` | `{"field": {"$lt": 10}}` | `FieldFilter("meta.field", "<", 10)` |
| `$lte` | `{"field": {"$lte": 10}}` | `FieldFilter("meta.field", "<=", 10)` |
| `$in` | `{"field": {"$in": [1, 2]}}` | `FieldFilter("meta.field", "in", [1, 2])` |
| `$nin` | `{"field": {"$nin": [1, 2]}}` | `FieldFilter("meta.field", "not-in", [1, 2])` |
| `$and` | `{"$and": [{...}, {...}]}` | Chained `.where()` calls |
| `$or` | `{"$or": [{...}, {...}]}` | `Or(filters=[...])` composite |

Nested composites work: `{"$or": [{"$and": [{"wing": "x"}, {"score": {"$gte": 5}}]}, {"wing": "y"}]}`

Multi-operator dicts like `{"$gte": 5, "$lte": 10}` raise `ValueError`, matching ChromaDB. Use `$and` instead:
```python
{"$and": [{"score": {"$gte": 5}}, {"score": {"$lte": 10}}]}
```

### where_document (Content Filtering)

ChromaDB supports `where_document` for full-text content matching in `delete`. Firestore has no native full-text search, so this is **simulated via client-side filtering** — all documents are streamed and matched in Python.

```python
col.delete(where_document={"$contains": "hello"})      # delete docs containing "hello"
col.delete(where_document={"$not_contains": "hello"})   # delete docs NOT containing "hello"
```

**Performance note**: This is O(n) reads where n is the total number of documents. Fine for small collections, avoid on large ones.

### Validation

Matches ChromaDB's error behavior:

| Call | Behavior |
|---|---|
| `get(ids=[])` | Raises `ValueError` |
| `delete(ids=[])` | Raises `ValueError` |
| `delete()` (no args) | Raises `ValueError` |
| `{"field": {"$gte": 5, "$lte": 10}}` | Raises `ValueError` (one operator per expression) |

## Batch Write Limits

Firestore limits batch writes to 500 operations. The `_BatchWriter` automatically chunks at 450 operations (with margin), committing and starting a new batch transparently. This applies to `add`, `upsert`, `update`, and `delete`.

## Knowledge Graph

`FirestoreKnowledgeGraph` is a drop-in replacement for the SQLite-based `KnowledgeGraph`. Same interface, backed by Firestore subcollections.

### Collections

```
{base_path}/entities/{entity_id}    — entity nodes
{base_path}/triples/{triple_id}     — relationship triples
```

### Deterministic Triple IDs

Active triples use deterministic IDs: `t_{subject}_{predicate}_{object}`. This enables:
- Document-level reads inside Firestore transactions (transactions don't support queries)
- Natural dedup — concurrent adds of the same triple hit the same document

### Transactions

Write operations use Firestore transactions for atomicity:

| Operation | What's Transactional | Why |
|---|---|---|
| `add_triple` | Read existing + write new | Prevents duplicate active triples on concurrent adds |
| `invalidate` | Read active + set valid_to | Prevents race between checking "still active" and updating |

Entity auto-creation (`add_entity`) happens outside the transaction — it's idempotent via `set(merge=True)`.

### Entity Name Resolution

`_resolve_name(entity_id)` resolves entity IDs to display names with an in-memory cache. Prevents N+1 reads when `query_entity`, `query_relationship`, or `timeline` return multiple results referencing the same entities.

## Tunnel Storage

`FirestoreTunnelStore` replaces the JSON file-based tunnel storage. Tunnels are stored at `{base_path}/tunnels/{tunnel_id}`.

### Symmetric IDs

Tunnel IDs are symmetric: `create_tunnel(A→B)` and `create_tunnel(B→A)` produce the same ID via `sha256(sorted(endpoints))`. This prevents duplicate tunnels.

### Transactions

`create_tunnel` uses a transaction to atomically check for an existing tunnel (preserving its `created_at` on update) and write the new/updated data.

## Firestore Indexes

**Important**: If your `palace_path` uses subcollections (e.g. `users/{id}/memory`), use `COLLECTION_GROUP` scope for all composite indexes. The default `COLLECTION` scope only applies to top-level collections and will not be used for queries scoped to subcollection paths — queries will fail at runtime with `FailedPrecondition: The query requires an index`.

### Vector indexes (drawers and closets)

```json
{
  "collectionGroup": "mempalace_drawers",
  "queryScope": "COLLECTION",
  "fields": [
    { "fieldPath": "embedding",
      "vectorConfig": { "dimension": 384, "flat": {} }
    }
  ]
}
```

Same shape for `mempalace_closets`. Vector indexes work with `COLLECTION` scope because `find_nearest` targets a specific collection reference, not a collection group.

### Knowledge graph composite indexes

All KG queries combine a `where` clause with `order_by` on a different field, so they require composite indexes. Because triples live at `{base_path}/triples`, use `COLLECTION_GROUP` scope:

```json
[
  { "collectionGroup": "triples", "queryScope": "COLLECTION_GROUP",
    "fields": [
      { "fieldPath": "subject", "order": "ASCENDING" },
      { "fieldPath": "valid_to", "order": "ASCENDING" }
    ]
  },
  { "collectionGroup": "triples", "queryScope": "COLLECTION_GROUP",
    "fields": [
      { "fieldPath": "object", "order": "ASCENDING" },
      { "fieldPath": "valid_to", "order": "ASCENDING" }
    ]
  },
  { "collectionGroup": "triples", "queryScope": "COLLECTION_GROUP",
    "fields": [
      { "fieldPath": "subject", "order": "ASCENDING" },
      { "fieldPath": "valid_from", "order": "ASCENDING" }
    ]
  },
  { "collectionGroup": "triples", "queryScope": "COLLECTION_GROUP",
    "fields": [
      { "fieldPath": "object", "order": "ASCENDING" },
      { "fieldPath": "valid_from", "order": "ASCENDING" }
    ]
  },
  { "collectionGroup": "triples", "queryScope": "COLLECTION_GROUP",
    "fields": [
      { "fieldPath": "predicate", "order": "ASCENDING" },
      { "fieldPath": "valid_from", "order": "ASCENDING" }
    ]
  }
]
```

### Deploying indexes

Put the above in `firestore.indexes.json` and deploy with the Firebase CLI:

```bash
firebase deploy --only firestore:indexes --project <project-id>
```

Indexes take several minutes to build on first deployment.

## Configuration

### Environment Variable

```bash
export MEMPALACE_BACKEND=firestore  # default: "chroma"
```

### Programmatic

```python
from mempalace.palace import set_backend, get_backend

# Override before first collection access
set_backend(my_backend)

# Get the current backend (lazy-initialized from env if not set)
backend = get_backend()
```

## Package Layout

```
mempalace/backends/firestore/
  __init__.py           # lazy re-exports (no eager google-cloud-firestore import)
  collection.py         # FirestoreCollection, FirestoreBackend
  knowledge_graph.py    # FirestoreKnowledgeGraph
  tunnels.py            # FirestoreTunnelStore
```

All public classes are re-exported from `mempalace.backends.firestore`, so normal imports work:

```python
from mempalace.backends.firestore import (
    FirestoreBackend,
    FirestoreCollection,
    FirestoreKnowledgeGraph,
    FirestoreTunnelStore,
)
```

The `__init__.py` uses `__getattr__` for lazy loading so `google-cloud-firestore` and `sentence-transformers` only get imported if you actually touch the Firestore backend — ChromaDB-only users are unaffected.

## Testing

### Unit tests (mocked)

128 tests cover all operations, edge cases, and ChromaDB compatibility. No real Firestore needed:

```bash
python -m pytest tests/test_firestore_backend.py tests/test_firestore_kg.py \
    tests/test_firestore_tunnels.py tests/test_palace_backend_config.py -v
```

### Static type checking

Pyright is run in CI over all Firestore files. Run locally with:

```bash
pip install pyright google-cloud-firestore sentence-transformers
pyright
```

`pyrightconfig.json` is scoped to the Firestore files only — ChromaDB code is not type-checked.

### Integration tests

End-to-end tests against a live Firestore database are documented in
[`FIRESTORE_INTEGRATION_TEST_RESULTS.md`](./FIRESTORE_INTEGRATION_TEST_RESULTS.md),
including a complete drag-and-drop FastAPI reference server.
