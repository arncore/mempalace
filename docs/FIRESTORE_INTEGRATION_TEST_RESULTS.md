# Firestore Backend — Integration Test Results

This document records end-to-end integration testing of the Firestore storage
backend against a live Firestore database, so that future contributors can
replicate the test run.

## Reproducing the Test Environment

### Service Setup

A minimal FastAPI HTTP server wraps the Firestore backend and exposes the
MemPalace tool surface as REST endpoints. The complete reference implementation
used for these tests is provided below — drop it into a `server.py` file and it
will run with `uvicorn server:app --host 0.0.0.0 --port 8080`.

```python
"""Minimal FastAPI server exposing the MemPalace Firestore backend.

Run locally:
    export MEMPALACE_BACKEND=firestore
    export GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json
    uvicorn server:app --host 0.0.0.0 --port 8080
"""

import hashlib
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from mempalace.config import sanitize_content, sanitize_name
from mempalace.palace import get_collection, set_backend
from mempalace.searcher import search_memories
from mempalace.backends.firestore import (
    FirestoreBackend,
    FirestoreKnowledgeGraph,
    FirestoreTunnelStore,
)

# ── Startup: configure Firestore backend ────────────────────────────────

# If credentials are provided as a JSON string, write to a file for ADC.
_sa_json = os.environ.get("FIREBASE_SERVICE_ACCOUNT_JSON", "")
if _sa_json and not os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
    _creds_path = "/tmp/firebase-service-account.json"
    Path(_creds_path).write_text(_sa_json)
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = _creds_path

from google.cloud import firestore  # noqa: E402

_db = firestore.Client()
set_backend(FirestoreBackend(_db))

app = FastAPI(title="MemPalace Firestore API", version="1.0.0")


def _kg(palace_path: str) -> FirestoreKnowledgeGraph:
    return FirestoreKnowledgeGraph(_db, base_path=palace_path)


def _tunnels(palace_path: str) -> FirestoreTunnelStore:
    return FirestoreTunnelStore(_db, base_path=palace_path)


# ── Request models ──────────────────────────────────────────────────────

class AddDrawerRequest(BaseModel):
    wing: str
    room: str
    content: str
    source_file: Optional[str] = None
    added_by: str = "api"


class UpdateDrawerRequest(BaseModel):
    drawer_id: str
    content: Optional[str] = None
    wing: Optional[str] = None
    room: Optional[str] = None


class DrawerIdRequest(BaseModel):
    drawer_id: str


class ListDrawersRequest(BaseModel):
    wing: Optional[str] = None
    room: Optional[str] = None
    limit: int = Field(default=20, ge=1, le=100)
    offset: int = Field(default=0, ge=0)


class SearchRequest(BaseModel):
    query: str = Field(max_length=250)
    limit: int = Field(default=5, ge=1, le=100)
    wing: Optional[str] = None
    room: Optional[str] = None
    max_distance: float = 1.5


class CheckDuplicateRequest(BaseModel):
    content: str
    threshold: float = 0.9


class KgAddRequest(BaseModel):
    subject: str
    predicate: str
    object: str
    valid_from: Optional[str] = None
    source_closet: Optional[str] = None


class KgQueryRequest(BaseModel):
    entity: str
    as_of: Optional[str] = None
    direction: str = "both"


class KgInvalidateRequest(BaseModel):
    subject: str
    predicate: str
    object: str
    ended: Optional[str] = None


class KgTimelineRequest(BaseModel):
    entity: Optional[str] = None


class CreateTunnelRequest(BaseModel):
    source_wing: str
    source_room: str
    target_wing: str
    target_room: str
    label: str = ""
    source_drawer_id: Optional[str] = None
    target_drawer_id: Optional[str] = None


class ListTunnelsRequest(BaseModel):
    wing: Optional[str] = None


class DiaryWriteRequest(BaseModel):
    agent_name: str
    entry: str
    topic: str = "general"


class DiaryReadRequest(BaseModel):
    agent_name: str
    last_n: int = Field(default=10, ge=1, le=100)


# ── Helpers ─────────────────────────────────────────────────────────────

def _fetch_all_metadata(col, where=None):
    """Paginate col.get() to fetch all metadata."""
    total = col.count()
    all_meta, offset = [], 0
    while offset < total:
        kwargs = {"include": ["metadatas"], "limit": 1000, "offset": offset}
        if where:
            kwargs["where"] = where
        batch = col.get(**kwargs)
        if not batch["metadatas"]:
            break
        all_meta.extend(batch["metadatas"])
        offset += len(batch["metadatas"])
    return all_meta


# ── Health ──────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok"}


# ── Drawers ─────────────────────────────────────────────────────────────

@app.post("/palace/{palace_path:path}/drawers/add")
async def add_drawer(palace_path: str, body: AddDrawerRequest):
    try:
        wing = sanitize_name(body.wing, "wing")
        room = sanitize_name(body.room, "room")
        content = sanitize_content(body.content)
    except ValueError as e:
        raise HTTPException(400, str(e))

    col = get_collection(palace_path, create=True)
    drawer_id = (
        f"drawer_{wing}_{room}_"
        f"{hashlib.sha256((wing + room + content).encode()).hexdigest()[:24]}"
    )

    # Idempotency
    try:
        existing = col.get(ids=[drawer_id])
        if existing and existing["ids"]:
            return {"success": True, "reason": "already_exists", "drawer_id": drawer_id}
    except Exception:
        pass

    col.upsert(
        ids=[drawer_id],
        documents=[content],
        metadatas=[{
            "wing": wing,
            "room": room,
            "source_file": body.source_file or "",
            "added_by": body.added_by,
            "filed_at": datetime.now().isoformat(),
            "chunk_index": 0,
        }],
    )
    return {"success": True, "drawer_id": drawer_id, "wing": wing, "room": room}


@app.post("/palace/{palace_path:path}/drawers/get")
async def get_drawer(palace_path: str, body: DrawerIdRequest):
    col = get_collection(palace_path, create=False)
    result = col.get(ids=[body.drawer_id], include=["documents", "metadatas"])
    if not result["ids"]:
        raise HTTPException(404, f"Drawer not found: {body.drawer_id}")
    return {
        "drawer_id": body.drawer_id,
        "content": result["documents"][0],
        "metadata": result["metadatas"][0],
    }


@app.post("/palace/{palace_path:path}/drawers/update")
async def update_drawer(palace_path: str, body: UpdateDrawerRequest):
    col = get_collection(palace_path, create=False)
    existing = col.get(ids=[body.drawer_id], include=["documents", "metadatas"])
    if not existing["ids"]:
        raise HTTPException(404, f"Drawer not found: {body.drawer_id}")

    new_meta = dict(existing["metadatas"][0])
    if body.wing is not None:
        new_meta["wing"] = sanitize_name(body.wing, "wing")
    if body.room is not None:
        new_meta["room"] = sanitize_name(body.room, "room")

    kwargs = {"ids": [body.drawer_id], "metadatas": [new_meta]}
    if body.content is not None:
        kwargs["documents"] = [sanitize_content(body.content)]
    col.update(**kwargs)
    return {"success": True, "drawer_id": body.drawer_id}


@app.post("/palace/{palace_path:path}/drawers/delete")
async def delete_drawer(palace_path: str, body: DrawerIdRequest):
    col = get_collection(palace_path, create=False)
    existing = col.get(ids=[body.drawer_id])
    if not existing["ids"]:
        raise HTTPException(404, f"Drawer not found: {body.drawer_id}")
    col.delete(ids=[body.drawer_id])
    return {"success": True, "drawer_id": body.drawer_id}


@app.post("/palace/{palace_path:path}/drawers/list")
async def list_drawers(palace_path: str, body: ListDrawersRequest):
    col = get_collection(palace_path, create=False)
    conds = []
    if body.wing:
        conds.append({"wing": body.wing})
    if body.room:
        conds.append({"room": body.room})
    where = conds[0] if len(conds) == 1 else {"$and": conds} if conds else None

    kwargs = {"include": ["documents", "metadatas"], "limit": body.limit, "offset": body.offset}
    if where:
        kwargs["where"] = where
    result = col.get(**kwargs)

    drawers = [{
        "drawer_id": did,
        "wing": meta.get("wing", ""),
        "room": meta.get("room", ""),
        "content_preview": (doc[:200] + "...") if len(doc) > 200 else doc,
    } for did, meta, doc in zip(result["ids"], result["metadatas"], result["documents"])]
    return {"drawers": drawers, "count": len(drawers)}


# ── Search ──────────────────────────────────────────────────────────────

@app.post("/palace/{palace_path:path}/search")
async def search(palace_path: str, body: SearchRequest):
    return search_memories(
        body.query,
        palace_path=palace_path,
        wing=body.wing,
        room=body.room,
        n_results=body.limit,
        max_distance=body.max_distance,
    )


@app.post("/palace/{palace_path:path}/check-duplicate")
async def check_duplicate(palace_path: str, body: CheckDuplicateRequest):
    col = get_collection(palace_path, create=False)
    results = col.query(
        query_texts=[body.content], n_results=5,
        include=["metadatas", "documents", "distances"],
    )
    duplicates = []
    if results["ids"] and results["ids"][0]:
        for i, did in enumerate(results["ids"][0]):
            similarity = round(1 - results["distances"][0][i], 3)
            if similarity >= body.threshold:
                meta = results["metadatas"][0][i]
                doc = results["documents"][0][i]
                duplicates.append({
                    "id": did,
                    "wing": meta.get("wing", "?"),
                    "room": meta.get("room", "?"),
                    "similarity": similarity,
                    "content": (doc[:200] + "...") if len(doc) > 200 else doc,
                })
    return {"is_duplicate": bool(duplicates), "matches": duplicates}


# ── Palace Overview ─────────────────────────────────────────────────────

@app.get("/palace/{palace_path:path}/status")
async def status(palace_path: str):
    col = get_collection(palace_path, create=False)
    count = col.count()
    wings, rooms = {}, {}
    for m in _fetch_all_metadata(col):
        wings[m.get("wing", "unknown")] = wings.get(m.get("wing", "unknown"), 0) + 1
        rooms[m.get("room", "unknown")] = rooms.get(m.get("room", "unknown"), 0) + 1
    return {"total_drawers": count, "wings": wings, "rooms": rooms, "palace_path": palace_path}


# ── Knowledge Graph ─────────────────────────────────────────────────────

@app.post("/palace/{palace_path:path}/kg/add")
async def kg_add(palace_path: str, body: KgAddRequest):
    triple_id = _kg(palace_path).add_triple(
        body.subject, body.predicate, body.object,
        valid_from=body.valid_from, source_closet=body.source_closet,
    )
    return {
        "success": True,
        "triple_id": triple_id,
        "fact": f"{body.subject} → {body.predicate} → {body.object}",
    }


@app.post("/palace/{palace_path:path}/kg/query")
async def kg_query(palace_path: str, body: KgQueryRequest):
    facts = _kg(palace_path).query_entity(body.entity, as_of=body.as_of, direction=body.direction)
    return {"entity": body.entity, "as_of": body.as_of, "facts": facts, "count": len(facts)}


@app.post("/palace/{palace_path:path}/kg/invalidate")
async def kg_invalidate(palace_path: str, body: KgInvalidateRequest):
    _kg(palace_path).invalidate(body.subject, body.predicate, body.object, ended=body.ended)
    return {
        "success": True,
        "fact": f"{body.subject} → {body.predicate} → {body.object}",
        "ended": body.ended or "today",
    }


@app.post("/palace/{palace_path:path}/kg/timeline")
async def kg_timeline(palace_path: str, body: KgTimelineRequest):
    facts = _kg(palace_path).timeline(body.entity)
    return {"entity": body.entity or "all", "timeline": facts, "count": len(facts)}


@app.get("/palace/{palace_path:path}/kg/stats")
async def kg_stats(palace_path: str):
    return _kg(palace_path).stats()


# ── Tunnels ─────────────────────────────────────────────────────────────

@app.post("/palace/{palace_path:path}/tunnels/create")
async def create_tunnel(palace_path: str, body: CreateTunnelRequest):
    return _tunnels(palace_path).create_tunnel(
        body.source_wing, body.source_room,
        body.target_wing, body.target_room,
        label=body.label,
        source_drawer_id=body.source_drawer_id,
        target_drawer_id=body.target_drawer_id,
    )


@app.post("/palace/{palace_path:path}/tunnels/list")
async def list_tunnels(palace_path: str, body: ListTunnelsRequest):
    return _tunnels(palace_path).list_tunnels(body.wing)


@app.post("/palace/{palace_path:path}/tunnels/delete")
async def delete_tunnel(palace_path: str, body: DrawerIdRequest):
    return _tunnels(palace_path).delete_tunnel(body.drawer_id)


# ── Diary ───────────────────────────────────────────────────────────────

@app.post("/palace/{palace_path:path}/diary/write")
async def diary_write(palace_path: str, body: DiaryWriteRequest):
    agent = sanitize_name(body.agent_name, "agent_name")
    entry = sanitize_content(body.entry)
    wing = f"wing_{agent.lower().replace(' ', '_')}"
    col = get_collection(palace_path, create=True)
    now = datetime.now()
    entry_id = (
        f"diary_{wing}_{now.strftime('%Y%m%d_%H%M%S%f')}_"
        f"{hashlib.sha256(entry.encode()).hexdigest()[:12]}"
    )
    col.add(
        ids=[entry_id], documents=[entry],
        metadatas=[{
            "wing": wing, "room": "diary", "topic": body.topic,
            "type": "diary_entry", "agent": agent,
            "filed_at": now.isoformat(), "date": now.strftime("%Y-%m-%d"),
        }],
    )
    return {"success": True, "entry_id": entry_id, "timestamp": now.isoformat()}


@app.post("/palace/{palace_path:path}/diary/read")
async def diary_read(palace_path: str, body: DiaryReadRequest):
    agent = sanitize_name(body.agent_name, "agent_name")
    wing = f"wing_{agent.lower().replace(' ', '_')}"
    col = get_collection(palace_path, create=False)
    results = col.get(
        where={"$and": [{"wing": wing}, {"room": "diary"}]},
        include=["documents", "metadatas"], limit=10000,
    )
    if not results["ids"]:
        return {"agent": agent, "entries": []}
    entries = sorted(
        [{
            "date": m.get("date", ""),
            "timestamp": m.get("filed_at", ""),
            "topic": m.get("topic", ""),
            "content": d,
        } for d, m in zip(results["documents"], results["metadatas"])],
        key=lambda x: x["timestamp"], reverse=True,
    )[:body.last_n]
    return {"agent": agent, "entries": entries, "total": len(results["ids"])}
```

Endpoints exposed (all scoped by an opaque `palace_path` prefix):

| Route | Maps to | Backend class |
|-------|---------|---------------|
| `POST /palace/{p}/drawers/add` | collection.upsert | FirestoreCollection |
| `POST /palace/{p}/drawers/get` | collection.get(ids) | FirestoreCollection |
| `POST /palace/{p}/drawers/update` | collection.update | FirestoreCollection |
| `POST /palace/{p}/drawers/delete` | collection.delete | FirestoreCollection |
| `POST /palace/{p}/drawers/list` | collection.get(where, limit, offset) | FirestoreCollection |
| `POST /palace/{p}/search` | collection.query | FirestoreCollection |
| `POST /palace/{p}/check-duplicate` | collection.query | FirestoreCollection |
| `POST /palace/{p}/kg/add` | kg.add_triple | FirestoreKnowledgeGraph |
| `POST /palace/{p}/kg/query` | kg.query_entity | FirestoreKnowledgeGraph |
| `POST /palace/{p}/kg/invalidate` | kg.invalidate | FirestoreKnowledgeGraph |
| `POST /palace/{p}/kg/timeline` | kg.timeline | FirestoreKnowledgeGraph |
| `GET /palace/{p}/kg/stats` | kg.stats | FirestoreKnowledgeGraph |
| `POST /palace/{p}/tunnels/create` | store.create_tunnel | FirestoreTunnelStore |
| `POST /palace/{p}/tunnels/list` | store.list_tunnels | FirestoreTunnelStore |
| `POST /palace/{p}/tunnels/delete` | store.delete_tunnel | FirestoreTunnelStore |
| `GET /palace/{p}/status` | collection.count / collection.get | FirestoreCollection |
| `POST /palace/{p}/diary/write` | collection.add | FirestoreCollection |
| `POST /palace/{p}/diary/read` | collection.get | FirestoreCollection |

### Container Image

```dockerfile
FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# Pre-download the embedding model at build time
RUN python -c "from sentence_transformers import SentenceTransformer; \
    SentenceTransformer('all-MiniLM-L6-v2')"
COPY server.py .
EXPOSE 8080
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8080"]
```

### requirements.txt

```
mempalace @ git+https://github.com/<owner>/mempalace.git@<commit>
fastapi>=0.115.0
uvicorn>=0.30.0
google-cloud-firestore>=2.19.0
sentence-transformers>=3.0.0
```

### Environment variables

- `MEMPALACE_BACKEND=firestore`
- `GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json`
- `GOOGLE_CLOUD_PROJECT=<firebase-project-id>`

### Required Firestore Indexes

Triples are stored in subcollections (e.g. `users/{id}/triples/{tid}`), so KG
queries need `COLLECTION_GROUP`-scope composite indexes:

```json
{
  "indexes": [
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
        { "fieldPath": "predicate", "order": "ASCENDING" },
        { "fieldPath": "valid_from", "order": "ASCENDING" }
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
    { "collectionGroup": "mempalace_drawers", "queryScope": "COLLECTION",
      "fields": [
        { "fieldPath": "embedding",
          "vectorConfig": { "dimension": 384, "flat": {} }
        }
      ]
    },
    { "collectionGroup": "mempalace_closets", "queryScope": "COLLECTION",
      "fields": [
        { "fieldPath": "embedding",
          "vectorConfig": { "dimension": 384, "flat": {} }
        }
      ]
    }
  ]
}
```

Deploy with: `firebase deploy --only firestore:indexes --project <project>`.

## Test Environment Used

- **Date**: 2026-04-16T02:52Z
- **Platform**: Single container running on a shared-CPU VM (1 GB memory)
- **Runtime**: Python 3.12 + uvicorn
- **Embedding Model**: all-MiniLM-L6-v2 (384 dimensions) — loaded once at startup
- **Firestore Region**: nam5 (multi-region)
- **Test Path**: `users/inttest-{timestamp}/memory`

## Methodology

Each operation was issued via the HTTP API, then independently verified by
reading the underlying Firestore documents through the
[Firestore REST API](https://firebase.google.com/docs/firestore/reference/rest)
using a gcloud access token. This two-sided verification confirms the backend
writes documents with the expected schema and reads them back correctly through
the collection adapter.

## Results

| # | Test | Status | Detail |
|---|------|--------|--------|
| | **Drawer CRUD** | | |
| 1.1 | Add drawer | PASS | Returns drawer_id, HTTP 200 |
| 1.2 | Firestore document exists | PASS | Found at `mempalace_drawers/{id}` |
| 1.2 | Firestore has embedding | PASS | 384-dim vector stored |
| 1.2 | Firestore metadata preserved | PASS | Nested under `meta` field |
| 1.3 | Duplicate add is idempotent | PASS | Returns `already_exists`, no duplicate created |
| 1.4 | Get drawer by ID | PASS | Content matches original |
| 1.5 | Update drawer content | PASS | HTTP 200, new embedding generated |
| 1.5 | Firestore content updated | PASS | Verified via REST API |
| 1.6 | List drawers with wing filter | PASS | Returns only matching wing (1 result) |
| 1.7 | Delete drawer | PASS | HTTP 200 |
| 1.7 | Firestore document deleted | PASS | REST API returns NOT_FOUND |
| | **Vector Search** | | |
| 2.1 | Semantic search — top result relevant | PASS | Query "Japanese food sushi" → top result contains sushi/ramen |
| 2.2 | Near-duplicate detection | PASS | `is_duplicate=true` at 0.8 similarity threshold |
| | **Knowledge Graph** | | |
| 3.1 | Add triple | PASS | Deterministic `triple_id` returned |
| 3.1 | Firestore triple document | PASS | Correct shape: subject, predicate, object, valid_from |
| 3.1 | Entity auto-created | PASS | `entities/{id}` created transparently |
| 3.2 | Duplicate triple dedup | PASS | Same `triple_id` returned, no duplicate stored |
| 3.3 | Query entity (outgoing) | PASS | Returns all 3 outgoing facts |
| 3.3 | Query entity (incoming) | PASS | Returns inbound fact correctly |
| 3.4 | Invalidate triple | PASS | HTTP 200 |
| 3.4 | Firestore `valid_to` set | PASS | Date written atomically |
| 3.5 | Stats — current facts | PASS | Decrements correctly after invalidation |
| 3.5 | Stats — expired facts | PASS | Increments correctly after invalidation |
| 3.6 | Timeline | PASS | Ordered by `valid_from`, returns correct chronology |
| | **Tunnels** | | |
| 4.1 | Create tunnel | PASS | Returns tunnel ID |
| 4.1 | Firestore tunnel document | PASS | Correct shape verified |
| 4.2 | Symmetric tunnel ID | PASS | Reversing source/target produces same ID |
| 4.3 | List tunnels | PASS | HTTP 200 |
| 4.4 | Delete tunnel | PASS | HTTP 200 |
| | **Palace Overview** | | |
| 5.1 | Status with drawers | PASS | total_drawers > 0, wings and rooms counted |
| | **Diary** | | |
| 6.1 | Write diary entry | PASS | `entry_id` returned |
| 6.2 | Read diary entries | PASS | Content matches written entry |
| | **Edge Cases** | | |
| 7.1 | Get nonexistent drawer | PASS | HTTP 404 with descriptive message |

## Summary

- **Total**: 33
- **Passed**: 33
- **Pass Rate**: 100%

## Firestore Document Schemas (Verified)

### Drawer — `{scope}/mempalace_drawers/{id}`
```json
{
  "document": "string",
  "embedding": "vector<float, 384>",
  "meta": {
    "wing": "string",
    "room": "string",
    "source_file": "string",
    "added_by": "string",
    "filed_at": "ISO timestamp",
    "chunk_index": 0
  }
}
```

### Triple — `{scope}/triples/{id}`
```json
{
  "subject": "normalized_entity_id",
  "predicate": "normalized_relationship",
  "object": "normalized_entity_id",
  "valid_from": "YYYY-MM-DD | null",
  "valid_to": "YYYY-MM-DD | null",
  "confidence": 1.0,
  "extracted_at": "ISO timestamp"
}
```

### Entity — `{scope}/entities/{id}`
```json
{
  "name": "Original Name With Casing",
  "type": "unknown",
  "properties": {},
  "created_at": "ISO timestamp"
}
```

### Tunnel — `{scope}/tunnels/{id}`
```json
{
  "id": "sha256-based symmetric ID",
  "source": { "wing": "string", "room": "string" },
  "target": { "wing": "string", "room": "string" },
  "label": "string",
  "created_at": "ISO timestamp"
}
```

## Behavioural Notes Observed During Testing

- **Deterministic IDs** make dedup cheap — triple IDs are `t_{subject}_{predicate}_{object}`, drawer IDs are a hash of `wing + room + content`. Re-sending the same payload is idempotent.
- **Firestore transactions** are used for `add_triple`, `invalidate`, and `create_tunnel`. Concurrent calls that would produce duplicates converge on the same document.
- **Entities are auto-created** on first reference in a triple — the caller never needs to call `add_entity` explicitly.
- **Symmetric tunnel IDs** mean creating `A → B` and `B → A` produce the same document. Useful for bidirectional relationships like "related to".
- **Subcollection scoping** lets a single Firestore database host arbitrarily many isolated palaces — every query is implicitly namespaced by the `palace_path` prefix.
