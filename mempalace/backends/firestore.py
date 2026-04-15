"""Firestore-backed MemPalace collection adapter with vector search.

Implements the same BaseCollection interface as the ChromaDB adapter,
using Firestore for document storage and find_nearest() for vector
similarity search. Embeddings are generated via sentence-transformers
using the same all-MiniLM-L6-v2 model ChromaDB uses internally, so
search quality is identical.

The backend is path-agnostic: ``palace_path`` is treated as an opaque
Firestore path prefix. The caller decides how to scope collections
(per-user, per-project, per-org, etc.). No collection names or ID
schemes are hardcoded.
"""

import logging
from typing import Any, Callable, Dict, List, Optional

from google.cloud.firestore_v1.base_vector_query import DistanceMeasure
from google.cloud.firestore_v1.vector import Vector

from .base import BaseCollection

logger = logging.getLogger("mempalace.backends.firestore")

# Lazy-loaded embedding model — only initialized on first use.
_embed_model = None


def _get_embed_model():
    """Load sentence-transformers model (same as ChromaDB default)."""
    global _embed_model
    if _embed_model is None:
        from sentence_transformers import SentenceTransformer

        _embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _embed_model


def default_embed_fn(texts: List[str]) -> List[List[float]]:
    """Embed texts using all-MiniLM-L6-v2 (384 dims, same as ChromaDB)."""
    model = _get_embed_model()
    embeddings = model.encode(texts, convert_to_numpy=True)
    return [e.tolist() for e in embeddings]


def _apply_where_filter(query, where: Dict[str, Any]):
    """Translate ChromaDB where-filter syntax to Firestore query chains.

    Supports:
      {"field": "value"}                  — equality
      {"$and": [{...}, {...}]}            — AND of filters
      {"field": {"$in": [...]}}           — IN filter
      {"field": {"$gte": v}}              — >= filter
      {"field": {"$lte": v}}              — <= filter
      {"field": {"$ne": v}}               — != filter
    """
    if not where:
        return query

    if "$and" in where:
        for sub in where["$and"]:
            query = _apply_where_filter(query, sub)
        return query

    if "$or" in where:
        # Firestore doesn't support OR natively in the same way.
        # For now, we only apply the first condition and log a warning.
        logger.warning("$or filters not fully supported in Firestore backend; using first condition only")
        if where["$or"]:
            query = _apply_where_filter(query, where["$or"][0])
        return query

    for key, value in where.items():
        if key.startswith("$"):
            continue
        if isinstance(value, dict):
            if "$in" in value:
                query = query.where(filter=("meta." + key, "in", value["$in"]))
            elif "$gte" in value:
                query = query.where(filter=("meta." + key, ">=", value["$gte"]))
            elif "$lte" in value:
                query = query.where(filter=("meta." + key, "<=", value["$lte"]))
            elif "$ne" in value:
                query = query.where(filter=("meta." + key, "!=", value["$ne"]))
        else:
            query = query.where(filter=("meta." + key, "==", value))

    return query


class FirestoreCollection(BaseCollection):
    """Firestore-backed collection with vector search.

    Document schema in Firestore:
      {
        "document": str,          # the text content
        "embedding": VectorValue, # 384-dim embedding
        "meta": { ... },          # all metadata fields (nested to avoid collisions)
      }

    Metadata is stored under a "meta" prefix to avoid collisions with
    top-level fields (document, embedding). The ChromaDB API surface
    passes metadata as flat dicts; we nest/unnest transparently.
    """

    _BATCH_LIMIT = 450  # Firestore batch limit is 500; leave margin

    def __init__(self, col_ref, embed_fn: Callable = None):
        self._col = col_ref
        self._embed = embed_fn or default_embed_fn

    class _BatchWriter:
        """Auto-chunking batch writer for Firestore."""

        def __init__(self, col_ref, limit: int):
            self._col = col_ref
            self._limit = limit
            self._batch = col_ref.firestore_client.batch()
            self._count = 0

        def set(self, doc_ref, data, merge=False):
            self._batch.set(doc_ref, data, merge=merge)
            self._maybe_flush()

        def update(self, doc_ref, data):
            self._batch.update(doc_ref, data)
            self._maybe_flush()

        def delete(self, doc_ref):
            self._batch.delete(doc_ref)
            self._maybe_flush()

        def _maybe_flush(self):
            self._count += 1
            if self._count >= self._limit:
                self._batch.commit()
                self._batch = self._col.firestore_client.batch()
                self._count = 0

        def commit(self):
            if self._count > 0:
                self._batch.commit()
                self._count = 0

    def _batch(self):
        return self._BatchWriter(self._col, self._BATCH_LIMIT)

    def add(self, *, documents: List[str], ids: List[str],
            metadatas: Optional[List[Dict[str, Any]]] = None) -> None:
        embeddings = self._embed(documents)
        writer = self._batch()
        for i, doc_id in enumerate(ids):
            data = {
                "document": documents[i],
                "embedding": Vector(embeddings[i]),
                "meta": metadatas[i] if metadatas and i < len(metadatas) else {},
            }
            writer.set(self._col.document(doc_id), data)
        writer.commit()

    def upsert(self, *, documents: List[str], ids: List[str],
               metadatas: Optional[List[Dict[str, Any]]] = None) -> None:
        embeddings = self._embed(documents)
        writer = self._batch()
        for i, doc_id in enumerate(ids):
            data = {
                "document": documents[i],
                "embedding": Vector(embeddings[i]),
                "meta": metadatas[i] if metadatas and i < len(metadatas) else {},
            }
            writer.set(self._col.document(doc_id), data, merge=True)
        writer.commit()

    def update(self, *, ids: List[str],
               documents: Optional[List[str]] = None,
               metadatas: Optional[List[Dict[str, Any]]] = None) -> None:
        """Update existing documents (not in BaseCollection, but used by mcp_server)."""
        embeddings = None
        if documents:
            embeddings = self._embed(documents)

        writer = self._batch()
        for i, doc_id in enumerate(ids):
            updates = {}
            if documents and i < len(documents):
                updates["document"] = documents[i]
                updates["embedding"] = Vector(embeddings[i])
            if metadatas and i < len(metadatas):
                updates["meta"] = metadatas[i]
            if updates:
                writer.update(self._col.document(doc_id), updates)
        writer.commit()

    def query(self, **kwargs) -> Dict[str, Any]:
        """Vector similarity search, compatible with ChromaDB response format.

        Expected kwargs:
          query_texts: List[str]  — texts to embed and search for
          n_results: int          — max results
          include: List[str]      — which fields to return
          where: dict             — optional metadata filter
        """
        query_texts = kwargs.get("query_texts", [])
        n_results = kwargs.get("n_results", 5)
        include = kwargs.get("include", ["documents", "metadatas", "distances"])
        where = kwargs.get("where")

        if not query_texts:
            return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}

        query_embedding = self._embed(query_texts[:1])[0]

        q = self._col
        if where:
            q = _apply_where_filter(q, where)

        results = q.find_nearest(
            vector_field="embedding",
            query_vector=Vector(query_embedding),
            distance_measure=DistanceMeasure.COSINE,
            limit=n_results,
            distance_result_field="vector_distance",
        ).get()

        ids = []
        documents = []
        metadatas = []
        distances = []

        for doc_snap in results:
            data = doc_snap.to_dict()
            ids.append(doc_snap.id)
            if "documents" in include:
                documents.append(data.get("document", ""))
            if "metadatas" in include:
                metadatas.append(data.get("meta", {}))
            if "distances" in include:
                distances.append(data.get("vector_distance", 1.0))

        result = {"ids": [ids]}
        if "documents" in include:
            result["documents"] = [documents]
        if "metadatas" in include:
            result["metadatas"] = [metadatas]
        if "distances" in include:
            result["distances"] = [distances]

        return result

    def get(self, **kwargs) -> Dict[str, Any]:
        """Retrieve documents by IDs or metadata filter.

        Expected kwargs:
          ids: List[str]          — document IDs to fetch
          where: dict             — metadata filter
          include: List[str]      — which fields to return
          limit: int              — max results
          offset: int             — skip first N results
        """
        doc_ids = kwargs.get("ids")
        where = kwargs.get("where")
        include = kwargs.get("include", ["documents", "metadatas"])
        limit = kwargs.get("limit")
        offset = kwargs.get("offset", 0)

        snapshots = []

        if doc_ids is not None:
            # Fetch specific documents by ID
            for doc_id in doc_ids:
                snap = self._col.document(doc_id).get()
                if snap.exists:
                    snapshots.append(snap)
        else:
            # Query with optional where filter and pagination
            q = self._col
            if where:
                q = _apply_where_filter(q, where)
            if offset and offset > 0:
                q = q.offset(offset)
            if limit:
                q = q.limit(limit)
            snapshots = list(q.stream())

        ids = []
        documents = []
        metadatas = []

        for snap in snapshots:
            data = snap.to_dict()
            ids.append(snap.id)
            if include and "documents" in include:
                documents.append(data.get("document", ""))
            if include and "metadatas" in include:
                metadatas.append(data.get("meta", {}))

        result = {"ids": ids}
        if include and "documents" in include:
            result["documents"] = documents
        if include and "metadatas" in include:
            result["metadatas"] = metadatas

        return result

    def delete(self, **kwargs) -> None:
        """Delete documents by IDs or metadata filter.

        Expected kwargs:
          ids: List[str]    — document IDs to delete
          where: dict       — metadata filter for bulk delete
        """
        doc_ids = kwargs.get("ids")
        where = kwargs.get("where")

        if doc_ids:
            writer = self._batch()
            for doc_id in doc_ids:
                writer.delete(self._col.document(doc_id))
            writer.commit()
        elif where:
            q = self._col
            q = _apply_where_filter(q, where)
            writer = self._batch()
            for snap in q.stream():
                writer.delete(snap.reference)
            writer.commit()

    def count(self) -> int:
        """Return total document count."""
        agg = self._col.count().get()
        return agg[0][0].value


class FirestoreBackend:
    """Factory for Firestore-backed palace collections.

    Path-agnostic: ``palace_path`` is an opaque prefix. The caller
    decides the scoping scheme. Collections are created at
    ``{palace_path}/{collection_name}``.

    Examples::

        from google.cloud import firestore
        db = firestore.Client()
        backend = FirestoreBackend(db)

        # Per-user scoping (vibetime style)
        col = backend.get_collection("users/abc123", "mempalace_drawers")

        # Per-project scoping
        col = backend.get_collection("projects/myapp", "mempalace_drawers")

        # Flat / single-tenant
        col = backend.get_collection("palace", "mempalace_drawers")
    """

    def __init__(self, db, embed_fn: Callable = None):
        self._db = db
        self._embed = embed_fn or default_embed_fn

    def get_collection(
        self,
        palace_path: str,
        collection_name: str = "mempalace_drawers",
        create: bool = False,
    ) -> FirestoreCollection:
        """Get a Firestore collection at ``{palace_path}/{collection_name}``.

        ``palace_path`` is treated as an opaque Firestore path prefix.
        ``create`` is accepted for interface compatibility but is a no-op
        (Firestore collections are created implicitly on first write).
        """
        col_ref = self._db.collection(f"{palace_path}/{collection_name}")
        return FirestoreCollection(col_ref, embed_fn=self._embed)
