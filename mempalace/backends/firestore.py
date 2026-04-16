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

from google.cloud.firestore_v1.base_query import FieldFilter, Or
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


def _build_field_filter(key: str, value) -> FieldFilter:
    """Build a single FieldFilter from a ChromaDB field condition.

    Handles both plain equality (``{"field": "value"}``) and operator
    forms (``{"field": {"$gte": 5}}``).

    ChromaDB raises if more than one operator is present in a single
    expression dict (e.g. ``{"$gte": 5, "$lte": 10}``).
    """
    if isinstance(value, dict):
        op_map = {
            "$eq": "==",
            "$ne": "!=",
            "$gt": ">",
            "$gte": ">=",
            "$lt": "<",
            "$lte": "<=",
            "$in": "in",
            "$nin": "not-in",
        }
        if len(value) > 1:
            raise ValueError(
                f"Expected operator expression to have exactly one operator, got {value}"
            )
        for op, v in value.items():
            if op in op_map:
                return FieldFilter("meta." + key, op_map[op], v)
        raise ValueError(f"Unsupported operator in where filter: {list(value.keys())}")
    return FieldFilter("meta." + key, "==", value)


def _collect_field_filters(where: Dict[str, Any]) -> List[FieldFilter]:
    """Collect FieldFilter objects from a where dict, handling nested composites.

    Supports plain field conditions and recursively expands ``$and`` blocks
    so that ``$or`` containing ``$and`` sub-filters works correctly.
    """
    filters = []
    for key, value in where.items():
        if key == "$and":
            for sub in value:
                filters.extend(_collect_field_filters(sub))
        elif key.startswith("$"):
            continue
        else:
            filters.append(_build_field_filter(key, value))
    return filters


def _apply_where_filter(query, where: Dict[str, Any]):
    """Translate ChromaDB where-filter syntax to Firestore query chains.

    Supports:
      {"field": "value"}                  — equality
      {"field": {"$eq": v}}               — equality (explicit)
      {"field": {"$ne": v}}               — != filter
      {"field": {"$gt": v}}               — > filter
      {"field": {"$gte": v}}              — >= filter
      {"field": {"$lt": v}}               — < filter
      {"field": {"$lte": v}}              — <= filter
      {"field": {"$in": [...]}}           — IN filter
      {"field": {"$nin": [...]}}          — NOT-IN filter
      {"$and": [{...}, {...}]}            — AND of filters
      {"$or": [{...}, {...}]}             — OR of filters (Firestore Or)
    """
    if not where:
        return query

    if "$and" in where:
        for sub in where["$and"]:
            query = _apply_where_filter(query, sub)
        return query

    if "$or" in where:
        or_filters = []
        for sub in where["$or"]:
            or_filters.extend(_collect_field_filters(sub))
        return query.where(filter=Or(filters=or_filters))

    for key, value in where.items():
        if key.startswith("$"):
            continue
        query = query.where(filter=_build_field_filter(key, value))

    return query


def _matches_where_document(doc_text: str, where_document: Dict[str, Any]) -> bool:
    """Evaluate a ChromaDB where_document filter against document text.

    Supports:
      {"$contains": "substring"}         — text contains substring
      {"$not_contains": "substring"}     — text does not contain substring
      {"$and": [{...}, {...}]}           — all conditions match
      {"$or": [{...}, {...}]}            — any condition matches
    """
    if "$and" in where_document:
        return all(_matches_where_document(doc_text, sub) for sub in where_document["$and"])
    if "$or" in where_document:
        return any(_matches_where_document(doc_text, sub) for sub in where_document["$or"])
    if "$contains" in where_document:
        return where_document["$contains"] in doc_text
    if "$not_contains" in where_document:
        return where_document["$not_contains"] not in doc_text
    return False


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

    def __init__(self, col_ref, db_client, embed_fn: Optional[Callable] = None):
        self._col = col_ref
        self._db = db_client
        self._embed = embed_fn or default_embed_fn

    class _BatchWriter:
        """Auto-chunking batch writer for Firestore."""

        def __init__(self, db_client, limit: int):
            self._db = db_client
            self._limit = limit
            self._batch = db_client.batch()
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
                self._batch = self._db.batch()
                self._count = 0

        def commit(self):
            if self._count > 0:
                self._batch.commit()
                self._count = 0

    def _batch(self):
        return self._BatchWriter(self._db, self._BATCH_LIMIT)

    def add(
        self,
        *,
        documents: List[str],
        ids: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        embeddings = self._embed(documents)
        writer = self._batch()
        for i, doc_id in enumerate(ids):
            doc_ref = self._col.document(doc_id)
            # ChromaDB silently ignores duplicate IDs — skip if doc exists.
            if doc_ref.get().exists:
                continue
            data = {
                "document": documents[i],
                "embedding": Vector(embeddings[i]),
                "meta": metadatas[i] if metadatas and i < len(metadatas) else {},
            }
            writer.set(doc_ref, data)
        writer.commit()

    def upsert(
        self,
        *,
        documents: List[str],
        ids: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
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

    def update(
        self,
        *,
        ids: List[str],
        documents: Optional[List[str]] = None,
        metadatas: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Update existing documents (not in BaseCollection, but used by mcp_server).

        ChromaDB silently skips nonexistent IDs — we check existence first.
        """
        embeddings: Optional[List[List[float]]] = None
        if documents:
            embeddings = self._embed(documents)

        writer = self._batch()
        for i, doc_id in enumerate(ids):
            updates = {}
            if documents and embeddings and i < len(documents):
                updates["document"] = documents[i]
                updates["embedding"] = Vector(embeddings[i])
            if metadatas and i < len(metadatas):
                updates["meta"] = metadatas[i]
            if updates:
                doc_ref = self._col.document(doc_id)
                # ChromaDB silently ignores updates to nonexistent docs.
                if not doc_ref.get().exists:
                    continue
                writer.update(doc_ref, updates)
        writer.commit()

    def query(self, **kwargs) -> Dict[str, Any]:
        """Vector similarity search, compatible with ChromaDB response format.

        Expected kwargs:
          query_texts: List[str]  — texts to embed and search for
          n_results: int          — max results
          include: List[str]      — which fields to return
          where: dict             — optional metadata filter

        Returns one result set per query text (nested lists), matching
        the ChromaDB response shape::

            {"ids": [["a"], ["b"]], "distances": [[0.2], [0.3]]}
        """
        query_texts = kwargs.get("query_texts", [])
        n_results = kwargs.get("n_results", 5)
        include = kwargs.get("include", ["documents", "metadatas", "distances"])
        where = kwargs.get("where")

        if not query_texts:
            return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}

        query_embeddings = self._embed(query_texts)

        all_ids: List[List[str]] = []
        all_documents: List[List[str]] = []
        all_metadatas: List[List[Dict[str, Any]]] = []
        all_distances: List[List[float]] = []

        for query_embedding in query_embeddings:
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

            ids: List[str] = []
            documents: List[str] = []
            metadatas: List[Dict[str, Any]] = []
            distances: List[float] = []

            for doc_snap in results:
                data = doc_snap.to_dict()
                ids.append(doc_snap.id)
                if "documents" in include:
                    documents.append(data.get("document", ""))
                if "metadatas" in include:
                    metadatas.append(data.get("meta", {}))
                if "distances" in include:
                    distances.append(data.get("vector_distance", 1.0))

            all_ids.append(ids)
            all_documents.append(documents)
            all_metadatas.append(metadatas)
            all_distances.append(distances)

        # ChromaDB always returns all keys; non-included fields are None.
        result: Dict[str, Any] = {"ids": all_ids}
        result["documents"] = all_documents if "documents" in include else None
        result["metadatas"] = all_metadatas if "metadatas" in include else None
        result["distances"] = all_distances if "distances" in include else None

        return result

    def get(self, **kwargs) -> Dict[str, Any]:
        """Retrieve documents by IDs or metadata filter.

        Expected kwargs:
          ids: List[str]          — document IDs to fetch
          where: dict             — metadata filter
          include: List[str]      — which fields to return
          limit: int              — max results
          offset: int             — skip first N results

        ChromaDB raises ValueError on ``ids=[]``. Non-included fields
        are returned as ``None`` (not omitted).
        """
        doc_ids = kwargs.get("ids")
        where = kwargs.get("where")
        include = kwargs.get("include", ["documents", "metadatas"])
        limit = kwargs.get("limit")
        offset = kwargs.get("offset", 0)

        if doc_ids is not None and len(doc_ids) == 0:
            raise ValueError("Expected IDs to be a non-empty list, got 0 IDs")

        snapshots = []

        if doc_ids is not None:
            # Batch-fetch documents by ID
            doc_refs = [self._col.document(doc_id) for doc_id in doc_ids]
            snapshots = [snap for snap in self._db.get_all(doc_refs) if snap.exists]
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

        # ChromaDB always returns all keys; non-included fields are None.
        result: Dict[str, Any] = {"ids": ids}
        result["documents"] = documents if (include and "documents" in include) else None
        result["metadatas"] = metadatas if (include and "metadatas" in include) else None

        return result

    def delete(self, **kwargs) -> None:
        """Delete documents by IDs or metadata filter.

        Expected kwargs:
          ids: List[str]           — document IDs to delete
          where: dict              — metadata filter for bulk delete
          where_document: dict     — document content filter

        Raises ValueError if none of ids, where, or where_document is provided,
        matching ChromaDB behaviour. Also raises ValueError on ``ids=[]``
        and NotImplementedError for ``where_document`` (Firestore cannot do
        full-text search).
        """
        doc_ids = kwargs.get("ids")
        where = kwargs.get("where")
        where_document = kwargs.get("where_document")

        # ChromaDB raises on empty ID list
        if doc_ids is not None and len(doc_ids) == 0:
            raise ValueError("Expected IDs to be a non-empty list, got 0 IDs")

        if not doc_ids and not where and not where_document:
            raise ValueError(
                "At least one of ids, where, or where_document must be provided in delete."
            )

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
        elif where_document:
            # Firestore has no full-text search. Simulate by streaming all
            # documents and filtering client-side. O(n) reads.
            writer = self._batch()
            for snap in self._col.stream():
                doc_text = snap.to_dict().get("document", "")
                if _matches_where_document(doc_text, where_document):
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

    def __init__(self, db, embed_fn: Optional[Callable] = None):
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
        return FirestoreCollection(col_ref, db_client=self._db, embed_fn=self._embed)
