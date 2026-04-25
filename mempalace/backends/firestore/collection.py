"""Firestore-backed MemPalace collection adapter with vector search.

Implements the RFC 001 ``BaseCollection`` / ``BaseBackend`` contract using
Firestore for document storage and ``find_nearest()`` for vector similarity
search. Embeddings are generated via sentence-transformers using the same
``all-MiniLM-L6-v2`` model ChromaDB uses internally so search quality is
identical.

The backend is path-agnostic: ``palace_path`` (legacy positional) or
``PalaceRef.namespace`` / ``PalaceRef.id`` (new kwargs-only path) is treated
as an opaque Firestore prefix. The caller decides scoping (per-user,
per-project, per-org, etc.).
"""

import logging
from typing import Any, Callable, ClassVar, Dict, List, Optional

from google.cloud.firestore_v1.base_query import FieldFilter, Or
from google.cloud.firestore_v1.base_vector_query import DistanceMeasure
from google.cloud.firestore_v1.vector import Vector

from ..base import (
    BackendClosedError,
    BaseBackend,
    BaseCollection,
    GetResult,
    HealthStatus,
    PalaceNotFoundError,
    PalaceRef,
    QueryResult,
    UnsupportedFilterError,
    _IncludeSpec,
)

logger = logging.getLogger("mempalace.backends.firestore")

# Lazy-loaded embedding model — only initialized on first use.
_embed_model = None


_OP_MAP = {
    "$eq": "==",
    "$ne": "!=",
    "$gt": ">",
    "$gte": ">=",
    "$lt": "<",
    "$lte": "<=",
    "$in": "in",
    "$nin": "not-in",
}

# Operators we accept inside `where` (metadata filter).
_SUPPORTED_WHERE_OPERATORS = frozenset({"$and", "$or"} | set(_OP_MAP.keys()))
# Operators we accept inside `where_document`.
_SUPPORTED_WHERE_DOCUMENT_OPERATORS = frozenset(
    {"$contains", "$not_contains", "$and", "$or"}
)


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


def _validate_where(where: Optional[dict], *, allowed: frozenset) -> None:
    """Walk a where-clause and raise ``UnsupportedFilterError`` on unknown ops.

    RFC 001 §1.4 forbids silently dropping unknown ``$``-prefixed operators.
    """
    if not where:
        return
    stack: list = [where]
    while stack:
        node = stack.pop()
        if not isinstance(node, dict):
            continue
        for k, v in node.items():
            if k.startswith("$") and k not in allowed:
                raise UnsupportedFilterError(
                    f"operator {k!r} not supported by firestore backend"
                )
            if isinstance(v, dict):
                stack.append(v)
            elif isinstance(v, list):
                stack.extend(x for x in v if isinstance(x, dict))


def _build_field_filter(key: str, value) -> FieldFilter:
    """Build a single FieldFilter from a ChromaDB field condition.

    Handles plain equality (``{"field": "value"}``) and operator forms
    (``{"field": {"$gte": 5}}``). ChromaDB raises if a single expression
    dict has more than one operator.
    """
    if isinstance(value, dict):
        if len(value) > 1:
            raise ValueError(
                f"Expected operator expression to have exactly one operator, got {value}"
            )
        for op, v in value.items():
            if op in _OP_MAP:
                return FieldFilter("meta." + key, _OP_MAP[op], v)
        raise ValueError(f"Unsupported operator in where filter: {list(value.keys())}")
    return FieldFilter("meta." + key, "==", value)


def _collect_field_filters(where: Dict[str, Any]) -> List[FieldFilter]:
    """Collect FieldFilter objects from a where dict, expanding nested $and."""
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
    """Translate ChromaDB where-filter syntax to Firestore query chains."""
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
    """Evaluate a ChromaDB where_document filter against document text."""
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

    Document schema in Firestore::

      {
        "document": str,          # the text content
        "embedding": VectorValue, # 384-dim embedding
        "meta": { ... },          # all metadata fields (nested to avoid collisions)
      }

    Metadata is stored under ``meta.`` to avoid collisions with the top-level
    ``document`` and ``embedding`` fields. The MemPalace API surface passes
    metadata as flat dicts; we nest/unnest transparently.
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

    # ------------------------------------------------------------------
    # Writes
    # ------------------------------------------------------------------

    def add(
        self,
        *,
        documents: List[str],
        ids: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        embeddings: Optional[List[List[float]]] = None,
    ) -> None:
        embeds = embeddings if embeddings is not None else self._embed(documents)
        writer = self._batch()
        for i, doc_id in enumerate(ids):
            doc_ref = self._col.document(doc_id)
            # ChromaDB silently ignores duplicate IDs — skip if doc exists.
            if doc_ref.get().exists:
                continue
            data = {
                "document": documents[i],
                "embedding": Vector(embeds[i]),
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
        embeddings: Optional[List[List[float]]] = None,
    ) -> None:
        embeds = embeddings if embeddings is not None else self._embed(documents)
        writer = self._batch()
        for i, doc_id in enumerate(ids):
            data = {
                "document": documents[i],
                "embedding": Vector(embeds[i]),
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
        embeddings: Optional[List[List[float]]] = None,
    ) -> None:
        """Atomic per-id update.

        Overrides the BaseCollection default (get + merge + upsert) with a
        single-round-trip Firestore update for documents that exist. Unlike
        the default, this implementation tolerates length-mismatched
        ``documents`` / ``metadatas`` lists, applying each only where its
        index is in range. ChromaDB silently skips updates to nonexistent
        IDs; we do the same.
        """
        embeds: Optional[List[List[float]]] = embeddings
        if documents and embeds is None:
            embeds = self._embed(documents)

        writer = self._batch()
        for i, doc_id in enumerate(ids):
            updates: dict[str, Any] = {}
            if documents and embeds and i < len(documents):
                updates["document"] = documents[i]
                updates["embedding"] = Vector(embeds[i])
            if metadatas and i < len(metadatas):
                updates["meta"] = metadatas[i]
            if updates:
                doc_ref = self._col.document(doc_id)
                # ChromaDB silently ignores updates to nonexistent docs.
                if not doc_ref.get().exists:
                    continue
                writer.update(doc_ref, updates)
        writer.commit()

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------

    def query(
        self,
        *,
        query_texts: Optional[List[str]] = None,
        query_embeddings: Optional[List[List[float]]] = None,
        n_results: int = 10,
        where: Optional[dict] = None,
        where_document: Optional[dict] = None,
        include: Optional[List[str]] = None,
    ) -> QueryResult:
        _validate_where(where, allowed=_SUPPORTED_WHERE_OPERATORS)
        _validate_where(where_document, allowed=_SUPPORTED_WHERE_DOCUMENT_OPERATORS)

        spec = _IncludeSpec.resolve(include, default_distances=True)

        if (query_texts is None) == (query_embeddings is None):
            raise ValueError("query requires exactly one of query_texts or query_embeddings")
        chosen = query_texts if query_texts is not None else query_embeddings
        if not chosen:
            raise ValueError("query input must be a non-empty list")

        if query_embeddings is None:
            assert query_texts is not None
            query_embeddings = self._embed(query_texts)

        all_ids: List[List[str]] = []
        all_documents: List[List[str]] = []
        all_metadatas: List[List[Dict[str, Any]]] = []
        all_distances: List[List[float]] = []
        all_embeddings: List[List[List[float]]] = []

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
            embeddings_inner: List[List[float]] = []

            for doc_snap in results:
                data = doc_snap.to_dict()
                if where_document and not _matches_where_document(
                    data.get("document", ""), where_document
                ):
                    continue
                ids.append(doc_snap.id)
                documents.append(data.get("document", ""))
                metadatas.append(data.get("meta", {}))
                distances.append(data.get("vector_distance", 1.0))
                if spec.embeddings:
                    raw_embed = data.get("embedding")
                    # Vector values may be exposed via .values or be list-like.
                    embed_values = (
                        list(raw_embed.values)
                        if hasattr(raw_embed, "values")
                        else list(raw_embed) if raw_embed is not None
                        else []
                    )
                    embeddings_inner.append(embed_values)

            all_ids.append(ids)
            # Per RFC 001 §1.3: fields not in include= preserve the outer
            # query dimension but carry empty inner lists. embeddings is
            # the only field that flips to None when not requested.
            all_documents.append(documents if spec.documents else [])
            all_metadatas.append(metadatas if spec.metadatas else [])
            all_distances.append(distances if spec.distances else [])
            all_embeddings.append(embeddings_inner)

        return QueryResult(
            ids=all_ids,
            documents=all_documents,
            metadatas=all_metadatas,
            distances=all_distances,
            embeddings=all_embeddings if spec.embeddings else None,
        )

    def get(
        self,
        *,
        ids: Optional[List[str]] = None,
        where: Optional[dict] = None,
        where_document: Optional[dict] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        include: Optional[List[str]] = None,
    ) -> GetResult:
        _validate_where(where, allowed=_SUPPORTED_WHERE_OPERATORS)
        _validate_where(where_document, allowed=_SUPPORTED_WHERE_DOCUMENT_OPERATORS)

        spec = _IncludeSpec.resolve(include, default_distances=False)

        if ids is not None and len(ids) == 0:
            raise ValueError("Expected IDs to be a non-empty list, got 0 IDs")

        snapshots: list = []

        if ids is not None:
            doc_refs = [self._col.document(doc_id) for doc_id in ids]
            snapshots = [snap for snap in self._db.get_all(doc_refs) if snap.exists]
        else:
            q = self._col
            if where:
                q = _apply_where_filter(q, where)
            if offset and offset > 0:
                q = q.offset(offset)
            if limit:
                q = q.limit(limit)
            snapshots = list(q.stream())

        # where_document is a client-side filter regardless of how we got here.
        if where_document:
            snapshots = [
                s for s in snapshots
                if _matches_where_document(s.to_dict().get("document", ""), where_document)
            ]

        out_ids: List[str] = []
        out_docs: List[str] = []
        out_metas: List[Dict[str, Any]] = []
        out_embeds: List[List[float]] = []

        for snap in snapshots:
            data = snap.to_dict()
            out_ids.append(snap.id)
            out_docs.append(data.get("document", ""))
            out_metas.append(data.get("meta", {}))
            if spec.embeddings:
                raw_embed = data.get("embedding")
                embed_values = (
                    list(raw_embed.values)
                    if hasattr(raw_embed, "values")
                    else list(raw_embed) if raw_embed is not None
                    else []
                )
                out_embeds.append(embed_values)

        # Pad doc/meta lists to match ids if data was sparse.
        if len(out_docs) < len(out_ids):
            out_docs = out_docs + [""] * (len(out_ids) - len(out_docs))
        if len(out_metas) < len(out_ids):
            out_metas = out_metas + [{}] * (len(out_ids) - len(out_metas))

        return GetResult(
            ids=out_ids,
            documents=out_docs if spec.documents else [],
            metadatas=out_metas if spec.metadatas else [],
            embeddings=out_embeds if spec.embeddings else None,
        )

    def delete(
        self,
        *,
        ids: Optional[List[str]] = None,
        where: Optional[dict] = None,
        where_document: Optional[dict] = None,
    ) -> None:
        _validate_where(where, allowed=_SUPPORTED_WHERE_OPERATORS)
        _validate_where(where_document, allowed=_SUPPORTED_WHERE_DOCUMENT_OPERATORS)

        if ids is not None and len(ids) == 0:
            raise ValueError("Expected IDs to be a non-empty list, got 0 IDs")

        if not ids and not where and not where_document:
            raise ValueError(
                "At least one of ids, where, or where_document must be provided in delete."
            )

        if ids:
            writer = self._batch()
            for doc_id in ids:
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
            # Firestore has no full-text search. Simulate by streaming the
            # collection and filtering client-side. O(n) reads.
            writer = self._batch()
            for snap in self._col.stream():
                doc_text = snap.to_dict().get("document", "")
                if _matches_where_document(doc_text, where_document):
                    writer.delete(snap.reference)
            writer.commit()

    def count(self) -> int:
        agg = self._col.count().get()
        return agg[0][0].value


# ---------------------------------------------------------------------------
# Backend
# ---------------------------------------------------------------------------


def _resolve_palace_prefix(palace: PalaceRef) -> str:
    """Pick the Firestore path prefix for a PalaceRef.

    ``namespace`` is the spec-blessed location for server-mode tenant
    routing; if absent, fall back to ``id``. ``local_path`` is ignored —
    Firestore is not a filesystem-rooted backend.
    """
    return palace.namespace or palace.id


class FirestoreBackend(BaseBackend):
    """RFC 001 ``BaseBackend`` implementation backed by Firestore.

    Two calling conventions are supported during the RFC 001 transition:

    * **New (preferred)**::

        backend = FirestoreBackend()  # uses google.cloud.firestore.Client()
        col = backend.get_collection(
            palace=PalaceRef(id="users/abc", namespace="users/abc"),
            collection_name="mempalace_drawers",
            create=True,
        )

    * **Legacy positional** (used by ``palace.get_collection`` and pre-RFC-001
      tests)::

        backend = FirestoreBackend(db, embed_fn=...)
        col = backend.get_collection("users/abc", "mempalace_drawers", create=True)

    Construction is lightweight by spec — when called with no ``db`` arg, the
    client is created lazily on first ``get_collection`` so registry-driven
    instantiation (``cls()``) does not perform I/O at import time.
    """

    name: ClassVar[str] = "firestore"
    spec_version: ClassVar[str] = "1.0"
    capabilities: ClassVar[frozenset[str]] = frozenset(
        {
            "supports_embeddings_in",
            "supports_embeddings_passthrough",
            "supports_embeddings_out",
            "supports_metadata_filters",
            "server_mode",
        }
    )

    def __init__(self, db=None, embed_fn: Optional[Callable] = None):
        self._db = db
        self._embed = embed_fn or default_embed_fn
        self._closed = False

    def _client(self):
        if self._closed:
            raise BackendClosedError("FirestoreBackend has been closed")
        if self._db is None:
            from google.cloud import firestore as firestore_mod

            self._db = firestore_mod.Client()
        return self._db

    def get_collection(
        self,
        *args,
        **kwargs,
    ) -> FirestoreCollection:
        """Obtain a Firestore-backed collection for a palace.

        Accepts both new (``palace=PalaceRef``) and legacy positional
        (``palace_path, collection_name, create=False``) signatures.
        """
        palace_ref, collection_name, _create, _options = _normalize_get_collection_args(
            args, kwargs
        )

        prefix = _resolve_palace_prefix(palace_ref)
        if not prefix:
            raise PalaceNotFoundError("FirestoreBackend requires PalaceRef.namespace or .id")

        db = self._client()
        col_ref = db.collection(f"{prefix}/{collection_name}")
        # Firestore creates collections implicitly on first write — `create`
        # is a no-op, kept for interface parity with chroma.
        return FirestoreCollection(col_ref, db_client=db, embed_fn=self._embed)

    def close(self) -> None:
        # Firestore client has no explicit close; mark and drop the handle.
        self._db = None
        self._closed = True

    def health(self, palace: Optional[PalaceRef] = None) -> HealthStatus:
        if self._closed:
            return HealthStatus.unhealthy("backend closed")
        return HealthStatus.healthy()

    @classmethod
    def detect(cls, path: str) -> bool:
        # Firestore is not on-disk; auto-detection from a local path is
        # never appropriate. Selection must come from explicit config or env.
        return False


def _normalize_get_collection_args(args, kwargs):
    """Unify legacy positional ``(palace_path, collection_name, create)`` calls
    with the new kwargs-only ``(palace=PalaceRef, collection_name=..., create=...)``.

    Returns ``(PalaceRef, collection_name, create, options)``.
    """
    if "palace" in kwargs:
        palace_ref = kwargs.pop("palace")
        if not isinstance(palace_ref, PalaceRef):
            raise TypeError("palace= must be a PalaceRef instance")
        collection_name = kwargs.pop("collection_name", "mempalace_drawers")
        create = bool(kwargs.pop("create", False))
        options = kwargs.pop("options", None)
        if kwargs:
            raise TypeError(f"unexpected kwargs: {sorted(kwargs)}")
        if args:
            raise TypeError("positional args not allowed with palace= kwarg")
        return palace_ref, collection_name, create, options

    # Legacy: first positional is a path string.
    if args:
        palace_path = args[0]
        rest = list(args[1:])
        collection_name = (
            kwargs.pop("collection_name", None)
            or (rest.pop(0) if rest else None)
            or "mempalace_drawers"
        )
        create = bool(kwargs.pop("create", False))
        if rest:
            create = bool(rest.pop(0))
        if rest:
            raise TypeError(f"unexpected positional args: {rest!r}")
        if kwargs:
            raise TypeError(f"unexpected kwargs: {sorted(kwargs)}")
        return (
            PalaceRef(id=palace_path, namespace=palace_path),
            collection_name,
            create,
            None,
        )

    if "palace_path" in kwargs:
        palace_path = kwargs.pop("palace_path")
        collection_name = kwargs.pop("collection_name", "mempalace_drawers")
        create = bool(kwargs.pop("create", False))
        if kwargs:
            raise TypeError(f"unexpected kwargs: {sorted(kwargs)}")
        return (
            PalaceRef(id=palace_path, namespace=palace_path),
            collection_name,
            create,
            None,
        )

    raise TypeError("get_collection requires palace= or a positional palace_path")
