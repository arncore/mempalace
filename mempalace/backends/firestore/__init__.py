"""Firestore storage backend for MemPalace (RFC 001).

Provides Firestore-backed implementations of the collection, knowledge graph,
and tunnel storage interfaces. ``FirestoreBackend`` implements the
``mempalace.backends.base.BaseBackend`` contract and is published via the
``mempalace.backends`` entry-point group, so registry discovery picks it up
when ``google-cloud-firestore`` and ``sentence-transformers`` are installed
(``pip install mempalace[firestore]``).

All components are path-agnostic — the caller controls scoping via the
``PalaceRef.namespace`` (or legacy ``palace_path``) prefix.

Usage::

    from mempalace.backends.firestore import (
        FirestoreBackend,
        FirestoreCollection,
        FirestoreKnowledgeGraph,
        FirestoreTunnelStore,
    )
"""


def __getattr__(name: str):
    if name in ("FirestoreBackend", "FirestoreCollection"):
        from .collection import FirestoreBackend, FirestoreCollection

        return FirestoreBackend if name == "FirestoreBackend" else FirestoreCollection
    if name == "FirestoreKnowledgeGraph":
        from .knowledge_graph import FirestoreKnowledgeGraph

        return FirestoreKnowledgeGraph
    if name == "FirestoreTunnelStore":
        from .tunnels import FirestoreTunnelStore

        return FirestoreTunnelStore
    if name == "register_firestore_backend":
        return _register_firestore_backend
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _register_firestore_backend() -> None:
    """Explicitly register ``FirestoreBackend`` with the in-process registry.

    Useful in editable / source checkouts where entry-point discovery has not
    been triggered (no ``pip install -e .``), or in tests that need the
    backend available before any installer hook runs. Callers that installed
    the package normally do not need this — entry-point discovery runs
    automatically the first time the registry is consulted.
    """
    from ..registry import register
    from .collection import FirestoreBackend

    register("firestore", FirestoreBackend)


__all__ = [
    "FirestoreBackend",
    "FirestoreCollection",
    "FirestoreKnowledgeGraph",
    "FirestoreTunnelStore",
    "register_firestore_backend",
]
