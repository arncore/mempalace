"""Firestore storage backend for MemPalace.

Provides Firestore-backed implementations of the collection, knowledge graph,
and tunnel storage interfaces. All components are path-agnostic — the caller
controls scoping via an opaque ``palace_path`` or ``base_path`` prefix.

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
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "FirestoreBackend",
    "FirestoreCollection",
    "FirestoreKnowledgeGraph",
    "FirestoreTunnelStore",
]
