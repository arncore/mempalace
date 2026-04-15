"""Storage backend implementations for MemPalace."""

from .base import BaseCollection
from .chroma import ChromaBackend, ChromaCollection

__all__ = ["BaseCollection", "ChromaBackend", "ChromaCollection"]

# Firestore backend is imported lazily to avoid hard dependency on
# google-cloud-firestore for users who only need ChromaDB.
# Use: from mempalace.backends.firestore import FirestoreBackend, FirestoreCollection
