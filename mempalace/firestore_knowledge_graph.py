"""
firestore_knowledge_graph.py — Firestore-backed Knowledge Graph
================================================================

Drop-in replacement for knowledge_graph.py's KnowledgeGraph class.
Same interface (add_entity, add_triple, invalidate, query_entity,
timeline, stats), backed by Firestore instead of SQLite.

Path-agnostic: the caller provides a ``base_path`` that determines
where entities and triples are stored. No collection names or
scoping schemes are hardcoded.

Examples::

    from google.cloud import firestore
    db = firestore.Client()

    # Per-user
    kg = FirestoreKnowledgeGraph(db, base_path="users/abc123")

    # Per-project
    kg = FirestoreKnowledgeGraph(db, base_path="projects/myapp")
"""

from datetime import date, datetime

from google.cloud.firestore_v1.transaction import transactional


class FirestoreKnowledgeGraph:
    """Knowledge graph backed by Firestore.

    Stores entities at ``{base_path}/entities/{id}`` and triples at
    ``{base_path}/triples/{id}``.
    """

    def __init__(self, db, base_path: str):
        self._db = db
        self._entities = db.collection(f"{base_path}/entities")
        self._triples = db.collection(f"{base_path}/triples")
        self._name_cache: dict = {}

    def _resolve_name(self, entity_id: str) -> str:
        """Resolve an entity ID to its display name, with caching."""
        if entity_id not in self._name_cache:
            snap = self._entities.document(entity_id).get()
            self._name_cache[entity_id] = (
                snap.to_dict().get("name", entity_id) if snap.exists else entity_id
            )
        return self._name_cache[entity_id]

    def _entity_id(self, name: str) -> str:
        return name.lower().replace(" ", "_").replace("'", "")

    # ── Write operations ─────────────────────────────────────────────────

    def add_entity(self, name: str, entity_type: str = "unknown", properties: dict = None):
        """Add or update an entity node."""
        eid = self._entity_id(name)
        self._entities.document(eid).set(
            {
                "name": name,
                "type": entity_type,
                "properties": properties or {},
                "created_at": datetime.now().isoformat(),
            },
            merge=True,
        )
        return eid

    def add_triple(
        self,
        subject: str,
        predicate: str,
        obj: str,
        valid_from: str = None,
        valid_to: str = None,
        confidence: float = 1.0,
        source_closet: str = None,
        source_file: str = None,
    ):
        """Add a relationship triple: subject -> predicate -> object.

        Auto-creates entities if they don't exist. Deduplicates: if an
        identical active triple (same subject/predicate/object, valid_to
        is None) already exists, returns its ID without creating a new one.

        Uses a deterministic document ID for active triples and a Firestore
        transaction for atomic dedup-check + write, preventing race conditions
        from concurrent adds of the same triple.
        """
        sub_id = self._entity_id(subject)
        obj_id = self._entity_id(obj)
        pred = predicate.lower().replace(" ", "_")

        # Auto-create entities (idempotent via merge=True)
        self._entities.document(sub_id).set({"name": subject}, merge=True)
        self._entities.document(obj_id).set({"name": obj}, merge=True)

        # Deterministic ID for the active triple
        triple_id = f"t_{sub_id}_{pred}_{obj_id}"
        doc_ref = self._triples.document(triple_id)

        data = {
            "subject": sub_id,
            "predicate": pred,
            "object": obj_id,
            "valid_from": valid_from,
            "valid_to": valid_to,
            "confidence": confidence,
            "source_closet": source_closet,
            "source_file": source_file,
            "extracted_at": datetime.now().isoformat(),
        }

        @transactional
        def _add_in_txn(transaction):
            snap = doc_ref.get(transaction=transaction)
            if snap.exists and snap.to_dict().get("valid_to") is None:
                return snap.id
            transaction.set(doc_ref, data)
            return triple_id

        transaction = self._db.transaction()
        return _add_in_txn(transaction)

    def invalidate(self, subject: str, predicate: str, obj: str, ended: str = None):
        """Mark a relationship as no longer valid (set valid_to date).

        Uses a Firestore transaction for atomic read + update, preventing
        race conditions from concurrent invalidations.
        """
        sub_id = self._entity_id(subject)
        obj_id = self._entity_id(obj)
        pred = predicate.lower().replace(" ", "_")
        ended = ended or date.today().isoformat()

        # Deterministic ID for the active triple
        triple_id = f"t_{sub_id}_{pred}_{obj_id}"
        doc_ref = self._triples.document(triple_id)

        @transactional
        def _invalidate_in_txn(transaction):
            snap = doc_ref.get(transaction=transaction)
            if snap.exists and snap.to_dict().get("valid_to") is None:
                transaction.update(doc_ref, {"valid_to": ended})

        transaction = self._db.transaction()
        _invalidate_in_txn(transaction)

    # ── Query operations ─────────────────────────────────────────────────

    def query_entity(self, name: str, as_of: str = None, direction: str = "outgoing"):
        """Get all relationships for an entity.

        direction: "outgoing" (entity → ?), "incoming" (? → entity), "both"
        as_of: date string — only return facts valid at that time
        """
        eid = self._entity_id(name)
        results = []

        if direction in ("outgoing", "both"):
            query = self._triples.where("subject", "==", eid)
            for row in query.stream():
                data = row.to_dict()
                if as_of:
                    vf = data.get("valid_from")
                    vt = data.get("valid_to")
                    if vf and vf > as_of:
                        continue
                    if vt and vt < as_of:
                        continue

                results.append(
                    {
                        "direction": "outgoing",
                        "subject": name,
                        "predicate": data["predicate"],
                        "object": self._resolve_name(data["object"]),
                        "valid_from": data.get("valid_from"),
                        "valid_to": data.get("valid_to"),
                        "confidence": data.get("confidence", 1.0),
                        "source_closet": data.get("source_closet"),
                        "current": data.get("valid_to") is None,
                    }
                )

        if direction in ("incoming", "both"):
            query = self._triples.where("object", "==", eid)
            for row in query.stream():
                data = row.to_dict()
                if as_of:
                    vf = data.get("valid_from")
                    vt = data.get("valid_to")
                    if vf and vf > as_of:
                        continue
                    if vt and vt < as_of:
                        continue

                results.append(
                    {
                        "direction": "incoming",
                        "subject": self._resolve_name(data["subject"]),
                        "predicate": data["predicate"],
                        "object": name,
                        "valid_from": data.get("valid_from"),
                        "valid_to": data.get("valid_to"),
                        "confidence": data.get("confidence", 1.0),
                        "source_closet": data.get("source_closet"),
                        "current": data.get("valid_to") is None,
                    }
                )

        return results

    def query_relationship(self, predicate: str, as_of: str = None):
        """Get all triples with a given relationship type."""
        pred = predicate.lower().replace(" ", "_")
        query = self._triples.where("predicate", "==", pred)

        results = []
        for row in query.stream():
            data = row.to_dict()
            if as_of:
                vf = data.get("valid_from")
                vt = data.get("valid_to")
                if vf and vf > as_of:
                    continue
                if vt and vt < as_of:
                    continue

            results.append(
                {
                    "subject": self._resolve_name(data["subject"]),
                    "predicate": pred,
                    "object": self._resolve_name(data["object"]),
                    "valid_from": data.get("valid_from"),
                    "valid_to": data.get("valid_to"),
                    "current": data.get("valid_to") is None,
                }
            )
        return results

    def timeline(self, entity_name: str = None):
        """Get all facts in chronological order, optionally filtered by entity."""
        if entity_name:
            eid = self._entity_id(entity_name)
            # Firestore can't do OR on different fields, so we query both
            # and merge.
            outgoing = list(
                self._triples.where("subject", "==", eid)
                .order_by("valid_from")
                .limit(100)
                .stream()
            )
            incoming = list(
                self._triples.where("object", "==", eid)
                .order_by("valid_from")
                .limit(100)
                .stream()
            )
            rows = outgoing + incoming
        else:
            rows = list(
                self._triples.order_by("valid_from").limit(100).stream()
            )

        results = []
        for row in rows:
            data = row.to_dict()
            results.append(
                {
                    "subject": self._resolve_name(data["subject"]),
                    "predicate": data["predicate"],
                    "object": self._resolve_name(data["object"]),
                    "valid_from": data.get("valid_from"),
                    "valid_to": data.get("valid_to"),
                    "current": data.get("valid_to") is None,
                }
            )

        # Sort by valid_from, nulls last
        results.sort(key=lambda r: r["valid_from"] or "9999-99-99")
        return results[:100]

    # ── Stats ────────────────────────────────────────────────────────────

    def stats(self):
        """Return graph statistics."""
        entity_count = self._entities.count().get()[0][0].value
        triple_count = self._triples.count().get()[0][0].value

        # Count current vs expired
        current = self._triples.where("valid_to", "==", None).count().get()[0][0].value
        expired = triple_count - current

        # Get distinct predicates
        predicates = set()
        for row in self._triples.select(["predicate"]).stream():
            predicates.add(row.to_dict().get("predicate", ""))

        return {
            "entities": entity_count,
            "triples": triple_count,
            "current_facts": current,
            "expired_facts": expired,
            "relationship_types": sorted(predicates),
        }

    def close(self):
        """No-op for Firestore (no connection to close)."""
        pass
