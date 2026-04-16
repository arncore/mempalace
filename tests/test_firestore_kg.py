"""Tests for FirestoreKnowledgeGraph.

All Firestore interactions are mocked — no real Firestore client is needed.
"""

import sys
from datetime import date
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# Mock google.cloud.firestore_v1.transaction before importing the module
# under test.  The @transactional decorator is replaced with a pass-through
# that simply calls the decorated function with its arguments.
# ---------------------------------------------------------------------------

_mock_txn_module = MagicMock()


def _fake_transactional(fn):
    """Replacement for @firestore.transactional — just calls fn directly."""

    def wrapper(txn, *args, **kwargs):
        return fn(txn, *args, **kwargs)

    return wrapper


_mock_txn_module.transactional = _fake_transactional

sys.modules.setdefault("google.cloud", MagicMock())
sys.modules.setdefault("google.cloud.firestore_v1", MagicMock())
sys.modules.setdefault("google.cloud.firestore_v1.transaction", _mock_txn_module)

from mempalace.firestore_knowledge_graph import FirestoreKnowledgeGraph  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_doc_snapshot(doc_id, data, exists=True):
    """Create a mock Firestore document snapshot."""
    snap = MagicMock()
    snap.id = doc_id
    snap.exists = exists
    snap.to_dict.return_value = data if exists else None
    snap.reference = MagicMock()
    return snap


def _make_agg_value(count):
    """Mock an aggregation result value."""
    v = MagicMock()
    v.value = count
    return v


def _make_routing_collection(name="col"):
    """Create a mock collection whose .document(id) returns a stable, per-ID mock."""
    col = MagicMock(name=name)
    doc_mocks = {}

    def _doc_router(doc_id):
        if doc_id not in doc_mocks:
            doc_mocks[doc_id] = MagicMock(name=f"{name}/doc({doc_id})")
        return doc_mocks[doc_id]

    col.document.side_effect = _doc_router
    col._doc_mocks = doc_mocks  # expose for test introspection
    return col


@pytest.fixture
def mock_db():
    """Firestore client mock with distinct entities and triples collections."""
    db = MagicMock()
    entities_col = _make_routing_collection("entities_col")
    triples_col = _make_routing_collection("triples_col")

    def _route_collection(path):
        if path.endswith("/entities"):
            return entities_col
        if path.endswith("/triples"):
            return triples_col
        return MagicMock()

    db.collection.side_effect = _route_collection
    db._entities_col = entities_col
    db._triples_col = triples_col
    return db


@pytest.fixture
def kg(mock_db):
    """FirestoreKnowledgeGraph backed by mock Firestore."""
    graph = FirestoreKnowledgeGraph(mock_db, base_path="test/user1")
    return graph


# ═══════════════════════════════════════════════════════════════════════════
# Entity tests
# ═══════════════════════════════════════════════════════════════════════════


class TestAddEntity:
    def test_returns_normalized_id(self, kg):
        eid = kg.add_entity("Alice Smith", entity_type="person")
        assert eid == "alice_smith"

    def test_special_chars(self, kg):
        eid = kg.add_entity("Bob's Place", entity_type="location")
        assert eid == "bobs_place"

    def test_spaces_replaced(self, kg):
        eid = kg.add_entity("New York City")
        assert eid == "new_york_city"

    def test_upserts_with_merge_true(self, kg):
        kg.add_entity("Alice")
        doc_ref = kg._entities.document("alice")
        doc_ref.set.assert_called_once()
        _, kwargs = doc_ref.set.call_args
        assert kwargs.get("merge") is True

    def test_empty_name(self, kg):
        """Empty string produces empty entity ID — documents the behavior."""
        eid = kg.add_entity("", entity_type="unknown")
        assert eid == ""
        # Still calls set on the entity document (with empty ID)
        kg._entities.document("").set.assert_called_once()


# ═══════════════════════════════════════════════════════════════════════════
# Triple tests
# ═══════════════════════════════════════════════════════════════════════════


class TestAddTriple:
    def _setup_no_existing(self, kg):
        """Configure the doc get inside the transaction to return non-existent."""
        triple_doc = kg._triples.document("t_alice_knows_bob")
        triple_doc.get.return_value = _make_doc_snapshot("t_alice_knows_bob", None, exists=False)

    def test_auto_creates_entities(self, kg):
        self._setup_no_existing(kg)

        kg.add_triple("Alice", "knows", "Bob")

        # Both entities should have been set with merge=True
        kg._entities.document("alice").set.assert_called()
        kg._entities.document("bob").set.assert_called()

    def test_returns_deterministic_id(self, kg):
        """Triple IDs are now deterministic: t_{sub}_{pred}_{obj}."""
        self._setup_no_existing(kg)

        tid = kg.add_triple("Alice", "knows", "Bob")

        assert tid == "t_alice_knows_bob"

    def test_deduplicates_active_triples(self, kg):
        """If an identical active triple exists, return its ID without creating."""
        existing_snap = _make_doc_snapshot(
            "t_alice_knows_bob",
            {"valid_to": None, "subject": "alice", "predicate": "knows", "object": "bob"},
            exists=True,
        )
        kg._triples.document("t_alice_knows_bob").get.return_value = existing_snap

        tid = kg.add_triple("Alice", "knows", "Bob")

        assert tid == "t_alice_knows_bob"
        # The transaction should NOT have called set (dedup hit)
        mock_txn = kg._db.transaction()
        mock_txn.set.assert_not_called()

    def test_allows_new_triple_after_invalidation(self, kg):
        """An invalidated triple (valid_to != None) should not block a new one."""
        # Existing doc has valid_to set (invalidated) — should overwrite
        invalidated_snap = _make_doc_snapshot(
            "t_alice_knows_bob",
            {"valid_to": "2024-12-31", "subject": "alice", "predicate": "knows", "object": "bob"},
            exists=True,
        )
        kg._triples.document("t_alice_knows_bob").get.return_value = invalidated_snap

        tid = kg.add_triple("Alice", "knows", "Bob")

        assert tid == "t_alice_knows_bob"

    def test_with_valid_from_and_valid_to(self, kg):
        # Setup: no existing doc
        triple_doc = kg._triples.document("t_alice_works_at_acme")
        triple_doc.get.return_value = _make_doc_snapshot(
            "t_alice_works_at_acme", None, exists=False
        )

        tid = kg.add_triple(
            "Alice",
            "works_at",
            "Acme",
            valid_from="2020-01-01",
            valid_to="2024-12-31",
        )

        assert tid == "t_alice_works_at_acme"
        # The transaction mock's set should have been called with the data
        mock_txn = kg._db.transaction()
        mock_txn.set.assert_called()
        data = mock_txn.set.call_args[0][1]
        assert data["valid_from"] == "2020-01-01"
        assert data["valid_to"] == "2024-12-31"

    def test_same_subject_predicate_different_object(self, kg):
        """Alice->likes->Rock and Alice->likes->Jazz are different triples."""
        # First triple: no existing
        kg._triples.document("t_alice_likes_rock").get.return_value = _make_doc_snapshot(
            "t_alice_likes_rock", None, exists=False
        )

        tid1 = kg.add_triple("Alice", "likes", "Rock")

        # Second triple: no existing
        kg._triples.document("t_alice_likes_jazz").get.return_value = _make_doc_snapshot(
            "t_alice_likes_jazz", None, exists=False
        )

        tid2 = kg.add_triple("Alice", "likes", "Jazz")

        assert tid1 != tid2
        assert tid1 == "t_alice_likes_rock"
        assert tid2 == "t_alice_likes_jazz"

    def test_concurrent_dedup_prevented_by_transaction(self, kg):
        """With deterministic IDs and transactions, concurrent adds of the same
        triple are serialized — the second sees the first's write and returns
        the existing ID instead of creating a duplicate."""
        # First call: doc doesn't exist yet
        kg._triples.document("t_alice_knows_bob").get.return_value = _make_doc_snapshot(
            "t_alice_knows_bob", None, exists=False
        )

        tid1 = kg.add_triple("Alice", "knows", "Bob")

        # Second call: doc now exists (from the first transaction)
        kg._triples.document("t_alice_knows_bob").get.return_value = _make_doc_snapshot(
            "t_alice_knows_bob",
            {"valid_to": None, "subject": "alice", "predicate": "knows", "object": "bob"},
            exists=True,
        )

        tid2 = kg.add_triple("Alice", "knows", "Bob")

        # Both return the same deterministic ID
        assert tid1 == tid2 == "t_alice_knows_bob"


# ═══════════════════════════════════════════════════════════════════════════
# Invalidation tests
# ═══════════════════════════════════════════════════════════════════════════


class TestInvalidate:
    def test_sets_valid_to_on_matching(self, kg):
        """invalidate reads by deterministic ID and updates valid_to in a transaction."""
        snap = _make_doc_snapshot(
            "t_alice_works_at_acme",
            {"valid_to": None, "subject": "alice", "predicate": "works_at", "object": "acme"},
        )
        kg._triples.document("t_alice_works_at_acme").get.return_value = snap

        kg.invalidate("Alice", "works_at", "Acme")

        mock_txn = kg._db.transaction()
        mock_txn.update.assert_called()
        doc_ref, update_data = mock_txn.update.call_args[0]
        assert "valid_to" in update_data
        assert update_data["valid_to"] == date.today().isoformat()

    def test_with_explicit_ended_date(self, kg):
        snap = _make_doc_snapshot(
            "t_alice_works_at_acme",
            {"valid_to": None, "subject": "alice", "predicate": "works_at", "object": "acme"},
        )
        kg._triples.document("t_alice_works_at_acme").get.return_value = snap

        kg.invalidate("Alice", "works_at", "Acme", ended="2024-12-31")

        mock_txn = kg._db.transaction()
        _, update_data = mock_txn.update.call_args[0]
        assert update_data["valid_to"] == "2024-12-31"

    def test_no_matching_triple_no_error(self, kg):
        """If the deterministic doc doesn't exist, invalidate is a no-op."""
        snap = _make_doc_snapshot("t_nobody_does_nothing", None, exists=False)
        kg._triples.document("t_nobody_does_nothing").get.return_value = snap

        # Should not raise
        kg.invalidate("Nobody", "does", "Nothing")

    def test_already_invalidated_triple_is_noop(self, kg):
        """If the triple already has a valid_to, don't overwrite it."""
        snap = _make_doc_snapshot(
            "t_alice_works_at_acme",
            {
                "valid_to": "2024-06-01",
                "subject": "alice",
                "predicate": "works_at",
                "object": "acme",
            },
        )
        kg._triples.document("t_alice_works_at_acme").get.return_value = snap

        kg.invalidate("Alice", "works_at", "Acme")

        # Transaction should NOT have called update (already invalidated)
        mock_txn = kg._db.transaction()
        mock_txn.update.assert_not_called()


# ═══════════════════════════════════════════════════════════════════════════
# query_entity tests
# ═══════════════════════════════════════════════════════════════════════════


class TestQueryEntity:
    def _make_triple_snap(self, data):
        snap = MagicMock()
        snap.to_dict.return_value = data
        return snap

    def _setup_entity_name(self, kg, entity_id, name):
        """Wire up _resolve_name for a given entity."""
        snap = _make_doc_snapshot(entity_id, {"name": name})
        kg._entities.document(entity_id).get.return_value = snap

    def test_outgoing_direction(self, kg):
        self._setup_entity_name(kg, "bob", "Bob")

        triple = self._make_triple_snap(
            {
                "subject": "alice",
                "predicate": "knows",
                "object": "bob",
                "valid_from": "2020-01-01",
                "valid_to": None,
                "confidence": 0.9,
                "source_closet": None,
            }
        )

        outgoing_query = MagicMock()
        outgoing_query.stream.return_value = [triple]
        kg._triples.where.return_value = outgoing_query

        results = kg.query_entity("Alice", direction="outgoing")

        assert len(results) == 1
        assert results[0]["direction"] == "outgoing"
        assert results[0]["subject"] == "Alice"
        assert results[0]["predicate"] == "knows"
        assert results[0]["object"] == "Bob"
        assert results[0]["current"] is True

    def test_incoming_direction(self, kg):
        self._setup_entity_name(kg, "charlie", "Charlie")

        triple = self._make_triple_snap(
            {
                "subject": "charlie",
                "predicate": "follows",
                "object": "alice",
                "valid_from": None,
                "valid_to": None,
                "confidence": 1.0,
                "source_closet": None,
            }
        )

        incoming_query = MagicMock()
        incoming_query.stream.return_value = [triple]
        kg._triples.where.return_value = incoming_query

        results = kg.query_entity("Alice", direction="incoming")

        assert len(results) == 1
        assert results[0]["direction"] == "incoming"
        assert results[0]["subject"] == "Charlie"
        assert results[0]["object"] == "Alice"

    def test_both_directions(self, kg):
        self._setup_entity_name(kg, "bob", "Bob")
        self._setup_entity_name(kg, "charlie", "Charlie")

        out_triple = self._make_triple_snap(
            {
                "subject": "alice",
                "predicate": "knows",
                "object": "bob",
                "valid_from": None,
                "valid_to": None,
                "confidence": 1.0,
                "source_closet": None,
            }
        )
        in_triple = self._make_triple_snap(
            {
                "subject": "charlie",
                "predicate": "follows",
                "object": "alice",
                "valid_from": None,
                "valid_to": None,
                "confidence": 1.0,
                "source_closet": None,
            }
        )

        out_q = MagicMock()
        out_q.stream.return_value = [out_triple]
        in_q = MagicMock()
        in_q.stream.return_value = [in_triple]

        # First where("subject", ...) -> outgoing, second where("object", ...) -> incoming
        kg._triples.where.side_effect = [out_q, in_q]

        results = kg.query_entity("Alice", direction="both")

        assert len(results) == 2
        directions = {r["direction"] for r in results}
        assert directions == {"outgoing", "incoming"}

    def test_empty_results(self, kg):
        """Entity exists but has no triples — returns empty list."""
        q = MagicMock()
        q.stream.return_value = []
        kg._triples.where.return_value = q

        results = kg.query_entity("Alice", direction="outgoing")

        assert results == []

    def test_entity_does_not_exist(self, kg):
        """Query for an unknown entity — returns empty list (no triples found)."""
        q = MagicMock()
        q.stream.return_value = []
        kg._triples.where.return_value = q

        results = kg.query_entity("UnknownPerson", direction="both")

        assert results == []
        # Both outgoing and incoming queries should have been issued
        assert kg._triples.where.call_count == 2

    def test_as_of_excludes_future_and_expired(self, kg):
        self._setup_entity_name(kg, "bob", "Bob")
        self._setup_entity_name(kg, "charlie", "Charlie")
        self._setup_entity_name(kg, "dave", "Dave")

        current = self._make_triple_snap(
            {
                "subject": "alice",
                "predicate": "knows",
                "object": "bob",
                "valid_from": "2020-01-01",
                "valid_to": None,
                "confidence": 1.0,
                "source_closet": None,
            }
        )
        future = self._make_triple_snap(
            {
                "subject": "alice",
                "predicate": "knows",
                "object": "charlie",
                "valid_from": "2030-01-01",
                "valid_to": None,
                "confidence": 1.0,
                "source_closet": None,
            }
        )
        expired = self._make_triple_snap(
            {
                "subject": "alice",
                "predicate": "knows",
                "object": "dave",
                "valid_from": "2015-01-01",
                "valid_to": "2019-12-31",
                "confidence": 1.0,
                "source_closet": None,
            }
        )

        q = MagicMock()
        q.stream.return_value = [current, future, expired]
        kg._triples.where.return_value = q

        results = kg.query_entity("Alice", as_of="2025-06-15", direction="outgoing")

        assert len(results) == 1
        assert results[0]["object"] == "Bob"


class TestResolveNameCaching:
    def test_caches_entity_lookups(self, kg):
        snap = _make_doc_snapshot("alice", {"name": "Alice"})
        kg._entities.document("alice").get.return_value = snap

        # Call _resolve_name twice
        name1 = kg._resolve_name("alice")
        name2 = kg._resolve_name("alice")

        assert name1 == "Alice"
        assert name2 == "Alice"
        # Firestore .get() should only be called once due to caching
        kg._entities.document("alice").get.assert_called_once()


# ═══════════════════════════════════════════════════════════════════════════
# query_relationship tests
# ═══════════════════════════════════════════════════════════════════════════


class TestQueryRelationship:
    def test_returns_matching_predicate(self, kg):
        snap1 = MagicMock()
        snap1.to_dict.return_value = {
            "subject": "alice",
            "predicate": "knows",
            "object": "bob",
            "valid_from": None,
            "valid_to": None,
        }
        snap2 = MagicMock()
        snap2.to_dict.return_value = {
            "subject": "charlie",
            "predicate": "knows",
            "object": "dave",
            "valid_from": None,
            "valid_to": None,
        }

        q = MagicMock()
        q.stream.return_value = [snap1, snap2]
        kg._triples.where.return_value = q

        # Wire up entity name resolution
        for eid, name in [
            ("alice", "Alice"),
            ("bob", "Bob"),
            ("charlie", "Charlie"),
            ("dave", "Dave"),
        ]:
            s = _make_doc_snapshot(eid, {"name": name})
            kg._entities.document(eid).get.return_value = s

        results = kg.query_relationship("knows")

        assert len(results) == 2
        assert all(r["predicate"] == "knows" for r in results)

    def test_filters_by_as_of(self, kg):
        snap = MagicMock()
        snap.to_dict.return_value = {
            "subject": "alice",
            "predicate": "works_at",
            "object": "acme",
            "valid_from": "2020-01-01",
            "valid_to": "2024-12-31",
        }

        q = MagicMock()
        q.stream.return_value = [snap]
        kg._triples.where.return_value = q

        # as_of 2025 — after expiry
        results = kg.query_relationship("works_at", as_of="2025-06-01")
        assert len(results) == 0


# ═══════════════════════════════════════════════════════════════════════════
# Timeline tests
# ═══════════════════════════════════════════════════════════════════════════


class TestTimeline:
    def _make_row(self, subject, predicate, obj, valid_from, valid_to=None):
        snap = MagicMock()
        snap.to_dict.return_value = {
            "subject": subject,
            "predicate": predicate,
            "object": obj,
            "valid_from": valid_from,
            "valid_to": valid_to,
        }
        return snap

    def _setup_resolve(self, kg, mapping):
        for eid, name in mapping.items():
            s = _make_doc_snapshot(eid, {"name": name})
            kg._entities.document(eid).get.return_value = s

    def test_sorted_by_valid_from(self, kg):
        self._setup_resolve(kg, {"alice": "Alice", "bob": "Bob", "charlie": "Charlie"})

        rows = [
            self._make_row("bob", "knows", "charlie", "2023-01-01"),
            self._make_row("alice", "knows", "bob", "2020-01-01"),
        ]

        chain = MagicMock()
        chain.limit.return_value = chain
        chain.stream.return_value = rows
        kg._triples.order_by.return_value = chain

        results = kg.timeline()

        assert len(results) == 2
        assert results[0]["valid_from"] == "2020-01-01"
        assert results[1]["valid_from"] == "2023-01-01"

    def test_nulls_sort_last(self, kg):
        self._setup_resolve(kg, {"a": "A", "b": "B", "c": "C"})

        rows = [
            self._make_row("a", "x", "b", None),
            self._make_row("a", "y", "c", "2020-01-01"),
        ]

        chain = MagicMock()
        chain.limit.return_value = chain
        chain.stream.return_value = rows
        kg._triples.order_by.return_value = chain

        results = kg.timeline()

        assert results[0]["valid_from"] == "2020-01-01"
        assert results[1]["valid_from"] is None

    def test_empty_graph(self, kg):
        """Timeline on empty graph returns empty list."""
        chain = MagicMock()
        chain.limit.return_value = chain
        chain.stream.return_value = []
        kg._triples.order_by.return_value = chain

        results = kg.timeline()

        assert results == []

    def test_filters_by_entity(self, kg):
        self._setup_resolve(kg, {"alice": "Alice", "bob": "Bob"})

        out_row = self._make_row("alice", "knows", "bob", "2020-01-01")
        in_row = self._make_row("bob", "follows", "alice", "2021-01-01")

        out_chain = MagicMock()
        out_chain.order_by.return_value = out_chain
        out_chain.limit.return_value = out_chain
        out_chain.stream.return_value = [out_row]

        in_chain = MagicMock()
        in_chain.order_by.return_value = in_chain
        in_chain.limit.return_value = in_chain
        in_chain.stream.return_value = [in_row]

        kg._triples.where.side_effect = [out_chain, in_chain]

        results = kg.timeline(entity_name="Alice")

        assert len(results) == 2
        # Should be merged and sorted
        assert results[0]["valid_from"] == "2020-01-01"
        assert results[1]["valid_from"] == "2021-01-01"


# ═══════════════════════════════════════════════════════════════════════════
# Stats tests
# ═══════════════════════════════════════════════════════════════════════════


class TestStats:
    def test_empty_graph(self, kg):
        """All counts zero, empty predicate list."""
        kg._entities.count.return_value.get.return_value = [[_make_agg_value(0)]]
        kg._triples.count.return_value.get.return_value = [[_make_agg_value(0)]]

        current_chain = MagicMock()
        current_chain.count.return_value.get.return_value = [[_make_agg_value(0)]]
        kg._triples.where.return_value = current_chain

        kg._triples.select.return_value.stream.return_value = []

        result = kg.stats()

        assert result["entities"] == 0
        assert result["triples"] == 0
        assert result["current_facts"] == 0
        assert result["expired_facts"] == 0
        assert result["relationship_types"] == []

    def test_returns_correct_counts(self, kg):
        kg._entities.count.return_value.get.return_value = [[_make_agg_value(10)]]
        kg._triples.count.return_value.get.return_value = [[_make_agg_value(25)]]

        # Current triples count
        current_chain = MagicMock()
        current_chain.count.return_value.get.return_value = [[_make_agg_value(20)]]
        kg._triples.where.return_value = current_chain

        # Predicate scan
        pred_snaps = []
        for p in ["knows", "works_at", "follows"]:
            s = MagicMock()
            s.to_dict.return_value = {"predicate": p}
            pred_snaps.append(s)
        kg._triples.select.return_value.stream.return_value = pred_snaps

        result = kg.stats()

        assert result["entities"] == 10
        assert result["triples"] == 25
        assert result["current_facts"] == 20
        assert result["expired_facts"] == 5
        assert result["relationship_types"] == ["follows", "knows", "works_at"]


# ═══════════════════════════════════════════════════════════════════════════
# Edge cases
# ═══════════════════════════════════════════════════════════════════════════


class TestTransactions:
    """Verify that write operations use Firestore transactions for atomicity."""

    def test_add_triple_uses_transaction(self, kg, mock_db):
        """add_triple should use db.transaction() for atomic dedup + write."""
        # Setup: no existing active triple
        kg._triples.document("t_alice_knows_bob").get.return_value = _make_doc_snapshot(
            "t_alice_knows_bob", None, exists=False
        )

        kg.add_triple("Alice", "knows", "Bob")

        mock_db.transaction.assert_called()

    def test_invalidate_uses_transaction(self, kg, mock_db):
        """invalidate should use db.transaction() for atomic read + update."""
        snap = _make_doc_snapshot(
            "t_alice_works_at_acme",
            {"valid_to": None, "subject": "alice", "predicate": "works_at", "object": "acme"},
        )
        kg._triples.document("t_alice_works_at_acme").get.return_value = snap

        kg.invalidate("Alice", "works_at", "Acme")

        mock_db.transaction.assert_called()


class TestEdgeCases:
    def test_close_is_noop(self, kg):
        kg.close()  # should not raise

    def test_entity_id_normalization(self, kg):
        assert kg._entity_id("Hello World") == "hello_world"
        assert kg._entity_id("It's") == "its"
        assert kg._entity_id("UPPER") == "upper"
        assert kg._entity_id("a b c") == "a_b_c"
        assert kg._entity_id("") == ""
