"""Tests for FirestoreTunnelStore.

All Firestore interactions are mocked — no real Firestore client is needed.
"""

import sys
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

from mempalace.firestore_tunnels import (  # noqa: E402
    FirestoreTunnelStore,
    _canonical_tunnel_id,
    _endpoint_key,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_doc_snapshot(doc_id, data, exists=True):
    """Create a mock Firestore document snapshot."""
    snap = MagicMock()
    snap.id = doc_id
    snap.exists = exists
    snap.to_dict.return_value = data if exists else None
    return snap


@pytest.fixture
def mock_db():
    """Mock Firestore client."""
    return MagicMock()


@pytest.fixture
def store(mock_db):
    """FirestoreTunnelStore backed by mock Firestore."""
    return FirestoreTunnelStore(mock_db, base_path="test/user1")


# ═══════════════════════════════════════════════════════════════════════════
# create_tunnel tests
# ═══════════════════════════════════════════════════════════════════════════


class TestCreateTunnel:
    def test_stores_correct_data(self, store, mock_db):
        doc_ref = MagicMock()
        doc_ref.get.return_value = _make_doc_snapshot("tid", None, exists=False)
        store._col.document.return_value = doc_ref

        store.create_tunnel(
            source_wing="project",
            source_room="backend",
            target_wing="notes",
            target_room="planning",
            label="related",
        )

        # Write now goes through the transaction, not doc_ref directly
        mock_txn = mock_db.transaction()
        mock_txn.set.assert_called()
        _, data = mock_txn.set.call_args[0]

        assert data["source"]["wing"] == "project"
        assert data["source"]["room"] == "backend"
        assert data["target"]["wing"] == "notes"
        assert data["target"]["room"] == "planning"
        assert data["label"] == "related"
        assert "created_at" in data
        assert "updated_at" not in data  # new tunnel has no updated_at

    def test_with_drawer_ids(self, store, mock_db):
        doc_ref = MagicMock()
        doc_ref.get.return_value = _make_doc_snapshot("tid", None, exists=False)
        store._col.document.return_value = doc_ref

        store.create_tunnel(
            source_wing="a",
            source_room="b",
            target_wing="c",
            target_room="d",
            source_drawer_id="drawer_1",
            target_drawer_id="drawer_2",
        )

        mock_txn = mock_db.transaction()
        _, data = mock_txn.set.call_args[0]
        assert data["source"]["drawer_id"] == "drawer_1"
        assert data["target"]["drawer_id"] == "drawer_2"

    def test_updates_existing_preserves_created_at(self, store, mock_db):
        existing_data = {
            "created_at": "2025-01-01T00:00:00",
            "source": {"wing": "a", "room": "b"},
            "target": {"wing": "c", "room": "d"},
        }
        doc_ref = MagicMock()
        doc_ref.get.return_value = _make_doc_snapshot("tid", existing_data, exists=True)
        store._col.document.return_value = doc_ref

        store.create_tunnel(
            source_wing="a",
            source_room="b",
            target_wing="c",
            target_room="d",
            label="updated",
        )

        mock_txn = mock_db.transaction()
        _, data = mock_txn.set.call_args[0]
        assert data["created_at"] == "2025-01-01T00:00:00"
        assert "updated_at" in data
        assert data["label"] == "updated"

    def test_symmetric_id(self):
        """create(A->B) and create(B->A) produce the same tunnel_id."""
        id_ab = _canonical_tunnel_id("wing1", "room1", "wing2", "room2")
        id_ba = _canonical_tunnel_id("wing2", "room2", "wing1", "room1")
        assert id_ab == id_ba


# ═══════════════════════════════════════════════════════════════════════════
# list_tunnels tests
# ═══════════════════════════════════════════════════════════════════════════


class TestListTunnels:
    def test_returns_all(self, store):
        t1 = _make_doc_snapshot(
            "t1",
            {
                "id": "t1",
                "source": {"wing": "a", "room": "x"},
                "target": {"wing": "b", "room": "y"},
                "label": "",
            },
        )
        t2 = _make_doc_snapshot(
            "t2",
            {
                "id": "t2",
                "source": {"wing": "c", "room": "z"},
                "target": {"wing": "d", "room": "w"},
                "label": "",
            },
        )
        store._col.stream.return_value = [t1, t2]

        result = store.list_tunnels()

        assert len(result) == 2

    def test_filtered_by_wing_source(self, store):
        t1 = _make_doc_snapshot(
            "t1",
            {
                "source": {"wing": "project", "room": "backend"},
                "target": {"wing": "notes", "room": "planning"},
            },
        )
        t2 = _make_doc_snapshot(
            "t2",
            {
                "source": {"wing": "other", "room": "x"},
                "target": {"wing": "other2", "room": "y"},
            },
        )
        store._col.stream.return_value = [t1, t2]

        result = store.list_tunnels(wing="project")

        assert len(result) == 1
        assert result[0]["source"]["wing"] == "project"

    def test_filtered_by_wing_target(self, store):
        t1 = _make_doc_snapshot(
            "t1",
            {
                "source": {"wing": "other", "room": "x"},
                "target": {"wing": "project", "room": "backend"},
            },
        )
        store._col.stream.return_value = [t1]

        result = store.list_tunnels(wing="project")

        assert len(result) == 1

    def test_no_matches_returns_empty(self, store):
        t1 = _make_doc_snapshot(
            "t1",
            {
                "source": {"wing": "a", "room": "x"},
                "target": {"wing": "b", "room": "y"},
            },
        )
        store._col.stream.return_value = [t1]

        result = store.list_tunnels(wing="nonexistent")

        assert result == []

    def test_empty_collection(self, store):
        """Stream returns nothing — empty list result."""
        store._col.stream.return_value = []

        result = store.list_tunnels()

        assert result == []


# ═══════════════════════════════════════════════════════════════════════════
# delete_tunnel tests
# ═══════════════════════════════════════════════════════════════════════════


class TestDeleteTunnel:
    def test_removes_by_id(self, store):
        store.delete_tunnel("abc123")

        store._col.document("abc123").delete.assert_called_once()

    def test_returns_deleted_id(self, store):
        result = store.delete_tunnel("abc123")

        assert result == {"deleted": "abc123"}


# ═══════════════════════════════════════════════════════════════════════════
# follow_tunnels tests
# ═══════════════════════════════════════════════════════════════════════════


class TestFollowTunnels:
    def test_finds_outgoing(self, store):
        t1 = _make_doc_snapshot(
            "t1",
            {
                "id": "t1",
                "source": {"wing": "project", "room": "backend"},
                "target": {"wing": "notes", "room": "planning"},
                "label": "related",
            },
        )
        store._col.stream.return_value = [t1]

        result = store.follow_tunnels("project", "backend")

        assert len(result) == 1
        assert result[0]["direction"] == "outgoing"
        assert result[0]["connected_wing"] == "notes"
        assert result[0]["connected_room"] == "planning"
        assert result[0]["label"] == "related"

    def test_finds_incoming(self, store):
        t1 = _make_doc_snapshot(
            "t1",
            {
                "id": "t1",
                "source": {"wing": "other", "room": "stuff"},
                "target": {"wing": "project", "room": "backend"},
                "label": "depends",
            },
        )
        store._col.stream.return_value = [t1]

        result = store.follow_tunnels("project", "backend")

        assert len(result) == 1
        assert result[0]["direction"] == "incoming"
        assert result[0]["connected_wing"] == "other"
        assert result[0]["connected_room"] == "stuff"

    def test_with_drawer_previews(self, store):
        t1 = _make_doc_snapshot(
            "t1",
            {
                "id": "t1",
                "source": {"wing": "project", "room": "backend"},
                "target": {"wing": "notes", "room": "planning", "drawer_id": "d1"},
                "label": "",
            },
        )
        store._col.stream.return_value = [t1]

        drawers_col = MagicMock()
        drawers_col.get.return_value = {
            "ids": ["d1"],
            "documents": ["This is the drawer content preview text"],
        }

        result = store.follow_tunnels("project", "backend", drawers_col=drawers_col)

        assert len(result) == 1
        assert "drawer_preview" in result[0]
        assert result[0]["drawer_preview"].startswith("This is the drawer")

    def test_preview_fetch_fails_gracefully(self, store):
        """If drawers_col.get raises an exception, connections still returned without previews."""
        t1 = _make_doc_snapshot(
            "t1",
            {
                "id": "t1",
                "source": {"wing": "project", "room": "backend"},
                "target": {"wing": "notes", "room": "planning", "drawer_id": "d1"},
                "label": "related",
            },
        )
        store._col.stream.return_value = [t1]

        drawers_col = MagicMock()
        drawers_col.get.side_effect = Exception("Firestore unavailable")

        result = store.follow_tunnels("project", "backend", drawers_col=drawers_col)

        assert len(result) == 1
        assert result[0]["direction"] == "outgoing"
        assert result[0]["connected_wing"] == "notes"
        assert "drawer_preview" not in result[0]

    def test_no_connections_returns_empty(self, store):
        t1 = _make_doc_snapshot(
            "t1",
            {
                "id": "t1",
                "source": {"wing": "other", "room": "x"},
                "target": {"wing": "other2", "room": "y"},
                "label": "",
            },
        )
        store._col.stream.return_value = [t1]

        result = store.follow_tunnels("project", "backend")

        assert result == []


# ═══════════════════════════════════════════════════════════════════════════
# Helper function tests
# ═══════════════════════════════════════════════════════════════════════════


class TestTransactions:
    """Verify that create_tunnel uses Firestore transactions for atomicity."""

    def test_create_tunnel_uses_transaction(self, store, mock_db):
        """create_tunnel should use db.transaction() for atomic check-exists + write."""
        # Setup: mock transaction and doc ref
        mock_txn = MagicMock()
        mock_db.transaction.return_value = mock_txn

        doc_ref = MagicMock()
        doc_ref.get.return_value = _make_doc_snapshot("tid", None, exists=False)
        store._col.document.return_value = doc_ref

        store.create_tunnel(
            source_wing="project",
            source_room="backend",
            target_wing="notes",
            target_room="planning",
            label="related",
        )

        mock_db.transaction.assert_called()


class TestHelperFunctions:
    def test_endpoint_key(self):
        assert _endpoint_key("wing", "room") == "wing/room"

    def test_canonical_tunnel_id_is_deterministic(self):
        id1 = _canonical_tunnel_id("a", "b", "c", "d")
        id2 = _canonical_tunnel_id("a", "b", "c", "d")
        assert id1 == id2

    def test_canonical_tunnel_id_is_symmetric(self):
        id_fwd = _canonical_tunnel_id("a", "b", "c", "d")
        id_rev = _canonical_tunnel_id("c", "d", "a", "b")
        assert id_fwd == id_rev

    def test_canonical_tunnel_id_length(self):
        tid = _canonical_tunnel_id("a", "b", "c", "d")
        assert len(tid) == 16
