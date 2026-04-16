"""Tests for FirestoreCollection, FirestoreBackend, and _BatchWriter.

All Firestore interactions are mocked — no real Firestore client is needed.
Tests are written to match ChromaDB's actual behaviour (verified experimentally).
"""

import sys
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# Mock google.cloud.firestore_v1 before importing the module under test
# ---------------------------------------------------------------------------

_mock_firestore_v1 = MagicMock()


class _FakeDistanceMeasure:
    COSINE = "COSINE"


class _FakeVector:
    """Thin stand-in for google.cloud.firestore_v1.vector.Vector."""

    def __init__(self, values):
        self.values = values

    def __eq__(self, other):
        return isinstance(other, _FakeVector) and self.values == other.values

    def __repr__(self):
        return f"_FakeVector({self.values!r})"


class _FakeFieldFilter:
    """Stand-in for google.cloud.firestore_v1.base_query.FieldFilter."""

    def __init__(self, field, op, value):
        self.field = field
        self.op = op
        self.value = value

    def __eq__(self, other):
        return (
            isinstance(other, _FakeFieldFilter)
            and self.field == other.field
            and self.op == other.op
            and self.value == other.value
        )

    def __repr__(self):
        return f"FieldFilter({self.field!r}, {self.op!r}, {self.value!r})"


class _FakeOr:
    """Stand-in for google.cloud.firestore_v1.base_query.Or."""

    def __init__(self, filters):
        self.filters = filters

    def __eq__(self, other):
        return isinstance(other, _FakeOr) and self.filters == other.filters

    def __repr__(self):
        return f"Or(filters={self.filters!r})"


_mock_firestore_v1.base_vector_query.DistanceMeasure = _FakeDistanceMeasure
_mock_firestore_v1.vector.Vector = _FakeVector
_mock_firestore_v1.base_query.FieldFilter = _FakeFieldFilter
_mock_firestore_v1.base_query.Or = _FakeOr

sys.modules.setdefault("google.cloud.firestore_v1", _mock_firestore_v1)
sys.modules.setdefault("google.cloud.firestore_v1.base_query", _mock_firestore_v1.base_query)
sys.modules.setdefault(
    "google.cloud.firestore_v1.base_vector_query", _mock_firestore_v1.base_vector_query
)
sys.modules.setdefault("google.cloud.firestore_v1.vector", _mock_firestore_v1.vector)

from mempalace.backends.firestore import FirestoreBackend, FirestoreCollection  # noqa: E402
from mempalace.backends.firestore.collection import (  # noqa: E402
    _apply_where_filter,
    _build_field_filter,
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
    snap.reference = MagicMock()
    return snap


def _fake_embed(texts):
    """Deterministic fake embeddings — one float per text character count."""
    return [[float(len(t))] * 3 for t in texts]


def _make_col_ref():
    """Build a mock Firestore CollectionReference and a mock db client."""
    col_ref = MagicMock()
    db_client = MagicMock()
    batch_mock = MagicMock()
    db_client.batch.return_value = batch_mock
    return col_ref, db_client, batch_mock


# ═══════════════════════════════════════════════════════════════════════════
# _BatchWriter tests
# ═══════════════════════════════════════════════════════════════════════════


class TestBatchWriter:
    """FirestoreCollection._BatchWriter auto-chunking behaviour."""

    def _make_writer(self, limit=450):
        col_ref, db_client, batch_mock = _make_col_ref()
        writer = FirestoreCollection._BatchWriter(db_client, limit)
        return writer, col_ref, db_client, batch_mock

    def test_commits_at_limit(self):
        """Batch commits exactly when count reaches the limit (450)."""
        writer, col_ref, db_client, first_batch = self._make_writer(limit=450)
        second_batch = MagicMock()
        db_client.batch.side_effect = [second_batch]

        doc_ref = MagicMock()
        for _ in range(450):
            writer.set(doc_ref, {"x": 1})

        first_batch.commit.assert_called_once()
        # After flush, counter resets so final commit on second batch is a no-op
        writer.commit()
        second_batch.commit.assert_not_called()

    def test_zero_operations(self):
        """Commit with 0 operations is a no-op."""
        writer, _, _, batch_mock = self._make_writer()
        writer.commit()
        batch_mock.commit.assert_not_called()

    def test_one_operation(self):
        """A single set should not auto-flush; only commit() triggers it."""
        writer, _, _, batch_mock = self._make_writer()
        writer.set(MagicMock(), {"a": 1})
        batch_mock.commit.assert_not_called()
        writer.commit()
        batch_mock.commit.assert_called_once()

    def test_451_operations_two_commits(self):
        """451 ops => flush at 450 + final commit for the 1 leftover."""
        writer, col_ref, db_client, first_batch = self._make_writer(limit=450)
        second_batch = MagicMock()
        db_client.batch.side_effect = [second_batch]

        doc_ref = MagicMock()
        for _ in range(451):
            writer.set(doc_ref, {"x": 1})

        first_batch.commit.assert_called_once()  # flushed at 450
        writer.commit()
        second_batch.commit.assert_called_once()  # leftover 1

    def test_set_update_delete_all_work(self):
        """All three write operations go through the batch."""
        writer, _, _, batch_mock = self._make_writer()
        doc = MagicMock()

        writer.set(doc, {"a": 1}, merge=True)
        batch_mock.set.assert_called_once_with(doc, {"a": 1}, merge=True)

        writer.update(doc, {"b": 2})
        batch_mock.update.assert_called_once_with(doc, {"b": 2})

        writer.delete(doc)
        batch_mock.delete.assert_called_once_with(doc)

        writer.commit()
        batch_mock.commit.assert_called_once()


# ═══════════════════════════════════════════════════════════════════════════
# FirestoreCollection tests
# ═══════════════════════════════════════════════════════════════════════════


class TestFirestoreCollectionAdd:
    def test_add_creates_documents_with_correct_schema(self):
        col_ref, db_client, batch_mock = _make_col_ref()
        fc = FirestoreCollection(col_ref, db_client=db_client, embed_fn=_fake_embed)

        # Mock doc refs that don't exist yet
        doc1 = MagicMock()
        doc1.get.return_value = _make_doc_snapshot("id1", None, exists=False)
        doc2 = MagicMock()
        doc2.get.return_value = _make_doc_snapshot("id2", None, exists=False)
        col_ref.document.side_effect = [doc1, doc2]

        fc.add(
            documents=["hello", "world"],
            ids=["id1", "id2"],
            metadatas=[{"k": "v1"}, {"k": "v2"}],
        )

        assert batch_mock.set.call_count == 2
        first_call_data = batch_mock.set.call_args_list[0][0][1]
        assert first_call_data["document"] == "hello"
        assert "embedding" in first_call_data
        assert first_call_data["meta"] == {"k": "v1"}

    def test_add_duplicate_id_is_silently_skipped(self):
        """ChromaDB behaviour: duplicate add is silently ignored, original kept."""
        col_ref, db_client, batch_mock = _make_col_ref()
        fc = FirestoreCollection(col_ref, db_client=db_client, embed_fn=_fake_embed)

        # First doc exists, second does not
        doc1 = MagicMock()
        doc1.get.return_value = _make_doc_snapshot(
            "id1", {"document": "original", "meta": {"k": "v1"}}, exists=True
        )
        doc2 = MagicMock()
        doc2.get.return_value = _make_doc_snapshot("id2", None, exists=False)
        col_ref.document.side_effect = [doc1, doc2]

        fc.add(
            documents=["new text", "world"],
            ids=["id1", "id2"],
            metadatas=[{"k": "overwritten?"}, {"k": "v2"}],
        )

        # Only the non-existing doc should have been written
        assert batch_mock.set.call_count == 1
        written_data = batch_mock.set.call_args_list[0][0][1]
        assert written_data["document"] == "world"
        assert written_data["meta"] == {"k": "v2"}

    def test_add_empty_list(self):
        col_ref, db_client, batch_mock = _make_col_ref()
        fc = FirestoreCollection(col_ref, db_client=db_client, embed_fn=_fake_embed)

        fc.add(documents=[], ids=[], metadatas=[])
        batch_mock.set.assert_not_called()

    def test_add_without_metadatas(self):
        col_ref, db_client, batch_mock = _make_col_ref()
        fc = FirestoreCollection(col_ref, db_client=db_client, embed_fn=_fake_embed)

        doc1 = MagicMock()
        doc1.get.return_value = _make_doc_snapshot("id1", None, exists=False)
        col_ref.document.return_value = doc1

        fc.add(documents=["doc"], ids=["id1"])
        data = batch_mock.set.call_args_list[0][0][1]
        assert data["meta"] == {}

    def test_add_single_doc(self):
        col_ref, db_client, batch_mock = _make_col_ref()
        fc = FirestoreCollection(col_ref, db_client=db_client, embed_fn=_fake_embed)

        doc1 = MagicMock()
        doc1.get.return_value = _make_doc_snapshot("single", None, exists=False)
        col_ref.document.return_value = doc1

        fc.add(documents=["only one"], ids=["single"], metadatas=[{"tag": "solo"}])

        batch_mock.set.assert_called_once()
        data = batch_mock.set.call_args_list[0][0][1]
        assert data["document"] == "only one"
        assert data["meta"] == {"tag": "solo"}
        assert "embedding" in data
        batch_mock.commit.assert_called_once()

    def test_add_all_duplicates_skipped(self):
        """When all IDs already exist, nothing is written."""
        col_ref, db_client, batch_mock = _make_col_ref()
        fc = FirestoreCollection(col_ref, db_client=db_client, embed_fn=_fake_embed)

        doc1 = MagicMock()
        doc1.get.return_value = _make_doc_snapshot("id1", {"document": "old"}, exists=True)
        doc2 = MagicMock()
        doc2.get.return_value = _make_doc_snapshot("id2", {"document": "old2"}, exists=True)
        col_ref.document.side_effect = [doc1, doc2]

        fc.add(
            documents=["new1", "new2"],
            ids=["id1", "id2"],
        )

        batch_mock.set.assert_not_called()


class TestFirestoreCollectionUpsert:
    def test_upsert_uses_merge_true(self):
        col_ref, db_client, batch_mock = _make_col_ref()
        fc = FirestoreCollection(col_ref, db_client=db_client, embed_fn=_fake_embed)

        fc.upsert(
            documents=["hi"],
            ids=["id1"],
            metadatas=[{"m": 1}],
        )

        batch_mock.set.assert_called_once()
        _, kwargs = batch_mock.set.call_args
        assert kwargs["merge"] is True

    def test_upsert_empty_list(self):
        col_ref, db_client, batch_mock = _make_col_ref()
        fc = FirestoreCollection(col_ref, db_client=db_client, embed_fn=_fake_embed)

        fc.upsert(documents=[], ids=[], metadatas=[])

        batch_mock.set.assert_not_called()
        batch_mock.commit.assert_not_called()

    def test_upsert_without_metadatas(self):
        col_ref, db_client, batch_mock = _make_col_ref()
        fc = FirestoreCollection(col_ref, db_client=db_client, embed_fn=_fake_embed)

        fc.upsert(documents=["doc"], ids=["id1"])

        data = batch_mock.set.call_args[0][1]
        assert data["meta"] == {}

    def test_upsert_over_batch_limit(self):
        col_ref, db_client, _ = _make_col_ref()
        first_batch = MagicMock()
        second_batch = MagicMock()
        db_client.batch.side_effect = [first_batch, second_batch]

        fc = FirestoreCollection(col_ref, db_client=db_client, embed_fn=_fake_embed)

        docs = [f"doc_{i}" for i in range(460)]
        ids = [f"id_{i}" for i in range(460)]

        fc.upsert(documents=docs, ids=ids)

        first_batch.commit.assert_called_once()
        second_batch.commit.assert_called_once()


class TestFirestoreCollectionUpdate:
    def test_update_with_documents_reembeds(self):
        col_ref, db_client, batch_mock = _make_col_ref()
        fc = FirestoreCollection(col_ref, db_client=db_client, embed_fn=_fake_embed)

        doc_ref = MagicMock()
        doc_ref.get.return_value = _make_doc_snapshot("id1", {"document": "old"}, exists=True)
        col_ref.document.return_value = doc_ref

        fc.update(ids=["id1"], documents=["new text"], metadatas=[{"m": 2}])

        batch_mock.update.assert_called_once()
        data = batch_mock.update.call_args[0][1]
        assert data["document"] == "new text"
        assert "embedding" in data
        assert data["meta"] == {"m": 2}

    def test_update_metadata_only_no_reembed(self):
        col_ref, db_client, batch_mock = _make_col_ref()
        fc = FirestoreCollection(col_ref, db_client=db_client, embed_fn=_fake_embed)

        doc_ref = MagicMock()
        doc_ref.get.return_value = _make_doc_snapshot("id1", {"document": "old"}, exists=True)
        col_ref.document.return_value = doc_ref

        fc.update(ids=["id1"], metadatas=[{"m": 3}])

        data = batch_mock.update.call_args[0][1]
        assert "document" not in data
        assert "embedding" not in data
        assert data["meta"] == {"m": 3}

    def test_update_with_nothing(self):
        col_ref, db_client, batch_mock = _make_col_ref()
        fc = FirestoreCollection(col_ref, db_client=db_client, embed_fn=_fake_embed)

        fc.update(ids=["id1"])
        batch_mock.update.assert_not_called()

    def test_update_nonexistent_id_is_silent_noop(self):
        """ChromaDB behaviour: update with nonexistent ID is a silent no-op."""
        col_ref, db_client, batch_mock = _make_col_ref()
        fc = FirestoreCollection(col_ref, db_client=db_client, embed_fn=_fake_embed)

        doc_ref = MagicMock()
        doc_ref.get.return_value = _make_doc_snapshot("id_missing", None, exists=False)
        col_ref.document.return_value = doc_ref

        # Should not raise
        fc.update(ids=["id_missing"], metadatas=[{"m": 99}])

        batch_mock.update.assert_not_called()

    def test_update_mixed_existing_and_nonexistent(self):
        """Only existing docs are updated; nonexistent ones are silently skipped."""
        col_ref, db_client, batch_mock = _make_col_ref()
        fc = FirestoreCollection(col_ref, db_client=db_client, embed_fn=_fake_embed)

        doc_existing = MagicMock()
        doc_existing.get.return_value = _make_doc_snapshot("id1", {"document": "old"}, exists=True)
        doc_missing = MagicMock()
        doc_missing.get.return_value = _make_doc_snapshot("id2", None, exists=False)
        col_ref.document.side_effect = [doc_existing, doc_missing]

        fc.update(
            ids=["id1", "id2"],
            metadatas=[{"m": 1}, {"m": 2}],
        )

        # Only id1 should have been updated
        assert batch_mock.update.call_count == 1
        data = batch_mock.update.call_args[0][1]
        assert data["meta"] == {"m": 1}

    def test_update_partial_docs_and_meta(self):
        """Some entries have docs, some have meta, some have both."""
        col_ref, db_client, batch_mock = _make_col_ref()
        fc = FirestoreCollection(col_ref, db_client=db_client, embed_fn=_fake_embed)

        # All three exist
        doc1 = MagicMock()
        doc1.get.return_value = _make_doc_snapshot("id1", {"document": "old1"}, exists=True)
        doc2 = MagicMock()
        doc2.get.return_value = _make_doc_snapshot("id2", {"document": "old2"}, exists=True)
        doc3 = MagicMock()
        doc3.get.return_value = _make_doc_snapshot("id3", {"document": "old3"}, exists=True)
        col_ref.document.side_effect = [doc1, doc2, doc3]

        # 3 IDs: first has doc+meta, second only meta (doc list shorter), third has neither
        fc.update(
            ids=["id1", "id2", "id3"],
            documents=["new text"],  # only 1 doc — only id1 gets doc+embedding
            metadatas=[{"m": 1}, {"m": 2}],  # 2 metas — id1 and id2 get meta
        )

        assert batch_mock.update.call_count == 2  # id3 has no updates

        # First call: id1 gets document + embedding + meta
        first_data = batch_mock.update.call_args_list[0][0][1]
        assert first_data["document"] == "new text"
        assert "embedding" in first_data
        assert first_data["meta"] == {"m": 1}

        # Second call: id2 gets only meta (no doc because i=1 >= len(documents)=1)
        second_data = batch_mock.update.call_args_list[1][0][1]
        assert "document" not in second_data
        assert "embedding" not in second_data
        assert second_data["meta"] == {"m": 2}


class TestFirestoreCollectionQuery:
    def test_query_embeds_and_calls_find_nearest(self):
        col_ref, db_client, _ = _make_col_ref()
        fc = FirestoreCollection(col_ref, db_client=db_client, embed_fn=_fake_embed)

        snap = _make_doc_snapshot(
            "d1",
            {
                "document": "hello",
                "meta": {"k": "v"},
                "vector_distance": 0.1,
            },
        )
        # find_nearest returns a query object whose .get() returns docs
        nearest_query = MagicMock()
        nearest_query.get.return_value = [snap]
        col_ref.find_nearest.return_value = nearest_query

        result = fc.query(query_texts=["search term"], n_results=5)

        col_ref.find_nearest.assert_called_once()
        call_kwargs = col_ref.find_nearest.call_args[1]
        assert call_kwargs["vector_field"] == "embedding"
        assert call_kwargs["distance_measure"] == _FakeDistanceMeasure.COSINE
        assert call_kwargs["limit"] == 5

        assert result["ids"] == [["d1"]]
        assert result["documents"] == [["hello"]]
        assert result["metadatas"] == [[{"k": "v"}]]
        assert result["distances"] == [[0.1]]

    def test_query_empty_query_texts(self):
        col_ref, db_client, _ = _make_col_ref()
        fc = FirestoreCollection(col_ref, db_client=db_client, embed_fn=_fake_embed)

        result = fc.query(query_texts=[])

        assert result == {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}

    def test_query_with_where_filter(self):
        col_ref, db_client, _ = _make_col_ref()
        fc = FirestoreCollection(col_ref, db_client=db_client, embed_fn=_fake_embed)

        filtered_ref = MagicMock()
        col_ref.where.return_value = filtered_ref
        nearest_q = MagicMock()
        nearest_q.get.return_value = []
        filtered_ref.find_nearest.return_value = nearest_q

        fc.query(query_texts=["q"], where={"wing": "project"})

        col_ref.where.assert_called_once()
        ff = col_ref.where.call_args[1]["filter"]
        assert ff == _FakeFieldFilter("meta.wing", "==", "project")
        filtered_ref.find_nearest.assert_called_once()

    def test_query_multiple_query_texts(self):
        """Multiple query_texts produce one result set per text (nested lists)."""
        col_ref, db_client, _ = _make_col_ref()
        fc = FirestoreCollection(col_ref, db_client=db_client, embed_fn=_fake_embed)

        snap1 = _make_doc_snapshot(
            "d1",
            {
                "document": "first",
                "meta": {"k": "v1"},
                "vector_distance": 0.1,
            },
        )
        snap2 = _make_doc_snapshot(
            "d2",
            {
                "document": "second",
                "meta": {"k": "v2"},
                "vector_distance": 0.3,
            },
        )

        nearest_q1 = MagicMock()
        nearest_q1.get.return_value = [snap1]
        nearest_q2 = MagicMock()
        nearest_q2.get.return_value = [snap2]

        col_ref.find_nearest.side_effect = [nearest_q1, nearest_q2]

        result = fc.query(query_texts=["alpha", "beta"], n_results=5)

        assert col_ref.find_nearest.call_count == 2
        assert result["ids"] == [["d1"], ["d2"]]
        assert result["documents"] == [["first"], ["second"]]
        assert result["metadatas"] == [[{"k": "v1"}], [{"k": "v2"}]]
        assert result["distances"] == [[0.1], [0.3]]

    def test_query_no_results(self):
        col_ref, db_client, _ = _make_col_ref()
        fc = FirestoreCollection(col_ref, db_client=db_client, embed_fn=_fake_embed)

        nearest_q = MagicMock()
        nearest_q.get.return_value = []
        col_ref.find_nearest.return_value = nearest_q

        result = fc.query(query_texts=["something"], n_results=5)

        assert result["ids"] == [[]]
        assert result["documents"] == [[]]
        assert result["metadatas"] == [[]]
        assert result["distances"] == [[]]

    def test_query_where_with_multi_text(self):
        """Where filter combined with multiple query texts."""
        col_ref, db_client, _ = _make_col_ref()
        fc = FirestoreCollection(col_ref, db_client=db_client, embed_fn=_fake_embed)

        filtered_ref = MagicMock()
        col_ref.where.return_value = filtered_ref

        snap1 = _make_doc_snapshot(
            "d1",
            {
                "document": "first",
                "meta": {"wing": "notes"},
                "vector_distance": 0.1,
            },
        )
        snap2 = _make_doc_snapshot(
            "d2",
            {
                "document": "second",
                "meta": {"wing": "notes"},
                "vector_distance": 0.2,
            },
        )

        nearest_q1 = MagicMock()
        nearest_q1.get.return_value = [snap1]
        nearest_q2 = MagicMock()
        nearest_q2.get.return_value = [snap2]
        filtered_ref.find_nearest.side_effect = [nearest_q1, nearest_q2]

        result = fc.query(
            query_texts=["alpha", "beta"],
            n_results=5,
            where={"wing": "notes"},
        )

        # where should have been applied for each query text
        assert col_ref.where.call_count == 2
        assert filtered_ref.find_nearest.call_count == 2
        assert result["ids"] == [["d1"], ["d2"]]

    def test_query_include_filtering_returns_none_for_excluded(self):
        """ChromaDB behaviour: non-included fields are None, not omitted."""
        col_ref, db_client, _ = _make_col_ref()
        fc = FirestoreCollection(col_ref, db_client=db_client, embed_fn=_fake_embed)

        snap = _make_doc_snapshot(
            "d1",
            {
                "document": "hello",
                "meta": {"k": "v"},
                "vector_distance": 0.1,
            },
        )
        nearest_q = MagicMock()
        nearest_q.get.return_value = [snap]
        col_ref.find_nearest.return_value = nearest_q

        result = fc.query(
            query_texts=["search"],
            n_results=5,
            include=["documents"],
        )

        assert result["documents"] == [["hello"]]
        # ChromaDB returns None for non-included fields, not omitting them
        assert result["metadatas"] is None
        assert result["distances"] is None
        # ids are always included
        assert result["ids"] == [["d1"]]


class TestFirestoreCollectionGet:
    def test_get_by_ids(self):
        col_ref, db_client, _ = _make_col_ref()
        fc = FirestoreCollection(col_ref, db_client=db_client, embed_fn=_fake_embed)

        existing = _make_doc_snapshot("id1", {"document": "text", "meta": {"a": 1}})
        db_client.get_all.return_value = [existing]

        result = fc.get(ids=["id1"])

        db_client.get_all.assert_called_once()
        assert result["ids"] == ["id1"]
        assert result["documents"] == ["text"]
        assert result["metadatas"] == [{"a": 1}]

    def test_get_by_ids_some_missing(self):
        col_ref, db_client, _ = _make_col_ref()
        fc = FirestoreCollection(col_ref, db_client=db_client, embed_fn=_fake_embed)

        existing = _make_doc_snapshot("id1", {"document": "hi", "meta": {}})
        missing = _make_doc_snapshot("id2", None, exists=False)

        db_client.get_all.return_value = [existing, missing]

        result = fc.get(ids=["id1", "id2"])

        assert result["ids"] == ["id1"]
        assert len(result["documents"]) == 1

    def test_get_with_where_limit_offset(self):
        col_ref, db_client, _ = _make_col_ref()
        fc = FirestoreCollection(col_ref, db_client=db_client, embed_fn=_fake_embed)

        snap = _make_doc_snapshot("d1", {"document": "text", "meta": {}})
        chain = MagicMock()
        col_ref.where.return_value = chain
        chain.offset.return_value = chain
        chain.limit.return_value = chain
        chain.stream.return_value = [snap]

        result = fc.get(where={"wing": "notes"}, limit=10, offset=5)

        col_ref.where.assert_called_once()
        chain.offset.assert_called_once_with(5)
        chain.limit.assert_called_once_with(10)
        assert result["ids"] == ["d1"]

    def test_get_no_args_returns_all(self):
        col_ref, db_client, _ = _make_col_ref()
        fc = FirestoreCollection(col_ref, db_client=db_client, embed_fn=_fake_embed)

        s1 = _make_doc_snapshot("a", {"document": "one", "meta": {}})
        s2 = _make_doc_snapshot("b", {"document": "two", "meta": {}})
        col_ref.stream.return_value = [s1, s2]

        result = fc.get()

        col_ref.stream.assert_called_once()
        assert result["ids"] == ["a", "b"]

    def test_get_empty_ids_raises_valueerror(self):
        """ChromaDB behaviour: get(ids=[]) raises ValueError."""
        col_ref, db_client, _ = _make_col_ref()
        fc = FirestoreCollection(col_ref, db_client=db_client, embed_fn=_fake_embed)

        with pytest.raises(ValueError, match="Expected IDs to be a non-empty list"):
            fc.get(ids=[])

    def test_get_all_ids_missing(self):
        """ChromaDB behaviour: all missing IDs returns empty lists."""
        col_ref, db_client, _ = _make_col_ref()
        fc = FirestoreCollection(col_ref, db_client=db_client, embed_fn=_fake_embed)

        missing1 = _make_doc_snapshot("id1", None, exists=False)
        missing2 = _make_doc_snapshot("id2", None, exists=False)
        db_client.get_all.return_value = [missing1, missing2]

        result = fc.get(ids=["id1", "id2"])

        assert result["ids"] == []
        assert result["documents"] == []
        assert result["metadatas"] == []

    def test_get_include_filtering_returns_none_for_excluded(self):
        """ChromaDB behaviour: non-included fields are None, not omitted."""
        col_ref, db_client, _ = _make_col_ref()
        fc = FirestoreCollection(col_ref, db_client=db_client, embed_fn=_fake_embed)

        existing = _make_doc_snapshot("id1", {"document": "text", "meta": {"a": 1}})
        db_client.get_all.return_value = [existing]

        result = fc.get(ids=["id1"], include=["documents"])

        assert result["ids"] == ["id1"]
        assert result["documents"] == ["text"]
        # ChromaDB returns None for non-included fields
        assert result["metadatas"] is None

    def test_get_include_metadatas_only(self):
        """include=['metadatas'] returns metadatas, documents is None."""
        col_ref, db_client, _ = _make_col_ref()
        fc = FirestoreCollection(col_ref, db_client=db_client, embed_fn=_fake_embed)

        existing = _make_doc_snapshot("id1", {"document": "text", "meta": {"a": 1}})
        db_client.get_all.return_value = [existing]

        result = fc.get(ids=["id1"], include=["metadatas"])

        assert result["ids"] == ["id1"]
        assert result["documents"] is None
        assert result["metadatas"] == [{"a": 1}]


class TestFirestoreCollectionDelete:
    def test_delete_by_ids(self):
        col_ref, db_client, batch_mock = _make_col_ref()
        fc = FirestoreCollection(col_ref, db_client=db_client, embed_fn=_fake_embed)

        doc1 = MagicMock()
        doc2 = MagicMock()
        col_ref.document.side_effect = [doc1, doc2]

        fc.delete(ids=["x", "y"])

        assert batch_mock.delete.call_count == 2
        batch_mock.commit.assert_called_once()

    def test_delete_by_where_filter(self):
        col_ref, db_client, batch_mock = _make_col_ref()
        fc = FirestoreCollection(col_ref, db_client=db_client, embed_fn=_fake_embed)

        s1 = _make_doc_snapshot("d1", {})
        s2 = _make_doc_snapshot("d2", {})
        chain = MagicMock()
        col_ref.where.return_value = chain
        chain.stream.return_value = [s1, s2]

        fc.delete(where={"source_file": "old.py"})

        assert batch_mock.delete.call_count == 2
        batch_mock.commit.assert_called_once()

    def test_delete_no_args_raises(self):
        col_ref, db_client, _ = _make_col_ref()
        fc = FirestoreCollection(col_ref, db_client=db_client, embed_fn=_fake_embed)

        with pytest.raises(ValueError, match="At least one of"):
            fc.delete()

    def test_delete_over_batch_limit_ids(self):
        col_ref, db_client, _ = _make_col_ref()
        first_batch = MagicMock()
        second_batch = MagicMock()
        db_client.batch.side_effect = [first_batch, second_batch]

        fc = FirestoreCollection(col_ref, db_client=db_client, embed_fn=_fake_embed)

        ids = [f"id_{i}" for i in range(460)]
        fc.delete(ids=ids)

        first_batch.commit.assert_called_once()
        second_batch.commit.assert_called_once()

    def test_delete_empty_ids_raises_valueerror(self):
        """ChromaDB behaviour: delete(ids=[]) raises ValueError."""
        col_ref, db_client, batch_mock = _make_col_ref()
        fc = FirestoreCollection(col_ref, db_client=db_client, embed_fn=_fake_embed)

        with pytest.raises(ValueError, match="Expected IDs to be a non-empty list"):
            fc.delete(ids=[])

    def test_delete_where_no_matches(self):
        """Where filter matches nothing — commit with zero ops is a no-op."""
        col_ref, db_client, batch_mock = _make_col_ref()
        fc = FirestoreCollection(col_ref, db_client=db_client, embed_fn=_fake_embed)

        chain = MagicMock()
        col_ref.where.return_value = chain
        chain.stream.return_value = []

        fc.delete(where={"source_file": "nonexistent.py"})

        batch_mock.delete.assert_not_called()
        # commit with 0 operations is a no-op
        batch_mock.commit.assert_not_called()

    def test_delete_where_document_contains(self):
        """where_document $contains: simulated via client-side filtering."""
        col_ref, db_client, batch_mock = _make_col_ref()
        fc = FirestoreCollection(col_ref, db_client=db_client, embed_fn=_fake_embed)

        snap_match = MagicMock()
        snap_match.to_dict.return_value = {"document": "hello world"}
        snap_match.reference = MagicMock()

        snap_no_match = MagicMock()
        snap_no_match.to_dict.return_value = {"document": "goodbye moon"}
        snap_no_match.reference = MagicMock()

        col_ref.stream.return_value = iter([snap_match, snap_no_match])

        fc.delete(where_document={"$contains": "hello"})

        batch_mock.delete.assert_called_once_with(snap_match.reference)

    def test_delete_where_document_not_contains(self):
        """where_document $not_contains: deletes docs NOT containing the string."""
        col_ref, db_client, batch_mock = _make_col_ref()
        fc = FirestoreCollection(col_ref, db_client=db_client, embed_fn=_fake_embed)

        snap_a = MagicMock()
        snap_a.to_dict.return_value = {"document": "hello world"}
        snap_a.reference = MagicMock()

        snap_b = MagicMock()
        snap_b.to_dict.return_value = {"document": "goodbye moon"}
        snap_b.reference = MagicMock()

        col_ref.stream.return_value = iter([snap_a, snap_b])

        fc.delete(where_document={"$not_contains": "hello"})

        batch_mock.delete.assert_called_once_with(snap_b.reference)


class TestFirestoreCollectionCount:
    def test_count(self):
        col_ref, db_client, _ = _make_col_ref()
        fc = FirestoreCollection(col_ref, db_client=db_client, embed_fn=_fake_embed)

        agg_val = MagicMock()
        agg_val.value = 42
        col_ref.count.return_value.get.return_value = [[agg_val]]

        assert fc.count() == 42

    def test_count_empty_collection(self):
        col_ref, db_client, _ = _make_col_ref()
        fc = FirestoreCollection(col_ref, db_client=db_client, embed_fn=_fake_embed)

        agg_val = MagicMock()
        agg_val.value = 0
        col_ref.count.return_value.get.return_value = [[agg_val]]

        assert fc.count() == 0


# ═══════════════════════════════════════════════════════════════════════════
# FirestoreBackend tests
# ═══════════════════════════════════════════════════════════════════════════


class TestFirestoreBackend:
    def test_get_collection_creates_at_correct_path(self):
        db = MagicMock()
        backend = FirestoreBackend(db, embed_fn=_fake_embed)

        col = backend.get_collection("users/abc123")

        db.collection.assert_called_once_with("users/abc123/mempalace_drawers")
        assert isinstance(col, FirestoreCollection)

    def test_get_collection_custom_name(self):
        db = MagicMock()
        backend = FirestoreBackend(db, embed_fn=_fake_embed)

        backend.get_collection("projects/x", collection_name="mempalace_closets")

        db.collection.assert_called_once_with("projects/x/mempalace_closets")

    def test_create_param_is_noop(self):
        db = MagicMock()
        backend = FirestoreBackend(db, embed_fn=_fake_embed)

        col = backend.get_collection("p", create=True)
        assert isinstance(col, FirestoreCollection)


# ═══════════════════════════════════════════════════════════════════════════
# _apply_where_filter tests
# ═══════════════════════════════════════════════════════════════════════════


class TestApplyWhereFilter:
    def test_empty_where(self):
        q = MagicMock()
        result = _apply_where_filter(q, {})
        assert result is q
        q.where.assert_not_called()

    def test_none_where(self):
        q = MagicMock()
        result = _apply_where_filter(q, None)
        assert result is q

    def test_simple_equality(self):
        q = MagicMock()
        q.where.return_value = q

        result = _apply_where_filter(q, {"wing": "project"})

        q.where.assert_called_once()
        ff = q.where.call_args[1]["filter"]
        assert ff == _FakeFieldFilter("meta.wing", "==", "project")
        assert result is q

    def test_and_filter(self):
        q = MagicMock()
        q.where.return_value = q

        _apply_where_filter(q, {"$and": [{"wing": "project"}, {"room": "backend"}]})

        assert q.where.call_count == 2
        q.where.assert_any_call(filter=_FakeFieldFilter("meta.wing", "==", "project"))
        q.where.assert_any_call(filter=_FakeFieldFilter("meta.room", "==", "backend"))

    def test_in_filter(self):
        q = MagicMock()
        q.where.return_value = q

        _apply_where_filter(q, {"wing": {"$in": ["a", "b"]}})

        q.where.assert_called_once()
        ff = q.where.call_args[1]["filter"]
        assert ff == _FakeFieldFilter("meta.wing", "in", ["a", "b"])

    def test_or_filter_applies_or_composite(self):
        q = MagicMock()
        q.where.return_value = q

        _apply_where_filter(q, {"$or": [{"wing": "a"}, {"wing": "b"}]})

        q.where.assert_called_once()
        call_kwargs = q.where.call_args
        or_filter = call_kwargs[1]["filter"]
        assert isinstance(or_filter, _FakeOr)
        assert len(or_filter.filters) == 2
        assert or_filter.filters[0] == _FakeFieldFilter("meta.wing", "==", "a")
        assert or_filter.filters[1] == _FakeFieldFilter("meta.wing", "==", "b")

    def test_nested_and_with_in(self):
        q = MagicMock()
        q.where.return_value = q

        _apply_where_filter(
            q,
            {
                "$and": [
                    {"wing": {"$in": ["a", "b"]}},
                    {"room": "backend"},
                ]
            },
        )

        assert q.where.call_count == 2
        q.where.assert_any_call(filter=_FakeFieldFilter("meta.wing", "in", ["a", "b"]))
        q.where.assert_any_call(filter=_FakeFieldFilter("meta.room", "==", "backend"))

    def test_gte_filter(self):
        q = MagicMock()
        q.where.return_value = q

        _apply_where_filter(q, {"score": {"$gte": 5}})

        q.where.assert_called_once()
        ff = q.where.call_args[1]["filter"]
        assert ff == _FakeFieldFilter("meta.score", ">=", 5)

    def test_lte_filter(self):
        q = MagicMock()
        q.where.return_value = q

        _apply_where_filter(q, {"score": {"$lte": 10}})

        q.where.assert_called_once()
        ff = q.where.call_args[1]["filter"]
        assert ff == _FakeFieldFilter("meta.score", "<=", 10)

    def test_ne_filter(self):
        q = MagicMock()
        q.where.return_value = q

        _apply_where_filter(q, {"status": {"$ne": "deleted"}})

        q.where.assert_called_once()
        ff = q.where.call_args[1]["filter"]
        assert ff == _FakeFieldFilter("meta.status", "!=", "deleted")

    def test_nin_filter(self):
        q = MagicMock()
        q.where.return_value = q

        _apply_where_filter(q, {"status": {"$nin": ["deleted", "archived"]}})

        q.where.assert_called_once()
        ff = q.where.call_args[1]["filter"]
        assert ff == _FakeFieldFilter("meta.status", "not-in", ["deleted", "archived"])

    def test_gt_filter(self):
        q = MagicMock()
        q.where.return_value = q

        _apply_where_filter(q, {"score": {"$gt": 5}})

        q.where.assert_called_once()
        ff = q.where.call_args[1]["filter"]
        assert ff == _FakeFieldFilter("meta.score", ">", 5)

    def test_lt_filter(self):
        q = MagicMock()
        q.where.return_value = q

        _apply_where_filter(q, {"score": {"$lt": 10}})

        q.where.assert_called_once()
        ff = q.where.call_args[1]["filter"]
        assert ff == _FakeFieldFilter("meta.score", "<", 10)

    def test_eq_filter(self):
        q = MagicMock()
        q.where.return_value = q

        _apply_where_filter(q, {"wing": {"$eq": "project"}})

        q.where.assert_called_once()
        ff = q.where.call_args[1]["filter"]
        assert ff == _FakeFieldFilter("meta.wing", "==", "project")

    def test_or_with_operators(self):
        """$or can contain sub-filters with operators, not just equality."""
        q = MagicMock()
        q.where.return_value = q

        _apply_where_filter(
            q,
            {
                "$or": [
                    {"score": {"$gte": 5}},
                    {"score": {"$lte": 1}},
                ]
            },
        )

        q.where.assert_called_once()
        or_filter = q.where.call_args[1]["filter"]
        assert isinstance(or_filter, _FakeOr)
        assert or_filter.filters[0] == _FakeFieldFilter("meta.score", ">=", 5)
        assert or_filter.filters[1] == _FakeFieldFilter("meta.score", "<=", 1)

    def test_multi_operator_dict_raises_valueerror(self):
        """ChromaDB behaviour: {"$gte": 5, "$lte": 10} raises ValueError."""
        with pytest.raises(
            ValueError, match="Expected operator expression to have exactly one operator"
        ):
            _build_field_filter("score", {"$gte": 5, "$lte": 10})

    def test_nested_and_inside_or(self):
        """ChromaDB supports $or containing $and sub-filters."""
        q = MagicMock()
        q.where.return_value = q

        _apply_where_filter(
            q,
            {
                "$or": [
                    {"$and": [{"wing": "a"}, {"room": "b"}]},
                    {"$and": [{"wing": "c"}, {"room": "d"}]},
                ]
            },
        )

        q.where.assert_called_once()
        or_filter = q.where.call_args[1]["filter"]
        assert isinstance(or_filter, _FakeOr)
        # Each $and branch should produce field filters that are included in the Or
        assert len(or_filter.filters) == 4
        assert _FakeFieldFilter("meta.wing", "==", "a") in or_filter.filters
        assert _FakeFieldFilter("meta.room", "==", "b") in or_filter.filters
        assert _FakeFieldFilter("meta.wing", "==", "c") in or_filter.filters
        assert _FakeFieldFilter("meta.room", "==", "d") in or_filter.filters
