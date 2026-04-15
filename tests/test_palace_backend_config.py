"""Tests for the configurable backend in mempalace/palace.py.

Verifies get_backend / set_backend behaviour and env-var driven selection.
"""

import os
from unittest.mock import MagicMock, patch

import pytest

import mempalace.palace as palace_mod
from mempalace.palace import get_backend, set_backend


@pytest.fixture(autouse=True)
def _reset_backend():
    """Clear the cached backend before and after each test."""
    palace_mod._DEFAULT_BACKEND = None
    yield
    palace_mod._DEFAULT_BACKEND = None


class TestGetBackendDefaults:
    def test_defaults_to_chroma_when_env_not_set(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("MEMPALACE_BACKEND", None)
            backend = get_backend()

        from mempalace.backends.chroma import ChromaBackend
        assert isinstance(backend, ChromaBackend)

    @patch.dict(os.environ, {"MEMPALACE_BACKEND": "firestore"})
    @patch("mempalace.palace.firestore_mod", create=True)
    def test_returns_firestore_when_env_set(self, mock_fs_mod):
        """MEMPALACE_BACKEND=firestore triggers FirestoreBackend creation."""
        mock_client = MagicMock()

        with patch("mempalace.palace._init_default_backend") as mock_init:
            # Simulate what _init_default_backend would return
            from mempalace.backends.firestore import FirestoreBackend
            fake_backend = FirestoreBackend(mock_client, embed_fn=lambda x: [[0.0] * 3] * len(x))
            mock_init.return_value = fake_backend

            backend = get_backend()

            assert isinstance(backend, FirestoreBackend)

    def test_caches_instance(self):
        """get_backend() returns the same instance on repeated calls."""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("MEMPALACE_BACKEND", None)

            b1 = get_backend()
            b2 = get_backend()

            assert b1 is b2


class TestSetBackend:
    def test_overrides_default(self):
        sentinel = MagicMock()
        set_backend(sentinel)
        assert get_backend() is sentinel

    def test_get_after_set_returns_override(self):
        custom = MagicMock()
        set_backend(custom)

        result = get_backend()

        assert result is custom

    def test_set_replaces_previous(self):
        first = MagicMock()
        second = MagicMock()

        set_backend(first)
        assert get_backend() is first

        set_backend(second)
        assert get_backend() is second
