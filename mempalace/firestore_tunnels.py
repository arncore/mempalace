"""
firestore_tunnels.py — Firestore-backed explicit tunnel storage
================================================================

Drop-in replacement for palace_graph.py's JSON file-based tunnel
storage. Same interface (create_tunnel, list_tunnels, delete_tunnel,
follow_tunnels), backed by Firestore.

Path-agnostic: the caller provides a ``base_path`` that determines
where tunnels are stored.
"""

import hashlib
from datetime import datetime, timezone


def _endpoint_key(wing: str, room: str) -> str:
    return f"{wing}/{room}"


def _canonical_tunnel_id(
    source_wing: str, source_room: str, target_wing: str, target_room: str
) -> str:
    """Compute a symmetric tunnel ID (same as palace_graph.py)."""
    src = _endpoint_key(source_wing, source_room)
    tgt = _endpoint_key(target_wing, target_room)
    a, b = sorted((src, tgt))
    return hashlib.sha256(f"{a}↔{b}".encode()).hexdigest()[:16]


class FirestoreTunnelStore:
    """Explicit tunnel CRUD backed by Firestore.

    Tunnels are stored at ``{base_path}/tunnels/{tunnel_id}``.
    """

    def __init__(self, db, base_path: str):
        self._db = db
        self._col = db.collection(f"{base_path}/tunnels")

    def create_tunnel(
        self,
        source_wing: str,
        source_room: str,
        target_wing: str,
        target_room: str,
        label: str = "",
        source_drawer_id: str = None,
        target_drawer_id: str = None,
    ):
        """Create or update an explicit symmetric tunnel."""
        tunnel_id = _canonical_tunnel_id(source_wing, source_room, target_wing, target_room)
        now = datetime.now(timezone.utc).isoformat()

        doc_ref = self._col.document(tunnel_id)
        existing = doc_ref.get()

        tunnel = {
            "id": tunnel_id,
            "source": {"wing": source_wing, "room": source_room},
            "target": {"wing": target_wing, "room": target_room},
            "label": label,
        }
        if source_drawer_id:
            tunnel["source"]["drawer_id"] = source_drawer_id
        if target_drawer_id:
            tunnel["target"]["drawer_id"] = target_drawer_id

        if existing.exists:
            tunnel["created_at"] = existing.to_dict().get("created_at", now)
            tunnel["updated_at"] = now
        else:
            tunnel["created_at"] = now

        doc_ref.set(tunnel)
        return tunnel

    def list_tunnels(self, wing: str = None):
        """List all explicit tunnels, optionally filtered by wing."""
        tunnels = []
        for snap in self._col.stream():
            data = snap.to_dict()
            if wing:
                src_wing = data.get("source", {}).get("wing")
                tgt_wing = data.get("target", {}).get("wing")
                if src_wing != wing and tgt_wing != wing:
                    continue
            tunnels.append(data)
        return tunnels

    def delete_tunnel(self, tunnel_id: str):
        """Delete an explicit tunnel by ID."""
        self._col.document(tunnel_id).delete()
        return {"deleted": tunnel_id}

    def follow_tunnels(self, wing: str, room: str, drawers_col=None):
        """Follow explicit tunnels from a room — returns connected info.

        Optionally fetches drawer content previews if ``drawers_col``
        (a BaseCollection) is provided.
        """
        connections = []

        for snap in self._col.stream():
            t = snap.to_dict()
            src = t.get("source", {})
            tgt = t.get("target", {})

            if src.get("wing") == wing and src.get("room") == room:
                connections.append(
                    {
                        "direction": "outgoing",
                        "connected_wing": tgt.get("wing"),
                        "connected_room": tgt.get("room"),
                        "label": t.get("label", ""),
                        "drawer_id": tgt.get("drawer_id"),
                        "tunnel_id": t.get("id"),
                    }
                )
            elif tgt.get("wing") == wing and tgt.get("room") == room:
                connections.append(
                    {
                        "direction": "incoming",
                        "connected_wing": src.get("wing"),
                        "connected_room": src.get("room"),
                        "label": t.get("label", ""),
                        "drawer_id": src.get("drawer_id"),
                        "tunnel_id": t.get("id"),
                    }
                )

        # Fetch drawer previews if collection provided
        if drawers_col and connections:
            drawer_ids = [c["drawer_id"] for c in connections if c.get("drawer_id")]
            if drawer_ids:
                try:
                    results = drawers_col.get(ids=drawer_ids, include=["documents", "metadatas"])
                    drawer_map = dict(zip(results["ids"], results["documents"]))
                    for c in connections:
                        did = c.get("drawer_id")
                        if did and did in drawer_map:
                            c["drawer_preview"] = drawer_map[did][:300]
                except Exception:
                    pass

        return connections
