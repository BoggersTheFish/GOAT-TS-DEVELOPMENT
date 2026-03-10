"""Graph management: node/edge CRUD, ingestion, persistence. Uses Rust for heavy traversal."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Literal

NodeType = Literal["Concept", "Document", "Observation", "Hypothesis"]
EdgeType = Literal["Causal", "Semantic", "Evidence", "relates_to"]


class KnowledgeGraph:
    """In-memory knowledge graph with optional SQLite persistence."""

    def __init__(self, db_path: str | Path | None = None, in_memory_only: bool = False):
        self.db_path = Path(db_path) if db_path else None
        self.in_memory_only = in_memory_only or (db_path is None)
        self._nodes: dict[int, dict[str, Any]] = {}
        self._edges: list[dict[str, Any]] = []
        self._id_to_index: dict[int, int] = {}
        if not self.in_memory_only and self.db_path:
            self._ensure_db()
            self._load_from_db()

    def _ensure_db(self) -> None:
        if not self.db_path:
            return
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self.db_path))
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS nodes (
                id INTEGER PRIMARY KEY,
                type TEXT NOT NULL,
                metadata TEXT,
                weight REAL DEFAULT 1.0,
                influence REAL DEFAULT 0.0
            );
            CREATE TABLE IF NOT EXISTS edges (
                from_id INTEGER NOT NULL,
                to_id INTEGER NOT NULL,
                type TEXT NOT NULL,
                strength REAL DEFAULT 1.0,
                PRIMARY KEY (from_id, to_id)
            );
            CREATE INDEX IF NOT EXISTS idx_edges_from ON edges(from_id);
            CREATE INDEX IF NOT EXISTS idx_edges_to ON edges(to_id);
        """)
        conn.commit()
        conn.close()

    def _load_from_db(self) -> None:
        if not self.db_path or not self.db_path.exists():
            return
        conn = sqlite3.connect(str(self.db_path))
        for row in conn.execute("SELECT id, type, metadata, weight, influence FROM nodes"):
            nid, ntype, meta, weight, influence = row
            self._nodes[nid] = {
                "id": nid,
                "type": ntype,
                "metadata": _json_load(meta),
                "weight": weight,
                "influence": influence,
            }
            self._id_to_index[nid] = len(self._id_to_index)
        for row in conn.execute("SELECT from_id, to_id, type, strength FROM edges"):
            self._edges.append({
                "from": row[0], "to": row[1], "type": row[2], "strength": row[3],
            })
        conn.close()

    def _save_to_db(self) -> None:
        if self.in_memory_only or not self.db_path:
            return
        conn = sqlite3.connect(str(self.db_path))
        conn.execute("DELETE FROM nodes")
        conn.execute("DELETE FROM edges")
        for nid, n in self._nodes.items():
            conn.execute(
                "INSERT INTO nodes (id, type, metadata, weight, influence) VALUES (?, ?, ?, ?, ?)",
                (nid, n["type"], _json_dump(n.get("metadata", {})), n.get("weight", 1.0), n.get("influence", 0.0)),
            )
        for e in self._edges:
            conn.execute(
                "INSERT INTO edges (from_id, to_id, type, strength) VALUES (?, ?, ?, ?)",
                (e["from"], e["to"], e["type"], e.get("strength", 1.0)),
            )
        conn.commit()
        conn.close()

    def add_node(
        self,
        node_id: int,
        node_type: NodeType = "Concept",
        metadata: dict[str, str] | None = None,
        weight: float = 1.0,
        influence: float = 0.0,
    ) -> None:
        if node_id not in self._id_to_index:
            self._id_to_index[node_id] = len(self._id_to_index)
        self._nodes[node_id] = {
            "id": node_id,
            "type": node_type,
            "metadata": metadata or {},
            "weight": weight,
            "influence": influence,
        }

    def add_edge(self, from_id: int, to_id: int, edge_type: EdgeType = "Semantic", strength: float = 1.0) -> None:
        if from_id not in self._nodes:
            self.add_node(from_id)
        if to_id not in self._nodes:
            self.add_node(to_id)
        self._edges.append({"from": from_id, "to": to_id, "type": edge_type, "strength": strength})

    def get_node(self, node_id: int) -> dict[str, Any] | None:
        return self._nodes.get(node_id)

    def set_influences(self, influences: dict[int, float]) -> None:
        for nid, inf in influences.items():
            if nid in self._nodes:
                self._nodes[nid]["influence"] = inf
        self._save_to_db()

    def to_dict(self) -> dict[str, Any]:
        """Export graph for Rust and CIG output."""
        nodes = [
            {
                "id": n["id"],
                "type": n["type"],
                "metadata": n.get("metadata", {}),
                "weight": n.get("weight", 1.0),
                "influence": n.get("influence", 0.0),
            }
            for n in self._nodes.values()
        ]
        edges = [
            {"from": e["from"], "to": e["to"], "type": e["type"], "strength": e.get("strength", 1.0)}
            for e in self._edges
        ]
        return {"nodes": nodes, "edges": edges}

    def persist(self) -> None:
        self._save_to_db()

    def node_count(self) -> int:
        return len(self._nodes)

    def edge_count(self) -> int:
        return len(self._edges)


def _json_load(s: str | None) -> dict:
    if not s:
        return {}
    try:
        return json.loads(s)
    except Exception:
        return {}


def _json_dump(obj: dict) -> str:
    return json.dumps(obj)
