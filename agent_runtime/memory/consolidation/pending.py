"""Durable, coalescing projection work queue for a single memory writer."""

import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path


class PendingProjections:
    def __init__(self, path: Path) -> None:
        self.path = path

    @contextmanager
    def connect(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path, timeout=5)
        connection.row_factory = sqlite3.Row
        try:
            connection.execute("CREATE TABLE IF NOT EXISTS pending ("
                               "memory_id TEXT NOT NULL, projection TEXT NOT NULL, "
                               "attempts INTEGER NOT NULL DEFAULT 0, error TEXT NOT NULL DEFAULT '', "
                               "due REAL NOT NULL DEFAULT 0, PRIMARY KEY(memory_id, projection))")
            connection.execute('CREATE INDEX IF NOT EXISTS pending_due_idx ON pending(projection,due,memory_id)')
            yield connection
            connection.commit()
        finally:
            connection.close()

    def enqueue(self, memory_id, projection):
        with self.connect() as db:
            db.execute("INSERT INTO pending(memory_id,projection) VALUES (?,?) "
                       "ON CONFLICT(memory_id,projection) DO UPDATE SET due=0",
                       (memory_id, projection))

    def complete(self, memory_id, projection):
        with self.connect() as db:
            db.execute("DELETE FROM pending WHERE memory_id=? AND projection=?", (memory_id, projection))

    def fail(self, memory_id, projection, error):
        with self.connect() as db:
            db.execute("UPDATE pending SET attempts=attempts+1,error=?,due=? "
                       "WHERE memory_id=? AND projection=?",
                       (str(error)[:2000], time.time()+60, memory_id, projection))

    def batch(self, projections, limit=16):
        with self.connect() as db:
            slots = ','.join('?' for _ in projections)
            return db.execute(f"SELECT * FROM pending WHERE projection IN ({slots}) "
                              "AND due<=? ORDER BY due,memory_id LIMIT ?",
                              (*projections, time.time(), limit)).fetchall()

    def has_pending(self, projection):
        with self.connect() as db:
            return db.execute("SELECT 1 FROM pending WHERE projection=? LIMIT 1", (projection,)).fetchone() is not None
