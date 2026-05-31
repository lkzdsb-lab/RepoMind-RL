"""Tests for the QTableStore envelope format and legacy compatibility."""

import json
import tempfile
from pathlib import Path

from agent_runtime.rl.trainer import QTableStore


class TestQTableStoreEnvelope:
    def test_save_and_load_envelope(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "q_table.json"
            store = QTableStore(path)
            q_table = {
                "state_a": {"action_1": 0.5, "action_2": -0.1},
                "state_b": {"action_1": 0.9},
            }
            store.save(
                q_table,
                encoder_version="state-encoder-v1",
                action_space_version="action-space-v1",
                reward_version="reward-v1",
            )
            loaded = store.load()
            assert loaded == q_table
            # Verify envelope structure on disk
            raw = json.loads(path.read_text(encoding="utf-8"))
            assert "metadata" in raw
            assert raw["metadata"]["encoder_version"] == "state-encoder-v1"
            assert raw["metadata"]["action_space_version"] == "action-space-v1"
            assert raw["metadata"]["reward_version"] == "reward-v1"
            assert "q_values" in raw
            assert raw["q_values"] == q_table

    def test_load_missing_file_returns_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "nonexistent.json"
            store = QTableStore(path)
            assert store.load() == {}

    def test_legacy_format_returns_empty(self):
        """Legacy dict format (no metadata) should return empty to avoid pollution."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "legacy_q.json"
            legacy = {
                "state_old": {"action_old": 0.8, "action_z": -0.5},
            }
            path.write_text(json.dumps(legacy), encoding="utf-8")
            store = QTableStore(path)
            loaded = store.load()
            assert loaded == {}, "Legacy Q-table should return empty dict"

    def test_envelope_empty_q_values(self):
        """Envelope with metadata but empty q_values should work."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "empty_env.json"
            envelope = {
                "metadata": {
                    "encoder_version": "v1",
                    "action_space_version": "v1",
                    "reward_version": "v1",
                },
                "q_values": {},
            }
            path.write_text(json.dumps(envelope), encoding="utf-8")
            store = QTableStore(path)
            loaded = store.load()
            assert loaded == {}

    def test_envelope_with_string_number_conversion(self):
        """Values loaded from JSON should be float."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "q_table.json"
            store = QTableStore(path)
            store.save(
                {"s": {"a": 1}},
                encoder_version="v1",
                action_space_version="v1",
                reward_version="v1",
            )
            loaded = store.load()
            assert isinstance(loaded["s"]["a"], float)
            assert loaded["s"]["a"] == 1.0

    def test_unreadable_file_returns_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.json"
            path.write_text("not json{{{", encoding="utf-8")
            store = QTableStore(path)
            loaded = store.load()
            assert loaded == {}

    def test_non_dict_envelope(self):
        """If JSON root is a list or scalar, return empty."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "list.json"
            path.write_text(json.dumps([1, 2, 3]), encoding="utf-8")
            store = QTableStore(path)
            loaded = store.load()
            assert loaded == {}
