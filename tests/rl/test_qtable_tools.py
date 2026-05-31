"""Tests for the qtable_tools CLI (inspect + wrap-legacy)."""

import argparse
import json
import sys
import tempfile
from pathlib import Path
from io import StringIO

import pytest

from agent_runtime.rl import qtable_tools
from agent_runtime.rl.action_space import ACTION_SPACE_VERSION
from agent_runtime.rl.reward import REWARD_VERSION
from agent_runtime.rl.state_encoder import ENCODER_VERSION


class TestInspect:
    def test_inspect_envelope(self):
        with tempfile.TemporaryDirectory() as tmp:
            q_path = Path(tmp) / "q.json"
            envelope = {
                "metadata": {
                    "encoder_version": ENCODER_VERSION,
                    "action_space_version": ACTION_SPACE_VERSION,
                    "reward_version": REWARD_VERSION,
                },
                "q_values": {
                    "s1": {"a1": 0.5, "a2": -0.1},
                    "s2": {"a1": 0.9},
                },
            }
            q_path.write_text(json.dumps(envelope), encoding="utf-8")
            result = qtable_tools._inspect(str(q_path))
            assert result["is_envelope"] is True
            assert result["state_count"] == 2
            assert result["action_entries"] == 3
            assert result["q_max"] == 0.9
            assert result["q_min"] == -0.1

    def test_inspect_legacy(self):
        with tempfile.TemporaryDirectory() as tmp:
            q_path = Path(tmp) / "q.json"
            q_path.write_text(json.dumps({"s1": {"a1": 0.5}}), encoding="utf-8")
            result = qtable_tools._inspect(str(q_path))
            assert result["is_envelope"] is False
            assert result["state_count"] == 1
            assert result["metadata"] == {}

    def test_inspect_missing_file(self):
        with pytest.raises(SystemExit):
            qtable_tools._inspect("/nonexistent/q_table_xyz.json")

    def test_inspect_json_flag(self, capsys):
        with tempfile.TemporaryDirectory() as tmp:
            q_path = Path(tmp) / "q.json"
            q_path.write_text(json.dumps({"metadata": {}, "q_values": {}}), encoding="utf-8")
            import argparse
            ns = argparse.Namespace(q_table=str(q_path), json=True)
            qtable_tools.inspect_command(ns)
            captured = capsys.readouterr()
            data = json.loads(captured.out)
            assert data["is_envelope"] is True


class TestWrapLegacy:
    def test_refuses_without_trust_legacy(self):
        with tempfile.TemporaryDirectory() as tmp:
            in_path = Path(tmp) / "old.json"
            out_path = Path(tmp) / "wrapped.json"
            in_path.write_text(json.dumps({"s1": {"a1": 0.5}}), encoding="utf-8")
            with pytest.raises(SystemExit) as exc_info:
                qtable_tools.wrap_legacy_command(
                    argparse.Namespace(
                        input=str(in_path),
                        output=str(out_path),
                        trust_legacy=False,
                        force=False,
                    )
                )
            assert exc_info.value.code == 1

    def test_wraps_with_trust_legacy(self):
        with tempfile.TemporaryDirectory() as tmp:
            in_path = Path(tmp) / "old.json"
            out_path = Path(tmp) / "wrapped.json"
            in_path.write_text(json.dumps({"s1": {"a1": 0.5}}), encoding="utf-8")
            qtable_tools.wrap_legacy_command(
                argparse.Namespace(
                    input=str(in_path),
                    output=str(out_path),
                    trust_legacy=True,
                    force=False,
                )
            )
            assert out_path.exists()
            wrapped = json.loads(out_path.read_text(encoding="utf-8"))
            assert "metadata" in wrapped
            assert wrapped["metadata"]["encoder_version"] == ENCODER_VERSION
            assert wrapped["metadata"]["migrated_from"] == "legacy/unversioned"
            assert (
                wrapped["metadata"]["migration_mode"]
                == "wrap_only_no_semantic_conversion"
            )
            assert wrapped["q_values"] == {"s1": {"a1": 0.5}}

    def test_refuses_already_envelope_without_force(self):
        with tempfile.TemporaryDirectory() as tmp:
            in_path = Path(tmp) / "q.json"
            out_path = Path(tmp) / "out.json"
            envelope = {"metadata": {"v": "1"}, "q_values": {"s": {"a": 0.5}}}
            in_path.write_text(json.dumps(envelope), encoding="utf-8")
            with pytest.raises(SystemExit) as exc_info:
                qtable_tools.wrap_legacy_command(
                    argparse.Namespace(
                        input=str(in_path),
                        output=str(out_path),
                        trust_legacy=True,
                        force=False,
                    )
                )
            assert exc_info.value.code == 1

    def test_overwrites_envelope_with_force(self):
        with tempfile.TemporaryDirectory() as tmp:
            in_path = Path(tmp) / "q.json"
            out_path = Path(tmp) / "out.json"
            envelope = {
                "metadata": {"encoder_version": "old"},
                "q_values": {"s": {"a": 0.5}},
            }
            in_path.write_text(json.dumps(envelope), encoding="utf-8")
            qtable_tools.wrap_legacy_command(
                argparse.Namespace(
                    input=str(in_path),
                    output=str(out_path),
                    trust_legacy=True,
                    force=True,
                )
            )
            wrapped = json.loads(out_path.read_text(encoding="utf-8"))
            assert wrapped["metadata"]["encoder_version"] == ENCODER_VERSION

    def test_missing_input(self):
        with pytest.raises(SystemExit) as exc_info:
            qtable_tools.wrap_legacy_command(
                argparse.Namespace(
                    input="/nonexistent/q.json",
                    output="/tmp/out.json",
                    trust_legacy=True,
                    force=False,
                )
            )
        assert exc_info.value.code == 1
