"""Tests for the RL offline evaluator."""

import json
import sys
import tempfile
from pathlib import Path
from io import StringIO

from agent_runtime.rl import evaluator
from agent_runtime.rl.action_space import ACTION_SPACE_VERSION
from agent_runtime.rl.reward import REWARD_VERSION
from agent_runtime.rl.state_encoder import ENCODER_VERSION


def _write_replay(path: Path, transitions: list[dict]) -> Path:
    with path.open("w", encoding="utf-8") as fp:
        for t in transitions:
            fp.write(json.dumps(t, ensure_ascii=False) + "\n")
    return path


def _write_q_table(path: Path, data: dict) -> Path:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


class TestEvaluatorTextOutput:
    def test_empty_replay(self):
        with tempfile.TemporaryDirectory() as tmp:
            replay = _write_replay(Path(tmp) / "replay.jsonl", [])
            q_table = _write_q_table(Path(tmp) / "q.json", {"metadata": {}, "q_values": {}})
            captured = StringIO()
            old = sys.stdout
            sys.stdout = captured
            try:
                evaluator.run(str(replay), str(q_table), "text")
            finally:
                sys.stdout = old
            output = captured.getvalue()
            assert "RL Evaluation Report" in output
            assert "Transitions" in output
            assert "0" in output

    def test_with_transitions(self):
        with tempfile.TemporaryDirectory() as tmp:
            transitions = [
                {
                    "state_key": "s1", "action": "search_code_context",
                    "reward": 0.3, "next_state_key": "s2", "done": False,
                    "task_id": "t1",
                    "encoder_version": ENCODER_VERSION,
                    "action_space_version": ACTION_SPACE_VERSION,
                    "reward_version": REWARD_VERSION,
                },
                {
                    "state_key": "s2", "action": "read_file",
                    "reward": 0.23, "next_state_key": "s3", "done": False,
                    "task_id": "t1",
                    "encoder_version": ENCODER_VERSION,
                    "action_space_version": ACTION_SPACE_VERSION,
                    "reward_version": REWARD_VERSION,
                },
                {
                    "state_key": "s3", "action": "run_tests",
                    "reward": 0.98, "next_state_key": "s4", "done": False,
                    "task_id": "t1",
                    "encoder_version": ENCODER_VERSION,
                    "action_space_version": ACTION_SPACE_VERSION,
                    "reward_version": REWARD_VERSION,
                },
                {
                    "state_key": "s4", "action": "finish",
                    "reward": 0.98, "next_state_key": "s5", "done": True,
                    "task_id": "t1",
                    "encoder_version": ENCODER_VERSION,
                    "action_space_version": ACTION_SPACE_VERSION,
                    "reward_version": REWARD_VERSION,
                },
            ]
            replay = _write_replay(Path(tmp) / "replay.jsonl", transitions)
            q_data = {
                "metadata": {
                    "encoder_version": ENCODER_VERSION,
                    "action_space_version": ACTION_SPACE_VERSION,
                    "reward_version": REWARD_VERSION,
                },
                "q_values": {"s2": {"read_file": 0.5}},
            }
            q_table = _write_q_table(Path(tmp) / "q.json", q_data)
            captured = StringIO()
            old = sys.stdout
            sys.stdout = captured
            try:
                evaluator.run(str(replay), str(q_table), "text")
            finally:
                sys.stdout = old
            output = captured.getvalue()
            assert "RL Evaluation Report" in output
            assert "search_code_context" in output
            assert "read_file" in output
            assert "run_tests" in output
            assert "finish" in output
            assert "Expected versions" in output
            assert ENCODER_VERSION in output
            assert "Metadata matches expected: True" in output
            assert "Replay version coverage" in output
            assert "100.0%" in output or "1.0" in output
            assert "Q-table:" in output

    def test_missing_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            captured = StringIO()
            old = sys.stdout
            sys.stdout = captured
            try:
                evaluator.run(str(Path(tmp) / "nope.jsonl"), str(Path(tmp) / "nope.json"), "text")
            finally:
                sys.stdout = old
            output = captured.getvalue()
            assert "RL Evaluation Report" in output
            # Should not crash


class TestEvaluatorJsonOutput:
    def test_json_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            transitions = [
                {
                    "state_key": "s1", "action": "finish",
                    "reward": 0.5, "next_state_key": "s2", "done": True,
                    "task_id": "t1",
                    "encoder_version": ENCODER_VERSION,
                    "action_space_version": ACTION_SPACE_VERSION,
                    "reward_version": REWARD_VERSION,
                },
            ]
            replay = _write_replay(Path(tmp) / "replay.jsonl", transitions)
            q_data = {
                "metadata": {
                    "encoder_version": ENCODER_VERSION,
                    "action_space_version": ACTION_SPACE_VERSION,
                    "reward_version": REWARD_VERSION,
                },
                "q_values": {},
            }
            q_table = _write_q_table(Path(tmp) / "q.json", q_data)
            captured = StringIO()
            old = sys.stdout
            sys.stdout = captured
            try:
                evaluator.run(str(replay), str(q_table), "json")
            finally:
                sys.stdout = old
            data = json.loads(captured.getvalue())
            assert data["transitions"] == 1
            assert data["episodes"] == 1
            assert "expected_versions" in data
            assert "q_table_metadata" in data
            assert "metadata_matches_expected" in data
            assert data["metadata_matches_expected"] is True
            assert "replay_version_coverage" in data
            assert data["replay_version_coverage"]["encoder_version"] == 1.0
            assert "behavior_metrics" in data
            assert "verification_pass_rate" in data["behavior_metrics"]


class TestFinishAfterTestsRatio:
    def test_recognises_run_shell_command_verification_exit_code(self):
        transitions = [
            {
                "state_key": "s1", "action": "run_shell_command",
                "reward": 0.0,
                "action_args": {"purpose": "verification", "exit_code": 0},
                "next_state_key": "s2", "done": False,
                "task_id": "t1",
                "encoder_version": "",
                "action_space_version": "",
                "reward_version": "",
            },
            {
                "state_key": "s2", "action": "finish",
                "reward": 0.5, "next_state_key": "s3", "done": True,
                "task_id": "t1",
                "encoder_version": "",
                "action_space_version": "",
                "reward_version": "",
            },
        ]
        ratio = evaluator._finish_after_tests_ratio(transitions)
        assert ratio == 1.0

    def test_recognises_legacy_run_shell_command_verification_reward(self):
        transitions = [
            {
                "state_key": "s1", "action": "run_shell_command",
                "reward": 1.0,
                "action_args": {"purpose": "verification"},
                "next_state_key": "s2", "done": False,
                "task_id": "t1",
                "encoder_version": "",
                "action_space_version": "",
                "reward_version": "",
            },
            {
                "state_key": "s2", "action": "finish",
                "reward": 0.5, "next_state_key": "s3", "done": True,
                "task_id": "t1",
                "encoder_version": "",
                "action_space_version": "",
                "reward_version": "",
            },
        ]
        ratio = evaluator._finish_after_tests_ratio(transitions)
        assert ratio == 1.0

    def test_recognises_run_tests_pass(self):
        transitions = [
            {
                "state_key": "s1", "action": "run_tests",
                "reward": 1.0,
                "next_state_key": "s2", "done": False,
                "task_id": "t1",
                "encoder_version": "",
                "action_space_version": "",
                "reward_version": "",
            },
            {
                "state_key": "s2", "action": "finish",
                "reward": 0.5, "next_state_key": "s3", "done": True,
                "task_id": "t1",
                "encoder_version": "",
                "action_space_version": "",
                "reward_version": "",
            },
        ]
        ratio = evaluator._finish_after_tests_ratio(transitions)
        assert ratio == 1.0

    def test_finish_without_verification(self):
        transitions = [
            {
                "state_key": "s1", "action": "read_file",
                "reward": 0.23, "next_state_key": "s2", "done": False,
                "task_id": "t1",
                "encoder_version": "",
                "action_space_version": "",
                "reward_version": "",
            },
            {
                "state_key": "s2", "action": "finish",
                "reward": -0.5, "next_state_key": "s3", "done": True,
                "task_id": "t1",
                "encoder_version": "",
                "action_space_version": "",
                "reward_version": "",
            },
        ]
        ratio = evaluator._finish_after_tests_ratio(transitions)
        assert ratio == 0.0


class TestBehaviorMetrics:
    def test_behavior_metrics_measure_search_reads_verification_and_steps(self):
        transitions = [
            {
                "task_id": "t1",
                "action": "search_code_context",
                "reward": 0.4,
                "tool_output_summary": {"candidate_count": 2},
            },
            {
                "task_id": "t1",
                "action": "search_code_context",
                "reward": -0.1,
                "tool_output_summary": {"candidate_count": 0},
            },
            {
                "task_id": "t1",
                "action": "read_file",
                "action_args": {"file_path": "a.py"},
            },
            {
                "task_id": "t1",
                "action": "read_file",
                "action_args": {"file_path": "a.py"},
            },
            {
                "task_id": "t2",
                "action": "read_file",
                "action_args": {"file_path": "b.py"},
            },
            {
                "task_id": "t2",
                "action": "run_tests",
                "tool_output_summary": {"exit_code": 0},
            },
            {
                "task_id": "t2",
                "action": "run_shell_command",
                "action_args": {"purpose": "verification"},
                "tool_output_summary": {"exit_code": 1},
            },
        ]

        metrics = evaluator._behavior_metrics(transitions)

        assert metrics["search_hit_rate"] == 0.5
        assert metrics["duplicate_read_ratio"] == 0.3333
        assert metrics["verification_pass_rate"] == 0.5
        assert metrics["steps_per_episode_avg"] == 3.5


class TestReplayVersionCoverage:
    def test_full_coverage(self):
        transitions = [
            {
                "state_key": "s1", "action": "a",
                "reward": 0.0, "next_state_key": "s2", "done": False,
                "task_id": "t1",
                "encoder_version": ENCODER_VERSION,
                "action_space_version": ACTION_SPACE_VERSION,
                "reward_version": REWARD_VERSION,
            },
        ]
        cov = evaluator._replay_version_coverage(transitions)
        assert cov["encoder_version"] == 1.0
        assert cov["action_space_version"] == 1.0
        assert cov["reward_version"] == 1.0

    def test_zero_coverage(self):
        transitions = [
            {
                "state_key": "s1", "action": "a",
                "reward": 0.0, "next_state_key": "s2", "done": False,
                "task_id": "t1",
                "encoder_version": "",
                "action_space_version": "",
                "reward_version": "",
            },
        ]
        cov = evaluator._replay_version_coverage(transitions)
        assert cov["encoder_version"] == 0.0

    def test_empty_replay(self):
        cov = evaluator._replay_version_coverage([])
        assert cov["encoder_version"] == 0.0


class TestLegacyQTable:
    def test_legacy_q_table_is_readable(self):
        """Legacy format (no metadata) should still load q_values."""
        with tempfile.TemporaryDirectory() as tmp:
            legacy = {"old_state": {"old_action": 0.7}}
            q_path = _write_q_table(Path(tmp) / "q.json", legacy)
            data = evaluator._load_q_table(str(q_path))
            # Legacy gets empty metadata
            assert data["metadata"] == {}
            assert data["q_values"] == legacy
