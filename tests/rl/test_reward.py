"""Tests for the reward-v1 RewardFunction."""

import pytest
from agent_runtime.rl.reward import REWARD_VERSION, RewardFunction
from model.agent.actions import Action


def _state(**overrides):
    """Minimal AgentState-like dict."""
    s = {
        "status": "running",
        "candidate_files": [],
        "tool_calls": [],
        "observations": [],
        "test_results": [],
        "verification_stale": False,
        "verification_required": True,
        "llm_action_inputs_enabled": False,
        "editing_enabled": False,
        "plan_mode": False,
        "plan_mode_approved": False,
        "memory_written": False,
        "patch_summary": None,
        "patch": None,
        "edited_files": [],
        "edit_results": [],
        "error": None,
        "loop_count": 0,
        "max_loops": 8,
    }
    s.update(overrides)
    return s


class TestRewardVersion:
    def test_version_constant(self):
        assert REWARD_VERSION == "reward-v1"


class TestStepCost:
    def test_step_cost(self):
        rf = RewardFunction()
        prev = _state()
        next_s = _state()
        action = Action("search_code_context", {})
        result = rf.compute(prev, action, next_s, {})
        # step_cost = -0.02, and search_code_context with no candidates = -0.15
        # Total = -0.17
        assert result.reward < 0


class TestReadFile:
    def test_first_read_bonus(self):
        rf = RewardFunction()
        prev = _state()
        next_s = _state()
        output = {"file_path": "src/main.py", "content": "print('hello')"}
        action = Action("read_file", {"file_path": "src/main.py"})
        result = rf.compute(prev, action, next_s, output)
        # step_cost=-0.02 + first_read=+0.25 = +0.23
        assert result.reward == pytest.approx(0.23)
        assert any("first_read" in r for r in result.reasons)

    def test_repeat_read_penalty(self):
        rf = RewardFunction()
        prev = _state(
            tool_calls=[
                {
                    "name": "read_file",
                    "input": {"file_path": "src/main.py"},
                    "output": {"file_path": "src/main.py", "content": "hello"},
                    "error": None,
                }
            ]
        )
        next_s = _state(
            tool_calls=[
                {
                    "name": "read_file",
                    "input": {"file_path": "src/main.py"},
                    "output": {"file_path": "src/main.py", "content": "hello"},
                    "error": None,
                }
            ]
        )
        output = {"file_path": "src/main.py", "content": "print('hello')"}
        action = Action("read_file", {"file_path": "src/main.py"})
        result = rf.compute(prev, action, next_s, output)
        # step_cost=-0.02 + repeat_read=-0.12 = -0.14
        assert result.reward == pytest.approx(-0.14)
        assert any("repeat_read" in r for r in result.reasons)

    def test_read_empty_or_failed(self):
        rf = RewardFunction()
        prev = _state()
        next_s = _state()
        output = {"file_path": "src/main.py"}  # no content
        action = Action("read_file", {"file_path": "src/main.py"})
        result = rf.compute(prev, action, next_s, output)
        # step_cost=-0.02 + read_empty=-0.1 = -0.12
        assert result.reward == pytest.approx(-0.12)
        assert any("read_empty" in r for r in result.reasons)


class TestRunTests:
    def test_tests_passed(self):
        rf = RewardFunction()
        prev = _state()
        next_s = _state(
            test_results=[{"command": "pytest", "exit_code": 0}]
        )
        output = {"exit_code": 0}
        action = Action("run_tests", {"command": "pytest"})
        result = rf.compute(prev, action, next_s, output)
        # step_cost=-0.02 + tests_passed=+1.0 = +0.98
        assert result.reward == pytest.approx(0.98)
        assert any("tests_passed" in r for r in result.reasons)

    def test_tests_failed(self):
        rf = RewardFunction()
        prev = _state()
        next_s = _state(
            test_results=[{"command": "pytest", "exit_code": 1}]
        )
        output = {"exit_code": 1}
        action = Action("run_tests", {"command": "pytest"})
        result = rf.compute(prev, action, next_s, output)
        # step_cost=-0.02 + tests_failed=-0.25 = -0.27
        assert result.reward == pytest.approx(-0.27)
        assert any("tests_failed" in r for r in result.reasons)

    def test_cleared_verification_stale_bonus(self):
        rf = RewardFunction()
        prev = _state(verification_stale=True)
        next_s = _state(
            verification_stale=False,
            test_results=[{"command": "pytest", "exit_code": 0}],
        )
        output = {"exit_code": 0}
        action = Action("run_tests", {"command": "pytest"})
        result = rf.compute(prev, action, next_s, output)
        # -0.02 + 1.0 + 0.5 = 1.48
        assert result.reward == pytest.approx(1.48)
        assert any("cleared_verification_stale" in r for r in result.reasons)


class TestSearchCodeContext:
    def test_new_candidates_bonus(self):
        rf = RewardFunction()
        prev = _state(candidate_files=[])
        next_s = _state(candidate_files=["a.py", "b.py", "c.py"])
        output = {"files": ["a.py", "b.py", "c.py"]}
        action = Action("search_code_context", {"query": "test"})
        result = rf.compute(prev, action, next_s, output)
        # -0.02 + min(0.4, 0.1*3) = -0.02 + 0.3 = 0.28
        assert result.reward == pytest.approx(0.28)
        assert any("new_candidate_files" in r for r in result.reasons)

    def test_no_candidates_penalty(self):
        rf = RewardFunction()
        prev = _state(candidate_files=[])
        next_s = _state(candidate_files=[])
        output = {}
        action = Action("search_code_context", {"query": "test"})
        result = rf.compute(prev, action, next_s, output)
        # -0.02 + -0.15 = -0.17
        assert result.reward == pytest.approx(-0.17)
        assert any("no_candidates" in r for r in result.reasons)


class TestFinish:
    def test_finish_with_passing_tests(self):
        rf = RewardFunction()
        prev = _state()
        next_s = _state(
            test_results=[{"command": "pytest", "exit_code": 0}],
            candidate_files=["a.py"],
        )
        action = Action("finish", {})
        result = rf.compute(prev, action, next_s, {})
        # -0.02 + 1.0 = 0.98
        assert result.reward == pytest.approx(0.98)
        assert any("finish_with_passing_tests" in r for r in result.reasons)

    def test_finish_with_stale_verification(self):
        rf = RewardFunction()
        prev = _state()
        next_s = _state(
            verification_stale=True,
            candidate_files=["a.py"],
        )
        action = Action("finish", {})
        result = rf.compute(prev, action, next_s, {})
        # -0.02 + -1.2 = -1.22
        assert result.reward == pytest.approx(-1.22)
        assert any("stale_verification" in r for r in result.reasons)

    def test_premature_finish(self):
        rf = RewardFunction()
        prev = _state()
        next_s = _state()  # no candidates, no tests, no diff
        action = Action("finish", {})
        result = rf.compute(prev, action, next_s, {})
        # -0.02 + -0.8 = -0.82
        assert result.reward == pytest.approx(-0.82)
        assert any("premature_finish" in r for r in result.reasons)


class TestApplyCodePatch:
    def test_edit_applied(self):
        rf = RewardFunction()
        prev = _state()
        next_s = _state()
        output = {"applied": True}
        action = Action("apply_code_patch", {})
        result = rf.compute(prev, action, next_s, output)
        # -0.02 + 0.3 = 0.28
        assert result.reward == pytest.approx(0.28)
        assert any("edit_applied" in r for r in result.reasons)

    def test_edit_guard_penalty(self):
        rf = RewardFunction()
        prev = _state()
        next_s = _state()
        output = {"needs_more_context": True}
        action = Action("apply_code_patch", {})
        result = rf.compute(prev, action, next_s, output)
        # -0.02 + -0.08 = -0.10
        assert result.reward == pytest.approx(-0.10)


class TestGitDiff:
    def test_real_diff_bonus(self):
        rf = RewardFunction()
        prev = _state()
        next_s = _state()
        output = {"diff": "--- a/file.py\n+++ b/file.py\n@@ -1 +1 @@\n-old\n+new"}
        action = Action("git_diff", {})
        result = rf.compute(prev, action, next_s, output)
        # -0.02 + 0.15 = 0.13
        assert result.reward == pytest.approx(0.13)
        assert any("real_diff" in r for r in result.reasons)


class TestError:
    def test_fatal_error(self):
        rf = RewardFunction()
        prev = _state()
        next_s = _state(error="something went wrong")
        action = Action("run_tests", {})
        result = rf.compute(prev, action, next_s, {"error": "crash"})
        # -0.02 + -1.0 = -1.02
        assert result.reward < -1.0
        assert any("error" in r for r in result.reasons)


class TestTerminalReward:
    def test_terminal_reward_separate_from_compute(self):
        """terminal_reward exists for final report, not replay."""
        rf = RewardFunction()
        state = _state(
            status="finished",
            test_results=[{"command": "pytest", "exit_code": 0}],
            memory_written=True,
        )
        result = rf.terminal_reward(state)
        # finished +0.5 + tests_passed +1.0 + memory +0.2 = 1.7
        assert result.reward == pytest.approx(1.7)
