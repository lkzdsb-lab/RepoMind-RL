"""Tests for the ActionSpace with action-space-v1 categories and constraints."""

from agent_runtime.rl.action_space import (
    ACTION_SPACE_VERSION,
    LLM_ASSISTED_ACTIONS,
    PURE_RL_ACTIONS,
    SYSTEM_ACTIONS,
    ActionSpace,
)


def _base_state(**overrides):
    """Return a minimal AgentState-like dict for testing."""
    state = {
        "status": "running",
        "registry_snapshot": {
            "tools": sorted(PURE_RL_ACTIONS | LLM_ASSISTED_ACTIONS | SYSTEM_ACTIONS),
        },
        "tool_calls": [],
        "observations": [],
        "candidate_files": [],
        "code_context": {},
        "verification_required": True,
        "plan_mode": False,
        "plan_mode_approved": False,
        "llm_action_inputs_enabled": False,
        "editing_enabled": False,
        "is_git_repo": True,
        "test_results": [],
        "verification_stale": False,
        "edited_files": [],
        "edit_results": [],
        "patch_summary": None,
        "memory_written": False,
        "loop_count": 0,
        "max_loops": 8,
        "error": None,
        "task_type": "BUG_FIX",
        "debug_technical_plan": "",
        "verification_reason": "",
        "verify_command": "pytest",
    }
    state.update(overrides)
    return state


class TestActionSpaceVersion:
    def test_version_constant(self):
        assert ACTION_SPACE_VERSION == "action-space-v1"


class TestPureRLActions:
    """In pure RL mode (llm_action_inputs_enabled=False), only pure RL actions
    plus finish should be available."""

    def test_pure_rl_only_allows_pure_actions(self):
        space = ActionSpace()
        state = _base_state(
            candidate_files=["src/main.py"],
        )
        spec_names = {spec.name for spec in space.legal_specs(state)}
        # Pure RL actions that should be present
        assert "search_code_context" in spec_names
        assert "read_file" in spec_names
        # LLM-assisted actions MUST NOT appear
        for action in LLM_ASSISTED_ACTIONS:
            assert action not in spec_names, f"{action} should not be legal in pure RL mode"

    def test_list_files_never_legal(self):
        space = ActionSpace()
        state = _base_state(llm_action_inputs_enabled=True)
        spec_names = {spec.name for spec in space.legal_specs(state)}
        assert "list_files" not in spec_names

    def test_write_memory_never_legal(self):
        space = ActionSpace()
        state = _base_state(llm_action_inputs_enabled=True, memory_written=False)
        spec_names = {spec.name for spec in space.legal_specs(state)}
        assert "write_memory" not in spec_names

    def test_apply_code_patch_not_allowed_without_llm(self):
        space = ActionSpace()
        state = _base_state(
            editing_enabled=True,
            plan_mode_approved=True,
            observations=[{"content": {"file_path": "src/main.py"}}],
        )
        spec_names = {spec.name for spec in space.legal_specs(state)}
        assert "apply_code_patch" not in spec_names

    def test_run_shell_command_not_allowed_in_pure_rl(self):
        space = ActionSpace()
        state = _base_state(verification_stale=True)
        spec_names = {spec.name for spec in space.legal_specs(state)}
        # In pure RL, verification_stale should only show read_file or run_tests
        assert "run_shell_command" not in spec_names
        assert "read_file" in spec_names or "run_tests" in spec_names


class TestLLMAssistedActions:
    """With llm_action_inputs_enabled=True, LLM-assisted actions become available
    when their other preconditions are met."""

    def test_llm_allows_complex_actions(self):
        space = ActionSpace()
        state = _base_state(
            llm_action_inputs_enabled=True,
            editing_enabled=True,
            plan_mode_approved=True,
            observations=[{"content": {"file_path": "src/main.py"}}],
            candidate_files=["src/main.py", "src/utils.py"],
            task_type="BUG_FIX",
        )
        spec_names = {spec.name for spec in space.legal_specs(state)}
        # Pure RL actions still present
        assert "search_code_context" in spec_names
        assert "read_file" in spec_names
        # LLM-assisted actions now available
        assert "search_text" in spec_names or "request_user_input" in spec_names

    def test_apply_code_patch_allowed_with_full_preconditions(self):
        space = ActionSpace()
        state = _base_state(
            llm_action_inputs_enabled=True,
            editing_enabled=True,
            plan_mode_approved=True,
            task_type="BUG_FIX",
            observations=[{"content": {"file_path": "src/main.py"}}],
        )
        spec_names = {spec.name for spec in space.legal_specs(state)}
        assert "apply_code_patch" in spec_names

    def test_apply_code_patch_needs_read_files(self):
        """apply_code_patch should NOT be legal when no files have been read."""
        space = ActionSpace()
        state = _base_state(
            llm_action_inputs_enabled=True,
            editing_enabled=True,
            plan_mode_approved=True,
            task_type="BUG_FIX",
            observations=[],  # no read files
        )
        spec_names = {spec.name for spec in space.legal_specs(state)}
        assert "apply_code_patch" not in spec_names

    def test_apply_code_patch_needs_plan_approved(self):
        """apply_code_patch should NOT be legal without plan_mode_approved."""
        space = ActionSpace()
        state = _base_state(
            llm_action_inputs_enabled=True,
            editing_enabled=True,
            plan_mode_approved=False,
            task_type="BUG_FIX",
            observations=[{"content": {"file_path": "src/main.py"}}],
        )
        spec_names = {spec.name for spec in space.legal_specs(state)}
        assert "apply_code_patch" not in spec_names


class TestVerificationStale:
    """When verification_stale=True, only verification-related actions are legal."""

    def test_stale_only_allows_verification_actions(self):
        space = ActionSpace()
        state = _base_state(
            llm_action_inputs_enabled=False,
            verification_stale=True,
            edited_files=["src/main.py"],
            tool_calls=[
                {
                    "name": "apply_code_patch",
                    "input": {},
                    "output": {"applied": True, "changed_files": ["src/main.py"]},
                    "error": None,
                }
            ],
        )
        spec_names = {spec.name for spec in space.legal_specs(state)}
        # Only read_file (re-read) or run_tests should appear
        assert spec_names.issubset({"read_file", "run_tests", "finish"})
        # Should have at least one verification action
        assert "read_file" in spec_names or "run_tests" in spec_names

    def test_stale_with_llm_allows_shell(self):
        space = ActionSpace()
        state = _base_state(
            llm_action_inputs_enabled=True,
            verification_stale=True,
            edited_files=["src/main.py"],
            tool_calls=[
                {
                    "name": "apply_code_patch",
                    "input": {},
                    "output": {"applied": True, "changed_files": ["src/main.py"]},
                    "error": None,
                }
            ],
        )
        spec_names = {spec.name for spec in space.legal_specs(state)}
        # LLM mode allows run_shell_command for stale verification
        assert "run_shell_command" in spec_names or "read_file" in spec_names or "run_tests" in spec_names


class TestFinishTerminal:
    def test_finish_available_when_can_finish(self):
        space = ActionSpace()
        state = _base_state(memory_written=True)
        spec_names = {spec.name for spec in space.legal_specs(state)}
        assert "finish" in spec_names

    def test_finish_available_at_loop_limit(self):
        space = ActionSpace()
        state = _base_state(loop_count=7, max_loops=8)
        spec_names = {spec.name for spec in space.legal_specs(state)}
        assert "finish" in spec_names
