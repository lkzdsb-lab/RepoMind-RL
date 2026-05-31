"""Reward shaping for the debug agent RL loop.

Version: reward-v1 — state-change-aware rewards.
"""

from __future__ import annotations

from typing import Any

from model.agent.actions import Action
from model.agent.graph import AgentState
from model.agent.reward import RewardBreakdown

REWARD_VERSION = "reward-v1"


class RewardFunction:
    """State-change-aware reward function for the debug agent RL loop.

    Terminal reward is kept for the final report only and should NOT be
    written into the replay buffer (double-counting avoidance).
    """

    def compute(
        self,
        prev_state: AgentState,
        action: Action,
        next_state: AgentState,
        output: dict[str, Any] | None = None,
    ) -> RewardBreakdown:
        output = output or {}
        reward = -0.02
        reasons = ["step_cost=-0.02"]

        # -- fatal / error -------------------------------------------------
        if (output.get("error") and not output.get("needs_more_context")) or next_state.get(
            "error"
        ):
            reward -= 1.0
            reasons.append("error=-1.0")

        # -- search_code_context -------------------------------------------
        if action.name == "search_code_context":
            prev_candidates = set(prev_state.get("candidate_files") or [])
            next_candidates = set(next_state.get("candidate_files") or [])
            new_candidates = next_candidates - prev_candidates
            if new_candidates:
                bonus = min(0.4, 0.1 * len(new_candidates))
                reward += bonus
                reasons.append(f"new_candidate_files=+{bonus:.2f}")
            if not next_candidates:
                reward -= 0.15
                reasons.append("no_candidates=-0.15")

        # -- search_text (LLM-assisted) ------------------------------------
        elif action.name == "search_text":
            matches = output.get("matches") or []
            if matches:
                reward += min(0.4, 0.05 * len(matches))
                reasons.append(f"text_matches=+{min(0.4, 0.05 * len(matches)):.2f}")
            else:
                reward -= 0.03
                reasons.append("no_text_matches=-0.03")

        # -- read_file -----------------------------------------------------
        elif action.name == "read_file":
            file_path = str(action.args.get("file_path", "")).strip()
            if output.get("content"):
                if self._is_first_read(prev_state, file_path):
                    reward += 0.25
                    reasons.append("first_read=+0.25")
                else:
                    reward -= 0.12
                    reasons.append("repeat_read=-0.12")
            else:
                reward -= 0.1
                reasons.append("read_empty_or_failed=-0.1")

        # -- run_tests -----------------------------------------------------
        elif action.name == "run_tests":
            exit_code = output.get("exit_code")
            if exit_code == 0:
                reward += 1.0
                reasons.append("tests_passed=+1.0")
                # Bonus: cleared verification_stale after an edit
                if (
                    prev_state.get("verification_stale")
                    and not next_state.get("verification_stale")
                ):
                    reward += 0.5
                    reasons.append("cleared_verification_stale=+0.5")
            elif exit_code is not None:
                reward -= 0.25
                reasons.append("tests_failed=-0.25")

        # -- run_shell_command (LLM-assisted) ------------------------------
        elif action.name == "run_shell_command":
            exit_code = output.get("exit_code")
            if output.get("purpose") == "verification":
                if exit_code == 0:
                    reward += 1.0
                    reasons.append("verification_passed=+1.0")
                    if (
                        prev_state.get("verification_stale")
                        and not next_state.get("verification_stale")
                    ):
                        reward += 0.5
                        reasons.append("cleared_verification_stale=+0.5")
                elif exit_code is not None:
                    reward -= 0.25
                    reasons.append("verification_failed=-0.25")
            elif exit_code == 0:
                reward += 0.05
                reasons.append("diagnostic_command_ok=+0.05")

        # -- EnterPlanMode -------------------------------------------------
        elif action.name == "EnterPlanMode":
            if output.get("entered"):
                reward += 0.15
                reasons.append("plan_mode_entered=+0.15")

        # -- ExitPlanMode --------------------------------------------------
        elif action.name == "ExitPlanMode":
            if output.get("approved") and output.get("exited"):
                reward += 0.2
                reasons.append("plan_mode_approved=+0.2")
            elif output.get("needs_user_input"):
                reward += 0.05
                reasons.append("plan_uncertainty_escalated=+0.05")
            else:
                reward -= 0.05
                reasons.append("plan_mode_not_ready=-0.05")

        # -- apply_code_patch (LLM-assisted, guarded) ----------------------
        elif action.name == "apply_code_patch":
            if output.get("applied"):
                reward += 0.3
                reasons.append("edit_applied=+0.3")
            elif output.get("needs_user_input"):
                reward += 0.05
                reasons.append("asked_before_uncertain_edit=+0.05")
            elif output.get("needs_more_context") or output.get("guard_error"):
                reward -= 0.08
                reasons.append("edit_guard_or_needs_context=-0.08")

        # -- request_user_input --------------------------------------------
        elif action.name == "request_user_input":
            reward += 0.05
            reasons.append("explicit_user_question=+0.05")

        # -- git_diff ------------------------------------------------------
        elif action.name == "git_diff":
            if self._has_real_diff(output):
                reward += 0.15
                reasons.append("real_diff=+0.15")
            if next_state.get("patch_summary") is not None:
                reward += 0.1
                reasons.append("patch_summary=+0.1")
            if next_state.get("patch"):
                reward += 0.1
                reasons.append("patch_present=+0.1")

        # -- write_memory (system action, rarely reached via policy) --------
        elif action.name == "write_memory":
            if next_state.get("memory_written"):
                reward += 0.3
                reasons.append("memory_written=+0.3")

        # -- finish ---------------------------------------------------------
        elif action.name == "finish":
            if self._tests_passed(next_state):
                reward += 1.0
                reasons.append("finish_with_passing_tests=+1.0")
            if next_state.get("verification_stale"):
                reward -= 1.2
                reasons.append("finish_with_stale_verification=-1.2")
            if not self._has_candidates_or_tests_or_diff(next_state):
                reward -= 0.8
                reasons.append("premature_finish=-0.8")

        return RewardBreakdown(reward=round(reward, 4), reasons=reasons)

    def terminal_reward(self, state: AgentState) -> RewardBreakdown:
        """Terminal reward for the final *report only*.

        This is intentionally NOT written into the replay buffer to avoid
        double-counting the same episode outcome.
        """
        reward = 0.0
        reasons: list[str] = []
        if state.get("status") == "finished":
            reward += 0.5
            reasons.append("status_finished=+0.5")
        if state.get("status") == "failed":
            reward -= 0.5
            reasons.append("status_failed=-0.5")
        if self._tests_passed(state):
            reward += 1.0
            reasons.append("terminal_tests_passed=+1.0")
        if state.get("verification_stale"):
            reward -= 1.0
            reasons.append("terminal_stale_verification=-1.0")
        if state.get("memory_written"):
            reward += 0.2
            reasons.append("terminal_memory_written=+0.2")
        if state.get("error"):
            reward -= 0.5
            reasons.append("terminal_error=-0.5")
        return RewardBreakdown(reward=round(reward, 4), reasons=reasons)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _tests_passed(self, state: AgentState) -> bool:
        tests = state.get("test_results") or []
        return bool(tests and tests[-1].get("exit_code") == 0)

    def _is_first_read(self, state: AgentState, file_path: str) -> bool:
        """Return True if *file_path* has NOT been successfully read before."""
        if not file_path:
            return False
        for call in state.get("tool_calls", []) or []:
            if not isinstance(call, dict) or call.get("name") != "read_file":
                continue
            output = call.get("output")
            if not isinstance(output, dict) or output.get("error"):
                continue
            prev_path = str(output.get("file_path") or "").strip()
            if not prev_path:
                call_input = call.get("input")
                if isinstance(call_input, dict):
                    prev_path = str(call_input.get("file_path") or "").strip()
            if prev_path == file_path:
                return False
        return True

    def _has_real_diff(self, output: dict[str, Any]) -> bool:
        """Return True if the git_diff output contains actual diff content."""
        if output.get("skipped") or output.get("error"):
            return False
        diff = output.get("diff", "")
        return bool(diff and diff.strip())

    def _has_candidates_or_tests_or_diff(self, state: AgentState) -> bool:
        """Return True if the state has any sign of meaningful progress."""
        return bool(
            state.get("candidate_files")
            or state.get("test_results")
            or state.get("patch_summary")
            or state.get("patch")
            or state.get("edited_files")
        )
