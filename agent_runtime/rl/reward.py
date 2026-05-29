"""Reward shaping for the debug agent RL loop."""

from __future__ import annotations

from typing import Any

from model.agent.actions import Action
from model.agent.graph import AgentState
from model.agent.reward import RewardBreakdown


class RewardFunction:
    """
        最简版奖励函数设置，考虑优化掉 if
    """
    def compute(
        self,
        prev_state: AgentState,
        action: Action,
        next_state: AgentState,
        output: dict[str, Any] | None = None,
    ) -> RewardBreakdown:
        output = output or {}
        reward = -0.01
        reasons = ["step_cost=-0.01"]

        if (output.get("error") and not output.get("needs_more_context")) or next_state.get("error"):
            reward -= 1.0
            reasons.append("error=-1.0")

        if action.name == "list_files":
            count = len(output.get("files", []))
            if count:
                reward += 0.05
                reasons.append("listed_files=+0.05")

        elif action.name == "search_code_context":
            candidates = next_state.get("candidate_files") or []
            if candidates:
                reward += min(0.5, 0.1 * len(candidates))
                reasons.append(f"candidate_files=+{min(0.5, 0.1 * len(candidates)):.2f}")
            else:
                reward -= 0.1
                reasons.append("no_candidates=-0.1")
            if output.get("api_routes"):
                reward += 0.1
                reasons.append("api_routes=+0.1")
            if output.get("db_models"):
                reward += 0.1
                reasons.append("db_models=+0.1")

        elif action.name == "search_text":
            matches = output.get("matches") or []
            if matches:
                reward += min(0.4, 0.05 * len(matches))
                reasons.append(f"text_matches=+{min(0.4, 0.05 * len(matches)):.2f}")
            else:
                reward -= 0.03
                reasons.append("no_text_matches=-0.03")

        elif action.name == "read_file":
            if output.get("content"):
                reward += 0.2
                reasons.append("read_context=+0.2")

        elif action.name == "run_tests":
            exit_code = output.get("exit_code")
            if exit_code == 0:
                reward += 1.0
                reasons.append("tests_passed=+1.0")
            elif exit_code is not None:
                reward -= 0.2
                reasons.append("tests_failed=-0.2")

        elif action.name == "run_shell_command":
            exit_code = output.get("exit_code")
            if output.get("purpose") == "verification":
                if exit_code == 0:
                    reward += 1.0
                    reasons.append("verification_passed=+1.0")
                elif exit_code is not None:
                    reward -= 0.2
                    reasons.append("verification_failed=-0.2")
            elif exit_code == 0:
                reward += 0.05
                reasons.append("diagnostic_command_ok=+0.05")

        elif action.name == "EnterPlanMode":
            if output.get("entered"):
                reward += 0.15
                reasons.append("plan_mode_entered=+0.15")

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

        elif action.name == "apply_code_patch":
            if output.get("applied"):
                reward += 0.4
                reasons.append("edit_applied=+0.4")
            elif output.get("needs_user_input"):
                reward += 0.05
                reasons.append("asked_before_uncertain_edit=+0.05")
            elif output.get("needs_more_context"):
                reward -= 0.05
                reasons.append("edit_needs_more_context=-0.05")

        elif action.name == "request_user_input":
            reward += 0.05
            reasons.append("explicit_user_question=+0.05")

        elif action.name == "git_diff":
            if next_state.get("patch_summary") is not None:
                reward += 0.1
                reasons.append("patch_summary=+0.1")
            if next_state.get("patch"):
                reward += 0.25
                reasons.append("patch_present=+0.25")

        elif action.name == "write_memory":
            if next_state.get("memory_written"):
                reward += 0.3
                reasons.append("memory_written=+0.3")
            promoted = len(next_state.get("promoted_memories", []))
            consolidated = len(next_state.get("consolidated_skills", []))
            if promoted:
                reward += 0.2 * promoted
                reasons.append(f"promoted_memories=+{0.2 * promoted:.2f}")
            if consolidated:
                reward += 0.3 * consolidated
                reasons.append(f"consolidated_skills=+{0.3 * consolidated:.2f}")

        elif action.name == "finish":
            if self._tests_passed(next_state):
                reward += 1.0
                reasons.append("finish_with_passing_tests=+1.0")
            if next_state.get("verification_stale"):
                reward -= 1.0
                reasons.append("finish_with_stale_verification=-1.0")
            if next_state.get("memory_written"):
                reward += 0.2
                reasons.append("finish_after_memory=+0.2")
            if not self._is_task_complete(next_state):
                reward -= 0.5
                reasons.append("finish_incomplete=-0.5")

        return RewardBreakdown(reward=round(reward, 4), reasons=reasons)

    def terminal_reward(self, state: AgentState) -> RewardBreakdown:
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

    def _tests_passed(self, state: AgentState) -> bool:
        tests = state.get("test_results") or []
        return bool(tests and tests[-1].get("exit_code") == 0)

    # 三个标志此次流程结束
    def _is_task_complete(self, state: AgentState) -> bool:
        return (
            bool(state.get("error"))
            or bool(state.get("memory_written"))
            or self._tests_passed(state)
        )
