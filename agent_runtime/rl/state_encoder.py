"""State encoding for lightweight RL policies."""

from __future__ import annotations

from typing import Any
from model.agent.encode import EncodedState
from model.agent.graph import AgentState


class StateEncoder:
    """
        把复杂的 AgentState 压缩成 RL policy 能学习的离散状态表示
    """
    def encode(self, state: AgentState) -> EncodedState:
        features = self.features(state)
        key = "|".join(f"{name}={features[name]}" for name in sorted(features))
        return EncodedState(key=key, features=features)

    # todo 将返回 model 化
    def features(self, state: AgentState) -> dict[str, Any]:
        """
            将运行时 state 转换成可学习的 format
        """
        test_results = state.get("test_results") or []
        latest_test = test_results[-1] if test_results else {}
        candidate_files = state.get("candidate_files") or []
        read_files = self._read_files(state)
        unread = [path for path in candidate_files if path not in read_files]
        loop_count = int(state.get("loop_count", 0))
        max_loops = max(int(state.get("max_loops", 1)), 1)
        return {
            "status": state.get("status", "created"),
            "review_only": bool(state.get("review_only")),
            # agent 的阶段记录，为了区分失败的性质，当进入高位 bucket 时，系统可以强制介入
            "loop_bucket": self._bucket(loop_count, [0, 1, 2, 4, max_loops]),
            # agent 的相对记录，控制成本和防止死循环。它让 Agent 意识到“预算有限”，从而在接近终点时变得更加谨慎或果断。
            "progress_bucket": min(4, int((loop_count / max_loops) * 4)),
            "has_candidates": bool(candidate_files),
            "unread_candidates": min(len(unread), 3),
            "has_code_context": bool(state.get("code_context")),
            "has_tests": bool(test_results),
            "tests_passed": latest_test.get("exit_code") == 0,
            "tests_failed": bool(test_results) and latest_test.get("exit_code") != 0,
            "has_patch_summary": state.get("patch_summary") is not None,
            "has_patch": bool(state.get("patch")),
            "memory_written": bool(state.get("memory_written")),
            "has_error": bool(state.get("error")),
            "has_memory_context": bool(state.get("memory_context")),
            "has_compressed_context": bool(state.get("compressed_context")),
        }

    # 从整个流程记录获取访问过的文件
    def _read_files(self, state: AgentState) -> set[str]:
        files: set[str] = set()
        for observation in state.get("observations", []):
            content = observation.get("content", {})
            if isinstance(content, dict) and content.get("file_path"):
                files.add(str(content["file_path"]))
        return files

    # 非线性分桶
    def _bucket(self, value: int, thresholds: list[int]) -> str:
        for threshold in thresholds:
            if value <= threshold:
                return str(threshold)
        return "many"
