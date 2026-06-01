import builtins

import pytest

from agent_runtime.executor import DebugAgent
from agent_runtime.rl import QLearningDebugPolicy
from config import DebugAgentConfig, LLMConfig
from model.agent.actions import Action


class TestExecutorImport:
    """Verify that importing the executor does not require openai."""

    def test_debug_agent_importable(self):
        """from agent_runtime.executor import DebugAgent must succeed."""
        # If we reach this line the import succeeded.
        assert DebugAgent is not None


class TestDebugAgentInitRlSmoke:
    def test_init_with_rl_enabled(self, tmp_path):
        config = _rl_smoke_config(repo_path=str(tmp_path))
        agent = DebugAgent(config)
        assert agent is not None
        assert agent.rl_enabled is True
        assert isinstance(agent.policy, QLearningDebugPolicy)
        # Q-table store and replay buffer should be initialised
        assert agent.rl_q_store is not None
        assert agent.rl_replay is not None
        assert agent.rl_trainer is not None

    def test_init_rl_table_paths(self, tmp_path):
        config = _rl_smoke_config(repo_path=str(tmp_path))
        agent = DebugAgent(config)
        # RL files create parent dirs on init
        assert agent.rl_q_store is not None
        assert agent.rl_replay is not None

    def test_rl_disabled_skips_replay_and_trainer(self, tmp_path):
        config = _rl_smoke_config(repo_path=str(tmp_path))
        config.rl_enabled = False
        agent = DebugAgent(config)
        assert agent.rl_enabled is False
        assert agent.rl_replay is None
        assert agent.rl_trainer is None

    def test_q_table_load_missing_file(self, tmp_path):
        config = _rl_smoke_config(repo_path=str(tmp_path))
        agent = DebugAgent(config)
        assert agent.rl_q_table == {}

    def test_q_table_loaded_when_llm_policy(self, tmp_path):
        """When action_policy_mode='llm', q-table is loaded even if rl_enabled=False."""
        config = _rl_smoke_config(repo_path=str(tmp_path))
        config.rl_enabled = False
        config.action_policy_mode = "llm"
        agent = DebugAgent(config)
        assert agent.rl_q_table == {}

    def test_record_transition_persists_tool_output_summary(self, tmp_path):
        config = _rl_smoke_config(repo_path=str(tmp_path))
        agent = DebugAgent(config)
        prev_state = {
            "task_id": "smoke-task",
            "status": "running",
            "loop_count": 0,
            "max_loops": 4,
            "tool_calls": [],
            "candidate_files": [],
        }
        next_state = {
            **prev_state,
            "status": "testing",
            "test_results": [{"exit_code": 0}],
        }
        output = {
            "exit_code": 0,
            "command": "python -m pytest tests/rl -q",
            "stdout": "large output should not be copied to replay",
        }

        updated = agent._record_rl_transition(
            prev_state,
            Action("run_tests", args={"command": "python -m pytest tests/rl -q"}),
            next_state,
            output,
            done=False,
        )

        transition_dict = updated["rl_transitions"][0]
        summary = transition_dict["tool_output_summary"]
        assert summary["exit_code"] == 0
        assert summary["command"] == "python -m pytest tests/rl -q"
        assert "stdout" not in summary

        replayed = agent.rl_replay.list()
        assert replayed[0].tool_output_summary["exit_code"] == 0


class TestLlmLazyImport:
    """Verify that the openai dependency is truly lazy."""

    def test_disabled_llm_client_no_import(self):
        """When provider is disabled, no import of openai should be attempted."""
        from agent_runtime.llm.llm import build_llm_client
        config = LLMConfig(provider="disabled", model="")
        client = build_llm_client(config)
        from agent_runtime.llm.llm import DisabledLLMClient
        assert isinstance(client, DisabledLLMClient)

    def test_openai_client_fails_cleanly_without_package(self):
        from agent_runtime.llm.llm import OpenAICompatibleLLMClient
        original_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "openai":
                raise ModuleNotFoundError("No module named 'openai'")
            return original_import(name, *args, **kwargs)

        config = LLMConfig(provider="openai", model="test-model")
        with pytest.MonkeyPatch.context() as monkeypatch:
            monkeypatch.setattr(builtins, "__import__", fake_import)
            with pytest.raises(RuntimeError, match="Install openai or disable LLM provider"):
                OpenAICompatibleLLMClient(config)


def _rl_smoke_config(repo_path: str) -> DebugAgentConfig:
    return DebugAgentConfig(
        repo_path=repo_path,
        rl_enabled=True,
        action_policy_mode="rl",
        max_loops=4,
        rl_q_table_path=".repomind/rl/q_table.json",
        rl_replay_path=".repomind/rl/replay.jsonl",
        # All LLM modes disabled so no openai import is triggered
        planner_mode="heuristic",
        task_analyzer_mode="disabled",
        observer_mode="disabled",
        completion_judge_mode="rule_based",
        final_reporter_mode="rule_based",
        memory_query_planner_mode="disabled",
        memory_reranker_mode="disabled",
        code_context_query_planner_mode="disabled",
        code_context_reranker_mode="disabled",
        skill_selector_mode="disabled",
    )
