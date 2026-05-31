"""Tests for the ReplayBuffer."""

import json
import tempfile
from pathlib import Path

from agent_runtime.rl.replay_buffer import ReplayBuffer
from model.agent.transition import Transition


class TestReplayBuffer:
    def test_append_and_list(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "replay.jsonl"
            buf = ReplayBuffer(path)
            t1 = Transition(
                state_key="s1",
                action="a1",
                reward=0.5,
                next_state_key="s2",
                done=False,
            )
            t2 = Transition(
                state_key="s2",
                action="finish",
                reward=1.0,
                next_state_key="s3",
                done=True,
            )
            buf.append(t1)
            buf.append(t2)
            items = buf.list()
            assert len(items) == 2
            assert items[0].action == "a1"
            assert items[1].action == "finish"

    def test_sample_respects_batch_size(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "replay.jsonl"
            buf = ReplayBuffer(path)
            for i in range(10):
                buf.append(
                    Transition(
                        state_key=f"s{i}",
                        action="a",
                        reward=float(i),
                        next_state_key=f"s{i+1}",
                        done=False,
                    )
                )
            sample = buf.sample(3)
            assert len(sample) == 3

    def test_sample_all_when_fewer_than_batch(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "replay.jsonl"
            buf = ReplayBuffer(path)
            for i in range(2):
                buf.append(
                    Transition(
                        state_key=f"s{i}",
                        action="a",
                        reward=float(i),
                        next_state_key=f"s{i+1}",
                        done=False,
                    )
                )
            sample = buf.sample(10)
            assert len(sample) == 2

    def test_compact_removes_old_entries(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "replay.jsonl"
            buf = ReplayBuffer(path, max_size=5)
            for i in range(10):
                buf.append(
                    Transition(
                        state_key=f"s{i}",
                        action="a",
                        reward=float(i),
                        next_state_key=f"s{i+1}",
                        done=False,
                    )
                )
            items = buf.list()
            assert len(items) == 5
            # Should keep the 5 most recent
            assert items[0].state_key == "s5"
            assert items[-1].state_key == "s9"

    def test_empty_list(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "nonexistent.jsonl"
            buf = ReplayBuffer(path)
            assert buf.list() == []

    def test_transition_roundtrip_with_version_fields(self):
        """Version fields and next_legal_actions survive to_dict/from_dict."""
        t = Transition(
            state_key="s1",
            action="a1",
            reward=0.3,
            next_state_key="s2",
            done=False,
            encoder_version="state-encoder-v1",
            action_space_version="action-space-v1",
            reward_version="reward-v1",
            next_legal_actions=["a1", "a2"],
            task_id="task-123",
        )
        d = t.to_dict()
        t2 = Transition.from_dict(d)
        assert t2.encoder_version == "state-encoder-v1"
        assert t2.action_space_version == "action-space-v1"
        assert t2.reward_version == "reward-v1"
        assert t2.next_legal_actions == ["a1", "a2"]
        assert t2.task_id == "task-123"
