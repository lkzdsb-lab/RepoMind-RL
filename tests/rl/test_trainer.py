"""Tests for QLearningTrainer with next_legal_actions constraint."""

from agent_runtime.rl.action_space import ActionSpace
from agent_runtime.rl.trainer import QLearningTrainer
from model.agent.transition import Transition


class TestTrainerNextMax:
    def test_uses_next_legal_actions(self):
        """next_max should only consider next_legal_actions, not all Q-table entries."""
        q_table = {
            "state_next": {"action_a": 0.9, "action_b": 0.1, "secret_action": 100.0},
        }
        trainer = QLearningTrainer(q_table, ActionSpace())
        transition = Transition(
            state_key="state_prev",
            action="action_x",
            reward=0.5,
            next_state_key="state_next",
            done=False,
            next_legal_actions=["action_a", "action_b"],  # secret_action NOT legal
        )
        updated = trainer.update(transition)
        # next_max = max(0.9, 0.1) = 0.9 (NOT 100.0)
        # target = 0.5 + 0.9 * 0.9 = 0.5 + 0.81 = 1.31
        # current = 0.0
        # updated = 0.0 + 0.2 * (1.31 - 0.0) = 0.262
        expected_target = 0.5 + 0.9 * 0.9  # = 1.31
        expected = 0.0 + 0.2 * (expected_target - 0.0)
        assert updated == expected
        assert q_table["state_prev"]["action_x"] == expected

    def test_next_max_zero_when_no_legal_actions(self):
        """When next_legal_actions is empty, next_max should be 0."""
        q_table = {
            "state_next": {"any_action": 50.0},
        }
        trainer = QLearningTrainer(q_table, ActionSpace())
        transition = Transition(
            state_key="state_prev",
            action="action_x",
            reward=0.5,
            next_state_key="state_next",
            done=False,
            next_legal_actions=[],  # empty
        )
        updated = trainer.update(transition)
        # next_max = 0
        # target = 0.5 + 0.9 * 0 = 0.5
        # current = 0.0
        # updated = 0.0 + 0.2 * (0.5 - 0.0) = 0.1
        assert updated == 0.1

    def test_next_max_zero_when_done(self):
        """When done=True, next_max should be 0 regardless of legal actions."""
        q_table = {
            "state_next": {"action_a": 100.0},
        }
        trainer = QLearningTrainer(q_table, ActionSpace())
        transition = Transition(
            state_key="state_prev",
            action="action_x",
            reward=1.0,
            next_state_key="state_next",
            done=True,
            next_legal_actions=["action_a"],
        )
        updated = trainer.update(transition)
        # next_max = 0 (because done=True)
        # target = 1.0 + 0.9 * 0 = 1.0
        # current = 0.0
        # updated = 0.0 + 0.2 * (1.0 - 0.0) = 0.2
        assert updated == 0.2

    def test_ignores_actions_not_in_legal_list(self):
        """Actions that exist in Q-table but NOT in next_legal_actions MUST be ignored."""
        q_table = {
            "state_next": {"legal_a": 0.3, "illegal": 999.0, "legal_b": 0.7},
        }
        trainer = QLearningTrainer(q_table, ActionSpace())
        transition = Transition(
            state_key="state_prev",
            action="action_x",
            reward=0.0,
            next_state_key="state_next",
            done=False,
            next_legal_actions=["legal_a", "legal_b"],
        )
        updated = trainer.update(transition)
        # next_max = max(0.3, 0.7) = 0.7 (NOT 999.0)
        # target = 0.0 + 0.9 * 0.7 = 0.63
        # updated = 0.0 + 0.2 * (0.63 - 0.0) = 0.126
        assert updated == 0.126


class TestTrainerQValue:
    def test_q_value_default_zero(self):
        trainer = QLearningTrainer({}, ActionSpace())
        assert trainer.q_value("unknown_state", "unknown_action") == 0.0

    def test_q_value_retrieval(self):
        q_table = {"s1": {"a1": 0.5, "a2": -0.3}}
        trainer = QLearningTrainer(q_table, ActionSpace())
        assert trainer.q_value("s1", "a1") == 0.5
        assert trainer.q_value("s1", "a2") == -0.3
        assert trainer.q_value("s1", "a3") == 0.0


class TestTrainerBatch:
    def test_train_batch(self):
        q_table = {}
        trainer = QLearningTrainer(q_table, ActionSpace())
        transitions = [
            Transition(
                state_key="s1",
                action="a1",
                reward=1.0,
                next_state_key="s2",
                done=False,
                next_legal_actions=["a1", "a2"],
            ),
            Transition(
                state_key="s2",
                action="a2",
                reward=-0.5,
                next_state_key="s3",
                done=True,
                next_legal_actions=[],
            ),
        ]
        count = trainer.train_batch(transitions)
        assert count == 2
        assert "s1" in q_table
        assert "s2" in q_table
