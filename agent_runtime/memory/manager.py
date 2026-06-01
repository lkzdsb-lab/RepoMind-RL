"""Layered memory orchestration.

Short-term memory lives inside AgentState, mid-term memory can be backed by
Redis or JSONL, and long-term memory is stored through a vector-store boundary.
Reward-gated promotion decides which memories move upward and when they should
be consolidated into skill material.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent_runtime.memory.cards import MemoryCard, MemoryContextPack, MemorySearchResult, utc_now
from agent_runtime.memory.store import (
    JsonlMemoryStore,
    LocalVectorMemoryStore,
    MemoryStore,
    RedisMemoryStore,
)
from agent_runtime.registry import RegistrySnapshot
from model.agent.graph import AgentState
from config import DebugAgentConfig
from loguru import logger


@dataclass
class MemoryPromotionPolicy:
    semantic_threshold: float = 0.7
    procedural_threshold: float = 1.2
    skill_threshold: float = 1.5


@dataclass
class MemoryWriteResult:
    written: list[MemoryCard]
    promoted: list[MemoryCard]
    consolidated: list[dict[str, Any]]
    feedback: list[dict[str, Any]] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "written": [card.to_dict() for card in self.written],
            "promoted": [card.to_dict() for card in self.promoted],
            "consolidated": self.consolidated,
            "feedback": self.feedback or [],
        }

"""
    记忆编排层
"""
class LayeredMemoryManager:
    def __init__(
        self,
        mid_store: MemoryStore,
        long_store: MemoryStore,
        skill_dir: str | Path,
        policy: MemoryPromotionPolicy | None = None,
    ) -> None:
        self.mid_store = mid_store
        self.long_store = long_store
        self.skill_dir = Path(skill_dir)
        self.policy = policy or MemoryPromotionPolicy()

    # 配置记忆存储
    @classmethod
    def from_config(cls, config: DebugAgentConfig) -> "LayeredMemoryManager":
        repo_path = Path(config.repo_path)
        policy = MemoryPromotionPolicy(
            semantic_threshold=config.semantic_promotion_threshold,
            procedural_threshold=config.procedural_promotion_threshold,
            skill_threshold=config.skill_consolidation_threshold,
        )
        mid_store: MemoryStore
        if config.memory_redis_url:
            mid_store = RedisMemoryStore(config.memory_redis_url)
        else:
            mid_store = JsonlMemoryStore(repo_path / config.mid_memory_path)
        logger.info(
            "memory manager configured mid_store={} long_store={} skill_dir={}",
            mid_store.__class__.__name__,
            LocalVectorMemoryStore.__name__,
            repo_path / config.skill_memory_dir,
        )
        return cls(
            mid_store=mid_store,
            long_store=LocalVectorMemoryStore(repo_path / config.long_memory_path),
            skill_dir=repo_path / config.skill_memory_dir,
            policy=policy,
        )

    def retrieve(
        self,
        query: str,
        state: AgentState,
        registry: RegistrySnapshot,
        limit: int = 5,
    ) -> MemoryContextPack:
        """
        description: 根据 query 结合当前 state 获取记忆
        """
        pack = MemoryContextPack(
            short_term=self._search_short_term(query, state, limit=limit),
            mid_term=self.mid_store.search_cards(query, limit=limit),
            long_term=self.long_store.search_cards(query, limit=limit),
            skill=self._retrieve_skill_memory(query, state, registry, limit=limit),
        )
        self._touch_retrieved(pack, state)
        logger.bind(task_id=state.get("task_id")).info(
            "layered memory retrieve query_chars={} short={} mid={} long={} skill={}",
            len(query),
            len(pack.short_term),
            len(pack.mid_term),
            len(pack.long_term),
            len(pack.skill),
        )
        return pack

    def record_task_memory(
        self,
        state: AgentState,
        registry: RegistrySnapshot,
    ) -> MemoryWriteResult:
        base = self._build_task_card(state)
        written = [self.mid_store.append_card(base)]
        feedback = self.record_reuse_feedback(state)
        promoted = self._promote(base, state)
        persisted_promotions = [self.long_store.append_card(card) for card in promoted]
        consolidated = self._consolidate_to_skills(persisted_promotions, state, registry)
        logger.bind(task_id=state.get("task_id")).info(
            "task memory recorded written={} promoted={} consolidated={} feedback={} reward={:.2f}",
            len(written),
            len(persisted_promotions),
            len(consolidated),
            len(feedback),
            base.reward_credit,
        )
        return MemoryWriteResult(
            written=written,
            promoted=persisted_promotions,
            consolidated=consolidated,
            feedback=feedback,
        )

    def record_reuse_feedback(self, state: AgentState) -> list[dict[str, Any]]:
        success = self._task_succeeded(state)
        feedback: list[dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()
        memories = state.get("retrieved_memories") or {}
        if not isinstance(memories, dict):
            return feedback

        for tier in ("mid_term", "long_term"):
            values = memories.get(tier, [])
            if not isinstance(values, list):
                continue
            store = self.mid_store if tier == "mid_term" else self.long_store
            for item in values:
                if not isinstance(item, dict):
                    continue
                memory_id = str(item.get("memory_id", ""))
                if not memory_id or (tier, memory_id) in seen:
                    continue
                seen.add((tier, memory_id))
                updated = store.record_reuse_feedback(memory_id, success=success)
                if updated is None:
                    continue
                feedback.append(
                    {
                        "memory_id": memory_id,
                        "tier": tier,
                        "success": success,
                        "reuse_success": updated.reuse_success,
                        "reuse_failure": updated.reuse_failure,
                        "conflict_score": updated.conflict_score,
                    }
                )
        return feedback

    def add_short_term(
        self,
        state: AgentState,
        trigger: str,
        content: str,
        memory_type: str = "episodic",
        tags: list[str] | None = None,
    ) -> AgentState:
        card = MemoryCard(
            type=memory_type,  # type: ignore[arg-type]
            tier="short_term",
            scope=state.get("repo_path", ""),
            trigger=trigger,
            content=content,
            source_task_id=state.get("task_id"),
            tags=tags or [],
            status="draft",
        )
        short_term = state.get("short_term_memories", []) + [card.to_dict()]
        logger.bind(task_id=state.get("task_id")).debug(
            "short term memory added trigger={} type={} tags={}",
            trigger,
            memory_type,
            tags or [],
        )
        return {**state, "short_term_memories": short_term}

    def _search_short_term(
        self,
        query: str,
        state: AgentState,
        limit: int,
    ) -> list[MemorySearchResult]:
        query_terms = set(_tokens(query))
        results: list[MemorySearchResult] = []
        for item in state.get("short_term_memories", []):
            card = MemoryCard.from_dict(item)
            score = _lexical_score(query_terms, card)
            if score > 0:
                results.append(MemorySearchResult(card=card, score=score, source="short_term"))
        results.sort(key=lambda item: item.score, reverse=True)
        return results[:limit]

    def _retrieve_skill_memory(
        self,
        query: str,
        state: AgentState,
        registry: RegistrySnapshot,
        limit: int,
    ) -> list[MemorySearchResult]:
        query_terms = set(_tokens(query))
        selected = set(state.get("selected_skills", []))
        results: list[MemorySearchResult] = []

        for skill in registry.skills.values():
            skill_terms = set(_tokens(" ".join([skill.name, skill.description, " ".join(skill.triggers)])))
            if selected and skill.name not in selected:
                continue
            if not selected and not query_terms.intersection(skill_terms):
                continue

            content_parts = [skill.description]
            for resource in skill.resources:
                content_parts.append(_read_resource_excerpt(resource, Path(state.get("repo_path", ""))))

            card = MemoryCard(
                type="procedural",
                tier="skill",
                scope=state.get("repo_path", ""),
                trigger=skill.name,
                content="\n".join(part for part in content_parts if part).strip(),
                tags=skill.triggers,
                skill_name=skill.name,
                status="verified",
            )
            score = _lexical_score(query_terms or skill_terms, card)
            if score > 0:
                results.append(MemorySearchResult(card=card, score=score, source="skill"))

        results.sort(key=lambda item: item.score, reverse=True)
        return results[:limit]

    def _touch_retrieved(self, pack: MemoryContextPack, state: AgentState) -> None:
        """

        """
        used_at = utc_now()
        self._touch_short_term(pack.short_term, state, used_at)
        self._touch_store_results(self.mid_store, pack.mid_term, used_at)
        self._touch_store_results(self.long_store, pack.long_term, used_at)
        self._touch_ephemeral_results(pack.skill, used_at)

    def _touch_short_term(
        self,
        results: list[MemorySearchResult],
        state: AgentState,
        used_at: str,
    ) -> None:
        if not results:
            return

        result_by_id = {result.card.memory_id: result for result in results}
        updated_items: list[dict[str, Any]] = []
        for item in state.get("short_term_memories", []):
            card = MemoryCard.from_dict(item)
            result = result_by_id.get(card.memory_id)
            if result is not None:
                card = card.mark_retrieved(used_at)
                result.card = card
            updated_items.append(card.to_dict())
        state["short_term_memories"] = updated_items

    def _touch_store_results(
        self,
        store: MemoryStore,
        results: list[MemorySearchResult],
        used_at: str,
    ) -> None:
        for result in results:
            updated = store.touch_card(result.card.memory_id, used_at)
            if updated is None:
                logger.debug(
                    "retrieved memory touch missed store={} memory_id={}",
                    store.__class__.__name__,
                    result.card.memory_id,
                )
            result.card = updated or result.card.mark_retrieved(used_at)

    def _touch_ephemeral_results(
        self,
        results: list[MemorySearchResult],
        used_at: str,
    ) -> None:
        for result in results:
            result.card = result.card.mark_retrieved(used_at)

    def _build_task_card(self, state: AgentState) -> MemoryCard:
        latest_test = (state.get("test_results") or [{}])[-1]
        passed = self._task_succeeded(state)
        has_patch = bool(state.get("patch"))
        # 计算当前状态的 reward
        reward = self._reward_credit(state)
        memory_type = "episodic" if passed else "anti_pattern"
        status = "verified" if passed else "draft"
        evidence = []
        if latest_test:
            evidence.append(f"verify_command={latest_test.get('command', '')}")
            evidence.append(f"exit_code={latest_test.get('exit_code')}")
        if has_patch:
            evidence.append("git_diff_present=true")
        if state.get("error"):
            evidence.append(f"error={state.get('error')}")

        return MemoryCard(
            type=memory_type,
            tier="mid_term",
            scope=state.get("repo_path", ""),
            trigger=state.get("title", ""),
            content=self._task_memory_content(state),
            evidence=evidence,
            tags=self._task_tags(state),
            reward_credit=reward,
            status=status,
            source_task_id=state.get("task_id"),
            metadata={
                "description": state.get("description", ""),
                "candidate_files": state.get("candidate_files", []),
                "patch_summary": state.get("patch_summary"),
            },
        )

    # 计算需要持久化的 memory
    def _promote(self, base: MemoryCard, state: AgentState) -> list[MemoryCard]:
        score = base.promotion_score()
        promoted: list[MemoryCard] = []

        if base.type == "anti_pattern":
            logger.info(
                "promoting anti-pattern memory memory_id={} score={:.2f}",
                base.memory_id,
                score,
            )
            promoted.append(
                base.with_updates(
                    tier="long_term",
                    promoted_from=base.memory_id,
                    content="Avoid repeating this failing path. " + base.content,
                )
            )
            return promoted

        if score >= self.policy.semantic_threshold:
            logger.info(
                "promoting semantic memory memory_id={} score={:.2f} threshold={:.2f}",
                base.memory_id,
                score,
                self.policy.semantic_threshold,
            )
            promoted.append(
                base.with_updates(
                    type="semantic",
                    tier="long_term",
                    promoted_from=base.memory_id,
                    content=self._semantic_content(state),
                )
            )

        if score >= self.policy.procedural_threshold:
            logger.info(
                "promoting procedural memory memory_id={} score={:.2f} threshold={:.2f}",
                base.memory_id,
                score,
                self.policy.procedural_threshold,
            )
            promoted.append(
                base.with_updates(
                    type="procedural",
                    tier="long_term",
                    promoted_from=base.memory_id,
                    content=self._procedural_content(state),
                )
            )

        return promoted

    # 将有用的记忆 covert to skill
    def _consolidate_to_skills(
        self,
        cards: list[MemoryCard],
        state: AgentState,
        registry: RegistrySnapshot,
    ) -> list[dict[str, Any]]:
        consolidated: list[dict[str, Any]] = []
        for card in cards:
            if card.promotion_score() < self.policy.skill_threshold:
                continue
            skill_name = self._select_skill_name(card, state, registry)
            path = self._append_skill_memory(skill_name, card)
            logger.info(
                "memory consolidated to skill skill={} memory_id={} path={}",
                skill_name,
                card.memory_id,
                path,
            )
            consolidated.append(
                {
                    "skill": skill_name,
                    "memory_id": card.memory_id,
                    "path": path.as_posix(),
                }
            )
        return consolidated

    def _select_skill_name(
        self,
        card: MemoryCard,
        state: AgentState,
        registry: RegistrySnapshot,
    ) -> str:
        selected = state.get("selected_skills", [])
        if selected:
            return _slug(selected[0])

        card_terms = set(_tokens(" ".join([card.trigger, card.content, " ".join(card.tags)])))
        best_name = ""
        best_score = 0
        for skill in registry.skills.values():
            skill_terms = set(_tokens(" ".join([skill.name, skill.description, " ".join(skill.triggers)])))
            score = len(card_terms.intersection(skill_terms))
            if score > best_score:
                best_name = skill.name
                best_score = score

        if best_name:
            return _slug(best_name)
        if "go" in card_terms:
            return "go_bug_localization"
        return "general_debugging"

    # skill extend 操作
    def _append_skill_memory(self, skill_name: str, card: MemoryCard) -> Path:
        self.skill_dir.mkdir(parents=True, exist_ok=True)
        path = self.skill_dir / f"{_slug(skill_name)}.md"
        section = (
            "\n\n"
            f"## Consolidated Memory: {card.memory_id}\n"
            f"- Type: {card.type}\n"
            f"- Trigger: {card.trigger}\n"
            f"- Reward: {card.reward_credit:.2f}\n"
            f"- Evidence: {', '.join(card.evidence) or 'none'}\n\n"
            f"{card.content}\n"
        )
        with path.open("a", encoding="utf-8") as fp:
            fp.write(section)
        return path

    # 根据运行 test 命令来给
    def _reward_credit(self, state: AgentState) -> float:
        tests = state.get("test_results") or []
        latest_exit = tests[-1].get("exit_code") if tests else None
        reward = 0.0
        if latest_exit == 0:
            reward += 1.0
        elif latest_exit is not None:
            reward -= 0.2
        if state.get("patch"):
            reward += 0.35
        if state.get("error"):
            reward -= 0.4
        return reward

    def _task_succeeded(self, state: AgentState) -> bool:
        tests = state.get("test_results") or []
        latest_exit = tests[-1].get("exit_code") if tests else None
        return latest_exit == 0 and not state.get("error")

    def _task_memory_content(self, state: AgentState) -> str:
        candidates = ", ".join(state.get("candidate_files", [])[:5]) or "none"
        tests = state.get("test_results") or []
        latest = tests[-1] if tests else {}
        latest_exit = latest.get("exit_code", "not_run")
        tools = ", ".join(call.get("name", "unknown") for call in state.get("tool_calls", [])[-8:])
        selected_skills = ", ".join(state.get("selected_skills", [])[:5]) or "none"
        code_context = state.get("code_context") or {}
        context_summary = "none"
        if isinstance(code_context, dict):
            context_summary = (
                f"files={len(code_context.get('files', []))}, "
                f"functions={len(code_context.get('functions', []))}, "
                f"routes={len(code_context.get('api_routes', []))}, "
                f"db_models={len(code_context.get('db_models', []))}"
            )
        return (
            f"Task: {state.get('title', '')}\n"
            f"Description: {state.get('description', '') or 'none'}\n"
            f"Candidate files: {candidates}\n"
            f"Tools used: {tools or 'none'}\n"
            f"Selected skills: {selected_skills}\n"
            f"Code context summary: {context_summary}\n"
            f"Verify command: {latest.get('command', state.get('verify_command', 'not_run'))}\n"
            f"Latest test exit code: {latest_exit}\n"
            f"Patch summary: {state.get('patch_summary') or 'no patch'}\n"
            f"Error: {state.get('error') or 'none'}"
        )

    def _semantic_content(self, state: AgentState) -> str:
        candidates = ", ".join(state.get("candidate_files", [])[:5]) or "none"
        return (
            f"When debugging `{state.get('title', '')}`, relevant code tends to be in: "
            f"{candidates}. Verified outcome: {self._latest_exit(state)}."
        )

    def _procedural_content(self, state: AgentState) -> str:
        command = state.get("verify_command", "pytest")
        return (
            f"Procedure for similar tasks: search using task-specific keywords, read the "
            f"top candidate files, run `{command}`, inspect `git_diff`, and only promote "
            f"the memory when verification evidence is present."
        )

    def _latest_exit(self, state: AgentState) -> Any:
        tests = state.get("test_results") or []
        return tests[-1].get("exit_code") if tests else "not_run"

    def _task_tags(self, state: AgentState) -> list[str]:
        text = " ".join(
            [
                state.get("title", ""),
                state.get("description", ""),
                " ".join(state.get("candidate_files", [])),
            ]
        )
        return sorted(set(list(_tokens(text))[:12]))


def _tokens(text: str) -> list[str]:
    return re.findall(r"[\w-]+", text.lower())

def _lexical_score(query_terms: set[str], card: MemoryCard) -> float:
    """
    评估一个“记忆卡片（MemoryCard）”与当前“查询词（Query）”之间的匹配程度
    """
    memory_terms = set(_tokens(" ".join([card.trigger, card.content, " ".join(card.tags)])))
    if not query_terms or not memory_terms:
        return 0.0
    overlap = len(query_terms.intersection(memory_terms))
    return overlap / max(len(query_terms), 1) + min(card.promotion_score(), 2.0) * 0.1


def _read_resource_excerpt(resource: str, repo_path: Path, max_chars: int = 2000) -> str:
    path = Path(resource)
    if not path.is_absolute():
        path = repo_path / path
    if not path.exists() or not path.is_file():
        return ""
    return path.read_text(encoding="utf-8")[:max_chars]

# 正则匹配 文件名
def _slug(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "_", value.strip().lower()).strip("_")
    return slug or "general_debugging"
