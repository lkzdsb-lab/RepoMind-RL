"""Search query planning for codebase context lookup."""

from __future__ import annotations

import re
from typing import Any

from model.agent.graph import AgentState
from model.agent.search import SearchQueryPlan
from config import SearchQueryConfig

config = SearchQueryConfig()


"""
    llm 决策时逻辑，提取 keyword，选择需要的信息
"""
class SearchQueryPlanner:
    def __init__(self, default_query: str = "TODO", max_terms: int = 8) -> None:
        self.default_query = default_query
        self.max_terms = max_terms

    # 分桶为相关信息打分
    def plan(self, state: AgentState) -> SearchQueryPlan:
        weighted: dict[str, float] = {}
        buckets: dict[str, list[str]] = {
            "identifiers": [],
            "domain_terms": [],
            "code_terms": [],
            "memory_terms": [],
            "skill_terms": [],
        }

        self._add_text(
            weighted,
            buckets,
            "domain_terms",
            f"{state.get('title', '')} {state.get('description', '')}",
            weight=3.0,
        )
        analysis = state.get("task_analysis") or {}
        if isinstance(analysis, dict):
            self._add_text(
                weighted,
                buckets,
                "domain_terms",
                " ".join(
                    [
                        str(analysis.get("task_category", "")),
                        " ".join(str(item) for item in _list_values(analysis.get("entities"))[:12]),
                        " ".join(str(item) for item in _list_values(analysis.get("search_hints"))[:12]),
                    ]
                ),
                weight=2.5,
            )
        self._add_text(
            weighted,
            buckets,
            "memory_terms",
            str(state.get("memory_context", ""))[:3000],
            weight=0.8,
        )
        self._add_text(
            weighted,
            buckets,
            "skill_terms",
            " ".join(state.get("selected_skills", [])),
            weight=1.5,
        )
        for skill in state.get("skill_context", [])[:3]:
            self._add_text(weighted, buckets, "skill_terms", str(skill)[:1000], weight=1.0)

        self._add_code_context(weighted, buckets, state.get("code_context", {}))

        identifiers = [
            term
            for term in weighted
            if _looks_like_identifier(term)
        ]
        for identifier in identifiers:
            if identifier not in buckets["identifiers"]:
                buckets["identifiers"].append(identifier)
            weighted[identifier] += 1.5

        ranked = sorted(
            weighted,
            key=lambda term: (weighted[term], _term_quality(term), len(term)),
            reverse=True,
        )
        terms = ranked[: self.max_terms]
        query = " ".join(terms) if terms else self.default_query
        return SearchQueryPlan(
            query=query,
            terms=terms,
            identifiers=_ordered_subset(buckets["identifiers"], terms),
            domain_terms=_ordered_subset(buckets["domain_terms"], terms),
            code_terms=_ordered_subset(buckets["code_terms"], terms),
            memory_terms=_ordered_subset(buckets["memory_terms"], terms),
            skill_terms=_ordered_subset(buckets["skill_terms"], terms),
        )

    # 提取有效代码访问记录并赋权重
    def _add_code_context(
        self,
        weighted: dict[str, float],
        buckets: dict[str, list[str]],
        code_context: dict[str, Any],
    ) -> None:
        if not isinstance(code_context, dict):
            return
        for key in ("functions", "symbols", "db_models", "api_routes", "files"):
            for item in code_context.get(key, [])[:10]:
                if not isinstance(item, dict):
                    continue
                text = " ".join(
                    str(item.get(field, ""))
                    for field in (
                        "name",
                        "full_name",
                        "path",
                        "file_path",
                        "table",
                        "handler",
                        "method",
                    )
                )
                if item.get("path"):
                    text += " " + " ".join(str(item["path"]).split("/"))
                if item.get("file_path"):
                    text += " " + " ".join(str(item["file_path"]).split("/"))
                self._add_text(weighted, buckets, "code_terms", text, weight=2.0)
                if item.get("path") and key == "api_routes":
                    self._add_text(weighted, buckets, "code_terms", str(item["path"]), weight=2.5)

    # 通过便利 text 将 term 打分并分类
    def _add_text(
        self,
        weighted: dict[str, float],
        buckets: dict[str, list[str]],
        bucket: str,
        text: str,
        weight: float,
    ) -> None:
        for token in _extract_terms(text):
            if token in config.STOP_WORDS:
                continue
            weighted[token] = weighted.get(token, 0.0) + weight + _term_quality(token)
            if token not in buckets[bucket]:
                buckets[bucket].append(token)

# 核心分词器，支持中英文，存储有用的小写词汇集合
def _extract_terms(text: str) -> list[str]:
    terms: list[str] = []
    for raw in re.findall(r"[A-Za-z0-9_./:-]+|[\u4e00-\u9fff]+", text):
        token = raw.strip(".,:;()[]{}<>`'\"")
        if not token:
            continue
        for part in _split_token(token):
            if _valid_term(part) and part not in terms:
                terms.append(part)
    return terms

# 将 str 通过符号分隔开，存入 token 列表
def _split_token(token: str) -> list[str]:
    pieces = [token]
    if "/" in token:
        pieces.extend(part for part in token.split("/") if part)
    if "_" in token or "-" in token or "." in token or ":" in token:
        pieces.extend(part for part in re.split(r"[_\-.:/]+", token) if part)
    camel_parts = re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?=[A-Z]|$)|\d+", token)
    if len(camel_parts) > 1:
        pieces.extend(part.lower() for part in camel_parts)
    return [_normalize(piece) for piece in pieces]


def _normalize(token: str) -> str:
    return token.strip().lower()

# 去除无用词汇
def _valid_term(token: str) -> bool:
    if not token or token in config.STOP_WORDS:
        return False
    if any(fragment in token for fragment in config.CHINESE_STOP_FRAGMENTS):
        return False
    if token.isdigit():
        return False
    if len(token) < 2 and not re.match(r"[\u4e00-\u9fff]", token):
        return False
    return True

# 关键词识别 eg. user_id 这种一定是关键字段
def _looks_like_identifier(term: str) -> bool:
    return (
        "_" in term
        or "." in term
        or "/" in term
        or bool(re.search(r"[a-z]+\d+|\d+[a-z]+", term))
    )

# 为每个 term 打分
def _term_quality(term: str) -> float:
    score = 0.0
    if len(term) >= 4:
        score += 0.15
    if re.search(r"(handler|service|repo|repository|model|controller|route|table)$", term):
        score += 0.2
    if re.search(r"(id|status|state|order|user|payment|paid|pending|callback)", term):
        score += 0.25
    if re.match(r"[\u4e00-\u9fff]+$", term):
        score += 0.1
    return score


def _ordered_subset(values: list[str], selected: list[str]) -> list[str]:
    selected_set = set(selected)
    return [value for value in values if value in selected_set]


def _list_values(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if value in (None, ""):
        return []
    return [value]
