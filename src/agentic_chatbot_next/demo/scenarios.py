from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple


_DEFAULT_ERROR_PHRASES: Tuple[str, ...] = (
    "encountered an error",
    "unexpected internal error",
    "demo prompt failed",
    "i hit an unexpected internal error",
    "i wasn't able to produce an answer",
)


@dataclass(frozen=True)
class DemoScenarioChecks:
    expected_citations_min: Optional[int] = None
    expected_keywords: Tuple[str, ...] = ()
    fail_on_error_phrases: Tuple[str, ...] = _DEFAULT_ERROR_PHRASES


@dataclass(frozen=True)
class DemoTurn:
    prompt: str
    force_agent: Optional[bool] = None
    expected_citations_min: Optional[int] = None
    expected_keywords: Tuple[str, ...] = ()
    notes: str = ""


@dataclass(frozen=True)
class DemoScenario:
    id: str
    title: str
    goal: str
    difficulty: str
    tool_focus: Tuple[str, ...]
    turns: Tuple[DemoTurn, ...]
    checks: DemoScenarioChecks
    notes: str = ""


@dataclass(frozen=True)
class DemoVerificationResult:
    status: str  # PASS | WARN | FAIL
    messages: Tuple[str, ...]


_DEFAULT_SCENARIOS: Dict[str, DemoScenario] = {
    "router_and_basic": DemoScenario(
        id="router_and_basic",
        title="Router and Basic Chat",
        goal="Show BASIC route behavior for low-complexity questions.",
        difficulty="easy",
        tool_focus=("router", "basic_chat"),
        turns=(
            DemoTurn(prompt="What is fan-out vs fan-in in agentic systems?"),
            DemoTurn(prompt="Give a concise explanation of retrieval-augmented generation."),
        ),
        checks=DemoScenarioChecks(),
        notes="Useful as a quick warm-up before tool-heavy scenarios.",
    ),
    "utility_and_memory": DemoScenario(
        id="utility_and_memory",
        title="Utility and Memory",
        goal="Exercise calculator and persistent memory tools.",
        difficulty="easy",
        tool_focus=("calculator", "memory_save", "memory_load", "memory_list"),
        turns=(
            DemoTurn(prompt="What is 18% of 2400?", expected_keywords=("432",)),
            DemoTurn(prompt="Remember that preferred_jurisdiction is England and Wales."),
            DemoTurn(prompt="What value is saved under preferred_jurisdiction?", expected_keywords=("England and Wales",)),
        ),
        checks=DemoScenarioChecks(),
        notes="Demonstrates cross-turn statefulness.",
    ),
}


def _as_str_list(value: Any) -> Tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    out: List[str] = []
    for item in value:
        text = str(item).strip()
        if text:
            out.append(text)
    return tuple(out)


def _parse_turn(raw: Any) -> Optional[DemoTurn]:
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return None
        return DemoTurn(prompt=text)

    if not isinstance(raw, Mapping):
        return None

    prompt = str(raw.get("prompt", "")).strip()
    if not prompt:
        return None

    force_agent_raw = raw.get("force_agent")
    force_agent: Optional[bool] = None
    if isinstance(force_agent_raw, bool):
        force_agent = force_agent_raw

    citations_raw = raw.get("expected_citations_min")
    expected_citations_min: Optional[int] = None
    if isinstance(citations_raw, int):
        expected_citations_min = max(0, citations_raw)

    return DemoTurn(
        prompt=prompt,
        force_agent=force_agent,
        expected_citations_min=expected_citations_min,
        expected_keywords=_as_str_list(raw.get("expected_keywords", [])),
        notes=str(raw.get("notes", "")).strip(),
    )


def _parse_checks(raw: Any) -> DemoScenarioChecks:
    if not isinstance(raw, Mapping):
        return DemoScenarioChecks()

    citations_raw = raw.get("expected_citations_min")
    expected_citations_min: Optional[int] = None
    if isinstance(citations_raw, int):
        expected_citations_min = max(0, citations_raw)

    fail_phrases = _as_str_list(raw.get("fail_on_error_phrases", [])) or _DEFAULT_ERROR_PHRASES

    return DemoScenarioChecks(
        expected_citations_min=expected_citations_min,
        expected_keywords=_as_str_list(raw.get("expected_keywords", [])),
        fail_on_error_phrases=fail_phrases,
    )


def _parse_scenario(raw: Any) -> Optional[DemoScenario]:
    if not isinstance(raw, Mapping):
        return None

    scenario_id = str(raw.get("id", "")).strip()
    title = str(raw.get("title", "")).strip()
    goal = str(raw.get("goal", "")).strip()
    difficulty = str(raw.get("difficulty", "")).strip().lower() or "medium"

    if not scenario_id:
        return None

    turns_raw = raw.get("turns", [])
    if not isinstance(turns_raw, list):
        return None

    turns: List[DemoTurn] = []
    for row in turns_raw:
        parsed = _parse_turn(row)
        if parsed is not None:
            turns.append(parsed)

    if not turns:
        return None

    return DemoScenario(
        id=scenario_id,
        title=title or scenario_id,
        goal=goal or "Run a curated scenario.",
        difficulty=difficulty,
        tool_focus=_as_str_list(raw.get("tool_focus", [])),
        turns=tuple(turns),
        checks=_parse_checks(raw.get("checks", {})),
        notes=str(raw.get("notes", "")).strip(),
    )


def _from_v1_dict(raw: Mapping[str, Any]) -> Dict[str, DemoScenario]:
    out: Dict[str, DemoScenario] = {}
    for name, prompts in raw.items():
        if not isinstance(name, str) or not isinstance(prompts, list):
            continue

        turns = tuple(
            DemoTurn(prompt=p.strip())
            for p in (str(x) for x in prompts)
            if p.strip()
        )
        if not turns:
            continue

        out[name] = DemoScenario(
            id=name,
            title=name.replace("_", " ").title(),
            goal="Legacy prompt-list scenario.",
            difficulty="medium",
            tool_focus=(),
            turns=turns,
            checks=DemoScenarioChecks(),
            notes="Loaded through v1 compatibility mode.",
        )
    return out


def parse_demo_scenarios(raw: Any) -> Dict[str, DemoScenario]:
    """Parse scenario JSON payload with v2 canonical + v1 compatibility.

    Supported shapes:
    - v2 canonical: {"version": "v2", "scenarios": [ ... ]}
    - v2 relaxed:   [ ...scenario objects... ]
    - v1 legacy:    {"scenario_name": ["prompt1", "prompt2"]}
    """

    out: Dict[str, DemoScenario] = {}

    # v2 canonical object
    if isinstance(raw, Mapping) and isinstance(raw.get("scenarios"), list):
        for item in raw["scenarios"]:
            parsed = _parse_scenario(item)
            if parsed is not None:
                out[parsed.id] = parsed
        return out

    # v2 relaxed list
    if isinstance(raw, list):
        for item in raw:
            parsed = _parse_scenario(item)
            if parsed is not None:
                out[parsed.id] = parsed
        return out

    # v1 legacy
    if isinstance(raw, Mapping):
        return _from_v1_dict(raw)

    return {}


def load_demo_scenarios(data_dir: Path) -> Dict[str, DemoScenario]:
    path = data_dir / "demo" / "demo_scenarios.json"
    if not path.exists():
        return dict(_DEFAULT_SCENARIOS)

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return dict(_DEFAULT_SCENARIOS)

    parsed = parse_demo_scenarios(raw)
    return parsed or dict(_DEFAULT_SCENARIOS)


def render_scenario_summary(scenario: DemoScenario) -> str:
    tools = ", ".join(scenario.tool_focus) if scenario.tool_focus else "(not specified)"
    return (
        f"{scenario.id}: {scenario.title}\n"
        f"  Difficulty: {scenario.difficulty}\n"
        f"  Tool focus: {tools}\n"
        f"  Goal: {scenario.goal}\n"
        f"  Turns: {len(scenario.turns)}"
    )


def _count_citations(text: str) -> int:
    # Count bulletized citations emitted by renderer:
    #   - [CITATION_ID] title (location)
    line_hits = re.findall(r"(?m)^\s*-\s+\[[^\]]+\]", text)
    if line_hits:
        return len(line_hits)

    # Fallback: inline citation ids such as (DOC_abc#chunk0004)
    inline_hits = re.findall(r"\([A-Za-z0-9_\-]+#chunk\d+\)", text)
    return len(inline_hits)


def _normalize_text(text: str) -> str:
    lowered = text.lower()
    cleaned = re.sub(r"[^a-z0-9\s]", " ", lowered)
    return re.sub(r"\s+", " ", cleaned).strip()


def _parse_number_token(value: str) -> Optional[float]:
    cleaned = value.strip().replace("$", "").replace(",", "")
    if not cleaned:
        return None
    if not re.fullmatch(r"-?\d+(?:\.\d+)?", cleaned):
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def _extract_numbers(text: str) -> Tuple[float, ...]:
    out: List[float] = []
    for token in re.findall(r"-?\$?\d[\d,]*(?:\.\d+)?", text):
        parsed = _parse_number_token(token)
        if parsed is not None:
            out.append(parsed)
    return tuple(out)


def _keyword_present(
    keyword: str,
    *,
    lowered_text: str,
    normalized_text: str,
    numbers: Tuple[float, ...],
) -> bool:
    raw = keyword.strip()
    if not raw:
        return True

    numeric = _parse_number_token(raw)
    if numeric is not None:
        return any(abs(n - numeric) < 1e-6 for n in numbers)

    normalized_keyword = _normalize_text(raw)
    if raw.lower() in lowered_text:
        return True
    if normalized_keyword and normalized_keyword in normalized_text:
        return True
    return False


def evaluate_response(
    response_text: str,
    *,
    scenario: DemoScenario,
    turn: DemoTurn,
) -> DemoVerificationResult:
    messages: List[str] = []
    status = "PASS"

    text = (response_text or "").strip()
    if not text:
        return DemoVerificationResult(status="FAIL", messages=("Empty response text.",))

    error_phrases = scenario.checks.fail_on_error_phrases
    lowered = text.lower()
    for phrase in error_phrases:
        if phrase.lower() in lowered:
            return DemoVerificationResult(
                status="FAIL",
                messages=(f"Hard failure phrase detected: '{phrase}'.",),
            )

    citations_target = turn.expected_citations_min
    if citations_target is None:
        citations_target = scenario.checks.expected_citations_min

    if citations_target is not None:
        citation_count = _count_citations(text)
        if citation_count < citations_target:
            status = "WARN"
            messages.append(
                f"Expected >= {citations_target} citations, found {citation_count}."
            )
        else:
            messages.append(
                f"Citation check passed ({citation_count} >= {citations_target})."
            )

    keyword_source: Iterable[str]
    if turn.expected_keywords:
        keyword_source = turn.expected_keywords
    else:
        keyword_source = scenario.checks.expected_keywords

    expected_keywords: List[str] = []
    seen = set()
    for keyword in keyword_source:
        if keyword not in seen:
            expected_keywords.append(keyword)
            seen.add(keyword)

    normalized = _normalize_text(text)
    numbers = _extract_numbers(text)
    missing = [
        keyword
        for keyword in expected_keywords
        if not _keyword_present(
            keyword,
            lowered_text=lowered,
            normalized_text=normalized,
            numbers=numbers,
        )
    ]
    if missing:
        status = "WARN" if status == "PASS" else status
        messages.append("Missing expected keywords: " + ", ".join(missing))

    if not messages:
        messages.append("Heuristic checks passed.")

    return DemoVerificationResult(status=status, messages=tuple(messages))
