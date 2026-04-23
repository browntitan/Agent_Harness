from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, List

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator


def default_router_patterns_path() -> Path:
    return Path(__file__).resolve().parents[3] / "data" / "router" / "intent_patterns.json"


def normalize_router_text(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", str(value or ""))
    stripped = "".join(char for char in decomposed if not unicodedata.combining(char))
    lowered = stripped.casefold()
    collapsed = re.sub(r"\s+", " ", lowered).strip()
    return collapsed


class IntentPatternGroupModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    phrases: List[str] = Field(default_factory=list)
    regexes: List[str] = Field(default_factory=list)

    @field_validator("phrases", "regexes", mode="before")
    @classmethod
    def _require_list(cls, value: Any) -> List[str]:
        if value is None:
            return []
        if not isinstance(value, list):
            raise TypeError("must be a JSON array of strings")
        return value

    @field_validator("phrases", "regexes")
    @classmethod
    def _normalise_items(cls, value: List[Any]) -> List[str]:
        cleaned: List[str] = []
        for item in value:
            text = str(item or "").strip()
            if not text:
                raise ValueError("must not contain empty values")
            cleaned.append(text)
        return cleaned


class RouterPatternsModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tool_or_multistep_intent: IntentPatternGroupModel
    data_analysis_intent: IntentPatternGroupModel
    citation_grounding_intent: IntentPatternGroupModel
    high_stakes_intent: IntentPatternGroupModel
    kb_inventory_intent: IntentPatternGroupModel
    coordinator_campaign_intent: IntentPatternGroupModel
    rag_grounding_intent: IntentPatternGroupModel


def _format_router_validation_error(path: Path, error: ValidationError) -> ValueError:
    lines = [f"Router pattern config {path} is invalid:"]
    for item in error.errors():
        location = ".".join(str(part) for part in item.get("loc", ())) or "config"
        message = str(item.get("msg") or "invalid value")
        lines.append(f"- {location}: {message}")
    return ValueError("\n".join(lines))


@dataclass(frozen=True)
class CompiledIntentPatternGroup:
    phrases: tuple[str, ...]
    phrase_patterns: tuple[re.Pattern[str], ...]
    regexes: tuple[re.Pattern[str], ...]

    @classmethod
    def from_model(cls, model: IntentPatternGroupModel) -> "CompiledIntentPatternGroup":
        normalized_phrases = tuple(normalize_router_text(item) for item in model.phrases if normalize_router_text(item))
        return cls(
            phrases=normalized_phrases,
            phrase_patterns=tuple(_compile_phrase_boundary_pattern(item) for item in normalized_phrases),
            regexes=tuple(re.compile(item, flags=re.IGNORECASE) for item in model.regexes),
        )

    def matches(self, text: str, normalized_text: str | None = None) -> bool:
        normalized = normalized_text if normalized_text is not None else normalize_router_text(text)
        if any(pattern.search(normalized) for pattern in self.phrase_patterns):
            return True
        return any(pattern.search(text) for pattern in self.regexes)


def _compile_phrase_boundary_pattern(phrase: str) -> re.Pattern[str]:
    escaped = re.escape(phrase)
    return re.compile(rf"(?<!\w){escaped}(?!\w)")


@dataclass(frozen=True)
class CompiledRouterPatterns:
    tool_or_multistep_intent: CompiledIntentPatternGroup
    data_analysis_intent: CompiledIntentPatternGroup
    citation_grounding_intent: CompiledIntentPatternGroup
    high_stakes_intent: CompiledIntentPatternGroup
    kb_inventory_intent: CompiledIntentPatternGroup
    coordinator_campaign_intent: CompiledIntentPatternGroup
    rag_grounding_intent: CompiledIntentPatternGroup

    @classmethod
    def from_model(cls, model: RouterPatternsModel) -> "CompiledRouterPatterns":
        return cls(
            tool_or_multistep_intent=CompiledIntentPatternGroup.from_model(model.tool_or_multistep_intent),
            data_analysis_intent=CompiledIntentPatternGroup.from_model(model.data_analysis_intent),
            citation_grounding_intent=CompiledIntentPatternGroup.from_model(model.citation_grounding_intent),
            high_stakes_intent=CompiledIntentPatternGroup.from_model(model.high_stakes_intent),
            kb_inventory_intent=CompiledIntentPatternGroup.from_model(model.kb_inventory_intent),
            coordinator_campaign_intent=CompiledIntentPatternGroup.from_model(model.coordinator_campaign_intent),
            rag_grounding_intent=CompiledIntentPatternGroup.from_model(model.rag_grounding_intent),
        )


def _coerce_path(path: str | Path | None) -> Path:
    if path is None or str(path).strip() == "":
        return default_router_patterns_path().resolve()
    return Path(path).expanduser().resolve()


@lru_cache(maxsize=16)
def load_router_patterns(path: str | Path | None = None) -> CompiledRouterPatterns:
    resolved = _coerce_path(path)
    raw = json.loads(resolved.read_text(encoding="utf-8"))
    try:
        model = RouterPatternsModel.model_validate(raw)
    except ValidationError as exc:
        raise _format_router_validation_error(resolved, exc) from exc
    return CompiledRouterPatterns.from_model(model)


def clear_router_patterns_cache() -> None:
    load_router_patterns.cache_clear()


def patterns_path_from_settings(settings: Any | None) -> Path:
    raw = getattr(settings, "router_patterns_path", None) if settings is not None else None
    return _coerce_path(raw)


__all__ = [
    "CompiledIntentPatternGroup",
    "CompiledRouterPatterns",
    "IntentPatternGroupModel",
    "RouterPatternsModel",
    "clear_router_patterns_cache",
    "default_router_patterns_path",
    "load_router_patterns",
    "normalize_router_text",
    "patterns_path_from_settings",
]
