from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List

from langchain_core.messages import HumanMessage, SystemMessage

from agentic_chatbot_next.providers import ProviderBundle, build_providers
from agentic_chatbot_next.utils.json_utils import extract_json

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class NlpBatchResult:
    rows: List[Dict[str, Any]]
    failed_batches: List[Dict[str, Any]]
    model_name: str


class DataAnalystNlpRunner:
    def __init__(self, settings: Any, providers: ProviderBundle | None) -> None:
        self.settings = settings
        self.base_providers = providers
        self._cached_providers: ProviderBundle | None = None

    def _provider_bundle(self) -> ProviderBundle:
        if self.base_providers is None:
            raise RuntimeError("NLP column tasks require configured providers.")
        if self._cached_providers is not None:
            return self._cached_providers
        override = str(getattr(self.settings, "data_analyst_nlp_chat_model", "") or "").strip()
        if not override:
            self._cached_providers = self.base_providers
            return self._cached_providers
        self._cached_providers = build_providers(
            self.settings,
            embeddings=self.base_providers.embeddings,
            chat_model_override=override,
            chat_temperature_override=float(getattr(self.settings, "data_analyst_nlp_temperature", 0.0) or 0.0),
        )
        return self._cached_providers

    def run_task(
        self,
        *,
        task: str,
        classification_rules: str,
        allowed_labels: List[str],
        items: List[Dict[str, Any]],
        batch_size: int,
        callbacks: Iterable[Any] | None = None,
    ) -> NlpBatchResult:
        clean_batch_size = max(1, int(batch_size or 1))
        rows: List[Dict[str, Any]] = []
        failed_batches: List[Dict[str, Any]] = []
        for index in range(0, len(items), clean_batch_size):
            batch = items[index : index + clean_batch_size]
            try:
                rows.extend(
                    self._invoke_batch(
                        task=task,
                        classification_rules=classification_rules,
                        allowed_labels=allowed_labels,
                        batch=batch,
                        callbacks=list(callbacks or []),
                    )
                )
            except Exception as exc:
                failed_batches.append(
                    {
                        "item_ids": [str(item.get("item_id") or "") for item in batch],
                        "error": str(exc),
                    }
                )
        return NlpBatchResult(rows=rows, failed_batches=failed_batches, model_name=self._model_name())

    def _model_name(self) -> str:
        override = str(getattr(self.settings, "data_analyst_nlp_chat_model", "") or "").strip()
        if override:
            return override
        return (
            getattr(self.settings, "ollama_chat_model", "")
            or getattr(self.settings, "azure_openai_chat_deployment", "")
            or getattr(self.settings, "nvidia_chat_model", "")
            or "default"
        )

    def _invoke_batch(
        self,
        *,
        task: str,
        classification_rules: str,
        allowed_labels: List[str],
        batch: List[Dict[str, Any]],
        callbacks: List[Any],
    ) -> List[Dict[str, Any]]:
        chat_model = self._provider_bundle().chat
        response = chat_model.invoke(
            [
                SystemMessage(content=self._system_prompt(task=task, classification_rules=classification_rules, allowed_labels=allowed_labels)),
                HumanMessage(content=self._user_prompt(task=task, batch=batch)),
            ],
            config={"callbacks": callbacks},
        )
        payload = self._validate_payload(
            task=task,
            allowed_labels=allowed_labels,
            batch=batch,
            payload=extract_json(getattr(response, "content", None) or str(response)),
        )
        if payload is not None:
            return payload

        repair_response = chat_model.invoke(
            [
                SystemMessage(
                    content=(
                        "Repair the payload into strict JSON only.\n"
                        "Return exactly one object with a `results` array and no markdown fences."
                    )
                ),
                HumanMessage(content=getattr(response, "content", None) or str(response)),
            ],
            config={"callbacks": callbacks},
        )
        repaired = self._validate_payload(
            task=task,
            allowed_labels=allowed_labels,
            batch=batch,
            payload=extract_json(getattr(repair_response, "content", None) or str(repair_response)),
        )
        if repaired is None:
            raise ValueError("Model response could not be repaired into the required JSON schema.")
        return repaired

    def _system_prompt(self, *, task: str, classification_rules: str, allowed_labels: List[str]) -> str:
        if task == "sentiment":
            result_shape = '{"results":[{"item_id":"row_1","label":"positive|neutral|negative","score":0.93}]}'
            default_rules = "Classify sentiment as positive, neutral, or negative."
        elif task == "categorize":
            result_shape = '{"results":[{"item_id":"row_1","label":"<allowed label>","score":0.93}]}'
            default_rules = "Assign the best matching category label."
        elif task == "keywords":
            result_shape = '{"results":[{"item_id":"row_1","keywords":["keyword1","keyword2"]}]}'
            default_rules = "Extract 1-5 concise keywords."
        else:
            result_shape = '{"results":[{"item_id":"row_1","summary":"One sentence summary."}]}'
            default_rules = "Summarize each text in one sentence."

        allowed = f"\nAllowed labels: {', '.join(allowed_labels)}" if allowed_labels else ""
        rules = classification_rules.strip() or default_rules
        return (
            "You are a bounded NLP task runner for spreadsheet rows.\n"
            "Return strict JSON only. Do not include prose, markdown, or explanations.\n"
            f"Task: {task}\n"
            f"Rules: {rules}{allowed}\n"
            f"Response shape: {result_shape}\n"
            "Each result must preserve the provided item_id exactly."
        )

    def _user_prompt(self, *, task: str, batch: List[Dict[str, Any]]) -> str:
        return json.dumps(
            {
                "task": task,
                "items": [{"item_id": item["item_id"], "text": item["text"]} for item in batch],
            },
            ensure_ascii=False,
        )

    def _validate_payload(
        self,
        *,
        task: str,
        allowed_labels: List[str],
        batch: List[Dict[str, Any]],
        payload: Any,
    ) -> List[Dict[str, Any]] | None:
        if not isinstance(payload, dict):
            return None
        results = payload.get("results")
        if not isinstance(results, list):
            return None
        expected_ids = {str(item["item_id"]) for item in batch}
        allowed_lookup = {label.lower(): label for label in allowed_labels}
        seen_ids: set[str] = set()
        validated: List[Dict[str, Any]] = []
        for item in results:
            if not isinstance(item, dict):
                return None
            item_id = str(item.get("item_id") or "").strip()
            if not item_id or item_id not in expected_ids or item_id in seen_ids:
                return None
            seen_ids.add(item_id)
            row: Dict[str, Any] = {"item_id": item_id}
            if task in {"sentiment", "categorize"}:
                label = str(item.get("label") or "").strip()
                if not label:
                    return None
                if allowed_lookup and label.lower() not in allowed_lookup:
                    return None
                row["label"] = allowed_lookup.get(label.lower(), label)
                try:
                    score = float(item.get("score", 0.0))
                except (TypeError, ValueError):
                    return None
                if score < 0.0 or score > 1.0:
                    return None
                row["score"] = round(score, 6)
            elif task == "keywords":
                keywords = item.get("keywords")
                if not isinstance(keywords, list):
                    return None
                row["keywords"] = [str(value).strip() for value in keywords if str(value).strip()][:5]
            else:
                summary = str(item.get("summary") or "").strip()
                if not summary:
                    return None
                row["summary"] = summary
            validated.append(row)
        if seen_ids != expected_ids:
            return None
        return validated

