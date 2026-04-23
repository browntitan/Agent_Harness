from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage

from agentic_chatbot_next.contracts.agents import AgentDefinition
from agentic_chatbot_next.contracts.messages import utc_now_iso
from agentic_chatbot_next.runtime.artifacts import register_workspace_artifact
from agentic_chatbot_next.sandbox.workspace import SessionWorkspace
from agentic_chatbot_next.utils.json_utils import extract_json

logger = logging.getLogger(__name__)

_BACKGROUND_WORD_THRESHOLD = 3000
_BACKGROUND_SECTION_THRESHOLD = 5
_MAX_SECTION_RETRIES = 2


def _normalize_positive_int(value: Any, default: int, *, minimum: int = 1, maximum: int = 100_000) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return max(minimum, min(parsed, maximum))


def _sanitize_text(value: Any, *, default: str = "") -> str:
    text = str(value or "").strip()
    return text or default


def _slugify(value: str, *, fallback: str = "long-output") -> str:
    text = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return text[:48] or fallback


def _message_text(message: Any) -> str:
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, str):
                text = item.strip()
            elif isinstance(item, dict):
                text = str(item.get("text") or item.get("content") or "").strip()
            else:
                text = str(item).strip()
            if text:
                parts.append(text)
        return "\n".join(parts).strip()
    return str(content or "").strip()


def _message_metadata(message: Any) -> Dict[str, Any]:
    metadata: Dict[str, Any] = {}
    metadata.update(dict(getattr(message, "response_metadata", {}) or {}))
    metadata.update(dict(getattr(message, "additional_kwargs", {}) or {}))
    return metadata


def _is_truncated(message: Any) -> bool:
    metadata = _message_metadata(message)
    reason = str(
        metadata.get("finish_reason")
        or metadata.get("stop_reason")
        or metadata.get("reason")
        or metadata.get("completion_reason")
        or ""
    ).strip().lower()
    return reason in {"length", "max_tokens", "max_output_tokens"}


@dataclass
class LongOutputOptions:
    enabled: bool = False
    target_words: int = 1800
    target_sections: int = 4
    delivery_mode: str = "hybrid"
    background_ok: bool = True
    output_format: str = "markdown"
    async_requested: bool = False

    @classmethod
    def from_metadata(cls, raw: Any) -> "LongOutputOptions":
        if not isinstance(raw, dict):
            return cls(enabled=False)
        delivery_mode = _sanitize_text(raw.get("delivery_mode"), default="hybrid").lower()
        if delivery_mode not in {"chat", "file", "hybrid"}:
            delivery_mode = "hybrid"
        output_format = _sanitize_text(raw.get("output_format"), default="markdown").lower()
        if output_format not in {"markdown", "text"}:
            output_format = "markdown"
        return cls(
            enabled=bool(raw.get("enabled", False)),
            target_words=_normalize_positive_int(raw.get("target_words"), 1800, minimum=200, maximum=20_000),
            target_sections=_normalize_positive_int(raw.get("target_sections"), 4, minimum=1, maximum=16),
            delivery_mode=delivery_mode,
            background_ok=bool(raw.get("background_ok", True)),
            output_format=output_format,
            async_requested=bool(raw.get("async_requested", False)),
        )

    def should_run_in_background(self) -> bool:
        if not self.enabled or not self.background_ok:
            return False
        return self.async_requested or self.target_words > _BACKGROUND_WORD_THRESHOLD or self.target_sections > _BACKGROUND_SECTION_THRESHOLD

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class LongOutputSection:
    index: int
    heading: str
    brief: str
    target_words: int
    summary: str = ""
    status: str = "pending"
    retries: int = 0
    filename: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "LongOutputSection":
        return cls(
            index=_normalize_positive_int(raw.get("index"), 1, minimum=1, maximum=128),
            heading=_sanitize_text(raw.get("heading"), default="Section"),
            brief=_sanitize_text(raw.get("brief")),
            target_words=_normalize_positive_int(raw.get("target_words"), 400, minimum=100, maximum=5000),
            summary=_sanitize_text(raw.get("summary")),
            status=_sanitize_text(raw.get("status"), default="pending"),
            retries=_normalize_positive_int(raw.get("retries"), 0, minimum=0, maximum=32),
            filename=_sanitize_text(raw.get("filename")),
        )


@dataclass
class LongOutputPlan:
    title: str
    executive_summary: str
    sections: List[LongOutputSection]


@dataclass
class LongOutputResult:
    summary_text: str
    output_filename: str
    manifest_filename: str
    artifact: Dict[str, Any]
    background: bool = False
    job_id: str = ""
    title: str = ""
    section_count: int = 0

    def to_metadata(self) -> Dict[str, Any]:
        payload = {
            "long_output": {
                "title": self.title,
                "output_filename": self.output_filename,
                "manifest_filename": self.manifest_filename,
                "section_count": self.section_count,
                "background": self.background,
            }
        }
        if self.job_id:
            payload["job_id"] = self.job_id
        return payload


class LongOutputComposer:
    def __init__(
        self,
        *,
        settings: Any,
        chat_llm: Any,
        agent: AgentDefinition,
        system_prompt: str,
        session_or_state: Any,
        callbacks: Optional[List[Any]] = None,
        progress_sink: Any | None = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.settings = settings
        self.chat_llm = chat_llm
        self.agent = agent
        self.system_prompt = system_prompt.strip()
        self.session_or_state = session_or_state
        self.callbacks = list(callbacks or [])
        self.progress_sink = progress_sink
        self.metadata = dict(metadata or {})
        self.workspace = self._resolve_workspace(session_or_state)

    def compose(
        self,
        *,
        user_text: str,
        options: LongOutputOptions,
    ) -> LongOutputResult:
        request_hash = hashlib.sha1(
            json.dumps(
                {
                    "user_text": user_text,
                    "agent": self.agent.name,
                    "options": options.to_dict(),
                },
                ensure_ascii=False,
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()[:12]
        manifest_filename = f"long_output_{request_hash}_manifest.json"
        title_slug = _slugify(user_text[:80])
        output_ext = "md" if options.output_format == "markdown" else "txt"
        output_filename = f"long_output_{request_hash}_{title_slug}.{output_ext}"
        manifest = self._load_manifest(manifest_filename)
        if manifest is None or str(manifest.get("request_hash") or "") != request_hash:
            plan = self._plan_document(user_text=user_text, options=options)
            manifest = self._new_manifest(
                request_hash=request_hash,
                user_text=user_text,
                options=options,
                output_filename=output_filename,
                manifest_filename=manifest_filename,
                plan=plan,
            )
            self._write_manifest(manifest_filename, manifest)
            self._write_output(output_filename, self._render_output(manifest, include_pending=False))
        else:
            self._emit_progress(
                "phase_update",
                label="Resuming long-form draft",
                detail=f"{sum(1 for item in manifest.get('sections', []) if str(item.get('status') or '') == 'completed')} section(s) already complete",
                agent=self.agent.name,
            )

        total_sections = len(manifest.get("sections") or [])
        self._emit_progress(
            "phase_start",
            label="Generating long-form output",
            detail=f"{total_sections} section(s) planned",
            agent=self.agent.name,
        )
        for raw_section in manifest.get("sections") or []:
            section = LongOutputSection.from_dict(raw_section)
            if section.status == "completed":
                continue
            self._emit_progress(
                "phase_update",
                label=f"Generating section {section.index}/{total_sections}",
                detail=section.heading,
                agent=self.agent.name,
            )
            section_text, section_summary, retries = self._generate_section(
                user_text=user_text,
                plan_title=str(manifest.get("title") or ""),
                sections=[LongOutputSection.from_dict(item) for item in (manifest.get("sections") or [])],
                current_section=section,
                output_format=options.output_format,
            )
            section.status = "completed"
            section.summary = section_summary
            section.retries = retries
            section.filename = section.filename or f"long_output_{request_hash}_section_{section.index:02d}.{output_ext}"
            self.workspace.write_text(section.filename, section_text)
            self._update_section(manifest, section, section_text=section_text)
            self._write_manifest(manifest_filename, manifest)
            self._write_output(output_filename, self._render_output(manifest, include_pending=False))

        manifest["status"] = "completed"
        manifest["completed_at"] = utc_now_iso()
        self._write_manifest(manifest_filename, manifest)
        rendered_output = self._render_output(manifest, include_pending=False)
        self._write_output(output_filename, rendered_output)
        artifact_label = f"{manifest.get('title') or 'Long-form output'} ({output_filename})"
        artifact = register_workspace_artifact(self.session_or_state, filename=output_filename, label=artifact_label)
        summary_text = self._build_summary_text(
            manifest=manifest,
            options=options,
            artifact=artifact,
        )
        self._emit_progress(
            "phase_end",
            label="Long-form artifact ready",
            detail=output_filename,
            agent=self.agent.name,
        )
        return LongOutputResult(
            summary_text=summary_text,
            output_filename=output_filename,
            manifest_filename=manifest_filename,
            artifact=artifact,
            title=str(manifest.get("title") or ""),
            section_count=len(manifest.get("sections") or []),
        )

    def _resolve_workspace(self, session_or_state: Any) -> SessionWorkspace:
        existing = getattr(session_or_state, "workspace", None)
        if existing is not None:
            existing.open()
            return existing
        workspace_root = Path(
            str(
                getattr(session_or_state, "workspace_root", "")
                or getattr(self.settings, "workspace_dir", Path("data") / "workspaces") / str(getattr(session_or_state, "session_id", "local-session"))
            )
        )
        workspace = SessionWorkspace(session_id=str(getattr(session_or_state, "session_id", "")), root=workspace_root)
        workspace.open()
        return workspace

    def _emit_progress(self, event_type: str, **payload: Any) -> None:
        sink = self.progress_sink
        if sink is None or not hasattr(sink, "emit_progress"):
            return
        sink.emit_progress(event_type, **payload)

    def _invoke(self, messages: List[Any]) -> Any:
        return self.chat_llm.invoke(messages, config={"callbacks": self.callbacks})

    def _plan_document(self, *, user_text: str, options: LongOutputOptions) -> LongOutputPlan:
        section_target = max(1, options.target_sections)
        outline_system = (
            "You are planning a long-form document that will be generated section by section.\n"
            "Return JSON only using this schema:\n"
            "{"
            '"title": "document title",'
            '"executive_summary": "one sentence summary",'
            '"sections": ['
            '{"heading": "Section heading", "brief": "what this section must cover", "target_words": 400}'
            "]}"
        )
        if self.system_prompt:
            outline_system += "\n\nRole instructions:\n" + self.system_prompt
        outline_prompt = (
            f"USER_REQUEST:\n{user_text}\n\n"
            f"TARGET_WORDS: {options.target_words}\n"
            f"TARGET_SECTIONS: {section_target}\n"
            f"OUTPUT_FORMAT: {options.output_format}\n\n"
            "Plan a coherent outline with enough coverage to satisfy the request. "
            "Keep each section scoped tightly enough to fit within its target word count."
        )
        try:
            response = self._invoke([SystemMessage(content=outline_system), HumanMessage(content=outline_prompt)])
            payload = extract_json(_message_text(response)) or {}
        except Exception as exc:
            logger.warning("Long-output outline planning failed: %s", exc)
            payload = {}
        sections_payload = payload.get("sections") if isinstance(payload, dict) else None
        sections: List[LongOutputSection] = []
        if isinstance(sections_payload, list):
            for index, raw in enumerate(sections_payload[:section_target], start=1):
                if not isinstance(raw, dict):
                    continue
                sections.append(
                    LongOutputSection(
                        index=index,
                        heading=_sanitize_text(raw.get("heading"), default=f"Section {index}"),
                        brief=_sanitize_text(raw.get("brief"), default=f"Cover section {index}."),
                        target_words=_normalize_positive_int(
                            raw.get("target_words"),
                            max(180, options.target_words // max(section_target, 1)),
                            minimum=100,
                            maximum=5000,
                        ),
                    )
                )
        if not sections:
            default_target = max(180, options.target_words // max(section_target, 1))
            sections = [
                LongOutputSection(
                    index=index,
                    heading=f"Section {index}",
                    brief="Advance the user's request with concrete, non-duplicative content.",
                    target_words=default_target,
                )
                for index in range(1, section_target + 1)
            ]
        title = _sanitize_text(payload.get("title") if isinstance(payload, dict) else "", default=f"Long-form response for {self.agent.name}")
        executive_summary = _sanitize_text(payload.get("executive_summary") if isinstance(payload, dict) else "")
        return LongOutputPlan(title=title, executive_summary=executive_summary, sections=sections)

    def _new_manifest(
        self,
        *,
        request_hash: str,
        user_text: str,
        options: LongOutputOptions,
        output_filename: str,
        manifest_filename: str,
        plan: LongOutputPlan,
    ) -> Dict[str, Any]:
        return {
            "request_hash": request_hash,
            "status": "in_progress",
            "created_at": utc_now_iso(),
            "updated_at": utc_now_iso(),
            "completed_at": "",
            "user_text": user_text,
            "agent_name": self.agent.name,
            "title": plan.title,
            "executive_summary": plan.executive_summary,
            "options": options.to_dict(),
            "output_filename": output_filename,
            "manifest_filename": manifest_filename,
            "section_outputs": {},
            "sections": [section.to_dict() for section in plan.sections],
        }

    def _load_manifest(self, manifest_filename: str) -> Dict[str, Any] | None:
        if not self.workspace.exists(manifest_filename):
            return None
        try:
            return json.loads(self.workspace.read_text(manifest_filename))
        except Exception as exc:
            logger.warning("Could not read long-output manifest %s: %s", manifest_filename, exc)
            return None

    def _write_manifest(self, manifest_filename: str, manifest: Dict[str, Any]) -> None:
        manifest["updated_at"] = utc_now_iso()
        self.workspace.write_text(manifest_filename, json.dumps(manifest, ensure_ascii=False, indent=2))

    def _write_output(self, output_filename: str, content: str) -> None:
        self.workspace.write_text(output_filename, content)

    def _generate_section(
        self,
        *,
        user_text: str,
        plan_title: str,
        sections: List[LongOutputSection],
        current_section: LongOutputSection,
        output_format: str,
    ) -> tuple[str, str, int]:
        prior_summaries = [
            f"{section.index}. {section.heading}: {section.summary}"
            for section in sections
            if section.index < current_section.index and section.summary
        ]
        outline_text = "\n".join(
            f"{section.index}. {section.heading} ({section.target_words} words): {section.brief}"
            for section in sections
        )
        section_system = (
            "You are writing one section of a longer document.\n"
            "Return only the content for the requested section.\n"
            "Do not repeat previous sections, do not include front-matter, and keep style consistent."
        )
        if self.system_prompt:
            section_system += "\n\nRole instructions:\n" + self.system_prompt
        section_prompt = (
            f"DOCUMENT_TITLE: {plan_title}\n"
            f"USER_REQUEST:\n{user_text}\n\n"
            f"FULL_OUTLINE:\n{outline_text}\n\n"
            f"COMPLETED_SECTION_SUMMARIES:\n{chr(10).join(prior_summaries) if prior_summaries else '(none yet)'}\n\n"
            f"CURRENT_SECTION: {current_section.index}. {current_section.heading}\n"
            f"SECTION_BRIEF: {current_section.brief}\n"
            f"TARGET_WORDS: {current_section.target_words}\n"
            f"OUTPUT_FORMAT: {output_format}\n\n"
            "Write this section now."
        )
        pieces: List[str] = []
        retries = 0
        response = self._invoke([SystemMessage(content=section_system), HumanMessage(content=section_prompt)])
        pieces.append(_message_text(response))
        while _is_truncated(response) and retries < _MAX_SECTION_RETRIES:
            retries += 1
            continue_prompt = (
                f"Continue section {current_section.index}: {current_section.heading}.\n"
                "Continue from exactly where the draft stopped. Do not repeat prior sentences.\n\n"
                f"CURRENT_PARTIAL_SECTION:\n{''.join(pieces)}"
            )
            response = self._invoke([SystemMessage(content=section_system), HumanMessage(content=continue_prompt)])
            continuation = _message_text(response)
            if continuation:
                pieces.append(continuation)
        section_text = "\n\n".join(part.strip() for part in pieces if part and part.strip()).strip()
        if not section_text:
            section_text = f"{current_section.heading}\n\nContent generation did not produce output."
        summary = self._summarize_section(
            plan_title=plan_title,
            current_section=current_section,
            section_text=section_text,
        )
        return section_text, summary, retries

    def _summarize_section(
        self,
        *,
        plan_title: str,
        current_section: LongOutputSection,
        section_text: str,
    ) -> str:
        system = (
            "Summarize the section in one concise sentence for downstream planning memory. "
            "Return plain text only."
        )
        prompt = (
            f"DOCUMENT_TITLE: {plan_title}\n"
            f"SECTION: {current_section.heading}\n\n"
            f"SECTION_TEXT:\n{section_text[:4000]}"
        )
        try:
            response = self._invoke([SystemMessage(content=system), HumanMessage(content=prompt)])
            text = _message_text(response)
            if text:
                return text[:500]
        except Exception as exc:
            logger.warning("Long-output section summarization failed: %s", exc)
        return f"{current_section.heading} completed."

    def _update_section(self, manifest: Dict[str, Any], section: LongOutputSection, *, section_text: str) -> None:
        updated_sections: List[Dict[str, Any]] = []
        for raw in manifest.get("sections") or []:
            existing = LongOutputSection.from_dict(raw)
            if existing.index == section.index:
                updated_sections.append(section.to_dict())
            else:
                updated_sections.append(existing.to_dict())
        manifest["sections"] = updated_sections
        outputs = dict(manifest.get("section_outputs") or {})
        outputs[str(section.index)] = section_text
        manifest["section_outputs"] = outputs

    def _render_output(self, manifest: Dict[str, Any], *, include_pending: bool) -> str:
        options = dict(manifest.get("options") or {})
        output_format = _sanitize_text(options.get("output_format"), default="markdown").lower()
        title = _sanitize_text(manifest.get("title"), default="Long-form response")
        sections = [LongOutputSection.from_dict(item) for item in (manifest.get("sections") or [])]
        outputs = dict(manifest.get("section_outputs") or {})
        if output_format == "text":
            parts = [title, ""]
            if manifest.get("executive_summary"):
                parts.extend([str(manifest.get("executive_summary") or ""), ""])
            for section in sections:
                text = str(outputs.get(str(section.index)) or "").strip()
                if not text and not include_pending:
                    continue
                parts.append(section.heading)
                parts.append("")
                parts.append(text or "[pending]")
                parts.append("")
            return "\n".join(parts).strip() + "\n"
        parts = [f"# {title}", ""]
        executive_summary = _sanitize_text(manifest.get("executive_summary"))
        if executive_summary:
            parts.extend([executive_summary, ""])
        for section in sections:
            text = str(outputs.get(str(section.index)) or "").strip()
            if not text and not include_pending:
                continue
            normalized = text
            if normalized.startswith("#"):
                normalized = normalized.lstrip("#").strip()
            parts.append(f"## {section.heading}")
            parts.append("")
            parts.append(normalized or "[pending]")
            parts.append("")
        return "\n".join(parts).strip() + "\n"

    def _build_summary_text(
        self,
        *,
        manifest: Dict[str, Any],
        options: LongOutputOptions,
        artifact: Dict[str, Any],
    ) -> str:
        sections = [LongOutputSection.from_dict(item) for item in (manifest.get("sections") or [])]
        title = _sanitize_text(manifest.get("title"), default="Long-form draft")
        artifact_label = _sanitize_text(artifact.get("label"), default=_sanitize_text(artifact.get("filename")))
        if options.delivery_mode == "file":
            return f"I created the full long-form document \"{title}\" as a downloadable file: {artifact_label}."
        summary_system = (
            "Write a concise user-facing completion note for a long-form document generation run. "
            "Mention the document title, how many sections were completed, and that the full draft is attached as a file. "
            "Keep it under 120 words."
        )
        summary_prompt = (
            f"DOCUMENT_TITLE: {title}\n"
            f"SECTION_COUNT: {len(sections)}\n"
            f"ARTIFACT_LABEL: {artifact_label}\n"
            f"USER_REQUEST:\n{manifest.get('user_text') or ''}\n\n"
            "If useful, briefly mention the output format."
        )
        try:
            response = self._invoke([SystemMessage(content=summary_system), HumanMessage(content=summary_prompt)])
            text = _message_text(response)
            if text:
                return text
        except Exception as exc:
            logger.warning("Long-output summary generation failed: %s", exc)
        return (
            f"I finished the long-form draft \"{title}\" across {len(sections)} section(s). "
            f"The full document is attached as {artifact_label}."
        )


def latest_assistant_metadata(messages: List[Any], *, keys: Optional[List[str]] = None) -> Dict[str, Any]:
    wanted = set(keys or [])
    for message in reversed(list(messages or [])):
        if getattr(message, "type", "") != "ai":
            continue
        metadata = dict(getattr(message, "additional_kwargs", {}) or {})
        if not metadata:
            return {}
        if not wanted:
            return metadata
        return {key: metadata[key] for key in wanted if key in metadata}
    return {}
