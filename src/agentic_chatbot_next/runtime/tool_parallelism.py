from __future__ import annotations

import asyncio
from collections.abc import Sequence
import inspect
from time import perf_counter
from typing import Any, cast

from langchain_core.messages import AIMessage, HumanMessage, ToolCall, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.runnables.config import get_config_list, get_executor_for_config
from langgraph.prebuilt.tool_node import ToolNode
from langgraph.store.base import BaseStore
from pydantic import BaseModel

from agentic_chatbot_next.contracts.messages import utc_now_iso
from agentic_chatbot_next.observability.events import RuntimeEvent


def _current_turn_messages(messages: Sequence[Any]) -> list[Any]:
    last_human_index = -1
    for index, message in enumerate(messages):
        if isinstance(message, HumanMessage):
            last_human_index = index
    if last_human_index < 0:
        return list(messages)
    return list(messages[last_human_index + 1 :])


def count_current_turn_tool_messages(messages: Sequence[Any]) -> int:
    return sum(1 for message in _current_turn_messages(messages) if isinstance(message, ToolMessage))


def count_current_turn_ai_messages(messages: Sequence[Any]) -> int:
    return sum(1 for message in _current_turn_messages(messages) if isinstance(message, AIMessage))


class PolicyAwareToolNode(ToolNode):
    """Schedule tool bursts conservatively while preserving ToolNode behavior."""

    def __init__(
        self,
        tools: Sequence[Any],
        *,
        max_tool_calls: int = 12,
        max_parallel_tool_calls: int = 4,
        context_budget_manager: Any | None = None,
        tool_context: Any | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(tools, **kwargs)
        self.max_tool_calls = max(0, int(max_tool_calls or 0))
        self.max_parallel_tool_calls = max(1, int(max_parallel_tool_calls or 1))
        self.context_budget_manager = context_budget_manager
        self.tool_context = tool_context

    def _func(
        self,
        input: list[Any] | dict[str, Any] | BaseModel,
        config: RunnableConfig,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        runtime = self._runtime_from_call_args(args, kwargs)
        store = self._store_from_call_args(args, kwargs, runtime)
        tool_calls, input_type = self._parse_input_compat(input, store)
        scheduled_calls, overflow_calls = self._partition_calls(tool_calls, input)
        config_list = get_config_list(config, len(scheduled_calls))
        outputs: list[Any] = [None] * len(tool_calls)

        with get_executor_for_config(config) as executor:
            for wave_index, wave in enumerate(self._build_waves(scheduled_calls), start=1):
                if not wave:
                    continue
                started_at = utc_now_iso()
                started_perf = perf_counter()
                self._emit_tool_parallel_group(
                    "tool_parallel_group_started",
                    wave=wave,
                    wave_index=wave_index,
                    status="running",
                    started_at=started_at,
                )
                wave_calls = [call for _, call in wave]
                wave_input_types = [input_type] * len(wave_calls)
                group_id = self._tool_wave_group_id(wave_index, wave)
                execution_mode = self._wave_execution_mode(wave)
                wave_configs = [
                    self._config_with_parallel_group(
                        config_list[index],
                        group_id=group_id,
                        execution_mode=execution_mode,
                        group_size=len(wave),
                        call_index=index,
                    )
                    for index, _ in wave
                ]
                wave_runtime_args = self._runtime_args_for_wave(
                    input,
                    wave_calls=wave_calls,
                    wave_configs=wave_configs,
                    runtime=runtime,
                )
                try:
                    wave_outputs = list(
                        executor.map(self._run_one, wave_calls, wave_input_types, wave_runtime_args)
                    )
                except Exception:
                    self._emit_tool_parallel_group(
                        "tool_parallel_group_completed",
                        wave=wave,
                        wave_index=wave_index,
                        status="error",
                        started_at=started_at,
                        completed_at=utc_now_iso(),
                        duration_ms=max(0, int((perf_counter() - started_perf) * 1000)),
                    )
                    raise
                for (index, _), output in zip(wave, wave_outputs):
                    outputs[index] = output
                self._emit_tool_parallel_group(
                    "tool_parallel_group_completed",
                    wave=wave,
                    wave_index=wave_index,
                    status="completed",
                    started_at=started_at,
                    completed_at=utc_now_iso(),
                    duration_ms=max(0, int((perf_counter() - started_perf) * 1000)),
                )

        for index, call in overflow_calls:
            outputs[index] = self._tool_budget_exceeded_message(
                call,
                remaining_budget=max(0, self._remaining_tool_budget(input)),
                requested_calls=len(tool_calls),
            )

        outputs = self._budget_tool_outputs(cast(list[Any], outputs))
        return self._combine_tool_outputs(cast(list[Any], outputs), input_type)

    async def _afunc(
        self,
        input: list[Any] | dict[str, Any] | BaseModel,
        config: RunnableConfig,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        runtime = self._runtime_from_call_args(args, kwargs)
        store = self._store_from_call_args(args, kwargs, runtime)
        tool_calls, input_type = self._parse_input_compat(input, store)
        scheduled_calls, overflow_calls = self._partition_calls(tool_calls, input)
        config_list = get_config_list(config, len(scheduled_calls))
        outputs: list[Any] = [None] * len(tool_calls)

        for wave_index, wave in enumerate(self._build_waves(scheduled_calls), start=1):
            if not wave:
                continue
            started_at = utc_now_iso()
            started_perf = perf_counter()
            self._emit_tool_parallel_group(
                "tool_parallel_group_started",
                wave=wave,
                wave_index=wave_index,
                status="running",
                started_at=started_at,
            )
            wave_calls = [call for _, call in wave]
            group_id = self._tool_wave_group_id(wave_index, wave)
            execution_mode = self._wave_execution_mode(wave)
            wave_configs = [
                self._config_with_parallel_group(
                    config_list[index],
                    group_id=group_id,
                    execution_mode=execution_mode,
                    group_size=len(wave),
                    call_index=index,
                )
                for index, _ in wave
            ]
            wave_runtime_args = self._runtime_args_for_wave(
                input,
                wave_calls=wave_calls,
                wave_configs=wave_configs,
                runtime=runtime,
            )
            try:
                wave_outputs = await asyncio.gather(
                    *(
                        self._arun_one(call, input_type, runtime_arg)
                        for call, runtime_arg in zip(wave_calls, wave_runtime_args)
                    )
                )
            except Exception:
                self._emit_tool_parallel_group(
                    "tool_parallel_group_completed",
                    wave=wave,
                    wave_index=wave_index,
                    status="error",
                    started_at=started_at,
                    completed_at=utc_now_iso(),
                    duration_ms=max(0, int((perf_counter() - started_perf) * 1000)),
                )
                raise
            for (index, _), output in zip(wave, wave_outputs):
                outputs[index] = output
            self._emit_tool_parallel_group(
                "tool_parallel_group_completed",
                wave=wave,
                wave_index=wave_index,
                status="completed",
                started_at=started_at,
                completed_at=utc_now_iso(),
                duration_ms=max(0, int((perf_counter() - started_perf) * 1000)),
            )

        for index, call in overflow_calls:
            outputs[index] = self._tool_budget_exceeded_message(
                call,
                remaining_budget=max(0, self._remaining_tool_budget(input)),
                requested_calls=len(tool_calls),
            )

        outputs = self._budget_tool_outputs(cast(list[Any], outputs))
        return self._combine_tool_outputs(cast(list[Any], outputs), input_type)

    def _budget_tool_outputs(self, outputs: list[Any]) -> list[Any]:
        manager = self.context_budget_manager
        if manager is None or not bool(getattr(manager, "enabled", False)):
            return outputs
        budgeted: list[Any] = []
        for output in outputs:
            if isinstance(output, ToolMessage):
                budgeted.append(manager.budget_tool_message(output, tool_context=self.tool_context))
            else:
                budgeted.append(output)
        return budgeted

    def _partition_calls(
        self,
        tool_calls: list[ToolCall],
        input: list[Any] | dict[str, Any] | BaseModel,
    ) -> tuple[list[tuple[int, ToolCall]], list[tuple[int, ToolCall]]]:
        remaining_budget = self._remaining_tool_budget(input)
        scheduled = [(index, call) for index, call in enumerate(tool_calls[:remaining_budget])]
        overflow = [(index, call) for index, call in enumerate(tool_calls[remaining_budget:], start=remaining_budget)]
        return scheduled, overflow

    def _remaining_tool_budget(self, input: list[Any] | dict[str, Any] | BaseModel) -> int:
        prior_tool_messages = count_current_turn_tool_messages(self._messages_from_input(input))
        return max(0, self.max_tool_calls - prior_tool_messages)

    def _messages_from_input(self, input: list[Any] | dict[str, Any] | BaseModel) -> list[Any]:
        if isinstance(input, list):
            if input and isinstance(input[-1], dict) and input[-1].get("type") == "tool_call":
                return []
            return list(input)
        messages_key = self._messages_key_name()
        if isinstance(input, dict):
            return list(input.get(messages_key, []) or [])
        return list(getattr(input, messages_key, []) or [])

    def _build_waves(self, tool_calls: list[tuple[int, ToolCall]]) -> list[list[tuple[int, ToolCall]]]:
        waves: list[list[tuple[int, ToolCall]]] = []
        wave_keys: list[set[str]] = []

        for indexed_call in tool_calls:
            _, call = indexed_call
            key = self._concurrency_key(call)
            for batch, batch_keys in zip(waves, wave_keys):
                if len(batch) >= self.max_parallel_tool_calls:
                    continue
                if key and key in batch_keys:
                    continue
                batch.append(indexed_call)
                if key:
                    batch_keys.add(key)
                break
            else:
                waves.append([indexed_call])
                wave_keys.append({key} if key else set())

        return waves

    def _concurrency_key(self, call: ToolCall) -> str:
        tool = self._get_tools_by_name().get(call["name"])
        metadata = dict(getattr(tool, "metadata", {}) or {})
        return str(metadata.get("concurrency_key") or "").strip()

    def _wave_execution_mode(self, wave: list[tuple[int, ToolCall]]) -> str:
        return "parallel" if len(wave) > 1 else "sequential"

    def _tool_wave_group_id(self, wave_index: int, wave: list[tuple[int, ToolCall]]) -> str:
        first_call = wave[0][1] if wave else {}
        first_call_id = str(first_call.get("id") or first_call.get("name") or wave_index)
        active_agent = str(getattr(self.tool_context, "active_agent", "") or "agent")
        return f"tool-wave-{active_agent}-{wave_index}-{first_call_id}"

    def _config_with_parallel_group(
        self,
        config: RunnableConfig,
        *,
        group_id: str,
        execution_mode: str,
        group_size: int,
        call_index: int,
    ) -> RunnableConfig:
        next_config = dict(config or {})
        metadata = dict(next_config.get("metadata") or {})
        metadata.update(
            {
                "agentic_parallel_group_id": group_id,
                "agentic_parallel_execution_mode": execution_mode,
                "agentic_parallel_group_size": group_size,
                "agentic_tool_call_index": call_index,
            }
        )
        next_config["metadata"] = metadata
        return cast(RunnableConfig, next_config)

    def _emit_tool_parallel_group(
        self,
        event_type: str,
        *,
        wave: list[tuple[int, ToolCall]],
        wave_index: int,
        status: str,
        started_at: str,
        completed_at: str = "",
        duration_ms: int | None = None,
    ) -> None:
        tool_context = self.tool_context
        event_sink = getattr(tool_context, "event_sink", None)
        session = getattr(tool_context, "session", None)
        session_id = str(getattr(session, "session_id", "") or "")
        if event_sink is None or not session_id:
            return
        members = [
            {
                "tool_call_id": str(call.get("id") or ""),
                "tool_name": str(call.get("name") or ""),
                "concurrency_key": self._concurrency_key(call),
            }
            for _, call in wave
        ]
        execution_mode = self._wave_execution_mode(wave)
        event_sink.emit(
            RuntimeEvent(
                event_type=event_type,
                session_id=session_id,
                agent_name=str(getattr(tool_context, "active_agent", "") or ""),
                job_id=str((getattr(tool_context, "metadata", {}) or {}).get("job_id") or ""),
                payload={
                    "conversation_id": str(getattr(session, "conversation_id", "") or ""),
                    "group_id": self._tool_wave_group_id(wave_index, wave),
                    "group_kind": "tool_wave",
                    "status": status,
                    "execution_mode": execution_mode,
                    "size": len(wave),
                    "members": members,
                    "reason": (
                        "Independent tool calls were scheduled in the same executor wave."
                        if execution_mode == "parallel"
                        else "One tool call was scheduled; details are shown on the tool row."
                    ),
                    "started_at": started_at,
                    "completed_at": completed_at,
                    "duration_ms": duration_ms,
                    "max_parallel": self.max_parallel_tool_calls,
                },
            )
        )

    def _tool_budget_exceeded_message(
        self,
        call: ToolCall,
        *,
        remaining_budget: int,
        requested_calls: int,
    ) -> ToolMessage:
        return ToolMessage(
            content=(
                "Error: tool-call budget exceeded for this turn. "
                f"Requested {requested_calls} tool calls with {remaining_budget} remaining "
                f"against a max budget of {self.max_tool_calls}."
            ),
            name=call["name"],
            tool_call_id=call["id"],
            status="error",
        )

    def _parse_input_compat(
        self,
        input: list[Any] | dict[str, Any] | BaseModel,
        store: BaseStore | None,
    ) -> tuple[list[ToolCall], str]:
        """Call ToolNode._parse_input across LangGraph minor-version signatures."""
        if not hasattr(self, "tools_by_name"):
            tools = getattr(self, "_tools_by_name", None)
            if isinstance(tools, dict):
                setattr(self, "tools_by_name", tools)
        try:
            return self._parse_input(input, store)
        except TypeError as exc:
            message = str(exc)
            if "_parse_input" not in message or "positional" not in message:
                raise
            return self._parse_input(input)
        except AttributeError as exc:
            if "messages_key" not in str(exc):
                raise
            setattr(self, "messages_key", self._messages_key_name())
            return self._parse_input(input, store)

    def _messages_key_name(self) -> str:
        return str(
            getattr(self, "messages_key", None)
            or getattr(self, "_messages_key", None)
            or "messages"
        )

    def _get_tools_by_name(self) -> dict[str, Any]:
        tools = getattr(self, "tools_by_name", None)
        if isinstance(tools, dict):
            return tools
        tools = getattr(self, "_tools_by_name", None)
        return tools if isinstance(tools, dict) else {}

    def _runtime_from_call_args(self, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any | None:
        runtime = kwargs.get("runtime")
        if runtime is not None:
            return runtime
        if args:
            candidate = args[0]
            runtime_attrs = {"context", "store", "stream_writer", "execution_info", "server_info"}
            if any(hasattr(candidate, attr) for attr in runtime_attrs):
                return candidate
        return None

    def _store_from_call_args(
        self,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        runtime: Any | None,
    ) -> BaseStore | None:
        if kwargs.get("store") is not None:
            return kwargs.get("store")
        if runtime is not None:
            return getattr(runtime, "store", None)
        if args:
            candidate = args[0]
            runtime_attrs = {"context", "store", "stream_writer", "execution_info", "server_info"}
            if not any(hasattr(candidate, attr) for attr in runtime_attrs):
                return candidate
        return None

    def _run_one_expects_tool_runtime(self) -> bool:
        try:
            params = list(inspect.signature(self._run_one).parameters)
        except (TypeError, ValueError):
            return False
        return bool(params and params[-1] in {"tool_runtime", "runtime"})

    def _runtime_args_for_wave(
        self,
        input: list[Any] | dict[str, Any] | BaseModel,
        *,
        wave_calls: list[ToolCall],
        wave_configs: list[RunnableConfig],
        runtime: Any | None,
    ) -> list[Any]:
        if not self._run_one_expects_tool_runtime():
            return list(wave_configs)
        return [
            self._build_tool_runtime(input, call=call, config=config, runtime=runtime)
            for call, config in zip(wave_calls, wave_configs)
        ]

    def _build_tool_runtime(
        self,
        input: list[Any] | dict[str, Any] | BaseModel,
        *,
        call: ToolCall,
        config: RunnableConfig,
        runtime: Any | None,
    ) -> Any:
        try:
            from langgraph.prebuilt.tool_node import ToolRuntime
        except Exception:
            return config
        state = self._extract_state_compat(input, config) if hasattr(self, "_extract_state") else {}
        return ToolRuntime(
            state=state,
            tool_call_id=call["id"],
            config=config,
            context=getattr(runtime, "context", None),
            store=getattr(runtime, "store", None),
            stream_writer=getattr(runtime, "stream_writer", None),
            execution_info=getattr(runtime, "execution_info", None),
            server_info=getattr(runtime, "server_info", None),
        )

    def _extract_state_compat(
        self,
        input: list[Any] | dict[str, Any] | BaseModel,
        config: RunnableConfig,
    ) -> Any:
        """Call ToolNode._extract_state across LangGraph minor-version signatures."""
        try:
            return self._extract_state(input, config)
        except TypeError as exc:
            message = str(exc)
            if "_extract_state" not in message and "positional" not in message:
                raise
            return self._extract_state(input)
