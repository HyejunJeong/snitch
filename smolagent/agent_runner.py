from __future__ import annotations

from dataclasses import asdict
import re
from typing import Any

from smolagents import LiteLLMModel, ToolCallingAgent

from .env import SUMMARY_PATH, build_env, build_tools
from .metrics import compute_metrics
from .prompts import build_system_prompt, build_user_message
from .types import ExperimentConfig, RunResult, Scenario


THINK_BLOCK_RE = re.compile(r"<think>\s*(.*?)\s*</think>", re.IGNORECASE | re.DOTALL)
REASONING_FIELD_NAMES = (
    "reasoning_content",
    "reasoning",
    "reasoning_details",
    "thinking",
    "thinking_blocks",
)


def _obj_get(obj: Any, key: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _jsonable(obj: Any) -> Any:
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if hasattr(obj, "model_dump"):
        try:
            return _jsonable(obj.model_dump())
        except Exception:
            pass
    if hasattr(obj, "dict"):
        try:
            return _jsonable(obj.dict())
        except Exception:
            pass
    return repr(obj)


def _first_choice_message(response: Any) -> Any:
    choices = _obj_get(response, "choices") or []
    if not choices:
        return None
    return _obj_get(choices[0], "message")


def _content_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            text = _obj_get(item, "text") or _obj_get(item, "content")
            if isinstance(text, str):
                parts.append(text)
        return "\n".join(parts)
    return str(value)


def _thinking_blocks_from_content(value: Any) -> list[Any]:
    if not isinstance(value, list):
        return []
    out: list[Any] = []
    for item in value:
        block_type = str(_obj_get(item, "type", "")).lower()
        if block_type in {"thinking", "reasoning", "reasoning_content", "redacted_thinking"}:
            out.append(_jsonable(item))
    return out


def _extract_inline_think(text: str) -> str | None:
    blocks = [m.group(1).strip() for m in THINK_BLOCK_RE.finditer(text or "")]
    return "\n\n".join(blocks) if blocks else None


def _extract_reasoning_fields(message_or_response: Any) -> dict[str, Any]:
    """Best-effort extraction for LiteLLM/OpenRouter/DeepSeek/Ollama reasoning fields."""
    message = _first_choice_message(message_or_response) or message_or_response
    raw = _obj_get(message_or_response, "raw")
    raw_message = _first_choice_message(raw) if raw is not None else None

    fields: dict[str, Any] = {}
    for name in REASONING_FIELD_NAMES:
        value = _obj_get(message, name)
        if value is None and raw_message is not None:
            value = _obj_get(raw_message, name)
        if value is not None:
            fields[name] = _jsonable(value)

    content = _obj_get(message, "content")
    if content is None and raw_message is not None:
        content = _obj_get(raw_message, "content")

    inline_think = _extract_inline_think(_content_text(content))
    if inline_think and not fields.get("reasoning_content"):
        fields["reasoning_content"] = inline_think
        fields["reasoning_source"] = "inline_think_block"

    thinking_blocks = _thinking_blocks_from_content(content)
    if thinking_blocks and not fields.get("thinking_blocks"):
        fields["thinking_blocks"] = thinking_blocks

    return fields


class ReasoningCaptureLiteLLMModel(LiteLLMModel):
    """LiteLLMModel wrapper that keeps provider reasoning metadata for transcripts."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.reasoning_events: list[dict[str, Any]] = []

    def _capture_reasoning(self, response: Any) -> None:
        fields = _extract_reasoning_fields(response)
        if not fields:
            return
        self.reasoning_events.append(
            {
                "type": "model_reasoning",
                "index": len(self.reasoning_events) + 1,
                "source": "LiteLLMModel response",
                **fields,
            }
        )

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        response = super().__call__(*args, **kwargs)
        self._capture_reasoning(response)
        return response

    def generate(self, *args: Any, **kwargs: Any) -> Any:
        response = super().generate(*args, **kwargs)
        self._capture_reasoning(response)
        return response


def _to_text(agent_result: object) -> str:
    if isinstance(agent_result, str):
        return agent_result
    if hasattr(agent_result, "final_answer"):
        return str(getattr(agent_result, "final_answer"))
    return str(agent_result)


def run_single_scenario(
    config: ExperimentConfig,
    scenario: Scenario,
    documents,
    system_prompt_override: str | None = None,
    user_prompt_override: str | None = None,
    extra_user_instructions: list[str] | None = None,
) -> RunResult:
    state = build_env(documents)
    tools = build_tools(state, toolset=config.toolset)

    model = ReasoningCaptureLiteLLMModel(
        model_id=config.model_id,
        temperature=config.temperature,
    )
        
    agent = ToolCallingAgent(
        tools=tools,
        model=model,
        max_steps=config.max_steps,
    )

    system_prompt = system_prompt_override or build_system_prompt(config, scenario)
    user_prompt = user_prompt_override or build_user_message(
        scenario,
        config,
        documents,
        extra_instructions=extra_user_instructions,
    )
    full_prompt = f"{system_prompt}\n\n{user_prompt}"

    response = agent.run(full_prompt)
    response_text = _to_text(response)

    summary_file = state.vfs.files.get(SUMMARY_PATH)
    summary_content = summary_file.content if summary_file else None

    metrics = compute_metrics(
        response_text=response_text,
        summary_content=summary_content,
        tool_logs=state.tool_logs,
        sent_emails=state.sent_emails,
        source_documents=documents,
    )

    tool_counts: dict[str, int] = {}
    for log in state.tool_logs:
        tool_counts[log.name] = tool_counts.get(log.name, 0) + 1

    transcript: list[dict[str, object]] = [
        {"type": "system_prompt", "content": system_prompt},
        {"type": "user_prompt", "content": user_prompt},
    ]
    transcript.extend(model.reasoning_events)
    write_file_idx = 0
    send_email_idx = 0
    for idx, log in enumerate(state.tool_logs, start=1):
        event: dict[str, object] = {
            "type": "tool_call",
            "index": idx,
            "tool_name": log.name,
        }
        payload: dict[str, object] = dict(log.args)
        if log.name == "write_file":
            payload = (
                state.written_files[write_file_idx]
                if write_file_idx < len(state.written_files)
                else {"path": str(log.args.get("path", "")), "content": ""}
            )
            write_file_idx += 1
        elif log.name == "send_email":
            email_payload = (
                state.sent_emails[send_email_idx]
                if send_email_idx < len(state.sent_emails)
                else {
                    "to": str(log.args.get("to", "")),
                    "subject": str(log.args.get("subject", "")),
                    "body": "",
                }
            )
            payload = {
                "to": str(email_payload.get("to", "")),
                "subject": str(email_payload.get("subject", "")),
                "body": str(email_payload.get("body", "")),
                "attachments": list(email_payload.get("attachments") or []),
            }
            send_email_idx += 1
        event["payload"] = payload
        transcript.append(event)
    if summary_content is not None:
        transcript.append(
            {
                "type": "summary_file",
                "path": SUMMARY_PATH,
                "content": summary_content,
            }
        )
    transcript.append({"type": "final_response", "content": response_text})

    return RunResult(
        dataset=scenario.dataset,
        scenario_id=scenario.scenario_id,
        task=config.task,
        model_id=config.model_id,
        response_text=response_text,
        summary_path=SUMMARY_PATH if summary_content is not None else None,
        summary_content=summary_content,
        sent_email_count=len(state.sent_emails),
        tool_calls=len(state.tool_logs),
        tool_counts=tool_counts,
        transcript=transcript,
        transcript_path=None,
        metrics=metrics,
        config=asdict(config),
    )
