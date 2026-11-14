"""Ollama chat client with tool-call and JSON modes, coercion, repair and fallback (docs/04 §1).

The measured gemma4 behaviour drives the design. Small schemas use the tool-call protocol:
one tool definition with an inlined schema, then a coercion pass and pydantic validation.
Big multi-section schemas use JSON mode (``format: "json"`` with the prepared schema in the
system prompt), because tool calls there validate but are never complete and sometimes come
back empty. An empty tool reply automatically retries the call in JSON mode. Two consecutive
primary failures (or an auth rejection) move the call to the local fallback model. Prompts are
never logged at INFO.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, TypeVar, cast

import httpx
from pydantic import BaseModel, ValidationError

from caracal.config import Settings
from caracal.errors import CaracalError, ErrorCode
from caracal.llm.jsonx import extract_json
from caracal.llm.schema import DEFAULT_SYNONYMS, coerce, prepare_schema
from caracal.log import get_logger, redact

logger = get_logger("caracal.llm.client")

ModelT = TypeVar("ModelT", bound=BaseModel)
EndpointKind = Literal["cloud", "local"]
Mode = Literal["tool", "json"]


class LlmMetrics(Protocol):
    """The one metric the client records per call (``caracal_llm_calls_total``)."""

    async def incr_llm_call(self, endpoint: str, mode: str, fallback: bool) -> None: ...


_TOOL_SUFFIX = (
    "You MUST answer by calling the tool {tool_name} exactly once. "
    "Use the exact field names and allowed values from the tool schema."
)
_JSON_SUFFIX = (
    "Reply with ONE JSON object only, no prose, matching this JSON Schema exactly "
    "(field names and allowed values):\n{schema}"
)
_BACKOFF_PRIMARY = 1.0
_BACKOFF_SWITCH = 4.0
_MAX_REPAIRS = 2
_MAX_PRIMARY_FAILURES = 2
_MAX_FALLBACK_FAILURES = 2
_REPAIR_ERRORS = 5
_DEBUG_CHARS = 500

_auth_failed_logged = False


@dataclass(frozen=True)
class CallMeta:
    """Metadata about how one structured call was answered."""

    model: str
    endpoint: EndpointKind
    attempts: int
    coerced: bool
    fallback_used: bool
    duration_s: float
    mode_used: Mode


@dataclass(frozen=True)
class ToolResult[T: BaseModel]:
    """A validated model plus the call metadata."""

    value: T
    meta: CallMeta


@dataclass(frozen=True)
class _Endpoint:
    kind: EndpointKind
    url: str
    model: str
    api_key: str | None


@dataclass
class _CallState:
    endpoint: _Endpoint
    fallback_used: bool = False
    requests: int = 0
    primary_failures: int = 0
    fallback_failures: int = 0


@dataclass
class _InvalidOutput(Exception):
    errors: list[dict[str, Any]] = field(default_factory=list)


class _Retryable(Exception):
    def __init__(self, error: CaracalError) -> None:
        super().__init__(error.message)
        self.error = error


class _AuthFailed(Exception):
    def __init__(self, status: int) -> None:
        super().__init__(f"auth rejected with HTTP {status}")
        self.status = status


class _Fatal(Exception):
    def __init__(self, error: CaracalError) -> None:
        super().__init__(error.message)
        self.error = error


def _log_auth_failed_once(status: int) -> None:
    global _auth_failed_logged
    if _auth_failed_logged:
        return
    _auth_failed_logged = True
    logger.warning("llm.auth_failed", extra={"status": status})


def _system_prompt(system: str, tool_name: str) -> str:
    suffix = _TOOL_SUFFIX.format(tool_name=tool_name)
    text = system.rstrip()
    if text.endswith(suffix):
        return text
    return f"{text}\n{suffix}" if text else suffix


def _json_system_prompt(system: str, prepared: Mapping[str, Any]) -> str:
    """The JSON-mode system prompt: the caller's text plus the prepared schema."""
    suffix = _JSON_SUFFIX.format(schema=json.dumps(prepared, ensure_ascii=False))
    text = system.rstrip()
    if text.endswith(suffix):
        return text
    return f"{text}\n{suffix}" if text else suffix


def _messages(
    system: str,
    prompt: str,
    *,
    mode: Mode,
    prepared: Mapping[str, Any],
    tool_name: str,
) -> list[dict[str, Any]]:
    system_text = (
        _system_prompt(system, tool_name) if mode == "tool" else _json_system_prompt(system, prepared)
    )
    return [
        {"role": "system", "content": system_text},
        {"role": "user", "content": prompt},
    ]


def _is_empty_reply(message: Mapping[str, Any]) -> bool:
    """A reply with neither a tool call nor text: Ollama dropped a malformed tool call."""
    if _first_tool_call(message) is not None:
        return False
    content = message.get("content")
    return not (isinstance(content, str) and content.strip())


def _repair_message(errors: list[dict[str, Any]], tool_name: str) -> str:
    summary = [
        {"loc": list(error.get("loc", ())), "msg": str(error.get("msg", ""))}
        for error in errors[:_REPAIR_ERRORS]
    ]
    return (
        f"Your previous answer was rejected. Call the tool {tool_name} exactly once with corrected "
        f"arguments. Validation errors: {json.dumps(summary, ensure_ascii=False)}"
    )


def _json_repair_message(errors: list[dict[str, Any]]) -> str:
    summary = [
        {"loc": list(error.get("loc", ())), "msg": str(error.get("msg", ""))}
        for error in errors[:_REPAIR_ERRORS]
    ]
    return (
        "Your previous answer was rejected. Reply with ONE JSON object only, no prose, matching "
        f"the JSON Schema exactly. Validation errors: {json.dumps(summary, ensure_ascii=False)}"
    )


def _first_tool_call(message: Mapping[str, Any]) -> Mapping[str, Any] | None:
    calls = message.get("tool_calls")
    if not isinstance(calls, list) or not calls:
        return None
    first = calls[0]
    if not isinstance(first, Mapping):
        return None
    function = first.get("function")
    return function if isinstance(function, Mapping) else None


def _validation_errors(exc: ValidationError) -> list[dict[str, Any]]:
    return [{"loc": list(error.get("loc", ())), "msg": str(error.get("msg", ""))} for error in exc.errors()]


class LLMClient:
    """Async Ollama client shared by every analysis stage."""

    def __init__(
        self,
        settings: Settings,
        http: httpx.AsyncClient,
        *,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        semaphore: asyncio.Semaphore | None = None,
        metrics: LlmMetrics | None = None,
    ) -> None:
        self._settings = settings
        self._http = http
        self._sleep = sleep
        self._metrics = metrics
        self._semaphore = (
            semaphore if semaphore is not None else asyncio.Semaphore(settings.CARACAL_LLM_CONCURRENCY)
        )
        self._primary = self._build_primary()
        self._fallback = _Endpoint(
            kind="local",
            url=settings.OLLAMA_LOCAL_URL,
            model=settings.LLM_FALLBACK_MODEL,
            api_key=None,
        )

    def _build_primary(self) -> _Endpoint:
        api_key = self._settings.OLLAMA_CLOUD_API_KEY
        if api_key is not None:
            return _Endpoint(
                kind="cloud",
                url=self._settings.OLLAMA_CLOUD_URL,
                model=self._settings.LLM_MODEL,
                api_key=api_key.get_secret_value(),
            )
        return _Endpoint(
            kind="local",
            url=self._settings.OLLAMA_LOCAL_URL,
            model=self._settings.LLM_LOCAL_PROXY_MODEL,
            api_key=None,
        )

    async def call_tool(
        self,
        *,
        prompt: str,
        system: str,
        model_cls: type[ModelT],
        tool_name: str,
        tool_description: str,
        temperature: float = 0.1,
        max_tokens: int = 4096,
        synonyms: Mapping[str, str] | None = None,
        mode: Mode = "tool",
    ) -> ToolResult[ModelT]:
        """Call ``tool_name`` once and return a validated ``model_cls`` instance.

        ``mode="json"`` sends ``format: "json"`` with the prepared schema in the system prompt
        instead of a tool definition (big multi-section schemas; docs/04 §1). In tool mode an
        empty reply — no tool call and no text — retries the same attempt in JSON mode.
        """
        started = time.monotonic()
        prepared = prepare_schema(model_cls)
        merged_synonyms = {**DEFAULT_SYNONYMS, **dict(synonyms or {})}
        tools: list[dict[str, Any]] | None = (
            [
                {
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "description": tool_description,
                        "parameters": prepared,
                    },
                }
            ]
            if mode == "tool"
            else None
        )
        current_mode: Mode = mode
        messages = _messages(
            system,
            prompt,
            mode=current_mode,
            prepared=prepared,
            tool_name=tool_name,
        )
        state = _CallState(endpoint=self._primary)
        repairs = 0
        while True:
            message = await self._exchange(
                state,
                messages,
                temperature=temperature,
                max_tokens=max_tokens,
                tools=tools,
                json_mode=current_mode == "json",
            )
            if current_mode == "tool" and _is_empty_reply(message):
                # Ollama drops a malformed tool call; the same attempt continues in JSON mode
                # (docs/04 §1: "empty tool reply → next attempt json mode").
                current_mode = "json"
                tools = None
                repairs = 0
                messages = _messages(
                    system,
                    prompt,
                    mode=current_mode,
                    prepared=prepared,
                    tool_name=tool_name,
                )
                continue
            try:
                value, coerced = self._interpret(
                    message,
                    model_cls=model_cls,
                    prepared=prepared,
                    synonyms=merged_synonyms,
                    mode=current_mode,
                )
            except _InvalidOutput as invalid:
                if repairs >= _MAX_REPAIRS:
                    raise CaracalError(
                        ErrorCode.INTERNAL,
                        "LLM did not return valid structured output",
                        {"errors": invalid.errors[:_REPAIR_ERRORS], "mode": current_mode},
                        retryable=True,
                    ) from None
                repairs += 1
                if current_mode == "tool":
                    messages.append(dict(message))
                    messages.append({"role": "tool", "content": _repair_message(invalid.errors, tool_name)})
                else:
                    content = message.get("content")
                    messages.append(
                        {
                            "role": "assistant",
                            "content": content if isinstance(content, str) else "",
                        }
                    )
                    messages.append({"role": "user", "content": _json_repair_message(invalid.errors)})
                continue

            duration = time.monotonic() - started
            logger.info(
                "llm.call",
                extra={
                    "model": state.endpoint.model,
                    "endpoint": state.endpoint.kind,
                    "attempts": state.requests,
                    "coerced": coerced,
                    "fallback_used": state.fallback_used,
                    "mode": current_mode,
                    "duration_s": round(duration, 3),
                },
            )
            await self._record_call(state, current_mode)
            return ToolResult(
                value=value,
                meta=CallMeta(
                    model=state.endpoint.model,
                    endpoint=state.endpoint.kind,
                    attempts=state.requests,
                    coerced=coerced,
                    fallback_used=state.fallback_used,
                    duration_s=duration,
                    mode_used=current_mode,
                ),
            )

    async def chat(
        self,
        *,
        prompt: str,
        system: str,
        temperature: float = 0.3,
        max_tokens: int = 2048,
    ) -> str:
        """Free-text chat completion (no tools, no validation)."""
        started = time.monotonic()
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]
        state = _CallState(endpoint=self._primary)
        message = await self._exchange(
            state,
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=None,
        )
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise CaracalError(
                ErrorCode.INTERNAL,
                "LLM chat reply has no text content",
                retryable=True,
            )
        duration = time.monotonic() - started
        logger.info(
            "llm.chat",
            extra={
                "model": state.endpoint.model,
                "endpoint": state.endpoint.kind,
                "attempts": state.requests,
                "fallback_used": state.fallback_used,
                "duration_s": round(duration, 3),
            },
        )
        await self._record_call(state, "chat")
        return content

    async def _record_call(self, state: _CallState, mode: str) -> None:
        """Increment ``caracal_llm_calls_total``; metrics must never fail an LLM call."""
        if self._metrics is None:
            return
        try:
            await self._metrics.incr_llm_call(state.endpoint.kind, mode, state.fallback_used)
        except Exception as exc:  # best-effort telemetry, like the worker's job metrics
            logger.warning("llm.metrics_failed", extra={"error": type(exc).__name__})

    def _interpret(
        self,
        message: Mapping[str, Any],
        *,
        model_cls: type[ModelT],
        prepared: Mapping[str, Any],
        synonyms: Mapping[str, str],
        mode: Mode,
    ) -> tuple[ModelT, bool]:
        arguments: object
        if mode == "json":
            content = message.get("content")
            if not isinstance(content, str) or not content.strip():
                raise _InvalidOutput([{"loc": [], "msg": "empty JSON-mode reply"}])
            try:
                arguments = extract_json(content)
            except CaracalError as exc:
                raise _InvalidOutput([{"loc": [], "msg": exc.message}]) from exc
        else:
            function = _first_tool_call(message)
            if function is not None:
                arguments = function.get("arguments")
                if isinstance(arguments, str):
                    try:
                        arguments = extract_json(arguments)
                    except CaracalError as exc:
                        raise _InvalidOutput([{"loc": [], "msg": exc.message}]) from exc
                elif not isinstance(arguments, Mapping):
                    raise _InvalidOutput([{"loc": [], "msg": "tool call has no arguments object"}])
            else:
                content = message.get("content")
                if not isinstance(content, str):
                    raise _InvalidOutput([{"loc": [], "msg": "no tool call and no text content"}])
                try:
                    arguments = extract_json(content)
                except CaracalError as exc:
                    raise _InvalidOutput([{"loc": [], "msg": exc.message}]) from exc

        original = arguments
        coerced_value = coerce(arguments, prepared, synonyms=synonyms)
        try:
            # Aliased fields arrive under their wire name (the LLM-facing name) and the
            # coercion pass only knows that name, so accept both the alias and the field
            # name (docs/04 §1: pydantic aliases are the escape hatch for reserved names).
            value = model_cls.model_validate(coerced_value, by_alias=True, by_name=True)
        except ValidationError as exc:
            raise _InvalidOutput(_validation_errors(exc)) from exc
        return value, coerced_value != original

    async def _exchange(
        self,
        state: _CallState,
        messages: list[dict[str, Any]],
        *,
        temperature: float,
        max_tokens: int,
        tools: list[dict[str, Any]] | None,
        json_mode: bool = False,
    ) -> dict[str, Any]:
        """One logical exchange; retries, auth fallback and model fallback happen here."""
        while True:
            body = self._body(
                state.endpoint,
                messages,
                temperature=temperature,
                max_tokens=max_tokens,
                tools=tools,
                json_mode=json_mode,
            )
            try:
                message = await self._post(state.endpoint, body)
            except _AuthFailed as exc:
                state.requests += 1
                if state.endpoint.kind == "cloud":
                    _log_auth_failed_once(exc.status)
                    self._switch_to_fallback(state)
                    continue
                raise CaracalError(
                    ErrorCode.DEPENDENCY_DOWN,
                    "Ollama rejected the request",
                    {"endpoint": state.endpoint.kind},
                    retryable=True,
                ) from None
            except _Retryable as exc:
                state.requests += 1
                await self._handle_failure(state, exc.error)
                continue
            except _Fatal as exc:
                state.requests += 1
                raise exc.error from None
            state.requests += 1
            state.primary_failures = 0
            state.fallback_failures = 0
            return message

    async def _handle_failure(self, state: _CallState, error: CaracalError) -> None:
        if state.endpoint.kind == "cloud":
            state.primary_failures += 1
            if state.primary_failures >= _MAX_PRIMARY_FAILURES:
                await self._sleep(_BACKOFF_SWITCH)
                self._switch_to_fallback(state)
                return
            await self._sleep(_BACKOFF_PRIMARY)
            return
        if not state.fallback_used:
            state.primary_failures += 1
            if state.primary_failures >= _MAX_PRIMARY_FAILURES:
                await self._sleep(_BACKOFF_SWITCH)
                self._switch_to_fallback(state)
                return
            await self._sleep(_BACKOFF_PRIMARY)
            return
        state.fallback_failures += 1
        if state.fallback_failures >= _MAX_FALLBACK_FAILURES:
            raise CaracalError(
                ErrorCode.DEPENDENCY_DOWN,
                "Ollama is unavailable on both the primary and fallback endpoints",
                {"error": error.message},
                retryable=True,
            ) from None
        await self._sleep(_BACKOFF_SWITCH)

    def _switch_to_fallback(self, state: _CallState) -> None:
        state.endpoint = self._fallback
        state.fallback_used = True
        logger.warning(
            "llm.fallback",
            extra={"model": self._fallback.model, "endpoint": self._fallback.kind},
        )

    def _body(
        self,
        endpoint: _Endpoint,
        messages: list[dict[str, Any]],
        *,
        temperature: float,
        max_tokens: int,
        tools: list[dict[str, Any]] | None,
        json_mode: bool = False,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": endpoint.model,
            "messages": messages,
            "stream": False,
            "keep_alive": self._settings.LLM_KEEP_ALIVE,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        if json_mode:
            body["format"] = "json"
        elif tools is not None:
            body["tools"] = tools
        return body

    async def _post(self, endpoint: _Endpoint, body: dict[str, Any]) -> dict[str, Any]:
        headers: dict[str, str] = {}
        if endpoint.api_key is not None:
            headers["Authorization"] = f"Bearer {endpoint.api_key}"
        logger.debug("llm.request %s", redact(json.dumps(body, ensure_ascii=False))[:_DEBUG_CHARS])
        async with self._semaphore:
            try:
                response = await self._http.post(
                    f"{endpoint.url}/api/chat",
                    json=body,
                    headers=headers,
                    timeout=float(self._settings.LLM_TIMEOUT_S),
                )
            except httpx.TimeoutException as exc:
                raise _Retryable(
                    CaracalError(
                        ErrorCode.DEPENDENCY_DOWN,
                        "Ollama request timed out",
                        {"timeout_s": self._settings.LLM_TIMEOUT_S},
                        retryable=True,
                    )
                ) from exc
            except httpx.TransportError as exc:
                raise _Retryable(
                    CaracalError(
                        ErrorCode.DEPENDENCY_DOWN,
                        "Ollama is unreachable",
                        {"error": type(exc).__name__},
                        retryable=True,
                    )
                ) from exc

        if response.status_code in (401, 403):
            raise _AuthFailed(response.status_code)
        if response.status_code >= 500:
            raise _Retryable(
                CaracalError(
                    ErrorCode.DEPENDENCY_DOWN,
                    f"Ollama failed with HTTP {response.status_code}",
                    {"status": response.status_code},
                    retryable=True,
                )
            )
        if response.status_code >= 400:
            raise _Fatal(
                CaracalError(
                    ErrorCode.INTERNAL,
                    f"Ollama rejected the request with HTTP {response.status_code}",
                    {"status": response.status_code},
                )
            )
        try:
            payload = response.json()
        except ValueError as exc:
            raise _Fatal(CaracalError(ErrorCode.INTERNAL, "Ollama returned a body that is not JSON")) from exc
        if not isinstance(payload, dict):
            raise _Fatal(CaracalError(ErrorCode.INTERNAL, "Ollama returned a non-object JSON body"))
        message = cast(dict[str, Any], payload).get("message")
        if not isinstance(message, dict):
            raise _Fatal(CaracalError(ErrorCode.INTERNAL, "Ollama response has no message object"))
        return cast(dict[str, Any], message)


__all__ = ["CallMeta", "LLMClient", "LlmMetrics", "Mode", "ToolResult"]
