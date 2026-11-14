"""Ollama chat client with tool-call protocol, coercion, repair and fallback (docs/04 §1).

The measured gemma4 behaviour drives the design: ``format`` is ignored, so every structured
call goes through a single tool definition with an inlined schema, then a coercion pass and
pydantic validation. Two consecutive primary failures (or an auth rejection) move the call
to the local fallback model. Prompts are never logged at INFO.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, TypeVar, cast

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

_TOOL_SUFFIX = (
    "You MUST answer by calling the tool {tool_name} exactly once. "
    "Use the exact field names and allowed values from the tool schema."
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


def _repair_message(errors: list[dict[str, Any]], tool_name: str) -> str:
    summary = [
        {"loc": list(error.get("loc", ())), "msg": str(error.get("msg", ""))}
        for error in errors[:_REPAIR_ERRORS]
    ]
    return (
        f"Your previous answer was rejected. Call the tool {tool_name} exactly once with corrected "
        f"arguments. Validation errors: {json.dumps(summary, ensure_ascii=False)}"
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
    ) -> None:
        self._settings = settings
        self._http = http
        self._sleep = sleep
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
    ) -> ToolResult[ModelT]:
        """Call ``tool_name`` once and return a validated ``model_cls`` instance."""
        started = time.monotonic()
        prepared = prepare_schema(model_cls)
        merged_synonyms = {**DEFAULT_SYNONYMS, **dict(synonyms or {})}
        tools = [
            {
                "type": "function",
                "function": {
                    "name": tool_name,
                    "description": tool_description,
                    "parameters": prepared,
                },
            }
        ]
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": _system_prompt(system, tool_name)},
            {"role": "user", "content": prompt},
        ]
        state = _CallState(endpoint=self._primary)
        repairs = 0
        while True:
            message = await self._exchange(
                state,
                messages,
                temperature=temperature,
                max_tokens=max_tokens,
                tools=tools,
            )
            try:
                value, coerced = self._interpret(
                    message,
                    model_cls=model_cls,
                    prepared=prepared,
                    synonyms=merged_synonyms,
                )
            except _InvalidOutput as invalid:
                if repairs >= _MAX_REPAIRS:
                    raise CaracalError(
                        ErrorCode.INTERNAL,
                        "LLM did not return valid structured output",
                        {"errors": invalid.errors[:_REPAIR_ERRORS]},
                        retryable=True,
                    ) from None
                repairs += 1
                messages.append(dict(message))
                messages.append({"role": "tool", "content": _repair_message(invalid.errors, tool_name)})
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
                    "duration_s": round(duration, 3),
                },
            )
            return ToolResult(
                value=value,
                meta=CallMeta(
                    model=state.endpoint.model,
                    endpoint=state.endpoint.kind,
                    attempts=state.requests,
                    coerced=coerced,
                    fallback_used=state.fallback_used,
                    duration_s=duration,
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
        return content

    def _interpret(
        self,
        message: Mapping[str, Any],
        *,
        model_cls: type[ModelT],
        prepared: Mapping[str, Any],
        synonyms: Mapping[str, str],
    ) -> tuple[ModelT, bool]:
        function = _first_tool_call(message)
        arguments: object
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
            value = model_cls.model_validate(coerced_value)
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
    ) -> dict[str, Any]:
        """One logical exchange; retries, auth fallback and model fallback happen here."""
        while True:
            body = self._body(
                state.endpoint,
                messages,
                temperature=temperature,
                max_tokens=max_tokens,
                tools=tools,
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
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": endpoint.model,
            "messages": messages,
            "stream": False,
            "keep_alive": self._settings.LLM_KEEP_ALIVE,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        if tools is not None:
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


__all__ = ["CallMeta", "LLMClient", "ToolResult"]
