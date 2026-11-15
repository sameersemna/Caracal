"""Ollama chat client with tool-call and JSON modes, coercion, repair and per-task routing.

The measured gemma4 behaviour drives the design. Small schemas use the tool-call protocol:
one tool definition with an inlined schema, then a coercion pass and pydantic validation.
Big multi-section schemas use JSON mode (``format: "json"`` with the prepared schema in the
system prompt), because tool calls there validate but are never complete and sometimes come
back empty. An empty tool reply automatically retries the call in JSON mode.

Every call names a :class:`~caracal.config.Route`; the route's ordered chain
(``LLM_ROUTE_<TASK>``, docs/02) is tried ref by ref. Within one ref the transport retries
and the repair attempts apply; a ref is abandoned after two consecutive transport/5xx/timeout
failures, on a 401/403, on a 404 model-not-found, or when its structured output is still
invalid after the repairs — then the next ref of the chain answers. A local ref gets a
cold-start timeout (``max(LLM_TIMEOUT_S, 240)``) unless it is the first ref. Prompts are
never logged at INFO.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, TypeVar, cast

import httpx
from pydantic import BaseModel, ValidationError

from caracal.config import ModelRef, Route, Settings
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

    async def incr_llm_call(
        self,
        endpoint: str,
        mode: str,
        fallback: bool,
        *,
        route: str = ...,
        model: str = ...,
    ) -> None: ...


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
_MAX_REF_FAILURES = 2
_REPAIR_ERRORS = 5
_DEBUG_CHARS = 500
#: A local model may be cold (measured 21-43 s to load); the chain's fallback gets this.
_LOCAL_COLD_START_TIMEOUT_S = 240.0

#: Model refs whose 401/403 was already logged (``llm.auth_failed`` is once per ref).
_auth_failed_refs: set[str] = set()


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
    route: Route = Route.DEFAULT
    model_ref: str = ""
    refs_tried: int = 1


@dataclass(frozen=True)
class ToolResult[T: BaseModel]:
    """A validated model plus the call metadata."""

    value: T
    meta: CallMeta


@dataclass(frozen=True)
class _Endpoint:
    """One model ref resolved against the settings (URL, key and timeout included)."""

    kind: EndpointKind
    url: str
    model: str
    api_key: str | None
    ref: ModelRef
    timeout_s: float


@dataclass
class _CallState:
    """Per-ref request accounting; ``fallback_used`` means a later ref of the chain answers."""

    endpoint: _Endpoint
    fallback_used: bool = False
    requests: int = 0
    failures: int = 0


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


class _ModelNotFound(Exception):
    def __init__(self, error: CaracalError) -> None:
        super().__init__(error.message)
        self.error = error


class _Fatal(Exception):
    def __init__(self, error: CaracalError) -> None:
        super().__init__(error.message)
        self.error = error


class _RefFailed(Exception):
    """One model ref is exhausted; the route's next ref is tried."""

    def __init__(self, error: CaracalError) -> None:
        super().__init__(error.message)
        self.error = error


def _log_auth_failed_once(model_ref: str, status: int) -> None:
    if model_ref in _auth_failed_refs:
        return
    _auth_failed_refs.add(model_ref)
    logger.warning("llm.auth_failed", extra={"model_ref": model_ref, "status": status})


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

    def _chain(self, route: Route) -> tuple[_Endpoint, ...]:
        """Resolve the route's refs: first ref keeps ``LLM_TIMEOUT_S``, later locals get 240 s."""
        endpoints: list[_Endpoint] = []
        for index, ref in enumerate(self._settings.llm_route_chain(route)):
            timeout = float(self._settings.LLM_TIMEOUT_S)
            if index > 0 and ref.endpoint == "local":
                timeout = max(timeout, _LOCAL_COLD_START_TIMEOUT_S)
            endpoints.append(self._endpoint(ref, timeout))
        return tuple(endpoints)

    def _endpoint(self, ref: ModelRef, timeout_s: float) -> _Endpoint:
        if ref.endpoint == "cloud":
            api_key = self._settings.OLLAMA_CLOUD_API_KEY
            if api_key is None:  # Settings rejects a cloud ref without a key at startup
                raise CaracalError(
                    ErrorCode.INVALID_INPUT,
                    "a cloud model reference needs OLLAMA_CLOUD_API_KEY",
                    {"model": ref.normalised},
                )
            return _Endpoint(
                kind="cloud",
                url=self._settings.OLLAMA_CLOUD_URL,
                model=ref.name,
                api_key=api_key.get_secret_value(),
                ref=ref,
                timeout_s=timeout_s,
            )
        return _Endpoint(
            kind="local",
            url=self._settings.OLLAMA_LOCAL_URL,
            model=ref.name,
            api_key=None,
            ref=ref,
            timeout_s=timeout_s,
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
        route: Route = Route.DEFAULT,
    ) -> ToolResult[ModelT]:
        """Call ``tool_name`` once and return a validated ``model_cls`` instance.

        ``mode="json"`` sends ``format: "json"`` with the prepared schema in the system prompt
        instead of a tool definition (big multi-section schemas; docs/04 §1). In tool mode an
        empty reply — no tool call and no text — retries the same attempt in JSON mode.
        ``route`` selects the ordered model chain (docs/02 ``LLM_ROUTE_*``).
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
        last: CaracalError | None = None
        refs_tried = 0
        attempts = 0
        for endpoint in self._chain(route):
            refs_tried += 1
            state = _CallState(endpoint=endpoint, fallback_used=refs_tried > 1)
            if refs_tried > 1:
                logger.warning(
                    "llm.fallback",
                    extra={"route": route.value, "model_ref": endpoint.ref.normalised},
                )
            try:
                value, coerced, mode_used = await self._tool_attempt(
                    state,
                    system=system,
                    prompt=prompt,
                    prepared=prepared,
                    synonyms=merged_synonyms,
                    model_cls=model_cls,
                    tool_name=tool_name,
                    tools=tools,
                    initial_mode=mode,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            except _RefFailed as failed:
                attempts += state.requests
                last = failed.error
                continue

            attempts += state.requests
            duration = time.monotonic() - started
            logger.info(
                "llm.call",
                extra={
                    "route": route.value,
                    "model": state.endpoint.model,
                    "model_ref": state.endpoint.ref.normalised,
                    "endpoint": state.endpoint.kind,
                    "attempts": attempts,
                    "coerced": coerced,
                    "fallback_used": state.fallback_used,
                    "refs_tried": refs_tried,
                    "mode": mode_used,
                    "duration_s": round(duration, 3),
                },
            )
            await self._record_call(state, mode_used, route)
            return ToolResult(
                value=value,
                meta=CallMeta(
                    model=state.endpoint.model,
                    endpoint=state.endpoint.kind,
                    attempts=attempts,
                    coerced=coerced,
                    fallback_used=state.fallback_used,
                    duration_s=duration,
                    mode_used=mode_used,
                    route=route,
                    model_ref=state.endpoint.ref.normalised,
                    refs_tried=refs_tried,
                ),
            )
        raise self._chain_exhausted(route, refs_tried, last) from None

    async def chat(
        self,
        *,
        prompt: str,
        system: str,
        temperature: float = 0.3,
        max_tokens: int = 2048,
        route: Route = Route.DEFAULT,
    ) -> str:
        """Free-text chat completion (no tools, no validation) along the route's chain."""
        started = time.monotonic()
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]
        last: CaracalError | None = None
        refs_tried = 0
        for endpoint in self._chain(route):
            refs_tried += 1
            state = _CallState(endpoint=endpoint, fallback_used=refs_tried > 1)
            if refs_tried > 1:
                logger.warning(
                    "llm.fallback",
                    extra={"route": route.value, "model_ref": endpoint.ref.normalised},
                )
            try:
                message = await self._exchange(
                    state,
                    messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    tools=None,
                )
            except _RefFailed as failed:
                last = failed.error
                continue
            content = message.get("content")
            if not isinstance(content, str) or not content.strip():
                last = CaracalError(
                    ErrorCode.INTERNAL,
                    "LLM chat reply has no text content",
                    {"model_ref": endpoint.ref.normalised},
                    retryable=True,
                )
                continue
            duration = time.monotonic() - started
            logger.info(
                "llm.chat",
                extra={
                    "route": route.value,
                    "model": state.endpoint.model,
                    "model_ref": state.endpoint.ref.normalised,
                    "endpoint": state.endpoint.kind,
                    "attempts": state.requests,
                    "fallback_used": state.fallback_used,
                    "refs_tried": refs_tried,
                    "duration_s": round(duration, 3),
                },
            )
            await self._record_call(state, "chat", route)
            return content
        raise self._chain_exhausted(route, refs_tried, last) from None

    @staticmethod
    def _chain_exhausted(route: Route, refs_tried: int, last: CaracalError | None) -> CaracalError:
        """The error of the last ref, re-labelled with the route that ran out of models."""
        details: dict[str, Any] = {"route": route.value, "refs_tried": refs_tried}
        if last is not None:
            details["error"] = last.message
            return CaracalError(
                last.code,
                f"every model of route {route.value} failed: {last.message}",
                details,
                retryable=True,
            )
        return CaracalError(
            ErrorCode.DEPENDENCY_DOWN,
            f"route {route.value} has no model to call",
            details,
            retryable=True,
        )

    async def _tool_attempt(
        self,
        state: _CallState,
        *,
        system: str,
        prompt: str,
        prepared: Mapping[str, Any],
        synonyms: Mapping[str, str],
        model_cls: type[ModelT],
        tool_name: str,
        tools: list[dict[str, Any]] | None,
        initial_mode: Mode,
        temperature: float,
        max_tokens: int,
    ) -> tuple[ModelT, bool, Mode]:
        """Run the tool/JSON protocol against one ref; raise ``_RefFailed`` when it is done."""
        current_mode: Mode = initial_mode
        current_tools = tools
        messages = _messages(
            system,
            prompt,
            mode=current_mode,
            prepared=prepared,
            tool_name=tool_name,
        )
        repairs = 0
        while True:
            message = await self._exchange(
                state,
                messages,
                temperature=temperature,
                max_tokens=max_tokens,
                tools=current_tools,
                json_mode=current_mode == "json",
            )
            if current_mode == "tool" and _is_empty_reply(message):
                # Ollama drops a malformed tool call; the same attempt continues in JSON mode
                # (docs/04 §1: "empty tool reply → next attempt json mode").
                current_mode = "json"
                current_tools = None
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
                    synonyms=synonyms,
                    mode=current_mode,
                )
            except _InvalidOutput as invalid:
                if repairs >= _MAX_REPAIRS:
                    raise _RefFailed(
                        CaracalError(
                            ErrorCode.INTERNAL,
                            "LLM did not return valid structured output",
                            {"errors": invalid.errors[:_REPAIR_ERRORS], "mode": current_mode},
                            retryable=True,
                        )
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
            return value, coerced, current_mode

    async def _record_call(self, state: _CallState, mode: str, route: Route) -> None:
        """Increment ``caracal_llm_calls_total``; metrics must never fail an LLM call."""
        if self._metrics is None:
            return
        try:
            await self._metrics.incr_llm_call(
                state.endpoint.kind,
                mode,
                state.fallback_used,
                route=route.value,
                model=state.endpoint.model,
            )
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
        """One logical exchange against ``state.endpoint``; transport retries happen here.

        Raises ``_RefFailed`` when this ref is exhausted: two consecutive transport/5xx/timeout
        failures, an auth rejection or a 404 model-not-found.
        """
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
                _log_auth_failed_once(state.endpoint.ref.normalised, exc.status)
                raise _RefFailed(
                    CaracalError(
                        ErrorCode.DEPENDENCY_DOWN,
                        "Ollama rejected the request",
                        {"endpoint": state.endpoint.kind, "model_ref": state.endpoint.ref.normalised},
                        retryable=True,
                    )
                ) from None
            except _ModelNotFound as exc:
                state.requests += 1
                raise _RefFailed(exc.error) from None
            except _Retryable as exc:
                state.requests += 1
                state.failures += 1
                if state.failures >= _MAX_REF_FAILURES:
                    await self._sleep(_BACKOFF_SWITCH)
                    raise _RefFailed(exc.error) from None
                await self._sleep(_BACKOFF_PRIMARY)
                continue
            except _Fatal as exc:
                # A 400/other 4xx or a malformed body is not model-specific: fail the call.
                state.requests += 1
                raise exc.error from None
            state.requests += 1
            state.failures = 0
            return message

    def _body(
        self,
        endpoint: _Endpoint,
        messages: Sequence[Mapping[str, Any]],
        *,
        temperature: float,
        max_tokens: int,
        tools: list[dict[str, Any]] | None,
        json_mode: bool = False,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": endpoint.model,
            "messages": list(messages),
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
                    timeout=endpoint.timeout_s,
                )
            except httpx.TimeoutException as exc:
                raise _Retryable(
                    CaracalError(
                        ErrorCode.DEPENDENCY_DOWN,
                        "Ollama request timed out",
                        {
                            "timeout_s": endpoint.timeout_s,
                            "model_ref": endpoint.ref.normalised,
                        },
                        retryable=True,
                    )
                ) from exc
            except httpx.TransportError as exc:
                raise _Retryable(
                    CaracalError(
                        ErrorCode.DEPENDENCY_DOWN,
                        "Ollama is unreachable",
                        {"error": type(exc).__name__, "model_ref": endpoint.ref.normalised},
                        retryable=True,
                    )
                ) from exc

        if response.status_code == 404:
            raise _ModelNotFound(
                CaracalError(
                    ErrorCode.INTERNAL,
                    f"model {endpoint.model!r} is not available on the {endpoint.kind} endpoint",
                    {"status": 404, "model_ref": endpoint.ref.normalised},
                )
            )
        if response.status_code in (401, 403):
            raise _AuthFailed(response.status_code)
        if response.status_code >= 500:
            raise _Retryable(
                CaracalError(
                    ErrorCode.DEPENDENCY_DOWN,
                    f"Ollama failed with HTTP {response.status_code}",
                    {"status": response.status_code, "model_ref": endpoint.ref.normalised},
                    retryable=True,
                )
            )
        if response.status_code >= 400:
            raise _Fatal(
                CaracalError(
                    ErrorCode.INTERNAL,
                    f"Ollama rejected the request with HTTP {response.status_code}",
                    {"status": response.status_code, "model_ref": endpoint.ref.normalised},
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


__all__ = ["CallMeta", "LLMClient", "LlmMetrics", "Mode", "Route", "ToolResult"]
