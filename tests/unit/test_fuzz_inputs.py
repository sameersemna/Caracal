"""Property-based hardening of the parsing and sanitising boundaries (card T4.3).

Every suite is bounded (``max_examples`` well under 200), deterministic (``derandomize=True``)
and database-free, so the gate stays under 30 seconds. The properties are the safety
invariants, not exact outputs: hostile input may be rejected, but it must never escape a
directory, resolve to a private address, crash the parser or turn report prose into markup.
"""

from __future__ import annotations

import asyncio
import ipaddress
import re
import string
from pathlib import Path

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from tests.fakes.resolver import FakeResolver

from caracal.asr import Segment, Transcript, Word, join_words
from caracal.asr.captions import CaptionWord, parse_vtt
from caracal.errors import CaracalError, ErrorCode
from caracal.ingest.ssrf import default_ip_policy, resolve_and_check
from caracal.ingest.upload import sanitize_filename
from caracal.llm.jsonx import extract_json
from caracal.llm.schema import coerce, prepare_schema
from caracal.report.render_md import md_escape
from caracal.subtitles.segmenter import cues_from_transcript

_FUZZ = settings(
    max_examples=80,
    derandomize=True,
    database=None,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)

_PRIVATE_HOSTS = st.sampled_from(
    [
        "127.0.0.1",
        "10.0.0.1",
        "172.16.5.4",
        "192.168.1.1",
        "169.254.1.1",
        "100.64.0.1",
        "224.0.0.1",
        "255.255.255.255",
        "0",
        "2130706433",
        "0177.0.0.1",
        "0x7f.1",
        "::1",
        "fc00::1",
        "fe80::1",
        "::ffff:127.0.0.1",
        "64:ff9b::7f00:1",
    ]
)

_PRIVATE_ADDRESSES = st.sampled_from(
    [
        "127.0.0.1",
        "10.20.30.40",
        "172.31.0.1",
        "192.168.0.10",
        "169.254.10.10",
        "100.64.1.1",
        "0.0.0.0",  # noqa: S104 - asserting the policy rejects it, not binding a socket
        "224.0.0.1",
        "::1",
        "fc00::abcd",
        "fe80::1",
        "::ffff:10.0.0.1",
    ]
)

_HOSTNAMES = st.from_regex(r"[a-z][a-z0-9-]{0,20}\.(?:com|net|org)", fullmatch=True)

_DANGEROUS_TEXT = st.one_of(
    st.text(max_size=200),
    st.sampled_from(
        [
            "<script>alert(1)</script>",
            "[label](http://evil.example/path)",
            '<a href="https://evil.example">click</a>',
            "![image](http://evil.example/x.png)",
            "`code` **bold** _italic_ # heading | table |",
        ]
    ),
)


@_FUZZ
@given(host=_PRIVATE_HOSTS)
def test_literal_private_ip_forms_are_never_allowed(host: str) -> None:
    with pytest.raises(CaracalError) as excinfo:
        asyncio.run(resolve_and_check(f"http://{host}/media", resolver=FakeResolver()))

    assert excinfo.value.code is ErrorCode.SOURCE_BLOCKED


@_FUZZ
@given(host=_HOSTNAMES, address=_PRIVATE_ADDRESSES)
def test_a_hostname_resolving_to_a_private_address_is_never_allowed(host: str, address: str) -> None:
    resolver = FakeResolver({host: [address]})

    with pytest.raises(CaracalError) as excinfo:
        asyncio.run(resolve_and_check(f"https://{host}/a", resolver=resolver))

    assert excinfo.value.code is ErrorCode.SOURCE_BLOCKED


@_FUZZ
@given(
    url=st.one_of(
        st.text(max_size=200),
        st.builds(
            lambda scheme, host, path: f"{scheme}://{host}/{path}",
            st.sampled_from(["http", "https", "file", "ftp", "HTTP", ""]),
            st.text(max_size=60),
            st.text(max_size=60),
        ),
    )
)
def test_arbitrary_urls_only_ever_raise_a_caracal_error(url: str) -> None:
    try:
        target = asyncio.run(resolve_and_check(url, resolver=FakeResolver()))
    except CaracalError:
        return
    address = ipaddress.ip_address(target.ip)
    assert default_ip_policy(address)


@_FUZZ
@given(name=st.text(max_size=200))
def test_sanitize_filename_never_escapes_the_upload_directory(name: str) -> None:
    safe = sanitize_filename(name)
    root = Path("/var/lib/caracal/uploads/0123456789abcdef")

    assert re.fullmatch(r"[A-Za-z0-9._-]{1,100}", safe), safe
    assert safe not in {".", ".."}
    assert (root / safe).resolve().parent == root.resolve()


@_FUZZ
@given(text=st.text(max_size=300))
def test_extract_json_only_raises_caracal_errors(text: str) -> None:
    try:
        value = extract_json(text)
    except CaracalError as exc:
        assert exc.code is ErrorCode.INTERNAL
        return
    assert isinstance(value, dict)


class _FuzzPayload(BaseModel):
    """A small model with every coercion-sensitive shape."""

    model_config = ConfigDict(extra="forbid")

    kind: str = "a"
    score: float = Field(default=0.0, ge=0.0, le=1.0)
    tags: list[str] = Field(default_factory=list)
    note: str | None = None


_JSON_VALUES: st.SearchStrategy[object] = st.recursive(
    st.none()
    | st.booleans()
    | st.integers(min_value=-(10**6), max_value=10**6)
    | st.floats(allow_nan=False, allow_infinity=False, width=32)
    | st.text(max_size=20),
    lambda children: (
        st.lists(children, max_size=4) | st.dictionaries(st.text(max_size=8), children, max_size=4)
    ),
    max_leaves=20,
)


@_FUZZ
@given(value=_JSON_VALUES)
def test_coerce_and_validate_only_ever_raise_validation_errors(value: object) -> None:
    prepared = prepare_schema(_FuzzPayload)
    coerced = coerce(value, prepared)
    try:
        _FuzzPayload.model_validate(coerced)
    except ValidationError:
        pass


@_FUZZ
@given(text=_DANGEROUS_TEXT)
def test_md_escape_never_emits_markup_or_live_links(text: str) -> None:
    escaped = md_escape(text)

    assert "<script" not in escaped.lower()
    assert "](http" not in escaped
    assert "<a " not in escaped.lower()


_VTT_TEXT = st.one_of(
    st.text(max_size=500),
    st.builds(
        lambda start, end, body: f"WEBVTT\n\n00:00:0{start}.000 --> 00:00:0{end}.500\n{body}\n",
        st.integers(min_value=0, max_value=9),
        st.integers(min_value=1, max_value=9),
        st.text(max_size=80),
    ),
)


@_FUZZ
@given(text=_VTT_TEXT)
def test_parse_vtt_never_raises_on_arbitrary_text(text: str) -> None:
    words = parse_vtt(text)

    assert isinstance(words, list)
    assert all(isinstance(word, CaptionWord) for word in words)
    assert all(word.start <= word.end for word in words)


_WORD_TEXTS = st.one_of(
    st.text(alphabet=string.ascii_letters, min_size=1, max_size=12),
    st.sampled_from(
        [
            "well,",
            "end.",
            "really?",
            "supercalifragilisticexpialidocious",
            "extraordinarily-long-unbroken-token-without-any-spaces",
        ]
    ),
)


@st.composite
def _transcripts(draw: st.DrawFn) -> Transcript:
    count = draw(st.integers(min_value=0, max_value=20))
    words: list[Word] = []
    clock = 0.0
    for _ in range(count):
        duration = draw(st.floats(min_value=0.05, max_value=1.5))
        text = draw(_WORD_TEXTS).strip() or "x"
        speaker = draw(st.sampled_from([None, "S1", "S2"]))
        words.append(
            Word(
                text=text,
                start=clock,
                end=clock + duration,
                prob=0.9,
                speaker=speaker,
                space_before=True,
            )
        )
        clock += duration + draw(st.floats(min_value=0.0, max_value=0.6))
    segments = (
        (
            Segment(
                id=0,
                start=words[0].start,
                end=words[-1].end,
                text=" ".join(word.text for word in words),
                words=tuple(words),
                avg_logprob=-0.1,
                no_speech_prob=0.0,
                lang="en",
            ),
        )
        if words
        else ()
    )
    return Transcript(segments=segments, language_code="en", dropped={})


@_FUZZ
@given(transcript=_transcripts())
def test_subtitle_segmentation_invariants(transcript: Transcript) -> None:
    words = transcript.segments[0].words if transcript.segments else ()
    cues = cues_from_transcript(transcript)

    if not words:
        assert cues == []
        return

    assert cues
    previous_end = float("-inf")
    for index, cue in enumerate(cues):
        first, last = cue.word_span
        assert cue.index == index + 1
        assert 0 <= first <= last < len(words)
        assert cue.start <= cue.end
        assert cue.end - cue.start <= 7.0 + 1e-9
        assert cue.start >= previous_end - 1e-9
        previous_end = cue.end
        assert 1 <= len(cue.lines) <= 2
        cue_words = words[first : last + 1]
        for line in cue.lines:
            assert line.strip()
            if len(line) > 42:
                assert any(len(word.text) > 42 for word in cue_words)
        if index == 0:
            assert first == 0
        else:
            assert first == cues[index - 1].word_span[1] + 1
        assert join_words(cue_words).strip()

    assert cues[-1].word_span[1] == len(words) - 1
