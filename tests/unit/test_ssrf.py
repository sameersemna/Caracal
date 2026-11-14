"""SSRF guard: literal forms, IPv4-in-IPv6 unwrapping and resolver policy (docs/06 S3)."""

from __future__ import annotations

import ipaddress

import pytest
from tests.fakes.resolver import PUBLIC_IP, FakeResolver

from caracal.errors import CaracalError, ErrorCode
from caracal.ingest.ssrf import PinnedTarget, default_ip_policy, parse_ip, resolve_and_check


def _resolver() -> FakeResolver:
    resolver = FakeResolver()
    resolver.add("localhost", "127.0.0.1")
    resolver.add("example.com", PUBLIC_IP)
    resolver.add("mixed.example", PUBLIC_IP, "10.0.0.1")
    return resolver


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1/",
        "http://localhost/",
        "http://[::1]/",
        "http://[::ffff:127.0.0.1]/",
        "http://[64:ff9b::7f00:1]/",
        "http://2130706433/",
        "http://0177.0.0.1/",
        "http://0x7f.1/",
        "http://0.0.0.0/",
        "http://10.1.2.3/",
        "http://169.254.169.254/latest/meta-data",
        "http://100.64.0.1/",
        "http://[fe80::1]/",
        "http://user:pw@example.com/",
        "ftp://example.com/",
        "http://mixed.example/",
    ],
)
async def test_blocked_destinations(url: str) -> None:
    with pytest.raises(CaracalError) as excinfo:
        await resolve_and_check(url, resolver=_resolver())
    assert excinfo.value.code is ErrorCode.SOURCE_BLOCKED


async def test_public_url_is_allowed_and_pinned() -> None:
    resolver = _resolver()
    target = await resolve_and_check("https://example.com/a.mp3", resolver=resolver)
    assert target == PinnedTarget(scheme="https", host="example.com", port=443, ip=PUBLIC_IP)
    assert resolver.calls == [("example.com", 443)]


async def test_http_defaults_to_port_80() -> None:
    target = await resolve_and_check("http://example.com/x", resolver=_resolver())
    assert target.port == 80
    assert target.scheme == "http"


async def test_dns_failure_is_source_unavailable() -> None:
    with pytest.raises(CaracalError) as excinfo:
        await resolve_and_check("https://missing.example/", resolver=_resolver())
    assert excinfo.value.code is ErrorCode.SOURCE_UNAVAILABLE
    assert excinfo.value.details == {"host": "missing.example"}


async def test_error_never_contains_the_query_string() -> None:
    query = "token=super-secret-value"
    with pytest.raises(CaracalError) as excinfo:
        await resolve_and_check(f"http://127.0.0.1/?{query}", resolver=_resolver())
    assert excinfo.value.code is ErrorCode.SOURCE_BLOCKED
    assert "super-secret" not in excinfo.value.message
    assert "super-secret" not in str(excinfo.value.details)
    assert excinfo.value.details == {"host": "127.0.0.1"}


@pytest.mark.parametrize("value", ["2130706433", "0177.0.0.1", "0x7f.1", "127.1"])
def test_inet_aton_forms_parse_to_loopback(value: str) -> None:
    assert parse_ip(value) == ipaddress.ip_address("127.0.0.1")


def test_default_policy_unwraps_embedded_ipv4() -> None:
    assert default_ip_policy(ipaddress.ip_address("::ffff:8.8.8.8"))
    assert not default_ip_policy(ipaddress.ip_address("::ffff:10.0.0.1"))
    assert not default_ip_policy(ipaddress.ip_address("64:ff9b::a00:1"))
    assert not default_ip_policy(ipaddress.ip_address("2002:0a00:0001::"))
    assert default_ip_policy(ipaddress.ip_address("2002:0808:0808::"))


def test_default_policy_rejects_special_ranges() -> None:
    for value in ("224.0.0.1", "240.0.0.1", "255.255.255.255", "fc00::1", "fe80::1", "::"):
        assert not default_ip_policy(ipaddress.ip_address(value)), value
    assert default_ip_policy(ipaddress.ip_address("8.8.8.8"))
