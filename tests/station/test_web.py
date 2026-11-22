import secrets

import pytest
from starlette.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from vxl.station import Station
from vxl.system import StationConfig, System
from vxl.web.app import _allowed_hosts, _parse_allowed_hosts, create_app


def test_web_control_requires_authentication_and_accepts_same_origin_session_cookie(
    station_config: StationConfig,
) -> None:
    password = secrets.token_urlsafe(24)
    app = create_app(Station(station_config), serve_static=False, password=password)
    with TestClient(app, base_url="http://localhost") as client:
        assert client.get("/api/stations").status_code == 401
        assert client.get("/api/stations", auth=("voxel", "wrong")).status_code == 401
        response = client.get("/api/stations", auth=("voxel", password))
        assert response.status_code == 200
        assert "HttpOnly" in response.headers["set-cookie"]
        assert "SameSite=Strict" in response.headers["set-cookie"]
        assert client.get("/api/stations").status_code == 200
        path = f"/api/stations/{station_config.id}/ws"
        with client.websocket_connect(f"ws://localhost{path}", headers={"origin": "http://localhost"}) as websocket:
            assert websocket.receive_bytes()
        with (
            pytest.raises(WebSocketDisconnect) as disconnected,
            client.websocket_connect(f"ws://localhost{path}", headers={"origin": "https://foreign.invalid"}),
        ):
            pass
        assert disconnected.value.code == 1008
        denied = client.post(
            f"/api/stations/{station_config.id}/instruments",
            json={"template": "simulated-local", "name": "foreign"},
            headers={"origin": "https://foreign.invalid"},
        )
        assert denied.status_code == 403
        assert client.get("/api/stations", headers={"host": "foreign.invalid"}).status_code == 400


def test_web_control_rejects_unauthenticated_websocket(station_config: StationConfig) -> None:
    app = create_app(Station(station_config), serve_static=False, password=secrets.token_urlsafe(24))
    with TestClient(app, base_url="http://localhost") as client:
        with (
            pytest.raises(WebSocketDisconnect) as disconnected,
            client.websocket_connect(f"ws://localhost/api/stations/{station_config.id}/ws"),
        ):
            pass
        assert disconnected.value.code == 1008


def test_allowed_hosts_includes_local_hostnames_and_interface_ips_when_remote(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert _allowed_hosts("127.0.0.1") == ("localhost", "127.0.0.1", "[::1]")
    monkeypatch.setattr(System, "local_ip", staticmethod(lambda: "192.168.1.10"))
    monkeypatch.setattr(System, "hostname", staticmethod(lambda: "BenchRig.lan"))
    hosts = _allowed_hosts("0.0.0.0")  # noqa: S104
    assert "192.168.1.10" in hosts
    assert "benchrig" in hosts
    assert "benchrig.local" in hosts
    assert "benchrig.lan" in hosts


def test_allowed_hosts_merges_extra_hosts_on_loopback() -> None:
    parsed = _parse_allowed_hosts(" Rig.Example.Net., other.internal , , rig.example.net ")
    assert parsed == ("rig.example.net", "other.internal")
    assert _allowed_hosts("127.0.0.1", extra_hosts=parsed) == (
        "localhost",
        "127.0.0.1",
        "[::1]",
        "rig.example.net",
        "other.internal",
    )


def test_reverse_proxy_https_loopback_enables_wss_and_enforces_trusted_loopback(
    station_config: StationConfig,
) -> None:
    app = create_app(
        Station(station_config),
        serve_static=False,
        allowed_hosts=_allowed_hosts("127.0.0.1", extra_hosts=("rig.example.net",)),
        proxy_headers=True,
    )
    with TestClient(app, base_url="http://rig.example.net", client=("127.0.0.1", 50000)) as client:
        headers = {"x-forwarded-proto": "https", "origin": "https://rig.example.net"}
        discovery = client.get(f"/api/stations/{station_config.id}/discovery", headers=headers)
        assert discovery.status_code == 200
        realtime = discovery.json()["realtime"]
        assert realtime["state_websocket_url"].startswith("wss://rig.example.net/")
        assert realtime["preview_websocket_url"].startswith("wss://rig.example.net/")
        assert realtime["log_websocket_url"].startswith("wss://rig.example.net/")

        path = f"/api/stations/{station_config.id}/ws"
        with client.websocket_connect(f"ws://rig.example.net{path}", headers=headers) as websocket:
            assert websocket.receive_bytes()

        assert client.get("/api/stations", headers={"host": "evil.invalid"}).status_code == 400
        denied = client.post(
            f"/api/stations/{station_config.id}/instruments",
            json={"template": "simulated-local", "name": "evil"},
            headers={"x-forwarded-proto": "https", "origin": "https://evil.invalid"},
        )
        assert denied.status_code == 403

    # Non-loopback client must not be allowed to spoof X-Forwarded-Proto.
    with TestClient(app, base_url="http://rig.example.net", client=("10.0.0.5", 50000)) as untrusted:
        spoofed = untrusted.get(
            "/api/stations",
            headers={"x-forwarded-proto": "https", "origin": "https://rig.example.net"},
        )
        assert spoofed.status_code == 403


def test_reverse_proxy_https_sets_secure_cookie_when_password_configured(
    station_config: StationConfig,
) -> None:
    password = secrets.token_urlsafe(24)
    app = create_app(
        Station(station_config),
        serve_static=False,
        allowed_hosts=_allowed_hosts("127.0.0.1", extra_hosts=("rig.example.net",)),
        password=password,
        proxy_headers=True,
    )
    with TestClient(app, base_url="http://rig.example.net", client=("127.0.0.1", 50000)) as client:
        headers = {"x-forwarded-proto": "https", "origin": "https://rig.example.net"}
        assert client.get("/api/stations", headers=headers).status_code == 401
        authed = client.get("/api/stations", auth=("voxel", password), headers=headers)
        assert authed.status_code == 200
        cookie = authed.headers["set-cookie"]
        assert "HttpOnly" in cookie
        assert "SameSite=Strict" in cookie
        assert "Secure" in cookie
