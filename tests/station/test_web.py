import secrets

import pytest
from starlette.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from vxl.station import Station
from vxl.system import StationConfig, System
from vxl.web.app import _allowed_hosts, create_app


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
