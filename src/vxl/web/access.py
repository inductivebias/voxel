"""Origin checks and optional browser authentication for microscope control."""

import base64
import binascii
import secrets
from http.cookies import SimpleCookie

from starlette.requests import HTTPConnection
from starlette.responses import PlainTextResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send


class ControlAccess:
    def __init__(self, app: ASGIApp, *, origins: tuple[str, ...], password: str | None) -> None:
        self.app = app
        self.origins = origins
        self.password = password.encode() if password else None
        self.session = secrets.token_urlsafe(32)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] not in {"http", "websocket"}:
            await self.app(scope, receive, send)
            return
        connection = HTTPConnection(scope)
        scheme = "https" if scope.get("scheme") in {"https", "wss"} else "http"
        origin = connection.headers.get("origin")
        expected = f"{scheme}://{connection.headers.get('host', '')}"
        denied_origin = origin is not None and origin != expected and origin not in self.origins
        authenticated = self.password is None or secrets.compare_digest(
            connection.cookies.get("voxel_session", "").encode(), self.session.encode()
        )
        basic = self._basic_auth(connection.headers.get("authorization", ""))
        if denied_origin or not (authenticated or basic):
            if scope["type"] == "websocket":
                await send({"type": "websocket.close", "code": 1008})
            else:
                status = 403 if denied_origin else 401
                await PlainTextResponse(
                    "Control access denied", status_code=status, headers={"WWW-Authenticate": 'Basic realm="Voxel"'}
                )(scope, receive, send)
            return

        async def send_authenticated(message: Message) -> None:
            if basic and message["type"] == "http.response.start":
                cookie: SimpleCookie = SimpleCookie()
                cookie["voxel_session"] = self.session
                cookie["voxel_session"]["httponly"] = True
                cookie["voxel_session"]["samesite"] = "Strict"
                cookie["voxel_session"]["path"] = "/"
                cookie["voxel_session"]["secure"] = scheme == "https"
                message.setdefault("headers", []).append((b"set-cookie", cookie.output(header="").strip().encode()))
            await send(message)

        await self.app(scope, receive, send_authenticated)

    def _basic_auth(self, authorization: str) -> bool:
        if self.password is None or not authorization.startswith("Basic "):
            return False
        try:
            username, separator, password = base64.b64decode(authorization[6:], validate=True).partition(b":")
        except (binascii.Error, ValueError):
            return False
        return (
            bool(separator)
            and secrets.compare_digest(username, b"voxel")
            and secrets.compare_digest(password, self.password)
        )
