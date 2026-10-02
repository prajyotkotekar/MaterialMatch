"""
backend/security.py - Request limits, security headers and the optional API key.

Set MM_API_KEY to require the header `X-API-Key: <key>` on the endpoints that store data
(POST /listings, POST /feedback). Without it the API stays open, which is fine on 127.0.0.1.
"""

from __future__ import annotations

import os
import secrets

from fastapi import Header, HTTPException
from starlette.types import ASGIApp, Receive, Scope, Send

MAX_BODY_BYTES = 4 * 10 * 1024 * 1024 + 1024 * 1024  # 4 photos of 10 MB + form fields

SECURITY_HEADERS = [
    (b"x-content-type-options", b"nosniff"),
    (b"x-frame-options", b"DENY"),
    (b"referrer-policy", b"no-referrer"),
]


def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    expected = os.environ.get("MM_API_KEY")
    if expected and not secrets.compare_digest(x_api_key or "", expected):
        raise HTTPException(401, "Missing or wrong X-API-Key header")


class LimitsMiddleware:
    """Refuse bodies over MAX_BODY_BYTES before they are read, and add security headers to every response."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        length = dict(scope["headers"]).get(b"content-length", b"0")
        if not length.isdigit() or int(length) > MAX_BODY_BYTES:
            await _reject(send, 413, b'{"detail":"Request body too large"}')
            return
        received = 0

        async def limited_receive():
            nonlocal received
            message = await receive()
            received += len(message.get("body", b""))
            if received > MAX_BODY_BYTES:  # chunked uploads have no content-length
                raise HTTPException(413, "Request body too large")
            return message

        async def send_with_headers(message):
            if message["type"] == "http.response.start":
                message["headers"] = list(message.get("headers", [])) + SECURITY_HEADERS
            await send(message)

        await self.app(scope, limited_receive, send_with_headers)


async def _reject(send: Send, status: int, body: bytes) -> None:
    await send({"type": "http.response.start", "status": status,
                "headers": [(b"content-type", b"application/json"), *SECURITY_HEADERS]})
    await send({"type": "http.response.body", "body": body})
