"""Stable error codes shared with clients (api/openapi.yaml ErrorCode)."""

from __future__ import annotations


class ApiError(Exception):
    def __init__(self, status: int, code: str, message: str):
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message

    def body(self) -> dict:
        return {"error": {"code": self.code, "message": self.message}}


def not_found(what: str = "not found") -> ApiError:
    return ApiError(404, "not_found", what)
