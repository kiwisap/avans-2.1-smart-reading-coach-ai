"""Errors as problem details (RFC 9457), with Dutch texts, like the Fastify backend."""

from __future__ import annotations

from http import HTTPStatus

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

PROBLEM_CONTENT_TYPE = "application/problem+json"

TITLES = {
    400: "Ongeldig verzoek",
    401: "Niet geautoriseerd",
    404: "Niet gevonden",
    405: "Methode niet toegestaan",
    409: "Al bezig",
    422: "Ongeldige invoer",
    500: "Interne fout",
    503: "Dienst niet beschikbaar",
}


class ProblemError(Exception):
    def __init__(self, status: int, detail: str) -> None:
        super().__init__(detail)
        self.status = status
        self.detail = detail


def problem(request: Request, status: int, detail: str) -> JSONResponse:
    body = {
        "type": "about:blank",
        "title": TITLES.get(status, HTTPStatus(status).phrase),
        "status": status,
        "detail": detail,
        "instance": request.url.path,
    }
    return JSONResponse(body, status_code=status, media_type=PROBLEM_CONTENT_TYPE)


def describe_validation(error: RequestValidationError) -> str:
    issues = error.errors()
    if not issues:
        return "De invoer is ongeldig."
    first = issues[0]
    field = ".".join(str(part) for part in first.get("loc", []) if part != "body") or "invoer"
    kind = first.get("type", "")
    if kind == "missing":
        return f"Het veld '{field}' is verplicht."
    if kind == "literal_error":
        return f"Het veld '{field}' heeft een waarde die niet is toegestaan."
    if kind in {"greater_than_equal", "less_than_equal"}:
        return f"Het veld '{field}' valt buiten het toegestane bereik."
    if kind == "value_error":
        return f"Het veld '{field}' is ongeldig: {first.get('ctx', {}).get('error', 'controleer de waarde')}."
    return f"Het veld '{field}' is ongeldig."


def register_problem_handlers(app: FastAPI) -> None:
    @app.exception_handler(ProblemError)
    async def handle_problem(request: Request, error: ProblemError) -> JSONResponse:
        return problem(request, error.status, error.detail)

    @app.exception_handler(RequestValidationError)
    async def handle_validation(request: Request, error: RequestValidationError) -> JSONResponse:
        return problem(request, 422, describe_validation(error))

    @app.exception_handler(StarletteHTTPException)
    async def handle_http(request: Request, error: StarletteHTTPException) -> JSONResponse:
        detail = {404: "Deze route bestaat niet.", 405: "Deze methode is niet toegestaan op deze route."}.get(
            error.status_code, str(error.detail)
        )
        return problem(request, error.status_code, detail)

    @app.exception_handler(Exception)
    async def handle_unexpected(request: Request, error: Exception) -> JSONResponse:
        return problem(request, 500, "Er ging iets mis bij het maken van het leesadvies.")
