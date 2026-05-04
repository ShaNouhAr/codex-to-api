"""
Passerelle devant codex_openai_server : corrige les corps POST /v1/chat/completions.

Problème en amont (voir codex_openai_server.openai_server.normalize_chat_content) : tout bloc
``{"type":"text",...}`` devient ``input_text``, ce qui convient aux messages utilisateur mais
pas à l'historique ``assistant`` — l'API Codex /responses renvoie alors 400 dès le 2ᵉ tour.

Réf. discussions sur erreurs 400 « Invalid Responses API request » / format d'input.

Fonctionnalités :
  - Correction automatique des messages assistant (content list → string)
  - Détection automatique des modèles Codex au démarrage
  - Rafraîchissement périodique du cache des modèles
  - Proxy transparent avec streaming pour /chat/completions et /responses
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
from contextlib import asynccontextmanager
from typing import Any

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import StreamingResponse

# ── Configuration ────────────────────────────────────────────────────────────

UPSTREAM = os.environ.get("UPSTREAM_URL", "http://127.0.0.1:8001").rstrip("/")
MODEL_REFRESH_SECONDS = int(os.environ.get("MODEL_REFRESH_SECONDS", "300"))
INTERNAL_API_KEY = os.environ.get("OPENAI_COMPAT_API_KEY", "")
PROXY_TIMEOUT = httpx.Timeout(600.0, connect=30.0)

# ── Logging ──────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-7s │ %(name)s │ %(message)s",
)
logger = logging.getLogger("gateway")

# ── Model cache ──────────────────────────────────────────────────────────────

_cached_models: dict | None = None
_refresh_task: asyncio.Task | None = None


async def fetch_models_from_upstream() -> dict | None:
    """Query upstream /v1/models and return the parsed JSON, or None on error."""
    headers = {}
    if INTERNAL_API_KEY:
        headers["Authorization"] = f"Bearer {INTERNAL_API_KEY}"
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            r = await client.get(f"{UPSTREAM}/v1/models", headers=headers)
            if r.status_code == 200:
                return r.json()
            logger.warning("Upstream /v1/models returned HTTP %d", r.status_code)
    except Exception as e:
        logger.warning("Could not reach upstream for model detection: %s", e)
    return None


def log_detected_models(data: dict) -> None:
    """Pretty-print detected models to the log."""
    models = data.get("data", [])
    logger.info("═" * 50)
    logger.info("  🔍 %d model(s) detected from Codex:", len(models))
    for m in models:
        logger.info("    • %-25s  (%s)", m.get("id", "?"), m.get("owned_by", ""))
    logger.info("═" * 50)


async def refresh_models_loop() -> None:
    """Background task: periodically refresh the model cache."""
    global _cached_models
    while True:
        await asyncio.sleep(MODEL_REFRESH_SECONDS)
        data = await fetch_models_from_upstream()
        if data:
            prev_ids = {m.get("id") for m in (_cached_models or {}).get("data", [])}
            new_ids = {m.get("id") for m in data.get("data", [])}
            _cached_models = data
            if new_ids != prev_ids:
                logger.info("Model list changed — refreshing cache")
                log_detected_models(data)
            else:
                logger.debug("Model cache refreshed (no changes)")


# ── Lifespan ─────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _cached_models, _refresh_task

    logger.info("Gateway starting — upstream: %s", UPSTREAM)
    logger.info("Detecting available Codex models…")

    for attempt in range(1, 4):
        data = await fetch_models_from_upstream()
        if data:
            _cached_models = data
            log_detected_models(data)
            break
        logger.warning("  Attempt %d/3 failed, retrying in 5s…", attempt)
        await asyncio.sleep(5)
    else:
        logger.error("Could not detect models at startup — will keep retrying in background")

    _refresh_task = asyncio.create_task(refresh_models_loop())
    logger.info("Model cache will auto-refresh every %ds", MODEL_REFRESH_SECONDS)

    yield

    if _refresh_task:
        _refresh_task.cancel()
        try:
            await _refresh_task
        except asyncio.CancelledError:
            pass
    logger.info("Gateway stopped")


# ── App ──────────────────────────────────────────────────────────────────────

app = FastAPI(title="codex-to-api chat gateway", version="2.0.0", lifespan=lifespan)


# ── Helpers ──────────────────────────────────────────────────────────────────

def fix_chat_messages(messages: list[Any]) -> list[Any]:
    """Aplatit le content ``assistant`` en chaîne (évite input_text sur les tours assistant)."""
    out: list[Any] = []
    for m in messages or []:
        if not isinstance(m, dict):
            out.append(m)
            continue
        role = m.get("role")
        c = m.get("content")
        if role == "assistant" and isinstance(c, list):
            texts: list[str] = []
            for part in c:
                if not isinstance(part, dict):
                    continue
                typ = part.get("type")
                if typ in ("text", "input_text", "output_text"):
                    tx = part.get("text")
                    if tx is not None:
                        texts.append(str(tx))
            if texts:
                joined = "\n".join(texts) if len(texts) > 1 else texts[0]
                out.append({**m, "content": joined})
                continue
        out.append(m)
    return out


def bearer_headers(request: Request) -> dict[str, str]:
    """Forward the client's Authorization header."""
    auth = request.headers.get("authorization")
    return {"Authorization": auth} if auth else {}


async def proxy_upstream(
    method: str,
    path: str,
    request: Request,
    body: bytes | None = None,
    *,
    is_stream: bool = False,
    extra_headers: dict[str, str] | None = None,
) -> Response | StreamingResponse:
    """Generic proxy: send *method* + *path* to upstream, handle stream/non-stream."""
    headers = {
        **bearer_headers(request),
        "Content-Type": request.headers.get("content-type", "application/json"),
        **(extra_headers or {}),
    }

    client = httpx.AsyncClient(timeout=PROXY_TIMEOUT)
    try:
        req = client.build_request(method, f"{UPSTREAM}{path}", headers=headers, content=body)
        r = await client.send(req, stream=True)
    except Exception:
        await client.aclose()
        raise

    if not is_stream:
        content = await r.aread()
        sc, ct = r.status_code, r.headers.get("content-type")
        await r.aclose()
        await client.aclose()
        return Response(content=content, status_code=sc, media_type=ct)

    async def body_iter() -> Any:
        try:
            async for chunk in r.aiter_bytes():
                yield chunk
        finally:
            await r.aclose()
            await client.aclose()

    return StreamingResponse(
        body_iter(),
        status_code=r.status_code,
        media_type=r.headers.get("content-type", "text/event-stream"),
        headers={
            k: v
            for k, v in r.headers.items()
            if k.lower()
            in ("cache-control", "x-request-id", "openai-processing-ms", "openai-version")
        },
    )


# ── Routes ───────────────────────────────────────────────────────────────────

@app.get("/health")
async def health() -> Response:
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.get(f"{UPSTREAM}/health")
    return Response(
        content=r.content,
        status_code=r.status_code,
        media_type=r.headers.get("content-type"),
    )


@app.get("/v1/models")
async def models(request: Request) -> Response:
    """Return models from cache (instant) or proxy upstream as fallback."""
    if _cached_models:
        content = json.dumps(_cached_models).encode("utf-8")
        return Response(content=content, status_code=200, media_type="application/json")
    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.get(f"{UPSTREAM}/v1/models", headers=bearer_headers(request))
    return Response(
        content=r.content,
        status_code=r.status_code,
        media_type=r.headers.get("content-type"),
    )


@app.post("/v1/chat/completions", response_model=None)
async def chat_completions(request: Request):
    raw = await request.body()
    stream = False
    try:
        data = json.loads(raw)
        stream = bool(data.get("stream"))
        if isinstance(data.get("messages"), list):
            data["messages"] = fix_chat_messages(data["messages"])
        raw = json.dumps(data, ensure_ascii=False).encode("utf-8")
    except (json.JSONDecodeError, TypeError, ValueError):
        pass

    return await proxy_upstream(
        "POST", "/v1/chat/completions", request, body=raw, is_stream=stream,
    )


@app.post("/v1/responses", response_model=None)
async def responses(request: Request):
    raw = await request.body()
    stream = False
    try:
        stream = bool(json.loads(raw).get("stream"))
    except (json.JSONDecodeError, TypeError):
        pass

    return await proxy_upstream(
        "POST", "/v1/responses", request, body=raw, is_stream=stream,
    )
