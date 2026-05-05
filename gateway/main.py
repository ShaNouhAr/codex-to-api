"""
Passerelle devant codex_openai_server : corrige les corps POST /v1/chat/completions.

Problème en amont (voir codex_openai_server.openai_server.normalize_chat_content) : tout bloc
``{"type":"text",...}`` devient ``input_text``, ce qui convient aux messages utilisateur mais
pas à l'historique ``assistant`` — l'API Codex /responses renvoie alors 400 dès le 2ᵉ tour.

Réf. discussions sur erreurs 400 « Invalid Responses API request » / format d'input.

Fonctionnalités :
  - Correction automatique des messages assistant (content list → string)
  - Retrait des paramètres refusés par l’API Responses Codex (temperature, max_tokens, etc.)
  - Détection automatique des modèles Codex au démarrage
  - Rafraîchissement périodique du cache des modèles
  - Proxy transparent avec streaming pour /chat/completions et /responses
  - Charset UTF-8 explicite sur les réponses JSON et SSE (accents côté client)
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

# Champs que ``/backend-api/codex/responses`` refuse souvent (unknown_parameter), alors que
# codex_openai_server les relaie depuis chat ou /v1/responses — cf. codex-lb #128.
_STRIP_RESPONSES_UPSTREAM: frozenset[str] = frozenset(
    {"temperature", "prompt_cache_retention", "max_output_tokens"}
)
_STRIP_CHAT_ONLY: frozenset[str] = frozenset({"max_tokens", "max_completion_tokens"})

JSON_UTF8 = "application/json; charset=utf-8"

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

def _mojibake_score(s: str) -> int:
    """Heuristique : UTF-8 lu comme cp1252/latin-1 produit souvent « Ã », « Â », etc."""
    return s.count("Ã") + s.count("Â") + s.count("â€")


def parse_json_body(raw: bytes) -> Any:
    """
    Parse un corps JSON en acceptant des encodages clients incorrects.

    UTF-8 strict d’abord. Si échec : ne pas décoder tout le buffer en Latin-1 (cela
    casse l’UTF-8 valide : « ç » → « Ã§ », « très » → « trÃ¨s »). On essaie plusieurs
    décodages puis on retient le parse JSON avec le moins de mojibake / de ``\\ufffd``.
    """
    if not raw:
        raise json.JSONDecodeError("Expecting value", "", 0)
    for encoding in ("utf-8-sig", "utf-8"):
        try:
            return json.loads(raw.decode(encoding))
        except UnicodeDecodeError:
            continue
        except json.JSONDecodeError:
            raise

    decoders = (
        lambda b: b.decode("utf-8", errors="replace"),
        lambda b: b.decode("cp1252"),
        lambda b: b.decode("iso-8859-1"),
    )
    candidates: list[tuple[int, int, Any]] = []
    for dec in decoders:
        try:
            text = dec(raw)
            obj = json.loads(text)
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
        candidates.append((_mojibake_score(text), text.count("\ufffd"), obj))

    if not candidates:
        raise json.JSONDecodeError("Expecting value", "", 0)

    candidates.sort(key=lambda t: (t[0], t[1]))
    return candidates[0][2]


def strip_upstream_unsupported(d: dict[str, Any], *, chat: bool) -> None:
    """Retire les paramètres incompatibles avec l’API Responses Codex en amont."""
    for k in _STRIP_RESPONSES_UPSTREAM:
        d.pop(k, None)
    if chat:
        for k in _STRIP_CHAT_ONLY:
            d.pop(k, None)


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


def upstream_json_content_type(request: Request) -> str:
    """Corps JSON vers le core : toujours annoncer UTF-8 pour ``application/json`` (bytes déjà UTF-8)."""
    ct = request.headers.get("content-type") or JSON_UTF8
    if ct.split(";")[0].strip().lower() == "application/json":
        return JSON_UTF8
    return ct


def response_media_type_with_charset(media_type: str | None, *, streaming: bool) -> str:
    """Réponses au client : annoncer UTF-8 pour JSON et SSE (accents dans le corps / les deltas)."""
    if not media_type:
        return "text/event-stream; charset=utf-8" if streaming else JSON_UTF8
    m = media_type.strip()
    base = m.split(";")[0].strip().lower()
    if "charset=" in m.lower():
        return m
    if base == "application/json":
        return JSON_UTF8
    if base == "text/event-stream":
        return "text/event-stream; charset=utf-8"
    return m


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
    extra = dict(extra_headers or {})
    ct = extra.pop("Content-Type", None) or upstream_json_content_type(request)
    headers = {**bearer_headers(request), **extra, "Content-Type": ct}

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
        return Response(
            content=content,
            status_code=sc,
            media_type=response_media_type_with_charset(ct, streaming=False),
        )

    async def body_iter() -> Any:
        try:
            async for chunk in r.aiter_bytes():
                yield chunk
        finally:
            await r.aclose()
            await client.aclose()

    stream_mt = r.headers.get("content-type", "text/event-stream")
    return StreamingResponse(
        body_iter(),
        status_code=r.status_code,
        media_type=response_media_type_with_charset(stream_mt, streaming=True),
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
        media_type=response_media_type_with_charset(r.headers.get("content-type"), streaming=False),
    )


@app.get("/v1/models")
async def models(request: Request) -> Response:
    """Return models from cache (instant) or proxy upstream as fallback."""
    if _cached_models:
        content = json.dumps(_cached_models, ensure_ascii=False).encode("utf-8")
        return Response(content=content, status_code=200, media_type=JSON_UTF8)
    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.get(f"{UPSTREAM}/v1/models", headers=bearer_headers(request))
    return Response(
        content=r.content,
        status_code=r.status_code,
        media_type=response_media_type_with_charset(r.headers.get("content-type"), streaming=False),
    )


@app.post("/v1/chat/completions", response_model=None)
async def chat_completions(request: Request):
    raw = await request.body()
    stream = False
    try:
        data = parse_json_body(raw)
        if not isinstance(data, dict):
            raise TypeError("chat completion body must be a JSON object")
        stream = bool(data.get("stream"))
        if isinstance(data.get("messages"), list):
            data["messages"] = fix_chat_messages(data["messages"])
        strip_upstream_unsupported(data, chat=True)
        raw = json.dumps(data, ensure_ascii=False).encode("utf-8")
    except (json.JSONDecodeError, TypeError, ValueError, UnicodeDecodeError):
        pass

    return await proxy_upstream(
        "POST", "/v1/chat/completions", request, body=raw, is_stream=stream,
    )


@app.post("/v1/responses", response_model=None)
async def responses(request: Request):
    raw = await request.body()
    stream = False
    try:
        data = parse_json_body(raw)
        if isinstance(data, dict):
            stream = bool(data.get("stream"))
            strip_upstream_unsupported(data, chat=False)
        raw = json.dumps(data, ensure_ascii=False).encode("utf-8")
    except (json.JSONDecodeError, TypeError, ValueError, UnicodeDecodeError):
        pass

    return await proxy_upstream(
        "POST", "/v1/responses", request, body=raw, is_stream=stream,
    )
