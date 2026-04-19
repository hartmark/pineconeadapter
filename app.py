import os
import logging
import functools
import time
from datetime import datetime, timezone
from flask import Flask, request, jsonify, redirect
from flasgger import Swagger

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


class FilterHealthOK(logging.Filter):
    def filter(self, record):
        msg = record.getMessage()
        if "GET /health" in msg and '" 200 -' in msg:
            return False
        return True


logging.getLogger("werkzeug").addFilter(FilterHealthOK())

# ── App & config ──────────────────────────────────────────────────────────────

app = Flask(__name__)

PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]
API_KEY          = os.environ.get("PINECONEADAPTER_API_KEY", "")
PORT             = int(os.environ.get("PORT", 11434))

if not API_KEY:
    log.warning("API_KEY not set — all endpoints are unprotected!")

import requests as http
from pinecone import Pinecone
from pinecone.exceptions import PineconeApiException

pc = Pinecone(api_key=PINECONE_API_KEY)
log.info(f"Pinecone client ready")

# ── Auth ──────────────────────────────────────────────────────────────────────

def require_api_key(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        if not API_KEY:
            return fn(*args, **kwargs)
        auth  = request.headers.get("Authorization", "")
        token = auth.removeprefix("Bearer ").strip() if auth.startswith("Bearer ") else ""
        key   = request.headers.get("X-API-Key", "").strip()
        if token != API_KEY and key != API_KEY:
            log.warning(f"Unauthorized request to {request.path} from {request.remote_addr}")
            return jsonify({"error": "Unauthorized"}), 401
        return fn(*args, **kwargs)
    return wrapper

# ── Swagger ───────────────────────────────────────────────────────────────────

swagger_config = {
    "headers": [],
    "specs": [{
        "endpoint": "apispec",
        "route": "/apispec.json",
        "rule_filter": lambda rule: True,
        "model_filter": lambda tag: True,
    }],
    "static_url_path": "/flasgger_static",
    "swagger_ui": True,
    "specs_route": "/apidocs",
}

swagger_template = {
    "swagger": "2.0",
    "info": {
        "title": "Pinecone Embedding Proxy",
        "description": """Ollama-compatible embedding proxy backed by Pinecone Inference API.

## Authentication
All `/api/` endpoints require an API key (except `/health`).
Pass it as either:
- `Authorization: Bearer <key>`
- `X-API-Key: <key>`

## Namespace conventions
Use namespaces to separate different memory types:
- `documents` — long-term document storage
- `sessions` — conversation or session history
- `archive` — older or archived content
- *(empty string)* — no namespace separation

## Recommended metadata fields
Always include these when upserting:
```json
{
  "text": "the chunk text (auto-included, max 1000 chars)",
  "source": "memory | session | reflection | web | custom",
  "filename": "document-chunk-001.md",
  "created_at": "2026-04-07T00:00:00Z",
  "tags": ["optional", "labels"]
}
```
`text` is critical — without it search results only return IDs with no readable content.
`filename` lets you navigate back to the source file and filter by it.
""",
        "version": "1.0.0",
    },
    "securityDefinitions": {
        "BearerAuth": {
            "type": "apiKey",
            "name": "Authorization",
            "in": "header",
            "description": "Bearer token. Example: `Authorization: Bearer your-api-key`",
        },
        "ApiKeyAuth": {
            "type": "apiKey",
            "name": "X-API-Key",
            "in": "header",
            "description": "Example: `X-API-Key: your-api-key`",
        },
    },
    "security": [{"BearerAuth": []}, {"ApiKeyAuth": []}],
    "tags": [
        {"name": "system", "description": "Health and metadata"},
        {"name": "ollama", "description": "Ollama-compatible routes — works with any Ollama client"},
        {"name": "openai", "description": "OpenAI-compatible routes — works with any OpenAI SDK client"},
        {"name": "ping",   "description": "Extended endpoints for embedding, search and index management"},
    ],
    "definitions": {
        "OllamaEmbedRequest": {
            "type": "object",
            "properties": {
                "model":  {"type": "string", "example": "llama-text-embed-v2"},
                "prompt": {"type": "string", "example": "What is the capital of France?"},
                "input":  {"type": "string", "description": "Alternative to prompt. Can also be a JSON array for batch."},
            },
        },
        "SingleEmbedResponse": {
            "type": "object",
            "example": {"embedding": [0.012, -0.034, 0.071, "...2048 floats total"]},
            "properties": {
                "embedding": {"type": "array", "items": {"type": "number"}},
            },
        },
        "EmbedRequest": {
            "type": "object",
            "required": ["input"],
            "properties": {
                "input": {
                    "type": "string",
                    "description": "Text or JSON array of texts to embed.",
                    "example": "Pinecone enables fast vector similarity search at scale",
                },
                "model": {
                    "type": "string",
                    "description": "Pinecone model to use.",
                },
            },
        },
        "EmbedResponse": {
            "type": "object",
            "example": {
                "embeddings": [[0.012, -0.034, "..."], [0.045, 0.021, "..."]],
                "count": 2,
                "dimensions": 2048,
                "model": "llama-text-embed-v2"
            },
            "properties": {
                "embeddings":  {"type": "array", "items": {"type": "array", "items": {"type": "number"}}},
                "count":       {"type": "integer", "example": 2},
                "dimensions":  {"type": "integer", "example": 2048},
                "model":       {"type": "string", "example": "llama-text-embed-v2"},
            },
        },
        "UpsertRecord": {
            "type": "object",
            "required": ["id", "text"],
            "properties": {
                "id":   {"type": "string", "example": "doc-chunk-001"},
                "text": {"type": "string", "example": "Pinecone is a managed vector database designed for AI applications."},
                "metadata": {
                    "type": "object",
                    "example": {
                        "source": "documents",
                        "filename": "pinecone-intro.md",
                        "created_at": "2026-04-07T00:00:00Z",
                        "tags": ["pinecone", "vector-db"],
                    },
                },
            },
        },
        "UpsertVectorRecord": {
            "type": "object",
            "required": ["id", "values"],
            "properties": {
                "id":     {"type": "string", "example": "doc-chunk-001"},
                "values": {"type": "array", "items": {"type": "number"}, "example": [0.1, 0.2, 0.3]},
                "text":   {"type": "string", "example": "Optional text for metadata readability"},
                "metadata": {
                    "type": "object",
                    "example": {
                        "source": "documents",
                        "filename": "pinecone-intro.md",
                    },
                },
            },
        },
        "UpsertRequest": {
            "type": "object",
            "required": ["index", "records"],
            "properties": {
                "index": {"type": "string", "example": "my-index"},
                "namespace": {
                    "type": "string",
                    "example": "documents",
                    "description": "Separate data by type or tenant. Default: ''",
                },
                "records": {"type": "array", "items": {"$ref": "#/definitions/UpsertRecord"}},
                "model": {
                    "type": "string",
                    "description": "Pinecone model to use.",
                },
            },
        },
        "UpsertVectorRequest": {
            "type": "object",
            "required": ["index", "records"],
            "properties": {
                "index": {"type": "string", "example": "my-index"},
                "namespace": {
                    "type": "string",
                    "example": "documents",
                    "description": "Separate data by type or tenant. Default: ''",
                },
                "records": {"type": "array", "items": {"$ref": "#/definitions/UpsertVectorRecord"}},
            },
        },
        "UpsertResponse": {
            "type": "object",
            "example": {
                "upserted": 3,
                "index": "my-index",
                "namespace": "documents"
            },
            "properties": {
                "upserted":  {"type": "integer", "example": 3},
                "index":     {"type": "string", "example": "my-index"},
                "namespace": {"type": "string", "example": "documents"},
            },
        },
        "SearchRequest": {
            "type": "object",
            "required": ["index"],
            "properties": {
                "index": {"type": "string", "example": "my-index"},
                "query": {
                    "type": "string",
                    "example": "How does vector search work?",
                    "description": "Natural language search query. The API embeds it for you.",
                },
                "vector": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "Pre-embedded vector to search with. Alternative to 'query'.",
                },
                "model": {
                    "type": "string",
                    "description": "Pinecone model to use for query embedding (if 'query' is natural language).",
                },
                "namespace": {
                    "type": "string",
                    "example": "documents",
                    "description": "Search within a specific namespace. Omit to search all.",
                },
                "top_k": {
                    "type": "integer",
                    "example": 5,
                    "default": 5,
                    "description": "Number of results to return (max 20).",
                },
                "min_score": {
                    "type": "number",
                    "example": 0.3,
                    "default": 0.0,
                    "description": "Minimum similarity score threshold (0.0–1.0).",
                },
                "filter": {
                    "type": "object",
                    "example": {"source": {"$eq": "documents"}},
                    "description": "Pinecone metadata filter. See Pinecone filter docs.",
                },
                "model": {
                    "type": "string",
                    "description": "Pinecone model to use.",
                }
            },
        },
        "SearchMatch": {
            "type": "object",
            "properties": {
                "id":       {"type": "string", "example": "doc-chunk-001"},
                "score":    {"type": "number", "example": 0.87},
                "text":     {"type": "string", "example": "Pinecone is a managed vector database..."},
                "metadata": {"type": "object"},
            },
        },
        "SearchResponse": {
            "type": "object",
            "example": {
                "matches": [
                    {
                        "id": "doc-chunk-001",
                        "score": 0.872,
                        "text": "Pinecone is a managed vector database designed for AI applications.",
                        "metadata": {"source": "documents", "filename": "pinecone-intro.md", "created_at": "2026-04-07T00:00:00Z"}
                    },
                    {
                        "id": "doc-chunk-002",
                        "score": 0.741,
                        "text": "Vector search enables semantic similarity matching across documents.",
                        "metadata": {"source": "documents", "filename": "vector-search-intro.md", "created_at": "2026-03-01T00:00:00Z"}
                    }
                ],
                "count": 2,
                "index": "my-index",
                "namespace": "documents",
                "query": "what is semantic search?"
            },
            "properties": {
                "matches":   {"type": "array", "items": {"$ref": "#/definitions/SearchMatch"}},
                "count":     {"type": "integer", "example": 2},
                "index":     {"type": "string", "example": "my-index"},
                "namespace": {"type": "string", "example": "documents"},
                "query":     {"type": "string", "example": "what is semantic search?"},
            },
        },
        "IndexInfo": {
            "type": "object",
            "example": {
                "name": "my-index",
                "dimension": 2048,
                "metric": "cosine",
                "status": "Ready"
            },
            "properties": {
                "name":      {"type": "string", "example": "my-index"},
                "dimension": {"type": "integer", "example": 2048},
                "metric":    {"type": "string", "example": "cosine"},
                "status":    {"type": "string", "example": "Ready"},
            },
        },
        "HealthResponse": {
            "type": "object",
            "example": {
                "status": "ok",
                "model": "llama-text-embed-v2",
                "dimensions": 2048
            },
            "properties": {
                "status":     {"type": "string", "example": "ok"},
                "model":      {"type": "string", "example": "llama-text-embed-v2"},
                "dimensions": {"type": "integer", "example": 2048},
            },
        },
    },
}

Swagger(app, config=swagger_config, template=swagger_template)

# ── Helpers ───────────────────────────────────────────────────────────────────

MAX_RETRIES = 5
RETRY_BASE  = 2.0   # seconds — doubles each attempt: 2, 4, 8, 16, 32


def sanitize(text: str) -> str:
    return text.encode("utf-8", errors="replace").decode("utf-8")


def get_model_dimensions(model_name: str) -> int:
    """Get dimensions for a model using Pinecone Inference API with local caching."""
    if not hasattr(get_model_dimensions, "_cache"):
        get_model_dimensions._cache = {}

    if model_name in get_model_dimensions._cache:
        return get_model_dimensions._cache[model_name]

    try:
        model_info = pc.inference.get_model(model_name=model_name)
        # Handle cases where dimension might be in different fields
        dims = getattr(model_info, "dimension", None)
        if dims is None and hasattr(model_info, "supported_dimensions"):
            # If multiple are supported, we might need a default or just pick one.
            # Usually for models like llama-text-embed-v2 it returns 1024.
            dims = model_info.supported_dimensions[0] if model_info.supported_dimensions else 1024
        
        dims = dims or 1024
        get_model_dimensions._cache[model_name] = dims
        return dims
    except Exception as e:
        log.warning(f"Could not get dimensions for {model_name} from Pinecone: {e}. Falling back to 1024.")
        return 1024


def format_pinecone_error(e: PineconeApiException) -> str:
    """Extract a clean error message from PineconeApiException."""
    try:
        import json
        if hasattr(e, "body") and e.body:
            data = json.loads(e.body)
            if "message" in data:
                return data["message"]
        # Fallback if body is not JSON or doesn't have "message"
        return str(e)
    except Exception:
        return str(e)


def _embed_with_retry(inputs: list[str], input_type: str, model: str, dimensions: int = None) -> list[list[float]]:
    if dimensions is None:
        dimensions = get_model_dimensions(model)

    params = {"input_type": input_type, "truncate": "END"}
    if model != "multilingual-e5-large":
        params["dimension"] = dimensions

    """Call Pinecone embed with exponential backoff on 429."""
    for attempt in range(MAX_RETRIES):
        try:
            result = pc.inference.embed(
                model=model,
                inputs=inputs,
                parameters=params,
            )
            return [item["values"] for item in result.data]
        except PineconeApiException as e:
            if e.status == 429 and attempt < MAX_RETRIES - 1:
                wait = RETRY_BASE ** attempt
                log.warning(f"Rate limited (429) — retrying in {wait:.1f}s (attempt {attempt + 1}/{MAX_RETRIES})")
                time.sleep(wait)
            else:
                raise
    raise RuntimeError("Max retries exceeded")


def do_embed(inputs: list[str], model: str, dimensions: int = None) -> list[list[float]]:
    inputs = [sanitize(t) for t in inputs]
    return _embed_with_retry(inputs, "passage", model, dimensions)


def do_embed_query(query: str, model: str, dimensions: int = None) -> list[float]:
    """Embed a single search query (input_type=query for asymmetric models)."""
    return _embed_with_retry([sanitize(query)], "query", model, dimensions)[0]


# ── System ────────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return redirect("/apidocs")


@app.get("/health")
def health():
    """
    Liveness and Pinecone connectivity check. No auth required.
    ---
    tags: [system]
    responses:
      200:
        description: Healthy
        schema:
          $ref: '#/definitions/HealthResponse'
      503:
        description: Pinecone unreachable or API key invalid
    """
    model = request.args.get("model") or "llama-text-embed-v2"
    dimensions = get_model_dimensions(model)

    params = {"input_type": "passage", "truncate": "END"}
    if model != "multilingual-e5-large":
        params["dimension"] = dimensions

    try:
        pc.inference.embed(
            model=model,
            inputs=["health check"],
            parameters=params,
        )
        # Verify connectivity to control plane too
        pc.list_indexes()
        return jsonify({"status": "ok", "model": model, "dimensions": dimensions})
    except Exception as e:
        log.error(f"Health check failed: {e}")
        return jsonify({"status": "error", "detail": str(e)}), 503


# ── Ollama compat ─────────────────────────────────────────────────────────────

@app.get("/api/tags")
@require_api_key
def api_tags():
    """
    List available models (Ollama compatibility).
    ---
    tags: [ollama]
    security:
      - BearerAuth: []
      - ApiKeyAuth: []
    responses:
      200:
        description: Model list
        examples:
          application/json:
            models:
              - name: llama-text-embed-v2
                model: llama-text-embed-v2
                details:
                  family: pinecone-inference
      401:
        description: Unauthorized
    """
    return jsonify({
        "models": [
            {"name": "llama-text-embed-v2", "model": "llama-text-embed-v2", "details": {"family": "pinecone-inference"}},
            {"name": "multilingual-e5-large-index", "model": "multilingual-e5-large-index", "details": {"family": "pinecone-inference"}},
        ]
    })


@app.post("/api/embeddings")
@require_api_key
def api_embeddings():
    """
    Generate embeddings — Ollama-compatible, used by OpenClaw.
    ---
    tags: [ollama]
    security:
      - BearerAuth: []
      - ApiKeyAuth: []
    parameters:
      - in: body
        name: body
        required: true
        schema:
          $ref: '#/definitions/OllamaEmbedRequest'
    responses:
      200:
        description: Single or batch embeddings
        schema:
          $ref: '#/definitions/SingleEmbedResponse'
      400:
        description: Missing input
      401:
        description: Unauthorized
    """
    body = request.get_json(force=True)

    log.info(f"=== REQUEST BODY KEYS: {list(body.keys())}")
    for k, v in body.items():
        if k in ("prompt", "input"):
            log.info(f"  {k}: {str(v)[:200]}")
        else:
            log.info(f"  {k}: {v}")

    raw = body.get("prompt") or body.get("input")
    if not raw:
        return jsonify({"error": "Missing 'prompt' or 'input'"}), 400

    model = body.get("model")
    if not model:
        return jsonify({"error": "Missing 'model'"}), 400

    is_batch = isinstance(raw, list)
    inputs   = raw if is_batch else [raw]

    try:
        vectors = do_embed(inputs, model)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    log.info(f"Got {len(vectors)} vector(s) — dim={len(vectors[0])}")

    return jsonify({"embeddings": vectors} if is_batch else {"embedding": vectors[0]})


# ── OpenAI-compatible routes (/v1/) ──────────────────────────────────────────

@app.get("/v1/models")
@require_api_key
def v1_models():
    """
    List available models — OpenAI-compatible.
    ---
    tags: [openai]
    security:
      - BearerAuth: []
      - ApiKeyAuth: []
    responses:
      200:
        description: Model list
        schema:
          type: object
          example:
            object: list
            data:
              - id: llama-text-embed-v2
                object: model
                created: 1744000000
                owned_by: pinecone
      401:
        description: Unauthorized
    """
    return jsonify({
        "object": "list",
        "data": [
            {
                "id":       "llama-text-embed-v2",
                "object":   "model",
                "created":  1744000000,
                "owned_by": "pinecone",
            },
            {
                "id": "bilingual-e5-large-index",
                "object": "model",
                "created": 1744000000,
                "owned_by": "pinecone",
            }
        ],
    })


@app.post("/v1/embeddings")
@require_api_key
def v1_embeddings():
    """
    Generate embeddings — OpenAI-compatible spec.
    Accepts a single string or list of strings as input.

    **Example request:**
    ```json
    { "model": "llama-text-embed-v2", "input": "hello world" }
    { "model": "llama-text-embed-v2", "input": ["hello", "world"] }
    ```
    ---
    tags: [openai]
    security:
      - BearerAuth: []
      - ApiKeyAuth: []
    parameters:
      - in: body
        name: body
        required: true
        schema:
          type: object
          required: [input]
          properties:
            model:
              type: string
              example: llama-text-embed-v2
              description: "Ignored — adapter always uses its configured model"
            input:
              type: string
              description: "Single string or JSON array of strings"
              example: "Pinecone enables fast vector similarity search at scale"
    responses:
      200:
        description: OpenAI-format embedding response
        schema:
          type: object
          example:
            object: list
            model: llama-text-embed-v2
            data:
              - object: embedding
                index: 0
                embedding: [0.012, -0.034, 0.071, "...2048 floats total"]
            usage:
              prompt_tokens: 6
              total_tokens: 6
      400:
        description: Missing input
      401:
        description: Unauthorized
    """
    body = request.get_json(force=True)
    raw  = body.get("input") or body.get("prompt")

    if not raw:
        return jsonify({"error": {"message": "Missing 'input'", "type": "invalid_request_error"}}), 400

    model = body.get("model")
    if not model:
        return jsonify({"error": {"message": "Missing 'model'", "type": "invalid_request_error"}}), 400

    is_batch = isinstance(raw, list)
    inputs   = raw if is_batch else [raw]
    try:
        vectors = do_embed(inputs, model)
    except ValueError as e:
        return jsonify({"error": {"message": str(e), "type": "invalid_request_error"}}), 400

    log.info(f"[v1/embeddings] {len(vectors)} vector(s) — dim={len(vectors[0])}")

    total_tokens = sum(len(t.split()) * 13 // 10 for t in inputs)

    return jsonify({
        "object": "list",
        "model":  model,
        "data": [
            {
                "object":    "embedding",
                "index":     i,
                "embedding": vec,
            }
            for i, vec in enumerate(vectors)
        ],
        "usage": {
            "prompt_tokens": total_tokens,
            "total_tokens":  total_tokens,
        },
    })


# ── Extended endpoints ────────────────────────────────────────────────────────

@app.get("/api/indexes")
@require_api_key
def api_indexes():
    """
    List all Pinecone indexes in this account.
    ---
    tags: [ping]
    security:
      - BearerAuth: []
      - ApiKeyAuth: []
    responses:
      200:
        description: Index list
        schema:
          type: object
          properties:
            indexes:
              type: array
              items:
                $ref: '#/definitions/IndexInfo'
      401:
        description: Unauthorized
      500:
        description: Pinecone error
    """
    try:
        indexes = pc.list_indexes()
        return jsonify({
            "indexes": [
                {
                    "name": idx.name,
                    "dimension": idx.dimension,
                    "metric": idx.metric,
                    "status": idx.status.get("state") if isinstance(idx.status, dict) else str(idx.status),
                }
                for idx in indexes
            ]
        })
    except Exception as e:
        log.error(f"Failed to list indexes: {e}")
        return jsonify({"error": str(e)}), 500


@app.post("/api/embed")
@require_api_key
def api_embed():
    """
    Generate embeddings — clean extended endpoint.
    ---
    tags: [ping]
    security:
      - BearerAuth: []
      - ApiKeyAuth: []
    parameters:
      - in: body
        name: body
        required: true
        schema:
          $ref: '#/definitions/EmbedRequest'
    responses:
      200:
        description: Embeddings with metadata
        schema:
          $ref: '#/definitions/EmbedResponse'
      400:
        description: Missing input
      401:
        description: Unauthorized
    """
    body = request.get_json(force=True)
    raw  = body.get("input")

    if not raw:
        return jsonify({"error": "Missing 'input'"}), 400

    is_batch = isinstance(raw, list)
    inputs   = raw if is_batch else [raw]
    model = body.get("model")
    if not model:
        return jsonify({"error": {"message": "Missing 'model'", "type": "invalid_request_error"}}), 400

    log.info(f"[api/embed] {len(inputs)} input(s)")
    try:
        vectors = do_embed(inputs, model)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    log.info(f"[api/embed] Got {len(vectors)} vector(s) — dim={len(vectors[0])}")

    return jsonify({
        "embeddings": vectors,
        "count": len(vectors),
        "dimensions": len(vectors[0]),
        "model": model,
    })


@app.post("/api/upsert")
@require_api_key
def api_upsert():
    """
    Embed records and upsert vectors into a named Pinecone index.
    The `text` field is always stored in metadata so search results are readable.

    **Recommended metadata fields:**
    ```json
    {
      "source": "documents | sessions | archive | web | custom",
      "filename": "document-chunk-001.md",
      "created_at": "2026-04-07T00:00:00Z",
      "tags": ["optional", "labels"]
    }
    ```
    `filename` lets you navigate back to the source file and filter by it.

    **Namespace conventions:**
    - `documents` — long-term document storage
    - `sessions` — conversation or session history
    - `archive` — archived or older content
    ---
    tags: [ping]
    security:
      - BearerAuth: []
      - ApiKeyAuth: []
    parameters:
      - in: body
        name: body
        required: true
        schema:
          $ref: '#/definitions/UpsertRequest'
    responses:
      200:
        description: Upsert result
        schema:
          $ref: '#/definitions/UpsertResponse'
      400:
        description: Missing or invalid fields
      401:
        description: Unauthorized
      404:
        description: Index not found
    """
    body       = request.get_json(force=True)
    index_name = body.get("index")
    records    = body.get("records")
    namespace  = body.get("namespace", "")

    if not index_name or not records:
        return jsonify({"error": "Missing 'index' or 'records'"}), 400

    if not isinstance(records, list) or not all("id" in r and "text" in r for r in records):
        return jsonify({"error": "Each record must have 'id' and 'text'"}), 400

    try:
        idx = pc.Index(index_name)
    except Exception as e:
        return jsonify({"error": f"Could not connect to index '{index_name}': {e}"}), 404

    now     = datetime.now(timezone.utc).isoformat()
    texts   = [r["text"] for r in records]
    model = body.get("model")

    if not model:
        return jsonify({"error": "Missing 'model'"}), 400

    try:
        vectors = do_embed(texts, model)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    upsert_vectors = [
        {
            "id": r["id"],
            "values": v,
            "metadata": {
                "created_at": now,
                **r.get("metadata", {}),
                "text": r["text"][:1000],
            },
        }
        for r, v in zip(records, vectors)
    ]

    log.info(f"[api/upsert] {len(upsert_vectors)} vectors → index={index_name} namespace={namespace!r}")
    try:
        idx.upsert(vectors=upsert_vectors, namespace=namespace)
    except PineconeApiException as e:
        log.error(f"Pinecone API error during upsert: {e}")
        # Map 4xx errors to 400 Bad Request, others to 500
        status_code = 400 if 400 <= e.status < 500 else 500
        return jsonify({"error": format_pinecone_error(e)}), status_code
    except Exception as e:
        log.error(f"Unexpected error during upsert: {e}")
        return jsonify({"error": str(e)}), 500
    log.info(f"[api/upsert] Done")

    return jsonify({"upserted": len(upsert_vectors), "index": index_name, "namespace": namespace})


@app.post("/api/upsert-vectors")
@require_api_key
def api_upsert_vectors():
    """
    Upsert pre-embedded vectors into a named Pinecone index.
    Skips the embedding step. Useful when you have vectors from an external source.

    **Example record:**
    ```json
    {
      "id": "doc-001",
      "values": [0.1, 0.2, 0.3, ...],
      "text": "The original text (optional, for metadata)",
      "metadata": { "source": "external" }
    }
    ```
    ---
    tags: [ping]
    security:
      - BearerAuth: []
      - ApiKeyAuth: []
    parameters:
      - in: body
        name: body
        required: true
        schema:
          $ref: '#/definitions/UpsertVectorRequest'
    responses:
      200:
        description: Upsert result
        schema:
          $ref: '#/definitions/UpsertResponse'
      400:
        description: Missing or invalid fields
      401:
        description: Unauthorized
      404:
        description: Index not found
    """
    body       = request.get_json(force=True)
    index_name = body.get("index")
    records    = body.get("records")
    namespace  = body.get("namespace", "")

    if not index_name or not records:
        return jsonify({"error": "Missing 'index' or 'records'"}), 400

    if not isinstance(records, list) or not all("id" in r and "values" in r for r in records):
        return jsonify({"error": "Each record must have 'id' and 'values'"}), 400

    try:
        idx = pc.Index(index_name)
    except Exception as e:
        return jsonify({"error": f"Could not connect to index '{index_name}': {e}"}), 404

    now = datetime.now(timezone.utc).isoformat()

    upsert_vectors = []
    for r in records:
        metadata = {
            "created_at": now,
            **r.get("metadata", {}),
        }
        if "text" in r:
            metadata["text"] = r["text"][:1000]

        upsert_vectors.append({
            "id": r["id"],
            "values": r["values"],
            "metadata": metadata,
        })

    log.info(f"[api/upsert-vectors] {len(upsert_vectors)} vectors → index={index_name} namespace={namespace!r}")
    try:
        idx.upsert(vectors=upsert_vectors, namespace=namespace)
    except PineconeApiException as e:
        log.error(f"Pinecone API error during upsert-vectors: {e}")
        # Map 4xx errors to 400 Bad Request, others to 500
        status_code = 400 if 400 <= e.status < 500 else 500
        return jsonify({"error": format_pinecone_error(e)}), status_code
    except Exception as e:
        log.error(f"Unexpected error during upsert-vectors: {e}")
        return jsonify({"error": str(e)}), 500
    log.info(f"[api/upsert-vectors] Done")

    return jsonify({"upserted": len(upsert_vectors), "index": index_name, "namespace": namespace})


@app.post("/api/search")
@require_api_key
def api_search():
    """
    Semantic search in a Pinecone index.
    The API embeds your query automatically — just pass natural language.
    Alternatively, pass a pre-embedded `vector`.

    **Example queries:**
    - `"What is Pinecone used for?"`
    - `"vector similarity search"`
    - `"how to filter by metadata"`

    **Filtering by metadata:**
    ```json
    { "filter": { "source": { "$eq": "documents" } } }
    { "filter": { "tags": { "$in": ["pinecone", "vector-db"] } } }
    ```

    **Namespace tip:** Omit `namespace` to search across all namespaces,
    or specify `documents`, `sessions`, or `archive` to narrow scope.
    ---
    tags: [ping]
    security:
      - BearerAuth: []
      - ApiKeyAuth: []
    parameters:
      - in: body
        name: body
        required: true
        schema:
          $ref: '#/definitions/SearchRequest'
    responses:
      200:
        description: Search results ordered by similarity score
        schema:
          $ref: '#/definitions/SearchResponse'
      400:
        description: Missing index, query/vector, model
      401:
        description: Unauthorized
      404:
        description: Index not found
    """
    body       = request.get_json(force=True)
    index_name = body.get("index")
    query      = body.get("query")
    vector     = body.get("vector")
    namespace  = body.get("namespace", "")
    top_k      = min(int(body.get("top_k", 5)), 20)
    min_score  = float(body.get("min_score", 0.0))
    filter_    = body.get("filter", None)
    model      = body.get("model")

    if not index_name:
        return jsonify({"error": "Missing 'index'"}), 400

    if not query and not vector:
        return jsonify({"error": "Missing 'query' or 'vector'"}), 400

    try:
        idx = pc.Index(index_name)
    except Exception as e:
        return jsonify({"error": f"Could not connect to index '{index_name}': {e}"}), 404

    if vector:
        query_vector = vector
        log.info(f"[api/search] vector search index={index_name} namespace={namespace!r} top_k={top_k}")
    else:
        if not model:
            return jsonify({"error": "Missing 'model' for text query"}), 400
        log.info(f"[api/search] query={query[:100]!r} index={index_name} namespace={namespace!r} top_k={top_k}")
        try:
            query_vector = do_embed_query(query, model)
        except ValueError as e:
            return jsonify({"error": str(e)}), 400

    query_kwargs = dict(
        vector=query_vector,
        top_k=top_k,
        include_metadata=True,
        namespace=namespace,
    )
    if filter_:
        query_kwargs["filter"] = filter_

    try:
        result = idx.query(**query_kwargs)
    except PineconeApiException as e:
        log.error(f"Pinecone API error during search: {e}")
        # Map 4xx errors to 400 Bad Request, others to 500
        status_code = 400 if 400 <= e.status < 500 else 500
        return jsonify({"error": format_pinecone_error(e)}), status_code
    except Exception as e:
        log.error(f"Unexpected error during search: {e}")
        return jsonify({"error": str(e)}), 500

    matches = [
        {
            "id":       m["id"],
            "score":    round(m["score"], 4),
            "text":     m.get("metadata", {}).get("text", ""),
            "metadata": {k: v for k, v in m.get("metadata", {}).items() if k != "text"},
        }
        for m in result["matches"]
        if m["score"] >= min_score
    ]

    log.info(f"[api/search] {len(matches)} matches above min_score={min_score}")

    return jsonify({
        "matches":   matches,
        "count":     len(matches),
        "index":     index_name,
        "namespace": namespace,
        "query":     query,
    })


@app.post("/api/indexes/create")
@require_api_key
def api_index_create():
    """
    Create a new serverless Pinecone index.
    Defaults to cosine metric and AWS us-east-1 (free tier compatible).
    Dimension defaults to the adapter's configured embedding dimension.

    **Example:**
    ```json
    {
      "name": "my-experiments",
      "dimension": 2048,
      "metric": "cosine",
      "cloud": "aws",
      "region": "us-east-1"
    }
    ```
    ---
    tags: [ping]
    security:
      - BearerAuth: []
      - ApiKeyAuth: []
    parameters:
      - in: body
        name: body
        required: true
        schema:
          type: object
          required: [name]
          properties:
            name:
              type: string
              example: my-experiments
            dimension:
              type: integer
              example: 2048
              description: "Defaults to adapter's EMBED_DIMENSIONS if omitted"
            metric:
              type: string
              example: cosine
              description: "cosine | euclidean | dotproduct (default: cosine)"
            cloud:
              type: string
              example: aws
              description: "Only 'aws' is available on the free Starter plan"
            region:
              type: string
              example: us-east-1
              description: "Only 'us-east-1' is available on the free Starter plan. Other regions require Standard/Enterprise."
    responses:
      201:
        description: Index created
        schema:
          type: object
          example:
            name: my-experiments
            dimension: 2048
            metric: cosine
            cloud: aws
            region: us-east-1
            status: created
          properties:
            name:      {type: string}
            dimension: {type: integer}
            metric:    {type: string}
            cloud:     {type: string}
            region:    {type: string}
            status:    {type: string}
      400:
        description: Missing name or invalid params
      409:
        description: Index already exists
    """
    from pinecone import ServerlessSpec

    body       = request.get_json(force=True)
    name      = body.get("name")
    dimension = body.get("dimension")
    if dimension:
        dimension = int(dimension)
    metric    = body.get("metric", "cosine")
    cloud     = body.get("cloud", "aws")
    region    = body.get("region", "us-east-1")

    if not name:
        return jsonify({"error": "Missing 'name'"}), 400
    if not dimension:
        return jsonify({"error": "Missing 'dimension'. Provide in body."}), 400

    try:
        pc.create_index(
            name=name,
            dimension=dimension,
            metric=metric,
            spec=ServerlessSpec(cloud=cloud, region=region),
        )
        log.info(f"[api/indexes/create] Created index '{name}' dim={dimension} metric={metric}")
        return jsonify({
            "name":      name,
            "dimension": dimension,
            "metric":    metric,
            "cloud":     cloud,
            "region":    region,
            "status":    "created",
        }), 201
    except Exception as e:
        err = str(e)
        if "ALREADY_EXISTS" in err or "already exists" in err.lower():
            return jsonify({"error": f"Index '{name}' already exists"}), 409
        log.error(f"[api/indexes/create] Failed: {e}")
        return jsonify({"error": err}), 500


@app.delete("/api/indexes/<index_name>")
@require_api_key
def api_index_delete(index_name):
    """
    Delete a Pinecone index permanently. This cannot be undone.

    **Warning:** All vectors and metadata in the index will be lost.
    Consider cloning first if you want a backup.
    ---
    tags: [ping]
    security:
      - BearerAuth: []
      - ApiKeyAuth: []
    parameters:
      - in: path
        name: index_name
        required: true
        type: string
        description: Name of the index to delete
    responses:
      200:
        description: Index deleted
        schema:
          type: object
          example:
            deleted: my-old-index
          properties:
            deleted: {type: string}
      404:
        description: Index not found
    """
    try:
        pc.delete_index(index_name)
        log.info(f"[api/indexes/delete] Deleted index '{index_name}'")
        return jsonify({"deleted": index_name})
    except Exception as e:
        err = str(e)
        if "NOT_FOUND" in err or "not found" in err.lower():
            return jsonify({"error": f"Index '{index_name}' not found"}), 404
        log.error(f"[api/indexes/delete] Failed: {e}")
        return jsonify({"error": err}), 500


@app.post("/api/indexes/clone")
@require_api_key
def api_index_clone():
    """
    Clone a Pinecone index by creating a new index and copying all vectors.

    **Note:** This works by fetching all vector IDs from the source index
    and re-upserting them into the destination. It is suitable for small-to-medium
    indexes (up to ~50k vectors). For large indexes, use Pinecone's native
    backup/restore feature (Standard/Enterprise plan required).

    The destination index is created automatically with the same dimension
    and metric as the source.

    **Example:**
    ```json
    {
      "source": "my-index",
      "destination": "my-index-backup",
      "namespace": "documents"
    }
    ```
    ---
    tags: [ping]
    security:
      - BearerAuth: []
      - ApiKeyAuth: []
    parameters:
      - in: body
        name: body
        required: true
        schema:
          type: object
          required: [source, destination]
          properties:
            source:
              type: string
              example: my-index
              description: Source index name
            destination:
              type: string
              example: my-index-backup
              description: Destination index name (will be created)
            namespace:
              type: string
              example: documents
              description: "Namespace to clone. Omit to clone all namespaces."
            cloud:
              type: string
              example: aws
              description: "Only 'aws' available on free Starter plan"
            region:
              type: string
              example: us-east-1
              description: "Only 'us-east-1' available on free Starter plan"
    responses:
      200:
        description: Clone completed
        schema:
          type: object
          example:
            source: my-index
            destination: my-index-backup
            copied: 2315
          properties:
            source:      {type: string}
            destination: {type: string}
            copied:      {type: integer}
      400:
        description: Missing source or destination
      404:
        description: Source index not found
    """
    from pinecone import ServerlessSpec

    body        = request.get_json(force=True)
    source      = body.get("source")
    destination = body.get("destination")
    namespace   = body.get("namespace", "")
    cloud       = body.get("cloud", "aws")
    region      = body.get("region", "us-east-1")

    if not source or not destination:
        return jsonify({"error": "Missing 'source' or 'destination'"}), 400

    try:
        src_info = pc.describe_index(source)
    except Exception as e:
        return jsonify({"error": f"Source index '{source}' not found: {e}"}), 404

    dimension = src_info.dimension
    metric    = src_info.metric

    try:
        pc.create_index(
            name=destination,
            dimension=dimension,
            metric=metric,
            spec=ServerlessSpec(cloud=cloud, region=region),
        )
        log.info(f"[api/indexes/clone] Created destination index '{destination}'")
    except Exception as e:
        if "ALREADY_EXISTS" not in str(e) and "already exists" not in str(e).lower():
            return jsonify({"error": f"Could not create destination index: {e}"}), 500
        log.info(f"[api/indexes/clone] Destination '{destination}' already exists, continuing")

    src_idx  = pc.Index(source)
    dst_idx  = pc.Index(destination)

    copied           = 0
    batch_size       = 100
    pagination_token = None

    log.info(f"[api/indexes/clone] Starting copy {source} → {destination} namespace={namespace!r}")

    while True:
        list_kwargs = {"namespace": namespace, "limit": batch_size}
        if pagination_token:
            list_kwargs["pagination_token"] = pagination_token

        list_result = src_idx.list(**list_kwargs)
        ids = list(list_result.vectors.keys()) if hasattr(list_result, "vectors") else list(list_result)

        if not ids:
            break

        # Fetch in small batches to avoid 414 Request-URI Too Large
        fetch_batch_size = 20
        vectors_to_copy  = []
        for i in range(0, len(ids), fetch_batch_size):
            batch        = ids[i:i + fetch_batch_size]
            fetch_result = src_idx.fetch(ids=batch, namespace=namespace)
            vectors_to_copy.extend([
                {"id": vid, "values": v.values, "metadata": v.metadata or {}}
                for vid, v in fetch_result.vectors.items()
            ])

        if vectors_to_copy:
            dst_idx.upsert(vectors=vectors_to_copy, namespace=namespace)
            copied += len(vectors_to_copy)
            log.info(f"[api/indexes/clone] Copied {copied} vectors so far...")

        pagination_token = getattr(list_result, "pagination", None)
        if not pagination_token:
            break

    log.info(f"[api/indexes/clone] Done — {copied} vectors copied")
    return jsonify({"source": source, "destination": destination, "copied": copied})


@app.post("/api/indexes/compare")
@require_api_key
def api_index_compare():
    """
    Run the same semantic query against multiple indexes and compare results side by side.
    Useful for evaluating whether a new index structure produces better results than the original.

    Returns matches from each index with scores, plus a diff showing which results
    are unique to each index and which overlap.

    **Example:**
    ```json
    {
      "indexes": ["my-index", "my-index-sandbox"],
      "query": "what is semantic search?",
      "namespace": "documents",
      "top_k": 5,
      "min_score": 0.3
    }
    ```
    ---
    tags: [ping]
    security:
      - BearerAuth: []
      - ApiKeyAuth: []
    parameters:
      - in: body
        name: body
        required: true
        schema:
          type: object
          required: [indexes, query]
          properties:
            indexes:
              type: array
              items:
                type: string
              example: ["my-index", "my-index-sandbox"]
              description: "List of 2+ index names to compare"
            query:
              type: string
              example: "what is semantic search?"
              description: "Natural language query — embedded automatically"
            namespace:
              type: string
              example: documents
              description: "Search within this namespace in all indexes"
            top_k:
              type: integer
              example: 5
              default: 5
            min_score:
              type: number
              example: 0.3
              default: 0.0
            model:
              type: string
              example: llama-text-embed-v2
              description: "Embedding model to use for the query"
            filter:
              type: object
              description: "Pinecone metadata filter applied to all indexes"
    responses:
      200:
        description: Per-index results with overlap/diff analysis
        schema:
          type: object
          example:
            query: "what is semantic search?"
            namespace: documents
            results:
              my-index:
                - id: doc-chunk-001
                  score: 0.872
                  text: "Semantic search finds results based on meaning."
                  metadata:
                    source: documents
                    filename: pinecone-intro.md
              my-index-sandbox:
                - id: doc-chunk-001
                  score: 0.891
                  text: "Semantic search finds results based on meaning."
                  metadata:
                    source: documents
                    filename: pinecone-intro.md
                - id: doc-chunk-002
                  score: 0.741
                  text: "Vector databases store high-dimensional embeddings."
                  metadata:
                    source: documents
            analysis:
              overlap:
                - doc-chunk-001
              unique:
                my-index: []
                my-index-sandbox:
                  - doc-chunk-002
              score_diff:
                - id: doc-chunk-001
                  scores:
                    my-index: 0.872
                    my-index-sandbox: 0.891
                  delta: 0.019
          properties:
            query:
              type: string
            namespace:
              type: string
            results:
              type: object
              description: "Map of index_name to list of matches."
            analysis:
              type: object
              description: "Overlap and diff analysis across indexes"
      400:
        description: Missing indexes or query, or fewer than 2 indexes specified
      500:
        description: Error querying one or more indexes
    """
    body      = request.get_json(force=True)
    indexes   = body.get("indexes", [])
    query     = body.get("query")
    namespace = body.get("namespace", "")
    top_k     = min(int(body.get("top_k", 5)), 20)
    min_score = float(body.get("min_score", 0.0))
    filter_   = body.get("filter", None)
    model = body.get("model")

    if not model:
        return jsonify({"error": "Provide the embedding model."}), 400

    if not indexes or len(indexes) < 2:
        return jsonify({"error": "Provide at least 2 index names in 'indexes'"}), 400
    if not query:
        return jsonify({"error": "Missing 'query'"}), 400

    log.info(f"[api/indexes/compare] query={query[:80]!r} indexes={indexes} namespace={namespace!r}")

    query_vector = do_embed_query(query, model)

    results = {}
    errors  = {}

    for index_name in indexes:
        try:
            idx = pc.Index(index_name)
            query_kwargs = dict(
                vector=query_vector,
                top_k=top_k,
                include_metadata=True,
                namespace=namespace,
            )
            if filter_:
                query_kwargs["filter"] = filter_

            result = idx.query(**query_kwargs)
            results[index_name] = [
                {
                    "id":       m["id"],
                    "score":    round(m["score"], 4),
                    "text":     m.get("metadata", {}).get("text", ""),
                    "metadata": {k: v for k, v in m.get("metadata", {}).items() if k != "text"},
                }
                for m in result["matches"]
                if m["score"] >= min_score
            ]
            log.info(f"[api/indexes/compare] {index_name}: {len(results[index_name])} matches")
        except Exception as e:
            log.error(f"[api/indexes/compare] Failed for index '{index_name}': {e}")
            errors[index_name] = str(e)

    if errors:
        return jsonify({"error": "Failed to query some indexes", "details": errors}), 500

    id_scores = {}
    for index_name, matches in results.items():
        for m in matches:
            if m["id"] not in id_scores:
                id_scores[m["id"]] = {}
            id_scores[m["id"]][index_name] = m["score"]

    overlap = [vid for vid, scores in id_scores.items() if len(scores) == len(indexes)]

    unique = {
        index_name: [vid for vid, scores in id_scores.items() if list(scores.keys()) == [index_name]]
        for index_name in indexes
    }

    score_diff = [
        {
            "id":     vid,
            "scores": scores,
            "delta":  round(max(scores.values()) - min(scores.values()), 4),
        }
        for vid, scores in id_scores.items()
        if len(scores) > 1
    ]
    score_diff.sort(key=lambda x: x["delta"], reverse=True)

    return jsonify({
        "query":     query,
        "namespace": namespace,
        "results":   results,
        "analysis": {
            "overlap":    overlap,
            "unique":     unique,
            "score_diff": score_diff,
        },
    })


@app.post("/api/delete")
@require_api_key
def api_delete():
    """
    Delete vectors by ID from a Pinecone index.

    **Example:**
    ```json
    {
      "index": "my-index",
      "namespace": "documents",
      "ids": ["doc-chunk-001", "doc-chunk-002"]
    }
    ```
    ---
    tags: [ping]
    security:
      - BearerAuth: []
      - ApiKeyAuth: []
    parameters:
      - in: body
        name: body
        required: true
        schema:
          type: object
          required: [index, ids]
          properties:
            index:
              type: string
              example: my-index
            namespace:
              type: string
              example: documents
              description: "Namespace to delete from. Default: ''"
            ids:
              type: array
              items:
                type: string
              example: ["doc-chunk-001", "doc-chunk-002"]
    responses:
      200:
        description: Vectors deleted
        schema:
          type: object
          example:
            deleted: 2
            index: my-index
            namespace: documents
            ids: ["doc-chunk-001", "doc-chunk-002"]
          properties:
            deleted:   {type: integer}
            index:     {type: string}
            namespace: {type: string}
            ids:       {type: array, items: {type: string}}
      400:
        description: Missing index or ids
      404:
        description: Index not found
    """
    body       = request.get_json(force=True)
    index_name = body.get("index")
    ids        = body.get("ids", [])
    namespace  = body.get("namespace", "")

    if not index_name or not ids:
        return jsonify({"error": "Missing 'index' or 'ids'"}), 400

    try:
        idx = pc.Index(index_name)
    except Exception as e:
        return jsonify({"error": f"Could not connect to index '{index_name}': {e}"}), 404

    idx.delete(ids=ids, namespace=namespace)
    log.info(f"[api/delete] Deleted {len(ids)} vectors from index={index_name} namespace={namespace!r}")

    return jsonify({"deleted": len(ids), "index": index_name, "namespace": namespace, "ids": ids})


@app.post("/api/delete-by-file")
@require_api_key
def api_delete_by_file():
    """
    Delete all vectors associated with a specific filename.

    Uses Pinecone's `fetch_by_metadata` endpoint (POST) to find matching vectors
    by `metadata.filename` — no URI length limits, no full index scan needed.

    **Example:**
    ```json
    {
      "index": "my-index",
      "namespace": "documents",
      "filename": "pinecone-intro.md"
    }
    ```
    ---
    tags: [ping]
    security:
      - BearerAuth: []
      - ApiKeyAuth: []
    parameters:
      - in: body
        name: body
        required: true
        schema:
          type: object
          required: [index, filename]
          properties:
            index:
              type: string
              example: my-index
            namespace:
              type: string
              example: documents
              description: "Namespace to search in. Default: ''"
            filename:
              type: string
              example: "pinecone-intro.md"
              description: "Exact filename to match against metadata.filename"
    responses:
      200:
        description: Vectors deleted
        schema:
          type: object
          example:
            deleted: 4
            index: my-index
            namespace: documents
            filename: "pinecone-intro.md"
            ids: ["doc-chunk-001", "doc-chunk-002", "doc-chunk-003", "doc-chunk-004"]
          properties:
            deleted:   {type: integer}
            index:     {type: string}
            namespace: {type: string}
            filename:  {type: string}
            ids:       {type: array, items: {type: string}}
      400:
        description: Missing index or filename
      404:
        description: Index not found
    """
    body       = request.get_json(force=True)
    index_name = body.get("index")
    filename   = body.get("filename")
    namespace  = body.get("namespace", "")

    if not index_name or not filename:
        return jsonify({"error": "Missing 'index' or 'filename'"}), 400

    try:
        idx = pc.Index(index_name)
    except Exception as e:
        return jsonify({"error": f"Could not connect to index '{index_name}': {e}"}), 404

    log.info(f"[api/delete-by-file] fetch_by_metadata filename={filename!r} index={index_name} namespace={namespace!r}")

    index_host       = pc.describe_index(index_name).host
    matched_ids      = []
    pagination_token = None

    while True:
        payload = {
            "namespace": namespace or "__default__",
            "filter":    {"filename": {"$eq": filename}},
            "limit":     100,
        }
        if pagination_token:
            payload["paginationToken"] = pagination_token

        resp = http.post(
            f"https://{index_host}/vectors/fetch_by_metadata",
            headers={
                "Api-Key":                PINECONE_API_KEY,
                "Content-Type":           "application/json",
                "X-Pinecone-Api-Version": "2025-10",
            },
            json=payload,
            timeout=30,
        )

        if resp.status_code != 200:
            log.error(f"[api/delete-by-file] fetch_by_metadata failed: {resp.status_code} {resp.text}")
            return jsonify({"error": f"Pinecone fetch_by_metadata failed: {resp.text}"}), 500

        data    = resp.json()
        vectors = data.get("vectors", {})
        matched_ids.extend(vectors.keys())
        log.info(f"[api/delete-by-file] Got {len(vectors)} matches this page")

        pagination_token = data.get("pagination", {}).get("next") if data.get("pagination") else None
        if not pagination_token:
            break

    if matched_ids:
        idx.delete(ids=matched_ids, namespace=namespace)
        log.info(f"[api/delete-by-file] Deleted {len(matched_ids)} vectors for filename={filename!r}")
    else:
        log.info(f"[api/delete-by-file] No vectors found for filename={filename!r}")

    return jsonify({
        "deleted":   len(matched_ids),
        "index":     index_name,
        "namespace": namespace,
        "filename":  filename,
        "ids":       matched_ids,
    })


# ── Entrypoint ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT)
