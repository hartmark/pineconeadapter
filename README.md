# Pinecone Embedding Proxy

A lightweight Flask service that acts as an **Ollama-compatible embedding proxy** backed by [Pinecone Inference API](https://docs.pinecone.io/guides/inference/generate-embeddings). It also exposes OpenAI-compatible endpoints and a set of extended index management endpoints.

## Why?

Running local embedding models (e.g. via Ollama) on CPU is slow — easily 30–90 seconds per request on low-powered hardware. This proxy offloads all embedding work to Pinecone's cloud infrastructure while keeping the same API surface that Ollama clients expect. The free Starter tier includes **5M embedding tokens/month**, which is more than enough for most personal or small-scale deployments.

## Features

- **Ollama-compatible** — drop-in replacement for any tool that calls `/api/embeddings`
- **OpenAI-compatible** — `/v1/embeddings` and `/v1/models` for OpenAI SDK clients
- **Pinecone index management** — create, delete, clone, compare indexes via REST
- **Semantic search** — embed and search in one call
- **Upsert with metadata** — store text + filename + source alongside vectors
- **Delete by filename** — uses `fetch_by_metadata` (POST) to avoid URI length limits
- **Swagger UI** at `/apidocs`
- **Bearer token / X-API-Key auth**

## Quick start

```
# Clone the repo

cp .env.sample .env
# edit .env to fit your environment

# Run
docker compose up -d --build pineconeadapter
```

The service starts on port `11434` (same as Ollama) by default.

## Configuration

| Variable | Default | Description |
|---|---|---|
| `PINECONE_API_KEY` | required | Your Pinecone API key |
| `PINECONEADAPTER_API_KEY` | — | Bearer token for protecting endpoints. If unset, all endpoints are open. |
| `PORT` | `11434` | Port to listen on |

## Endpoints

| Method | Route | Description |
|---|---|---|
| `GET` | `/health` | Liveness + Pinecone API key check (no auth) |
| `GET` | `/apidocs` | Swagger UI |
| `GET` | `/api/tags` | Ollama model list |
| `POST` | `/api/embeddings` | Ollama-compatible embeddings |
| `GET` | `/v1/models` | OpenAI model list |
| `POST` | `/v1/embeddings` | OpenAI-compatible embeddings |
| `GET` | `/api/indexes` | List Pinecone indexes |
| `POST` | `/api/embed` | Embed text (single or batch) |
| `POST` | `/api/upsert` | Embed + upsert to index |
| `POST` | `/api/upsert-vectors` | Upsert pre-embedded vectors |
| `POST` | `/api/search` | Semantic search (supports text or vectors) |
| `POST` | `/api/delete` | Delete vectors by ID |
| `POST` | `/api/delete-by-file` | Delete all vectors for a filename |
| `POST` | `/api/indexes/create` | Create a new index |
| `DELETE` | `/api/indexes/<name>` | Delete an index |
| `POST` | `/api/indexes/clone` | Clone an index |
| `POST` | `/api/indexes/compare` | Compare search results across indexes |

## Free tier notes

Pinecone Starter (free) is limited to:
- AWS `us-east-1` only
- 5 indexes, 2 GB storage
- 5M embedding tokens/month

All `create` and `clone` operations default to `aws` / `us-east-1`.

## Recommended metadata schema

When upserting, include these fields for best results:

```json
{
  "text":       "the chunk text (auto-included, max 1000 chars)",
  "source":     "documents | sessions | archive | web | custom",
  "filename":   "path/to/source-file.md",
  "created_at": "2026-04-17T00:00:00Z",
  "tags":       ["optional", "labels"]
}
```

`filename` is particularly useful — it enables `delete-by-file` to work without scanning every vector manually.

## Using with OpenAI SDK

```python
from openai import OpenAI

client = OpenAI(
    api_key="your-adapter-key",
    base_url="http://localhost:11434/v1"
)

response = client.embeddings.create(
    model="llama-text-embed-v2",
    input="Hello world"
)
print(response.data[0].embedding)
```

## Using with Ollama clients

Any tool configured to use Ollama at `http://localhost:11434` will work without changes, including [OpenClaw](https://openclaw.dev) and other Ollama-compatible clients. Just set the embedding model to `llama-text-embed-v2`.

## Development and Testing

The project uses `pytest` for testing.

### Running tests
```bash
# Run all tests using the provided script
./test/run_tests.sh

# To run system tests against real Pinecone
PINECONE_INDEX=my-index ./test/run_tests.sh --system
```

Alternatively, you can run them manually:
```bash
# Create a virtual environment and install dependencies
python3 -m venv venv
./venv/bin/pip install -r requirements.txt pytest

# Run all tests (system tests will be skipped if PINECONE_API_KEY is dummy)
./venv/bin/pytest test
```

### System tests
To run tests against the real Pinecone API, ensure your `.env` file has a valid `PINECONE_API_KEY`. 
For `upsert` and `search` system tests, also provide a `PINECONE_INDEX` name:

```bash
PINECONE_INDEX=my-index ./venv/bin/pytest test/test_system.py
```

## Related

- [Pinecone MCP Server](https://github.com/pinecone-io/pinecone-mcp) — official MCP server for Cursor/Claude Desktop (indexes with integrated inference only)
- [Pinecone Inference docs](https://docs.pinecone.io/guides/inference/generate-embeddings)
- [llama-text-embed-v2 model](https://www.pinecone.io/learn/nvidia-for-pinecone-inference/)
