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
| `PINECONEADAPTERAPI_KEY` | — | Bearer token for protecting endpoints. If unset, all endpoints are open. |
| `EMBED_MODEL` | `llama-text-embed-v2` | Pinecone inference model to use |
| `EMBED_DIMENSIONS` | `2048` | Embedding dimensions |
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
| `POST` | `/api/search` | Semantic search |
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

## Related

- [Pinecone MCP Server](https://github.com/pinecone-io/pinecone-mcp) — official MCP server for Cursor/Claude Desktop (indexes with integrated inference only)
- [Pinecone Inference docs](https://docs.pinecone.io/guides/inference/generate-embeddings)
- [llama-text-embed-v2 model](https://www.pinecone.io/learn/nvidia-for-pinecone-inference/)
