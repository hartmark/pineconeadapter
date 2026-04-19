import pytest
import os
from unittest.mock import MagicMock, patch

def test_health_ok(client, mock_pc):
    # Mock get_model to return a mock with dimension
    mock_model = MagicMock()
    mock_model.dimension = 1024
    mock_pc.inference.get_model.return_value = mock_model
    
    mock_pc.inference.embed.return_value = MagicMock()
    mock_pc.list_indexes.return_value = MagicMock()
    
    response = client.get("/health")
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "ok"
    assert "model" in data
    assert "dimensions" in data
    assert isinstance(data["dimensions"], int)

def test_health_error(client, mock_pc):
    # Mock get_model to return a mock with dimension
    mock_model = MagicMock()
    mock_model.dimension = 1024
    mock_pc.inference.get_model.return_value = mock_model
    
    mock_pc.inference.embed.side_effect = Exception("Pinecone down")
    response = client.get("/health")
    assert response.status_code == 503
    assert response.get_json()["status"] == "error"

def test_api_embeddings_ollama(client, mock_pc):
    # Mock return from Pinecone inference
    mock_pc.inference.embed.return_value.data = [{"values": [0.1, 0.2]}]
    
    headers = {"X-API-Key": os.environ.get("PINECONEADAPTER_API_KEY", "test-secret")}
    payload = {"model": "llama-text-embed-v2", "prompt": "hello"}
    
    response = client.post("/api/embeddings", json=payload, headers=headers)
    assert response.status_code == 200
    data = response.get_json()
    assert "embedding" in data
    assert data["embedding"] == [0.1, 0.2]

def test_v1_embeddings_openai(client, mock_pc):
    mock_pc.inference.embed.return_value.data = [{"values": [0.3, 0.4]}]
    
    headers = {"Authorization": f"Bearer {os.environ.get('PINECONEADAPTER_API_KEY', 'test-secret')}"}
    payload = {"model": "text-embedding-3-small", "input": "world"}
    
    response = client.post("/v1/embeddings", json=payload, headers=headers)
    assert response.status_code == 200
    data = response.get_json()
    assert data["object"] == "list"
    assert data["data"][0]["embedding"] == [0.3, 0.4]

def test_api_upsert(client, mock_pc):
    mock_pc.inference.embed.return_value.data = [{"values": [0.5, 0.6]}]
    mock_idx = MagicMock()
    mock_pc.Index.return_value = mock_idx
    
    headers = {"X-API-Key": os.environ.get("PINECONEADAPTER_API_KEY", "test-secret")}
    payload = {
        "index": "my-index",
        "model": "llama-text-embed-v2",
        "records": [{"id": "1", "text": "content"}]
    }
    
    response = client.post("/api/upsert", json=payload, headers=headers)
    assert response.status_code == 200
    assert response.get_json()["upserted"] == 1
    mock_idx.upsert.assert_called_once()

def test_api_upsert_vectors(client, mock_pc):
    mock_idx = MagicMock()
    mock_pc.Index.return_value = mock_idx
    
    headers = {"X-API-Key": os.environ.get("PINECONEADAPTER_API_KEY", "test-secret")}
    payload = {
        "index": "my-index",
        "records": [
            {
                "id": "v1", 
                "values": [0.1, 0.2], 
                "text": "original text",
                "metadata": {"custom": "value"}
            }
        ]
    }
    
    response = client.post("/api/upsert-vectors", json=payload, headers=headers)
    assert response.status_code == 200
    data = response.get_json()
    assert data["upserted"] == 1
    assert data["index"] == "my-index"
    
    # Verify what was sent to Pinecone
    mock_idx.upsert.assert_called_once()
    args, kwargs = mock_idx.upsert.call_args
    vectors = kwargs["vectors"]
    assert len(vectors) == 1
    assert vectors[0]["id"] == "v1"
    assert vectors[0]["values"] == [0.1, 0.2]
    assert vectors[0]["metadata"]["text"] == "original text"
    assert vectors[0]["metadata"]["custom"] == "value"
    assert "created_at" in vectors[0]["metadata"]
    
    # Verify that Pinecone inference (embedding) was NOT called
    mock_pc.inference.embed.assert_not_called()

def test_api_search(client, mock_pc):
    # Mock embedding the query
    mock_pc.inference.embed.return_value.data = [{"values": [0.7, 0.8]}]
    # Mock query result
    mock_idx = MagicMock()
    mock_idx.query.return_value = {
        "matches": [{"id": "1", "score": 0.9, "metadata": {"text": "found it"}}]
    }
    mock_pc.Index.return_value = mock_idx
    
    headers = {"X-API-Key": os.environ.get("PINECONEADAPTER_API_KEY", "test-secret")}
    payload = {"index": "my-index", "model": "llama-text-embed-v2", "query": "search query"}
    
    response = client.post("/api/search", json=payload, headers=headers)
    assert response.status_code == 200
    data = response.get_json()
    assert data["count"] == 1
    assert data["matches"][0]["text"] == "found it"

def test_api_search_vector(client, mock_pc):
    # Mock query result
    mock_idx = MagicMock()
    mock_idx.query.return_value = {
        "matches": [{"id": "v1", "score": 0.95, "metadata": {"text": "vector match"}}]
    }
    mock_pc.Index.return_value = mock_idx
    
    headers = {"X-API-Key": os.environ.get("PINECONEADAPTER_API_KEY", "test-secret")}
    payload = {
        "index": "my-index", 
        "vector": [0.1, 0.2], 
        "top_k": 1
    }
    
    response = client.post("/api/search", json=payload, headers=headers)
    assert response.status_code == 200
    data = response.get_json()
    assert data["count"] == 1
    assert data["matches"][0]["id"] == "v1"
    
    # Verify that Pinecone inference (embedding) was NOT called
    mock_pc.inference.embed.assert_not_called()
    # Verify idx.query was called with vector
    mock_idx.query.assert_called_once()
    args, kwargs = mock_idx.query.call_args
    assert kwargs["vector"] == [0.1, 0.2]

def test_unauthorized(client):
    response = client.get("/api/tags")
    assert response.status_code == 401
