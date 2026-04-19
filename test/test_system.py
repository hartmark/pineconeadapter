import pytest
import os
import time
from pinecone import Pinecone, ServerlessSpec

@pytest.fixture(scope="module")
def pc_client():
    api_key = os.environ.get("PINECONE_API_KEY")
    if not api_key:
        pytest.skip("PINECONE_API_KEY not set")
    return Pinecone(api_key=api_key)

@pytest.fixture(scope="module")
def temp_index(pc_client):
    """Fixture to create a temporary Pinecone index for testing."""
    index_name = f"test-sys-{int(time.time())}"
    dimensions = 1024
    
    # Create the index
    pc_client.create_index(
        name=index_name,
        dimension=dimensions,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
    
    # Wait for the index to be ready
    while not pc_client.describe_index(index_name).status['ready']:
        time.sleep(1)
    
    yield index_name
    
    # Cleanup: delete the index
    pc_client.delete_index(index_name)

def test_health_real(client):
    """Verify health check with real Pinecone connectivity."""
    response = client.get("/health", json={"model": "multilingual-e5-large"})
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "ok"

def test_api_embeddings_real(client):
    """Verify embeddings with real Pinecone inference."""
    headers = {"X-API-Key": os.environ.get("PINECONEADAPTER_API_KEY")}
    payload = {
        "model": "llama-text-embed-v2",
        "prompt": "This is a system test for Pinecone inference."
    }
    response = client.post("/api/embeddings", json=payload, headers=headers)
    assert response.status_code == 200
    data = response.get_json()
    assert "embedding" in data
    assert len(data["embedding"]) > 0

def test_api_indexes_real(client):
    """Verify listing indexes with real Pinecone."""
    headers = {"X-API-Key": os.environ.get("PINECONEADAPTER_API_KEY")}
    response = client.get("/api/indexes", headers=headers)
    assert response.status_code == 200
    data = response.get_json()
    assert "indexes" in data
    assert isinstance(data["indexes"], list)

def test_upsert_and_search_real(client, temp_index):
    """Verify upsert and search with real Pinecone using a temporary index."""
    headers = {"X-API-Key": os.environ.get("PINECONEADAPTER_API_KEY")}
    record_id = f"test-{int(time.time())}"
    text_content = "Pinecone system tests are running."
    
    # 1. Upsert
    upsert_payload = {
        "index": temp_index,
        "model": "multilingual-e5-large",
        "records": [{"id": record_id, "text": text_content}]
    }
    response = client.post("/api/upsert", json=upsert_payload, headers=headers)
    if response.status_code != 200:
        print(f"DEBUG: Upsert failed with {response.status_code}: {response.get_data(as_text=True)}")
    assert response.status_code == 200
    assert response.get_json()["upserted"] == 1
    
    # Wait a bit for eventual consistency
    time.sleep(5)
    
    # 2. Search
    search_payload = {
        "index": temp_index,
        "model": "multilingual-e5-large",
        "query": "system tests"
    }
    response = client.post("/api/search", json=search_payload, headers=headers)
    if response.status_code != 200:
        print(f"DEBUG: Search failed with {response.status_code}: {response.get_data(as_text=True)}")
    assert response.status_code == 200
    data = response.get_json()
    assert data["count"] > 0
    assert any(m["id"] == record_id for m in data["matches"])

def test_upsert_vectors_real(client, temp_index):
    """Verify upsert-vectors with real Pinecone using a temporary index."""
    headers = {"X-API-Key": os.environ.get("PINECONEADAPTER_API_KEY")}
    record_id = f"test-v-{int(time.time())}"
    # Use 1024 dimensions to match temp_index fixture
    vector = [0.1] * 1024
    
    payload = {
        "index": temp_index,
        "records": [
            {
                "id": record_id, 
                "values": vector, 
                "text": "Vector upsert test",
                "metadata": {"source": "system-test"}
            }
        ]
    }
    response = client.post("/api/upsert-vectors", json=payload, headers=headers)
    assert response.status_code == 200
    assert response.get_json()["upserted"] == 1
    
    # Wait a bit for eventual consistency
    time.sleep(5)
    
    # Verify by searching (using same vector)
    search_payload = {
        "index": temp_index,
        "vector": vector,
        "top_k": 1
    }
    # Note: /api/search might need vector support if not already there, 
    # but let's see if it works or if we should use pc_client directly
    response = client.post("/api/search", json=search_payload, headers=headers)
    assert response.status_code == 200
    data = response.get_json()
    assert data["count"] > 0
    assert data["matches"][0]["id"] == record_id
