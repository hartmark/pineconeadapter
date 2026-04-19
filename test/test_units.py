import pytest
from unittest.mock import MagicMock, patch
from app import sanitize, _embed_with_retry, do_embed, do_embed_query, get_model_dimensions
from pinecone.exceptions import PineconeApiException, PineconeApiTypeError

def test_sanitize():
    assert sanitize("hello") == "hello"
    # Just check it doesn't crash and returns something
    assert sanitize("hello\xff")

@patch("app.pc.inference.embed")
def test_embed_with_retry_success(mock_embed):
    mock_embed.return_value.data = [{"values": [0.1, 0.2]}]
    # Explicitly provide model and dimensions for unit tests
    res = _embed_with_retry(["test"], "passage", model="m1", dimensions=2)
    assert res == [[0.1, 0.2]]
    mock_embed.assert_called_once()

@patch("app.pc.inference.embed")
@patch("time.sleep", return_value=None)
def test_embed_with_retry_rate_limit(mock_sleep, mock_embed):
    # Simulate 429 then success
    error_429 = Exception()
    error_429.status = 429
    
    success_mock = MagicMock()
    success_mock.data = [{"values": [0.3]}]
    
    mock_embed.side_effect = [error_429, success_mock]
    
    # We must patch PineconeApiException in app to match our Exception
    with patch("app.PineconeApiException", Exception):
        res = _embed_with_retry(["test"], "passage", model="m1", dimensions=1)
        assert res == [[0.3]]
        assert mock_embed.call_count == 2
        mock_sleep.assert_called_once()

@patch("app._embed_with_retry")
def test_do_embed(mock_retry):
    mock_retry.return_value = [[0.1]]
    res = do_embed(["hello"], model="m", dimensions=1)
    assert res == [[0.1]]
    mock_retry.assert_called_with(["hello"], "passage", "m", 1)

@patch("app._embed_with_retry")
def test_do_embed_query(mock_retry):
    mock_retry.return_value = [[0.9]]
    res = do_embed_query("query", model="m", dimensions=1)
    assert res == [0.9]
    mock_retry.assert_called_with(["query"], "query", "m", 1)

@patch("app.pc.inference.get_model")
def test_get_model_dimensions_success(mock_get_model):
    mock_model_info = MagicMock()
    mock_model_info.dimension = 768
    mock_get_model.return_value = mock_model_info
    
    # Clear cache if it exists from other tests (though it shouldn't if we use a new model name)
    if hasattr(get_model_dimensions, "_cache"):
        get_model_dimensions._cache.pop("test-model", None)
    
    dims = get_model_dimensions("test-model")
    assert dims == 768
    mock_get_model.assert_called_with(model_name="test-model")

@patch("app.pc.inference.get_model")
def test_get_model_dimensions_fallback(mock_get_model):
    mock_get_model.side_effect = Exception("API Error")
    
    if hasattr(get_model_dimensions, "_cache"):
        get_model_dimensions._cache.pop("error-model", None)
        
    dims = get_model_dimensions("error-model")
    assert dims == 1024

@patch("app.pc.inference.embed")
@patch("app.get_model_dimensions")
def test_embed_multilingual_e5_large_no_dimension(mock_get_dims, mock_embed):
    mock_get_dims.return_value = 1024
    mock_embed.return_value.data = [{"values": [0.1, 0.2]}]
    _embed_with_retry(["test"], "passage", model="multilingual-e5-large")
    # Check that dimension is NOT in parameters
    args, kwargs = mock_embed.call_args
    assert "dimension" not in kwargs["parameters"]
    assert kwargs["parameters"]["input_type"] == "passage"

@patch("app.pc.Index")
def test_api_upsert_vectors_dimension_mismatch(mock_index_class, client):
    mock_idx = MagicMock()
    mock_index_class.return_value = mock_idx
    
    # Simulate PineconeApiException for dimension mismatch
    error_response = MagicMock()
    error_response.status = 400
    error_response.data = '{"code":3,"message":"Vector dimension 3 does not match the dimension of the index 10","details":[]}'
    # The SDK uses http_resp.data for body
    error_response.body = error_response.data
    
    error = PineconeApiException(http_resp=error_response)
    mock_idx.upsert.side_effect = error
    
    headers = {"X-API-Key": "test-secret"}
    payload = {
        "index": "my-index",
        "records": [{"id": "v1", "values": [0.1, 0.2, 0.3]}]
    }
    
    with patch("app.API_KEY", "test-secret"):
        response = client.post("/api/upsert-vectors", json=payload, headers=headers)
    
    assert response.status_code == 400
    # Should be the clean message now
    assert response.get_json()["error"] == "Vector dimension 3 does not match the dimension of the index 10"

@patch("app.pc.Index")
def test_api_upsert_vectors_hard_fault(mock_index_class, client):
    mock_idx = MagicMock()
    mock_index_class.return_value = mock_idx
    
    # Simulate PineconeApiException for a 500 error (hard fault)
    error_response = MagicMock()
    error_response.status = 500
    error_response.data = '{"message":"Internal Server Error"}'
    error_response.body = error_response.data
    
    error = PineconeApiException(http_resp=error_response)
    mock_idx.upsert.side_effect = error
    
    headers = {"X-API-Key": "test-secret"}
    payload = {
        "index": "my-index",
        "records": [{"id": "v1", "values": [0.1, 0.2, 0.3]}]
    }
    
    with patch("app.API_KEY", "test-secret"):
        response = client.post("/api/upsert-vectors", json=payload, headers=headers)
    
    assert response.status_code == 500
    assert response.get_json()["error"] == "Internal Server Error"
