import pytest
import os
import sys
from unittest.mock import patch, MagicMock
from dotenv import load_dotenv

# Load .env from project root if it exists
env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
if os.path.exists(env_path):
    load_dotenv(dotenv_path=env_path)

@pytest.fixture
def app():
    from app import app as flask_app
    flask_app.config.update({
        "TESTING": True,
    })
    return flask_app

@pytest.fixture
def client(app):
    return app.test_client()

@pytest.fixture
def mock_pc():
    """Fixture to mock the Pinecone client instance used in app.py."""
    with patch("app.pc") as mock:
        yield mock
