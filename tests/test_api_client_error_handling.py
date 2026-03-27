import json

import pytest

from gliner2.api_client import (
    AuthenticationError,
    GLiNER2API,
    GLiNER2APIError,
    ServerError,
)


class _FakeResponse:
    def __init__(self, status_code: int, content: bytes, json_data=None, json_error = None):
        self.status_code = status_code
        self.content = content
        self._json_data = json_data
        self._json_error = json_error

    @property
    def ok(self) -> bool:
        return 200 <= self.status_code < 300

    def json(self):
        if self._json_error is not None:
            raise self._json_error
        return self._json_data


@pytest.fixture
def api_client() -> GLiNER2API:
    return GLiNER2API(api_key="test-key")


def test_401_non_json_body_raises_authentication_error(api_client: GLiNER2API):
    response = _FakeResponse(
        status_code=401,
        content=b"<html>Unauthorized</html>",
        json_error=json.JSONDecodeError("Expecting value", "", 0),
    )
    api_client.session.post = lambda *args, **kwargs: response

    with pytest.raises(AuthenticationError) as exc_info:
        api_client.extract_entities("hello", ["person"])

    assert str(exc_info.value) == "Invalid or expired API key"
    assert exc_info.value.status_code == 401
    assert exc_info.value.response_data is None


def test_500_non_json_body_raises_server_error(api_client: GLiNER2API):
    response = _FakeResponse(
        status_code=500,
        content=b"<html>Server Error</html>",
        json_error=json.JSONDecodeError("Expecting value", "", 0),
    )
    api_client.session.post = lambda *args, **kwargs: response

    with pytest.raises(ServerError) as exc_info:
        api_client.extract_entities("hello", ["person"])

    assert str(exc_info.value) == "Server error occurred"
    assert exc_info.value.status_code == 500
    assert exc_info.value.response_data is None


def test_200_non_json_body_raises_api_error(api_client: GLiNER2API):
    response = _FakeResponse(
        status_code=200,
        content=b"<html>OK but not JSON</html>",
        json_error=json.JSONDecodeError("Expecting value", "", 0),
    )
    api_client.session.post = lambda *args, **kwargs: response

    with pytest.raises(GLiNER2APIError) as exc_info:
        api_client.extract_entities("hello", ["person"])

    assert "Invalid JSON response from API" in str(exc_info.value)
    assert exc_info.value.status_code == 200
