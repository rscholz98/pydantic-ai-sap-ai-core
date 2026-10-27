"""A fake SAP AI Core tenant served through `httpx2.MockTransport`.

The token endpoint, the deployment listing and the inference endpoints are all served from one
handler that records every request it sees, so tests can assert what went over the wire. Model
responses come from SAP's own recorded Azure OpenAI responses under `fixtures/sap_ai_sdk_js/`.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx2
import pytest

from pydantic_ai_sap_ai_core import SapAiCoreCredentials, SapAiCoreProvider

FIXTURES = Path(__file__).parent / 'fixtures' / 'sap_ai_sdk_js'

AUTH_URL = 'https://example.authentication.eu12.hana.ondemand.com/oauth/token'
API_URL = 'https://api.ai.ml.hana.ondemand.com/v2'
CREDENTIALS = SapAiCoreCredentials(
    client_id='sb-client',
    client_secret='top-secret',
    auth_url='https://example.authentication.eu12.hana.ondemand.com',
    base_url='https://api.ai.ml.hana.ondemand.com',
)

SERVICE_KEY: dict[str, Any] = {
    'clientid': 'sb-client',
    'clientsecret': 'top-secret',
    'url': 'https://example.authentication.eu12.hana.ondemand.com',
    'identityzone': 'examplezone',
    'appname': 'appname',
    'serviceurls': {'AI_API_URL': 'https://api.ai.ml.hana.ondemand.com'},
}


def deployment(
    model_name: str,
    deployment_id: str,
    *,
    created_at: str = '2026-01-01T00:00:00Z',
    executable_id: str | None = 'azure-openai',
    with_details: bool = True,
) -> dict[str, Any]:
    """One resource of a `GET /v2/lm/deployments` listing, shaped like SAP's Python SDK test data."""
    resource: dict[str, Any] = {
        'configurationId': f'config-{deployment_id}',
        'configurationName': model_name,
        'createdAt': created_at,
        'deploymentUrl': f'{API_URL}/inference/deployments/{deployment_id}',
        'id': deployment_id,
        'lastOperation': 'CREATE',
        'modifiedAt': created_at,
        'scenarioId': 'foundation-models',
        'status': 'RUNNING',
        'targetStatus': 'RUNNING',
    }
    if executable_id is not None:
        resource['executableId'] = executable_id
    if with_details:
        resource['details'] = {
            'resources': {'backend_details': {'model': {'name': model_name, 'version': 'latest'}}},
            'scaling': {'backend_details': {}},
        }
    return resource


DEPLOYMENTS = [
    deployment('gpt-4o', 'd768512472eae8f4'),
    deployment('gpt-4o-mini', 'dac9dca90be5213a'),
    deployment('anthropic--claude-4-sonnet', 'd1a82575de60e02a', executable_id='aws-bedrock'),
    deployment('text-embedding-ada-002', 'deebf33e5ec3450c'),
]


def load_fixture(name: str) -> bytes:
    return (FIXTURES / name).read_bytes()


def sse(name: str) -> httpx2.Response:
    """A streamed chat completion, served chunk by chunk like SAP AI Core does."""
    body = load_fixture(name)
    return httpx2.Response(200, content=body, headers={'content-type': 'text/event-stream'})


def chat_completion(name: str = 'azure-openai-chat-completion-success-response.json') -> httpx2.Response:
    return httpx2.Response(200, content=load_fixture(name), headers={'content-type': 'application/json'})


@dataclass
class MockSap:
    """The fake tenant: configurable responses plus a record of every request."""

    token_responses: list[httpx2.Response] = field(default_factory=list[httpx2.Response])
    """Served in order; the last one is repeated. Empty means a valid token every time."""
    expires_in: int = 43199
    deployments: list[dict[str, Any]] = field(default_factory=lambda: list(DEPLOYMENTS))
    listing_status: int = 200
    inference_responses: list[httpx2.Response] = field(default_factory=list[httpx2.Response])
    """Served in order to inference calls; the last one is repeated."""
    requests: list[httpx2.Request] = field(default_factory=list[httpx2.Request])
    tokens_issued: int = 0

    def handler(self, request: httpx2.Request) -> httpx2.Response:
        # Copy, because a replayed request is the same object with different headers.
        self.requests.append(
            httpx2.Request(request.method, request.url, headers=dict(request.headers), content=request.content)
        )
        url = str(request.url).split('?', 1)[0]
        if url == AUTH_URL:
            if self.token_responses:
                return self.token_responses.pop(0) if len(self.token_responses) > 1 else self.token_responses[0]
            self.tokens_issued += 1
            return httpx2.Response(
                200,
                json={
                    'access_token': f'token-{self.tokens_issued}',
                    'token_type': 'bearer',
                    'expires_in': self.expires_in,
                    'scope': 'aicore!b1.read',
                },
            )
        if url == f'{API_URL}/lm/deployments':
            if self.listing_status != 200:
                return httpx2.Response(self.listing_status, json={'error': {'message': 'forbidden'}})
            return httpx2.Response(200, json={'count': len(self.deployments), 'resources': self.deployments})
        if '/inference/deployments/' in url:
            if not self.inference_responses:
                return chat_completion()
            if len(self.inference_responses) > 1:
                return self.inference_responses.pop(0)
            return self.inference_responses[0]
        return httpx2.Response(404, json={'error': f'unexpected request to {url}'})  # pragma: no cover

    @property
    def transport(self) -> httpx2.MockTransport:
        return httpx2.MockTransport(self.handler)

    def client(self) -> httpx2.AsyncClient:
        return httpx2.AsyncClient(transport=self.transport)

    def requests_to(self, path_suffix: str) -> list[httpx2.Request]:
        return [r for r in self.requests if r.url.path.endswith(path_suffix)]

    def bodies_to(self, path_suffix: str) -> list[dict[str, Any]]:
        return [json.loads(r.content) for r in self.requests_to(path_suffix)]


@pytest.fixture
def sap() -> MockSap:
    return MockSap()


ProviderFactory = Callable[..., SapAiCoreProvider]


@pytest.fixture
def provider_factory(sap: MockSap) -> ProviderFactory:
    """Build a provider wired to the fake tenant; keyword arguments go to `SapAiCoreProvider`."""

    def factory(**kwargs: Any) -> SapAiCoreProvider:
        kwargs.setdefault('credentials', CREDENTIALS)
        kwargs.setdefault('http_client', sap.client())
        return SapAiCoreProvider(**kwargs)

    return factory


@pytest.fixture
def provider(provider_factory: ProviderFactory) -> SapAiCoreProvider:
    return provider_factory()


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[pytest.MonkeyPatch]:
    """No `AICORE_*` or `VCAP_SERVICES` variables, and a config home with no config file in it."""
    for name in (
        'AICORE_SERVICE_KEY',
        'AICORE_CLIENT_ID',
        'AICORE_CLIENT_SECRET',
        'AICORE_AUTH_URL',
        'AICORE_BASE_URL',
        'AICORE_RESOURCE_GROUP',
        'AICORE_CONFIG',
        'AICORE_PROFILE',
        'VCAP_SERVICES',
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv('AICORE_HOME', str(tmp_path / 'aicore-home'))
    yield monkeypatch
