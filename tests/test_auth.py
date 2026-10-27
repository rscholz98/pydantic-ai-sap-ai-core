"""The auth flow on its own: token caching and refresh, 401 replay, routing, and the errors it raises.

These go through a bare `httpx2.AsyncClient` rather than the OpenAI SDK so that each behaviour can be
driven directly; `test_agent.py` covers the same flow end to end.
"""

from __future__ import annotations

import time

import anyio
import httpx2
import pytest
from pydantic_ai.exceptions import UserError

from pydantic_ai_sap_ai_core import SapAiCoreAuth

from .conftest import API_URL, AUTH_URL, CREDENTIALS, MockSap, deployment

CHAT_URL = f'{API_URL}/inference/deployments/model/chat/completions'


def client(sap: MockSap, auth: SapAiCoreAuth | None = None) -> httpx2.AsyncClient:
    return httpx2.AsyncClient(transport=sap.transport, auth=auth or SapAiCoreAuth(CREDENTIALS))


async def test_token_is_fetched_once_and_reused(sap: MockSap):
    async with client(sap) as http:
        for _ in range(3):
            response = await http.post(CHAT_URL, json={'model': 'gpt-4o', 'messages': []})
            assert response.status_code == 200

    token_requests = sap.requests_to('/oauth/token')
    assert len(token_requests) == 1
    assert token_requests[0].method == 'POST'
    assert token_requests[0].headers['content-type'] == 'application/x-www-form-urlencoded'
    assert token_requests[0].content == b'grant_type=client_credentials&client_id=sb-client&client_secret=top-secret'
    assert 'authorization' not in token_requests[0].headers

    assert len(sap.requests_to('/lm/deployments')) == 1
    chat_requests = sap.requests_to('/chat/completions')
    assert len(chat_requests) == 3
    for request in chat_requests:
        assert str(request.url) == f'{API_URL}/inference/deployments/d768512472eae8f4/chat/completions'
        assert request.headers['authorization'] == 'Bearer token-1'
        assert request.headers['ai-resource-group'] == 'default'
        assert request.headers['ai-client-type'] == 'pydantic-ai-sap-ai-core'


async def test_token_is_refreshed_before_expiry(sap: MockSap, monkeypatch: pytest.MonkeyPatch):
    sap.expires_in = 120
    auth = SapAiCoreAuth(CREDENTIALS)
    async with client(sap, auth) as http:
        await http.post(CHAT_URL, json={'model': 'gpt-4o'})
        now = time.monotonic()
        # Thirty seconds in, with ninety left, the token is still used...
        monkeypatch.setattr(time, 'monotonic', lambda: now + 30)
        await http.post(CHAT_URL, json={'model': 'gpt-4o'})
        assert sap.tokens_issued == 1
        # ...but at a minute to expiry it is refreshed.
        monkeypatch.setattr(time, 'monotonic', lambda: now + 61)
        await http.post(CHAT_URL, json={'model': 'gpt-4o'})
        assert sap.tokens_issued == 2
        assert sap.requests_to('/chat/completions')[-1].headers['authorization'] == 'Bearer token-2'


async def test_401_refreshes_the_token_and_replays_once(sap: MockSap):
    sap.inference_responses = [
        httpx2.Response(401, json={'error': 'Jwt expired'}),
        httpx2.Response(200, json={'ok': True}),
    ]
    async with client(sap) as http:
        response = await http.post(CHAT_URL, json={'model': 'gpt-4o'})
    assert response.status_code == 200
    assert sap.tokens_issued == 2
    tokens_sent = [r.headers['authorization'] for r in sap.requests_to('/chat/completions')]
    assert tokens_sent == ['Bearer token-1', 'Bearer token-2']


async def test_second_401_is_returned(sap: MockSap):
    sap.inference_responses = [httpx2.Response(401, json={'error': 'nope'})]
    async with client(sap) as http:
        response = await http.post(CHAT_URL, json={'model': 'gpt-4o'})
    assert response.status_code == 401
    assert sap.tokens_issued == 2
    assert len(sap.requests_to('/chat/completions')) == 2


async def test_401_after_another_request_refreshed_reuses_the_new_token(sap: MockSap):
    """A 401 answered with a token that is already stale does not trigger a second refresh."""
    auth = SapAiCoreAuth(CREDENTIALS)
    sap.inference_responses = [httpx2.Response(401, json={'error': 'Jwt expired'}), httpx2.Response(200)]
    async with client(sap, auth) as http:
        await http.post(CHAT_URL, json={'model': 'gpt-4o'})  # fetches token-1, gets 401, refreshes to token-2
        assert sap.tokens_issued == 2
        # Simulate a concurrent flow having refreshed meanwhile by swapping the cached token under a 401.
        sap.inference_responses = [httpx2.Response(401, json={'error': 'Jwt expired'}), httpx2.Response(200)]
        stale = auth._token  # pyright: ignore[reportPrivateUsage]
        assert stale is not None
        original_apply = auth._apply  # pyright: ignore[reportPrivateUsage]

        def apply_and_rotate(request: httpx2.Request, token: str) -> None:
            original_apply(request, token)
            if auth._token is stale:  # pyright: ignore[reportPrivateUsage]
                auth._token = type(stale)('token-from-elsewhere', stale.expires_at)  # pyright: ignore[reportPrivateUsage]

        auth._apply = apply_and_rotate  # pyright: ignore[reportPrivateUsage]
        response = await http.post(CHAT_URL, json={'model': 'gpt-4o'})
    assert response.status_code == 200
    assert sap.tokens_issued == 2
    assert sap.requests_to('/chat/completions')[-1].headers['authorization'] == 'Bearer token-from-elsewhere'


async def test_token_endpoint_failure(sap: MockSap):
    sap.token_responses = [httpx2.Response(401, json={'error': 'unauthorized', 'error_description': 'Bad credentials'})]
    async with client(sap) as http:
        with pytest.raises(UserError, match=f'token request to {AUTH_URL} failed with HTTP 401: .*Bad credentials'):
            await http.post(CHAT_URL, json={'model': 'gpt-4o'})


async def test_token_response_without_access_token(sap: MockSap):
    sap.token_responses = [httpx2.Response(200, json={'token_type': 'bearer'})]
    async with client(sap) as http:
        with pytest.raises(UserError, match='is not a valid token'):
            await http.post(CHAT_URL, json={'model': 'gpt-4o'})


async def test_token_response_without_expires_in_defaults_to_an_hour(sap: MockSap, monkeypatch: pytest.MonkeyPatch):
    sap.token_responses = [httpx2.Response(200, json={'access_token': 'eternal'})]
    auth = SapAiCoreAuth(CREDENTIALS)
    async with client(sap, auth) as http:
        await http.post(CHAT_URL, json={'model': 'gpt-4o'})
        now = time.monotonic()
        monkeypatch.setattr(time, 'monotonic', lambda: now + 3600 - 59)
        await http.post(CHAT_URL, json={'model': 'gpt-4o'})
    assert len(sap.requests_to('/oauth/token')) == 2


async def test_requests_outside_the_placeholder_are_only_authenticated(sap: MockSap):
    sap.inference_responses = [httpx2.Response(200)]
    async with client(sap) as http:
        response = await http.get(f'{API_URL}/inference/deployments/d768512472eae8f4/chat/completions')
    assert response.status_code == 200
    assert sap.requests_to('/lm/deployments') == []
    request = sap.requests_to('/chat/completions')[0]
    assert request.headers['authorization'] == 'Bearer token-1'
    assert request.headers['ai-resource-group'] == 'default'


async def test_routing_keeps_endpoint_and_query(sap: MockSap):
    async with client(sap) as http:
        await http.post(f'{CHAT_URL}?api-version=2024-10-21', json={'model': 'gpt-4o-mini'})
    request = sap.requests_to('/chat/completions')[0]
    assert (
        str(request.url) == f'{API_URL}/inference/deployments/dac9dca90be5213a/chat/completions?api-version=2024-10-21'
    )
    assert request.headers['host'] == 'api.ai.ml.hana.ondemand.com'


async def test_routing_to_another_host_rewrites_the_host_header(sap: MockSap):
    sap.deployments = [
        {**deployment('gpt-4o', 'abc'), 'deploymentUrl': 'https://other.example.com/v2/inference/deployments/abc'}
    ]
    sap.inference_responses = [httpx2.Response(200)]
    async with client(sap) as http:
        await http.post(CHAT_URL, json={'model': 'gpt-4o'})
    request = sap.requests[-1]
    assert str(request.url) == 'https://other.example.com/v2/inference/deployments/abc/chat/completions'
    assert request.headers['host'] == 'other.example.com'


async def test_explicit_deployment_ids_skip_the_listing(sap: MockSap):
    auth = SapAiCoreAuth(CREDENTIALS, deployment_ids={'gpt-4o': 'explicit-id'})
    async with client(sap, auth) as http:
        await http.post(CHAT_URL, json={'model': 'gpt-4o'})
        await http.post(CHAT_URL, json={'model': 'gpt-4o-mini'})
    assert len(sap.requests_to('/lm/deployments')) == 1
    urls = [str(r.url) for r in sap.requests_to('/chat/completions')]
    assert urls == [
        f'{API_URL}/inference/deployments/explicit-id/chat/completions',
        f'{API_URL}/inference/deployments/dac9dca90be5213a/chat/completions',
    ]


async def test_listing_is_sent_with_the_right_query_and_headers(sap: MockSap):
    auth = SapAiCoreAuth(
        CREDENTIALS.__class__(**{**CREDENTIALS.__dict__, 'resource_group': 'team-a'}),
    )
    async with client(sap, auth) as http:
        await http.post(CHAT_URL, json={'model': 'gpt-4o'})
    listing = sap.requests_to('/lm/deployments')[0]
    assert listing.method == 'GET'
    assert dict(listing.url.params) == {'scenarioId': 'foundation-models', 'status': 'RUNNING'}
    assert listing.headers['authorization'] == 'Bearer token-1'
    assert listing.headers['ai-resource-group'] == 'team-a'


async def test_unknown_model_refreshes_the_listing_then_fails(sap: MockSap):
    async with client(sap) as http:
        await http.post(CHAT_URL, json={'model': 'gpt-4o'})
        with pytest.raises(UserError) as exc_info:
            await http.post(CHAT_URL, json={'model': 'gpt-5'})
    assert str(exc_info.value) == (
        "No running SAP AI Core deployment serves model 'gpt-5' in resource group 'default'. "
        'Running foundation-model deployments: anthropic--claude-4-sonnet, gpt-4o, gpt-4o-mini, text-embedding-ada-002. '
        'Deploy the model in SAP AI Launchpad, or map the name to a deployment id with '
        '`SapAiCoreProvider(deployment_ids={...})`.'
    )
    assert len(sap.requests_to('/lm/deployments')) == 2


async def test_model_deployed_meanwhile_is_found_on_the_refresh(sap: MockSap):
    async with client(sap) as http:
        await http.post(CHAT_URL, json={'model': 'gpt-4o'})
        sap.deployments.append(deployment('gpt-5', 'new-one'))
        await http.post(CHAT_URL, json={'model': 'gpt-5'})
    assert str(sap.requests[-1].url) == f'{API_URL}/inference/deployments/new-one/chat/completions'


async def test_listing_is_refetched_after_the_cache_ttl(sap: MockSap, monkeypatch: pytest.MonkeyPatch):
    auth = SapAiCoreAuth(CREDENTIALS, deployment_cache_ttl=10)
    async with client(sap, auth) as http:
        await http.post(CHAT_URL, json={'model': 'gpt-4o'})
        now = time.monotonic()
        monkeypatch.setattr(time, 'monotonic', lambda: now + 5)
        await http.post(CHAT_URL, json={'model': 'gpt-4o'})
        assert len(sap.requests_to('/lm/deployments')) == 1
        monkeypatch.setattr(time, 'monotonic', lambda: now + 11)
        await http.post(CHAT_URL, json={'model': 'gpt-4o'})
        assert len(sap.requests_to('/lm/deployments')) == 2


async def test_non_openai_executable_is_rejected(sap: MockSap):
    async with client(sap) as http:
        with pytest.raises(UserError, match="'anthropic--claude-4-sonnet' through the 'aws-bedrock' executable"):
            await http.post(CHAT_URL, json={'model': 'anthropic--claude-4-sonnet'})


async def test_listing_failure(sap: MockSap):
    sap.listing_status = 403
    async with client(sap) as http:
        with pytest.raises(UserError, match="failed with HTTP 403: .*forbidden.*resource group 'default'"):
            await http.post(CHAT_URL, json={'model': 'gpt-4o'})


async def test_listing_unparseable(sap: MockSap):
    sap.deployments = [{'id': 42}]  # a resource whose id is not a string
    async with client(sap) as http:
        with pytest.raises(UserError, match='deployment listing could not be parsed'):
            await http.post(CHAT_URL, json={'model': 'gpt-4o'})


async def test_body_without_model(sap: MockSap):
    async with client(sap) as http:
        with pytest.raises(UserError, match='names no `model`'):
            await http.post(CHAT_URL, json={'messages': []})
        with pytest.raises(UserError, match='names no `model`'):
            await http.post(CHAT_URL, content=b'not json')


def test_sync_client_is_rejected(sap: MockSap):
    with httpx2.Client(transport=httpx2.MockTransport(sap.handler), auth=SapAiCoreAuth(CREDENTIALS)) as http:
        with pytest.raises(UserError, match='requires an async HTTP client'):
            http.post(CHAT_URL, json={'model': 'gpt-4o'})


async def test_concurrent_cold_start_fetches_token_and_listing_once(sap: MockSap):
    """Two requests racing on a cold cache share one token fetch and one listing.

    The fake tenant holds the first token and listing responses until the test releases them, so the
    second request is certain to be waiting on the lock when the first one fills the cache.
    """
    reached_token, release_token = anyio.Event(), anyio.Event()
    reached_listing, release_listing = anyio.Event(), anyio.Event()

    async def slow_handler(request: httpx2.Request) -> httpx2.Response:
        if request.url.path.endswith('/oauth/token') and not reached_token.is_set():
            reached_token.set()
            await release_token.wait()
        elif request.url.path.endswith('/lm/deployments') and not reached_listing.is_set():
            reached_listing.set()
            await release_listing.wait()
        return sap.handler(request)

    auth = SapAiCoreAuth(CREDENTIALS)
    statuses: list[int] = []

    async def post() -> None:
        response = await http.post(CHAT_URL, json={'model': 'gpt-4o'})
        statuses.append(response.status_code)

    async with httpx2.AsyncClient(transport=httpx2.MockTransport(slow_handler), auth=auth) as http:
        async with anyio.create_task_group() as tg:
            tg.start_soon(post)
            tg.start_soon(post)
            await reached_token.wait()
            for _ in range(10):  # let the second request reach the lock
                await anyio.sleep(0)
            release_token.set()
            await reached_listing.wait()
            for _ in range(10):
                await anyio.sleep(0)
            release_listing.set()

    assert statuses == [200, 200]
    assert len(sap.requests_to('/oauth/token')) == 1
    assert len(sap.requests_to('/lm/deployments')) == 1
    assert len(sap.requests_to('/chat/completions')) == 2
