"""Constructing the provider: credential arguments, injected clients, lifecycle, and model profiles."""

from __future__ import annotations

import json

import httpx2
import pytest
from openai import AsyncOpenAI
from pydantic_ai._json_schema import InlineDefsJsonSchemaTransformer
from pydantic_ai.exceptions import UserError
from pydantic_ai.profiles.openai import OpenAIJsonSchemaTransformer

from pydantic_ai_sap_ai_core import DEFAULT_API_VERSION, SapAiCoreAuth, SapAiCoreProvider, sap_ai_core_model_profile

from .conftest import API_URL, CREDENTIALS, SERVICE_KEY, MockSap, ProviderFactory

pytestmark = pytest.mark.usefixtures('clean_env')


def test_provider_from_credentials(provider: SapAiCoreProvider):
    assert provider.name == 'sap-ai-core'
    assert provider.base_url == API_URL
    assert provider.credentials == CREDENTIALS
    assert provider.resource_group == 'default'
    assert isinstance(provider.client, AsyncOpenAI)
    assert str(provider.client.base_url) == f'{API_URL}/inference/deployments/model/'
    assert provider.client.api_key == 'sap-ai-core'
    assert provider.client._custom_query == {'api-version': DEFAULT_API_VERSION}  # pyright: ignore[reportPrivateUsage]
    assert repr(provider) == f'SapAiCoreProvider(name=sap-ai-core, base_url={API_URL})'


def test_provider_from_service_key_and_arguments(clean_env: pytest.MonkeyPatch):
    provider = SapAiCoreProvider(service_key=SERVICE_KEY, resource_group='team-a', api_version='2024-10-21')
    assert provider.credentials.client_id == 'sb-client'
    assert provider.resource_group == 'team-a'
    assert provider.client._custom_query == {'api-version': '2024-10-21'}  # pyright: ignore[reportPrivateUsage]

    clean_env.setenv('AICORE_SERVICE_KEY', json.dumps(SERVICE_KEY))
    assert SapAiCoreProvider().credentials.client_id == 'sb-client'

    provider = SapAiCoreProvider(
        client_id='a', client_secret='b', auth_url='https://auth.example.com', base_url='https://api.example.com'
    )
    assert provider.base_url == 'https://api.example.com/v2'


def test_provider_without_credentials():
    with pytest.raises(UserError, match='SAP AI Core credentials not found'):
        SapAiCoreProvider()


def test_credentials_and_arguments_are_exclusive():
    with pytest.raises(UserError, match='Cannot provide both `credentials` and the individual credential arguments'):
        SapAiCoreProvider(credentials=CREDENTIALS, client_id='x')


def test_openai_client_passthrough(sap: MockSap):
    openai_client = AsyncOpenAI(api_key='from-sap-sdk', base_url=f'{API_URL}/inference/deployments/abc')
    provider = SapAiCoreProvider(openai_client=openai_client)
    assert provider.client is openai_client
    assert provider.base_url == f'{API_URL}/inference/deployments/abc/'
    with pytest.raises(UserError, match='holds no SAP AI Core credentials'):
        provider.credentials
    # Mixing `openai_client` with anything else is a typing error too; it is rejected at runtime as well.
    with pytest.raises(UserError, match='Cannot provide both `openai_client` and `http_client`'):
        SapAiCoreProvider(openai_client=openai_client, http_client=sap.client())  # pyright: ignore[reportCallIssue]
    with pytest.raises(UserError, match='Cannot provide both `openai_client` and SAP AI Core credentials'):
        SapAiCoreProvider(openai_client=openai_client, credentials=CREDENTIALS)  # pyright: ignore[reportCallIssue]
    with pytest.raises(UserError, match='Cannot provide both `openai_client` and SAP AI Core credentials'):
        SapAiCoreProvider(openai_client=openai_client, client_id='x')  # pyright: ignore[reportCallIssue]


def test_http_client_gets_the_auth_attached(sap: MockSap):
    http_client = sap.client()
    provider = SapAiCoreProvider(credentials=CREDENTIALS, http_client=http_client)
    assert isinstance(http_client.auth, SapAiCoreAuth)
    assert http_client.auth.credentials == CREDENTIALS
    assert provider.client._client is http_client  # pyright: ignore[reportPrivateUsage]


def test_http_client_with_its_own_auth_is_rejected():
    http_client = httpx2.AsyncClient(auth=('user', 'pass'))
    with pytest.raises(UserError, match='already has an `auth`'):
        SapAiCoreProvider(credentials=CREDENTIALS, http_client=http_client)


async def test_own_http_client_lifecycle():
    provider = SapAiCoreProvider(credentials=CREDENTIALS)
    own = provider.client._client  # pyright: ignore[reportPrivateUsage]
    assert isinstance(own, httpx2.AsyncClient)
    assert isinstance(own.auth, SapAiCoreAuth)
    assert own.headers['user-agent'].startswith('pydantic-ai/')
    assert own.timeout == httpx2.Timeout(600, connect=5)

    async with provider:
        pass
    assert own.is_closed
    # Entering again replaces the closed client with a fresh, equally configured one.
    async with provider:
        replacement = provider.client._client  # pyright: ignore[reportPrivateUsage]
        assert replacement is not own
        assert isinstance(replacement, httpx2.AsyncClient)
        assert isinstance(replacement.auth, SapAiCoreAuth)
        assert not replacement.is_closed


async def test_injected_http_client_is_not_closed(sap: MockSap):
    http_client = sap.client()
    provider = SapAiCoreProvider(credentials=CREDENTIALS, http_client=http_client)
    async with provider:
        pass
    assert not http_client.is_closed
    await http_client.aclose()


@pytest.mark.parametrize(
    ('model_name', 'transformer', 'family_hint'),
    [
        ('gpt-4o', OpenAIJsonSchemaTransformer, 'openai'),
        ('GPT-4.1', OpenAIJsonSchemaTransformer, 'openai'),
        ('o3-mini', OpenAIJsonSchemaTransformer, 'openai'),
        ('meta--llama3.1-70b-instruct', InlineDefsJsonSchemaTransformer, 'meta'),
        ('mistralai--mistral-large-instruct', OpenAIJsonSchemaTransformer, 'mistral'),
        ('deepseek-r1-0528', OpenAIJsonSchemaTransformer, 'deepseek'),
        ('anthropic--claude-4-sonnet', OpenAIJsonSchemaTransformer, None),
        ('some-unknown-model', OpenAIJsonSchemaTransformer, None),
    ],
)
def test_model_profile(model_name: str, transformer: type, family_hint: str | None):
    profile = SapAiCoreProvider.model_profile(model_name)
    assert profile is not None
    assert profile == sap_ai_core_model_profile(model_name)
    fields = dict(profile)
    assert fields['json_schema_transformer'] is transformer
    # SAP AI Core rejects `strict` on several models, so it is off whatever the family says.
    assert fields['openai_supports_strict_tool_definition'] is False
    if family_hint == 'openai':
        assert profile.get('openai_supports_reasoning') is not None or profile.get('supports_json_schema_output')
    if family_hint is None:
        assert profile.get('supports_json_schema_output', False) is False


def test_provider_factory_fixture_defaults(provider_factory: ProviderFactory, sap: MockSap):
    provider = provider_factory(resource_group=None, deployment_ids={'gpt-4o': 'x'})
    assert provider.resource_group == 'default'
    assert provider.client._client.auth is not None  # pyright: ignore[reportPrivateUsage]
