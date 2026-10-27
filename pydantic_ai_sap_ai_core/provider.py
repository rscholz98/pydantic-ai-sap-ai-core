"""The SAP AI Core provider for Pydantic AI's OpenAI-based models."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, overload

import httpx2
from openai import AsyncOpenAI
from pydantic_ai.exceptions import UserError
from pydantic_ai.models import get_user_agent
from pydantic_ai.profiles import ModelProfile
from pydantic_ai.providers import Provider

from .auth import MODEL_PLACEHOLDER, SapAiCoreAuth
from .credentials import SapAiCoreCredentials, load_credentials
from .profiles import sap_ai_core_model_profile

__all__ = ('DEFAULT_API_VERSION', 'SapAiCoreProvider')

DEFAULT_API_VERSION = '2025-03-01-preview'
"""The Azure OpenAI `api-version` sent by default; the same one SAP's Python SDK uses."""

_PLACEHOLDER_API_KEY = 'sap-ai-core'
"""Given to the OpenAI SDK, which insists on a key; `SapAiCoreAuth` replaces the header it produces."""


class SapAiCoreProvider(Provider[AsyncOpenAI]):
    """Provider for [SAP AI Core](https://help.sap.com/docs/sap-ai-core) foundation models.

    Use it with [`OpenAIChatModel`][pydantic_ai.models.openai.OpenAIChatModel] or
    [`OpenAIResponsesModel`][pydantic_ai.models.openai.OpenAIResponsesModel] and the model name
    as SAP lists it, for example `gpt-4o` or `mistralai--mistral-large-instruct`:

    ```python
    from pydantic_ai import Agent
    from pydantic_ai.models.openai import OpenAIChatModel

    from pydantic_ai_sap_ai_core import SapAiCoreProvider

    model = OpenAIChatModel('gpt-4o', provider=SapAiCoreProvider())
    agent = Agent(model)
    ```

    The provider obtains OAuth2 tokens itself, sends the `AI-Resource-Group` header, and resolves
    each model name to the running deployment that serves it in the resource group, so the only
    configuration needed is the service key, read from the environment when nothing is passed.
    """

    @property
    def name(self) -> str:
        """The provider name, recorded as `provider_name` on every message part."""
        return 'sap-ai-core'

    @property
    def base_url(self) -> str:
        """The AI API URL, ending in `/v2`, or the base URL of a client passed in."""
        if self._credentials is not None:
            return self._credentials.base_url
        return str(self._client.base_url)

    @property
    def client(self) -> AsyncOpenAI:
        """The OpenAI SDK client, routed to SAP AI Core."""
        return self._client

    @property
    def credentials(self) -> SapAiCoreCredentials:
        """The credentials in use; raises if the provider was built from an `openai_client`."""
        if self._credentials is None:
            raise UserError(
                '`SapAiCoreProvider` was created with `openai_client=` and holds no SAP AI Core credentials.'
            )
        return self._credentials

    @property
    def resource_group(self) -> str:
        """The resource group whose deployments are used."""
        return self.credentials.resource_group

    @staticmethod
    def model_profile(model_name: str) -> ModelProfile | None:
        """The profile for a model name as SAP lists it; see [`sap_ai_core_model_profile`][pydantic_ai_sap_ai_core.sap_ai_core_model_profile]."""
        return sap_ai_core_model_profile(model_name)

    @overload
    def __init__(self, *, openai_client: AsyncOpenAI) -> None: ...

    @overload
    def __init__(
        self,
        *,
        credentials: SapAiCoreCredentials | None = None,
        service_key: str | Mapping[str, Any] | None = None,
        client_id: str | None = None,
        client_secret: str | None = None,
        auth_url: str | None = None,
        base_url: str | None = None,
        resource_group: str | None = None,
        api_version: str = DEFAULT_API_VERSION,
        deployment_ids: Mapping[str, str] | None = None,
        deployment_cache_ttl: float = 300.0,
        http_client: httpx2.AsyncClient | None = None,
    ) -> None: ...

    def __init__(
        self,
        *,
        credentials: SapAiCoreCredentials | None = None,
        service_key: str | Mapping[str, Any] | None = None,
        client_id: str | None = None,
        client_secret: str | None = None,
        auth_url: str | None = None,
        base_url: str | None = None,
        resource_group: str | None = None,
        api_version: str = DEFAULT_API_VERSION,
        deployment_ids: Mapping[str, str] | None = None,
        deployment_cache_ttl: float = 300.0,
        http_client: httpx2.AsyncClient | None = None,
        openai_client: AsyncOpenAI | None = None,
    ) -> None:
        """Create a new SAP AI Core provider.

        Args:
            credentials: Ready-made credentials. When given, the other credential arguments must be `None`.
            service_key: The SAP AI Core service key, as JSON text or parsed. Defaults to the
                `AICORE_SERVICE_KEY` environment variable.
            client_id: The OAuth2 client id. Defaults to `AICORE_CLIENT_ID`.
            client_secret: The OAuth2 client secret. Defaults to `AICORE_CLIENT_SECRET`.
            auth_url: The XSUAA token endpoint. Defaults to `AICORE_AUTH_URL`.
            base_url: The AI API URL. Defaults to `AICORE_BASE_URL`.
            resource_group: The resource group whose deployments are used. Defaults to
                `AICORE_RESOURCE_GROUP`, then `default`.
            api_version: The Azure OpenAI `api-version` query parameter sent with every request.
            deployment_ids: Model names mapped to deployment ids, for models that should not be
                looked up by listing deployments, or whose deployment lacks model details.
            deployment_cache_ttl: Seconds a deployment listing is reused before it is fetched again.
            http_client: An `httpx2.AsyncClient` to send requests with. It must not carry an `auth`
                of its own; the provider attaches [`SapAiCoreAuth`][pydantic_ai_sap_ai_core.SapAiCoreAuth] to it.
            openai_client: An existing `AsyncOpenAI` client that already reaches SAP AI Core, used
                as-is; for example one built by the SAP Cloud SDK for AI. If provided, all other
                arguments must be left at their defaults.

        See [`load_credentials`][pydantic_ai_sap_ai_core.load_credentials] for how the credential
        arguments and environment variables combine.
        """
        if openai_client is not None:
            if http_client is not None:
                raise UserError('Cannot provide both `openai_client` and `http_client`.')
            if credentials is not None or any((service_key, client_id, client_secret, auth_url, base_url)):
                raise UserError('Cannot provide both `openai_client` and SAP AI Core credentials.')
            self._credentials = None
            self._client = openai_client
            return

        if credentials is not None:
            if any((service_key, client_id, client_secret, auth_url, base_url, resource_group)):
                raise UserError('Cannot provide both `credentials` and the individual credential arguments.')
        else:
            credentials = load_credentials(
                client_id=client_id,
                client_secret=client_secret,
                auth_url=auth_url,
                base_url=base_url,
                resource_group=resource_group,
                service_key=service_key,
            )
        self._credentials = credentials
        self._auth = SapAiCoreAuth(
            credentials, deployment_ids=deployment_ids, deployment_cache_ttl=deployment_cache_ttl
        )

        if http_client is None:
            http_client = self._create_http_client()
            self._own_http_client = http_client
            self._http_client_factory = self._create_http_client
        else:
            if http_client.auth is not None:
                raise UserError(
                    'The `http_client` passed to `SapAiCoreProvider` already has an `auth`; '
                    'the provider needs to attach its own `SapAiCoreAuth`.'
                )
            http_client.auth = self._auth

        self._client = AsyncOpenAI(
            api_key=_PLACEHOLDER_API_KEY,
            base_url=f'{credentials.base_url}/inference/deployments/{MODEL_PLACEHOLDER}',
            default_query={'api-version': api_version},
            http_client=http_client,
        )

    def _create_http_client(self) -> httpx2.AsyncClient:
        return httpx2.AsyncClient(
            auth=self._auth,
            timeout=httpx2.Timeout(600, connect=5),
            headers={'User-Agent': get_user_agent()},
        )

    def _set_http_client(self, http_client: httpx2.AsyncClient) -> None:
        # OpenAI 3 keeps legacy HTTPX as a runtime-only escape hatch, outside its public type annotations.
        self._client._client = http_client  # pyright: ignore[reportPrivateUsage]
