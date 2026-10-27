"""An `httpx2.Auth` that turns OpenAI SDK requests into SAP AI Core requests.

Three things happen to every request on its way out:

1. An OAuth2 client-credentials token is fetched from the XSUAA endpoint (once, then cached until
   shortly before `expires_in`) and sent as `Authorization: Bearer ...`, replacing the placeholder
   key the OpenAI SDK put there. A `401` answer refreshes the token and replays the request once.
2. The `AI-Resource-Group` and `AI-Client-Type` headers are added.
3. The `model` placeholder segment in the path, which the provider sets as the OpenAI client's base
   URL, is replaced by the deployment URL of the model named in the JSON body.

All HTTP traffic, including the token and deployment-listing calls, goes through the client the
auth is attached to, so timeouts, proxies and instrumentation apply uniformly.
"""

from __future__ import annotations

import functools
import time
from collections.abc import AsyncGenerator, Generator, Mapping
from dataclasses import dataclass
from typing import Any

import anyio
import httpx2
from pydantic import BaseModel, TypeAdapter, ValidationError
from pydantic_ai.exceptions import UserError

from ._deployments import OPENAI_EXECUTABLE, DeploymentResolver
from ._errors import validation_summary
from .credentials import SapAiCoreCredentials

__all__ = ('MODEL_PLACEHOLDER', 'SapAiCoreAuth')

MODEL_PLACEHOLDER = 'model'
"""The path segment that stands in for the deployment id in the OpenAI client's base URL."""

CLIENT_TYPE = 'pydantic-ai-sap-ai-core'
"""Sent as `AI-Client-Type`, the header SAP's SDKs use to identify the calling library."""

_TOKEN_EXPIRY_BUFFER = 60.0
"""Seconds before `expires_in` at which a token is refreshed. XSUAA tokens usually last 12 hours."""


@dataclass(frozen=True)
class _Token:
    value: str
    expires_at: float

    def is_fresh(self) -> bool:
        return time.monotonic() < self.expires_at - _TOKEN_EXPIRY_BUFFER


class _TokenResponse(BaseModel):
    access_token: str
    expires_in: float = 3600


_request_body_adapter = TypeAdapter(dict[str, Any])


class SapAiCoreAuth(httpx2.Auth):
    """Authenticate requests to SAP AI Core and route them to the right deployment.

    Attach it to an `httpx2.AsyncClient` given to the OpenAI SDK, whose `base_url` ends in
    `/v2/inference/deployments/model`. It works with any OpenAI SDK endpoint served by an
    `azure-openai` deployment: chat completions, responses and embeddings.
    """

    def __init__(
        self,
        credentials: SapAiCoreCredentials,
        *,
        deployment_ids: Mapping[str, str] | None = None,
        deployment_cache_ttl: float = 300.0,
    ) -> None:
        """Create the auth.

        Args:
            credentials: The OAuth2 client, token endpoint, AI API URL and resource group.
            deployment_ids: Model names mapped to deployment ids, used instead of listing deployments.
            deployment_cache_ttl: Seconds a deployment listing is reused before it is fetched again.
        """
        self.credentials = credentials
        self._token: _Token | None = None
        self._deployments = DeploymentResolver(
            base_url=credentials.base_url,
            resource_group=credentials.resource_group,
            cache_ttl=deployment_cache_ttl,
            deployment_ids=dict(deployment_ids or {}),
        )

    @functools.cached_property
    def _lock(self) -> anyio.Lock:
        # Created on first use so it binds to the event loop that runs the requests.
        return anyio.Lock()

    def sync_auth_flow(self, request: httpx2.Request) -> Generator[httpx2.Request, httpx2.Response, None]:
        """Not supported: the provider is async-only, like Pydantic AI's own OAuth providers."""
        raise UserError('`SapAiCoreAuth` requires an async HTTP client; use `httpx2.AsyncClient`.')
        yield request  # pyright: ignore[reportUnreachable]  # makes this a generator, as `httpx2.Auth` expects

    async def async_auth_flow(self, request: httpx2.Request) -> AsyncGenerator[httpx2.Request, httpx2.Response]:
        """Authenticate and route `request`, sending the token and listing calls it depends on first."""
        token = self._token
        if token is None or not token.is_fresh():
            async with self._lock:
                token = self._token
                if token is None or not token.is_fresh():
                    response = yield self._token_request()
                    await response.aread()
                    token = self._update_token(response)

        if self._needs_routing(request):
            model_name = await self._model_name(request)
            url = self._deployments.explicit_url(model_name)
            if url is None:
                deployment = self._deployments.cached(model_name)
                if deployment is None:
                    async with self._lock:
                        deployment = self._deployments.cached(model_name)
                        if deployment is None:
                            response = yield self._deployments.list_request(token.value)
                            await response.aread()
                            self._deployments.update(response)
                            deployment = self._deployments.cached(model_name)
                            if deployment is None:
                                raise self._deployments.missing(model_name)
                if deployment.executable_id is not None and deployment.executable_id != OPENAI_EXECUTABLE:
                    raise UserError(
                        f'SAP AI Core serves model {model_name!r} through the {deployment.executable_id!r} executable, '
                        f'which does not speak the OpenAI-compatible API. This provider currently supports '
                        f'{OPENAI_EXECUTABLE!r} deployments only.'
                    )
                url = deployment.url
            _route(request, url)

        self._apply(request, token.value)
        response = yield request
        if response.status_code == 401:
            # The token was revoked or expired early: refresh once and replay.
            async with self._lock:
                if self._token is token:
                    response = yield self._token_request()
                    await response.aread()
                    token = self._update_token(response)
                else:
                    token = self._token
                    assert token is not None
            self._apply(request, token.value)
            yield request

    def _token_request(self) -> httpx2.Request:
        credentials = self.credentials
        return httpx2.Request(
            'POST',
            credentials.auth_url,
            data={
                'grant_type': 'client_credentials',
                'client_id': credentials.client_id,
                'client_secret': credentials.client_secret,
            },
        )

    def _update_token(self, response: httpx2.Response) -> _Token:
        if response.status_code != 200:
            raise UserError(
                f'The SAP AI Core token request to {self.credentials.auth_url} failed with HTTP '
                f'{response.status_code}: {response.text[:500]}. Check `AICORE_CLIENT_ID`, `AICORE_CLIENT_SECRET` '
                'and `AICORE_AUTH_URL`, or the service key they come from.'
            )
        try:
            payload = _TokenResponse.model_validate_json(response.content)
        except ValidationError as e:
            raise UserError(
                f'The SAP AI Core token response from {self.credentials.auth_url} is not a valid token: '
                f'{validation_summary(e)}'
            ) from e
        self._token = token = _Token(payload.access_token, time.monotonic() + payload.expires_in)
        return token

    def _apply(self, request: httpx2.Request, token: str) -> None:
        request.headers['Authorization'] = f'Bearer {token}'
        request.headers['AI-Resource-Group'] = self.credentials.resource_group
        request.headers.setdefault('AI-Client-Type', CLIENT_TYPE)

    def _needs_routing(self, request: httpx2.Request) -> bool:
        return f'/inference/deployments/{MODEL_PLACEHOLDER}/' in request.url.path

    async def _model_name(self, request: httpx2.Request) -> str:
        await request.aread()
        try:
            body = _request_body_adapter.validate_json(request.content)
        except ValidationError:
            body = {}
        model: Any = body.get('model')
        if not isinstance(model, str) or not model:
            raise UserError('Cannot route the request to an SAP AI Core deployment: the request body names no `model`.')
        return model


def _route(request: httpx2.Request, deployment_url: str) -> None:
    """Point `request` at `deployment_url`, keeping the endpoint path and query that follow the placeholder."""
    marker = f'/inference/deployments/{MODEL_PLACEHOLDER}/'
    endpoint = request.url.path.split(marker, 1)[1]
    target = httpx2.URL(f'{deployment_url}/{endpoint}', params=request.url.params)
    if target.host != request.url.host:
        request.headers['Host'] = target.netloc.decode('ascii')
    request.url = target
