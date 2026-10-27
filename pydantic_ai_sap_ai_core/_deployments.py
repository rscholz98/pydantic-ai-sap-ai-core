"""Resolve model names to SAP AI Core deployment URLs.

Generative models are served by per-model, per-resource-group deployments in the global
`foundation-models` scenario. Deployment ids are opaque and tenant-specific, so the provider lists
`GET /v2/lm/deployments?scenarioId=foundation-models&status=RUNNING` and matches each deployment's
`details.resources.backend_details.model.name`, the way SAP's own SDKs do, caching the listing for
a few minutes.
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass, field

import httpx2
from pydantic import BaseModel, Field, ValidationError
from pydantic_ai.exceptions import UserError

from ._errors import validation_summary

FOUNDATION_MODELS_SCENARIO = 'foundation-models'
OPENAI_EXECUTABLE = 'azure-openai'
"""The executable whose deployments speak the OpenAI-compatible (Azure OpenAI) dialect."""


class _BackendModel(BaseModel):
    name: str
    version: str | None = None


class _BackendDetails(BaseModel):
    model: _BackendModel | None = None


class _Resources(BaseModel):
    backend_details: _BackendDetails | None = None


class _Details(BaseModel):
    resources: _Resources | None = None


class _DeploymentResource(BaseModel):
    id: str
    deployment_url: str | None = Field(default=None, alias='deploymentUrl')
    executable_id: str | None = Field(default=None, alias='executableId')
    created_at: str = Field(default='', alias='createdAt')
    details: _Details | None = None


class _DeploymentList(BaseModel):
    resources: list[_DeploymentResource] = []


@dataclass(frozen=True)
class Deployment:
    """A running foundation-model deployment, as far as the provider needs it."""

    id: str
    url: str
    model_name: str
    model_version: str | None
    executable_id: str | None
    created_at: str


@dataclass
class DeploymentResolver:
    """Map model names to deployment URLs with a time-limited cache and an explicit override table."""

    base_url: str
    """The AI API URL, ending in `/v2`."""
    resource_group: str
    cache_ttl: float = 300.0
    """Seconds a listing stays fresh. The SAP JS SDK uses five minutes too."""
    deployment_ids: Mapping[str, str] = field(default_factory=dict[str, str])
    """Model names mapped to deployment ids, consulted before any listing."""

    _by_model: dict[str, Deployment] = field(default_factory=dict[str, Deployment], init=False, repr=False)
    _listed_at: float | None = field(default=None, init=False, repr=False)

    def explicit_url(self, model_name: str) -> str | None:
        """The deployment URL for a model configured in `deployment_ids`, if any."""
        deployment_id = self.deployment_ids.get(model_name)
        if deployment_id is None:
            return None
        return f'{self.base_url}/inference/deployments/{deployment_id}'

    def cached(self, model_name: str) -> Deployment | None:
        """The cached deployment for a model, if the listing is fresh and holds one."""
        if self._listed_at is None or time.monotonic() - self._listed_at > self.cache_ttl:
            return None
        return self._by_model.get(model_name)

    def list_request(self, token: str) -> httpx2.Request:
        """The request that lists the running foundation-model deployments of the resource group."""
        return httpx2.Request(
            'GET',
            f'{self.base_url}/lm/deployments',
            params={'scenarioId': FOUNDATION_MODELS_SCENARIO, 'status': 'RUNNING'},
            headers={'Authorization': f'Bearer {token}', 'AI-Resource-Group': self.resource_group},
        )

    def update(self, response: httpx2.Response) -> None:
        """Replace the cache with a listing response; the response body must already be read."""
        if response.status_code != 200:
            raise UserError(
                f'Listing SAP AI Core deployments failed with HTTP {response.status_code}: {response.text[:500]}. '
                f'Check that the service key may read deployments of resource group {self.resource_group!r}.'
            )
        try:
            listing = _DeploymentList.model_validate_json(response.content)
        except ValidationError as e:
            raise UserError(f'The SAP AI Core deployment listing could not be parsed: {validation_summary(e)}') from e

        by_model: dict[str, Deployment] = {}
        for resource in sorted(listing.resources, key=lambda r: r.created_at, reverse=True):
            deployment = _to_deployment(resource)
            if deployment is not None:
                by_model.setdefault(deployment.model_name, deployment)
        self._by_model = by_model
        self._listed_at = time.monotonic()

    def missing(self, model_name: str) -> UserError:
        """The error for a model with no running deployment, naming the ones that exist."""
        available = ', '.join(sorted(self._by_model)) or 'none'
        return UserError(
            f'No running SAP AI Core deployment serves model {model_name!r} in resource group '
            f'{self.resource_group!r}. Running foundation-model deployments: {available}. '
            'Deploy the model in SAP AI Launchpad, or map the name to a deployment id with '
            '`SapAiCoreProvider(deployment_ids={...})`.'
        )


def _to_deployment(resource: _DeploymentResource) -> Deployment | None:
    details = resource.details
    backend = details.resources.backend_details if details and details.resources else None
    model = backend.model if backend else None
    if model is None or not resource.deployment_url:
        return None
    return Deployment(
        id=resource.id,
        url=resource.deployment_url.rstrip('/'),
        model_name=model.name,
        model_version=model.version,
        executable_id=resource.executable_id,
        created_at=resource.created_at,
    )
