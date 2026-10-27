"""How a deployment listing is read: which resources count, and which deployment wins for a model."""

from __future__ import annotations

import httpx2

from pydantic_ai_sap_ai_core._deployments import DeploymentResolver

from .conftest import API_URL, deployment


def resolver() -> DeploymentResolver:
    return DeploymentResolver(base_url=API_URL, resource_group='default')


def listing(*resources: dict[str, object]) -> httpx2.Response:
    return httpx2.Response(200, json={'count': len(resources), 'resources': list(resources)})


def test_newest_deployment_wins_for_a_model():
    r = resolver()
    r.update(
        listing(
            deployment('gpt-4o', 'old', created_at='2025-01-01T00:00:00Z'),
            deployment('gpt-4o', 'new', created_at='2026-01-01T00:00:00Z'),
            deployment('gpt-4o', 'middle', created_at='2025-06-01T00:00:00Z'),
        )
    )
    found = r.cached('gpt-4o')
    assert found is not None
    assert found.id == 'new'
    assert found.url == f'{API_URL}/inference/deployments/new'
    assert found.model_version == 'latest'
    assert found.executable_id == 'azure-openai'


def test_resources_without_model_details_or_url_are_skipped():
    r = resolver()
    r.update(
        listing(
            deployment('gpt-4o', 'no-details', with_details=False),
            {**deployment('gpt-4o-mini', 'no-url'), 'deploymentUrl': None},
            {**deployment('o3-mini', 'empty-backend'), 'details': {'resources': {'backend_details': {}}}},
            {**deployment('o4-mini', 'no-resources'), 'details': {}},
            deployment('gpt-4.1', 'fine', executable_id=None),
        )
    )
    assert r.cached('gpt-4o') is None
    assert r.cached('gpt-4o-mini') is None
    assert r.cached('o3-mini') is None
    assert r.cached('o4-mini') is None
    fine = r.cached('gpt-4.1')
    assert fine is not None
    assert fine.executable_id is None
    assert 'Running foundation-model deployments: gpt-4.1.' in str(r.missing('gpt-4o'))


def test_empty_listing():
    r = resolver()
    r.update(httpx2.Response(200, json={'count': 0, 'resources': []}))
    assert r.cached('gpt-4o') is None
    assert 'Running foundation-model deployments: none.' in str(r.missing('gpt-4o'))


def test_nothing_cached_before_an_update():
    assert resolver().cached('gpt-4o') is None


def test_explicit_url():
    r = DeploymentResolver(base_url=API_URL, resource_group='default', deployment_ids={'gpt-4o': 'abc'})
    assert r.explicit_url('gpt-4o') == f'{API_URL}/inference/deployments/abc'
    assert r.explicit_url('gpt-4o-mini') is None
