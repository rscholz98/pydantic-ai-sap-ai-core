"""Smoke test against a live SAP AI Core tenant.

Run with the service key in the environment, for example from a `.env` file holding
`AICORE_SERVICE_KEY='{...}'` (or the four `AICORE_*` variables):

    uv run --env-file .env scripts/live_smoke.py            # first azure-openai deployment
    uv run --env-file .env scripts/live_smoke.py gpt-4o     # a specific model
    uv run --env-file .env scripts/live_smoke.py gpt-4o --api-version 2024-10-21

Steps, each reported as PASS/FAIL with the raw error body on failure:

1. token: fetch an OAuth2 token from the XSUAA endpoint
2. deployments: list RUNNING foundation-model deployments and their executables
3. run: a plain `Agent.run`
4. stream: `Agent.run_stream` (checks `stream_options` / usage on the last chunk)
5. tools: a tool call round trip
6. output: structured output via a Pydantic model

Nothing is recorded; this prints to stdout only.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import traceback
from collections.abc import Awaitable, Callable
from typing import Any

import httpx2
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel

from pydantic_ai_sap_ai_core import SapAiCoreProvider, load_credentials
from pydantic_ai_sap_ai_core._deployments import FOUNDATION_MODELS_SCENARIO, OPENAI_EXECUTABLE


class CityInfo(BaseModel):
    city: str
    country: str


async def fetch_token(http: httpx2.AsyncClient, creds: Any) -> str:
    response = await http.post(
        creds.auth_url,
        data={'grant_type': 'client_credentials'},
        auth=(creds.client_id, creds.client_secret),
    )
    response.raise_for_status()
    token = response.json()['access_token']
    print(f'  token ok ({len(token)} chars, expires_in={response.json().get("expires_in")})')
    return token


async def list_deployments(http: httpx2.AsyncClient, creds: Any, token: str) -> list[dict[str, Any]]:
    response = await http.get(
        f'{creds.base_url}/lm/deployments',
        params={'scenarioId': FOUNDATION_MODELS_SCENARIO, 'status': 'RUNNING'},
        headers={'Authorization': f'Bearer {token}', 'AI-Resource-Group': creds.resource_group},
    )
    response.raise_for_status()
    resources = response.json().get('resources', [])
    for r in resources:
        model = (((r.get('details') or {}).get('resources') or {}).get('backend_details') or {}).get('model') or {}
        print(
            f'  {r.get("id")}  executable={r.get("executableId")}  model={model.get("name")}  version={model.get("version")}'
        )
    if not resources:
        print(f'  no RUNNING deployments in resource group {creds.resource_group!r}')
    return resources


def pick_model(resources: list[dict[str, Any]]) -> str | None:
    for r in resources:
        if r.get('executableId') != OPENAI_EXECUTABLE:
            continue
        model = (((r.get('details') or {}).get('resources') or {}).get('backend_details') or {}).get('model') or {}
        if model.get('name'):
            return model['name']
    return None


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('model', nargs='?', help='model name as shown in AI Launchpad, e.g. gpt-4o')
    parser.add_argument('--api-version', default=None, help='Azure OpenAI api-version to send')
    args = parser.parse_args()

    creds = load_credentials()
    print(f'base_url={creds.base_url}  resource_group={creds.resource_group}')

    results: dict[str, bool] = {}

    async def step(name: str, fn: Callable[[], Awaitable[Any]]) -> Any:
        print(f'\n[{name}]')
        try:
            value = await fn()
        except httpx2.HTTPStatusError as e:
            print(f'  FAIL {e.response.status_code} {e.request.method} {e.request.url}')
            print(f'  {e.response.text[:2000]}')
            results[name] = False
            return None
        except Exception:  # noqa: BLE001
            print('  FAIL')
            traceback.print_exc()
            results[name] = False
            return None
        results[name] = True
        print('  PASS')
        return value

    async with httpx2.AsyncClient(timeout=30) as http:
        token = await step('token', lambda: fetch_token(http, creds))
        if token is None:
            return report(results)
        resources = await step('deployments', lambda: list_deployments(http, creds, token))

    model_name = args.model or pick_model(resources or [])
    if model_name is None:
        print('\nno azure-openai deployment found; pass a model name explicitly')
        return report(results)
    print(f'\nusing model {model_name!r}')

    provider_kwargs: dict[str, Any] = {}
    if args.api_version:
        provider_kwargs['api_version'] = args.api_version
    model = OpenAIChatModel(model_name, provider=SapAiCoreProvider(**provider_kwargs))

    async def run() -> None:
        agent = Agent(model, instructions='Answer in one short sentence.')
        result = await agent.run('What is SAP BTP?')
        print(f'  {result.output!r}')
        print(f'  usage={result.usage()}')

    async def stream() -> None:
        agent = Agent(model, instructions='Answer in one short sentence.')
        async with agent.run_stream('Name three SAP products.') as response:
            text = ''
            async for chunk in response.stream_text(delta=True):
                text += chunk
            print(f'  {text!r}')
            print(f'  usage={response.usage()}')

    async def tools() -> None:
        agent = Agent(model, instructions='Use the tool to look up the temperature; do not guess.')

        @agent.tool_plain
        def temperature(city: str) -> str:
            """Return the current temperature in a city."""
            return f'It is 21 degrees C in {city}.'

        result = await agent.run('How warm is it in Walldorf right now?')
        calls = [p for m in result.all_messages() for p in m.parts if p.part_kind == 'tool-call']
        print(f'  tool calls: {[(c.tool_name, c.args) for c in calls]}')
        print(f'  {result.output!r}')
        assert calls, 'model did not call the tool'

    async def output() -> None:
        agent = Agent(model, output_type=CityInfo)
        result = await agent.run('Where is SAP headquartered?')
        print(f'  {result.output!r}')

    await step('run', run)
    await step('stream', stream)
    await step('tools', tools)
    await step('output', output)
    return report(results)


def report(results: dict[str, bool]) -> int:
    print('\nsummary')
    for name, ok in results.items():
        print(f'  {"PASS" if ok else "FAIL"}  {name}')
    return 0 if all(results.values()) else 1


if __name__ == '__main__':
    sys.exit(asyncio.run(main()))
