"""Quick check against a live SAP AI Core tenant.

    python live_test.py              # list deployments, one chat call
    python live_test.py --full       # also streaming, a tool call, structured output
    python live_test.py gpt-4o       # pick the model (default: first azure-openai deployment)

Reads `.env` in this directory (or the real environment). One of the two credential forms:

    AICORE_SERVICE_KEY='{"clientid": "...", "clientsecret": "...", "url": "...", "serviceurls": {"AI_API_URL": "..."}}'

or

    AICORE_CLIENT_ID=...
    AICORE_CLIENT_SECRET=...
    AICORE_AUTH_URL=https://<subdomain>.authentication.<region>.hana.ondemand.com
    AICORE_BASE_URL=https://api.ai.<region>.ml.hana.ondemand.com

Optional: AICORE_RESOURCE_GROUP (default `default`), AICORE_API_VERSION (default 2025-03-01-preview).
"""

from __future__ import annotations

import asyncio
import os
import sys
import traceback
from collections.abc import Awaitable
from pathlib import Path
from typing import Any

import httpx2
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel

from pydantic_ai_sap_ai_core import SapAiCoreProvider, load_credentials

INDIVIDUAL = ('AICORE_CLIENT_ID', 'AICORE_CLIENT_SECRET', 'AICORE_AUTH_URL', 'AICORE_BASE_URL')


def load_dotenv(path: Path) -> None:
    """Put `KEY=value` lines of `path` into the environment without overriding what is already set."""
    if not path.is_file():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith('#') or '=' not in line:
            continue
        key, _, value = line.partition('=')
        key = key.strip().removeprefix('export ').strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in '\'"':
            value = value[1:-1]
        os.environ.setdefault(key, value)


def credential_source() -> str:
    has_key = bool(os.environ.get('AICORE_SERVICE_KEY'))
    individual = [name for name in INDIVIDUAL if os.environ.get(name)]
    if has_key and individual:
        print(f'note: both AICORE_SERVICE_KEY and {individual} are set; the service key wins')
    if has_key:
        return 'AICORE_SERVICE_KEY'
    if len(individual) == len(INDIVIDUAL):
        return 'AICORE_CLIENT_ID / AICORE_CLIENT_SECRET / AICORE_AUTH_URL / AICORE_BASE_URL'
    missing = [name for name in INDIVIDUAL if name not in individual]
    sys.exit(
        'no credentials found. Set either AICORE_SERVICE_KEY (the whole service key JSON) '
        f'or all four of {list(INDIVIDUAL)}; missing: {missing}'
    )


def dig(data: dict[str, Any], *keys: str) -> str | None:
    """Follow `keys` through nested dicts; `None` when any level is missing or not a string at the end."""
    current: Any = data
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
    return current if isinstance(current, str) else None


class CityInfo(BaseModel):
    city: str
    country: str


async def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    full = '--full' in sys.argv

    load_dotenv(Path(__file__).with_name('.env'))
    print(f'credentials: {credential_source()}')
    creds = load_credentials()
    print(f'base_url:    {creds.base_url}')
    print(f'auth_url:    {creds.auth_url}')
    print(f'client_id:   {creds.client_id}')
    secret = creds.client_secret
    print(f'secret:      {len(secret)} chars, starts {secret[:2]!r}, ends {secret[-2:]!r}')
    print(f'resource group: {creds.resource_group}')

    # 1. token + deployments, by hand, so a failure here is easy to read
    async with httpx2.AsyncClient(timeout=30) as http:
        r = await http.post(
            creds.auth_url, data={'grant_type': 'client_credentials'}, auth=(creds.client_id, creds.client_secret)
        )
        if r.status_code != 200:
            print(f'token FAILED {r.status_code}: {r.text[:1000]}')
            print('check the same credentials outside Python:')
            print(f"  curl -s -u '{creds.client_id}:<secret>' -d grant_type=client_credentials '{creds.auth_url}'")
            return 1
        token = r.json()['access_token']
        print('token:       ok')

        r = await http.get(
            f'{creds.base_url}/lm/deployments',
            params={'scenarioId': 'foundation-models', 'status': 'RUNNING'},
            headers={'Authorization': f'Bearer {token}', 'AI-Resource-Group': creds.resource_group},
        )
        if r.status_code != 200:
            print(f'deployments FAILED {r.status_code}: {r.text[:1000]}')
            return 1
        openai_models: list[str] = []
        print('deployments:')
        resources: list[dict[str, Any]] = r.json().get('resources', [])
        for d in resources:
            name = dig(d, 'details', 'resources', 'backend_details', 'model', 'name')
            version = dig(d, 'details', 'resources', 'backend_details', 'model', 'version')
            print(f'  {d.get("id")}  {d.get("executableId")}  {name} {version or ""}')
            if d.get('executableId') == 'azure-openai' and name:
                openai_models.append(name)
        if not openai_models and not args:
            print('no azure-openai deployment is RUNNING; pass a model name to try anyway')
            return 1

    model_name = args[0] if args else openai_models[0]
    api_version = os.environ.get('AICORE_API_VERSION')
    provider = SapAiCoreProvider(api_version=api_version) if api_version else SapAiCoreProvider()
    model = OpenAIChatModel(model_name, provider=provider)
    print(f'model:       {model_name}')

    ok = True

    async def step(name: str, coro: Awaitable[None]) -> None:
        nonlocal ok
        try:
            await coro
            print(f'{name}: PASS')
        except Exception as e:  # noqa: BLE001
            ok = False
            print(f'{name}: FAIL {type(e).__name__}: {e}')
            body = getattr(e, 'body', None)
            if body:
                print(f'  {body}')
            if '--trace' in sys.argv:
                traceback.print_exc()

    async def run() -> None:
        result = await Agent(model, instructions='One short sentence.').run('What is SAP BTP?')
        print(f'  {result.output!r}  {result.usage}')

    async def stream() -> None:
        async with Agent(model, instructions='One short sentence.').run_stream('Name three SAP products.') as s:
            text = ''.join([c async for c in s.stream_text(delta=True)])
            print(f'  {text!r}  {s.usage}')

    async def tools() -> None:
        agent = Agent(model, instructions='Use the tool; do not guess.')

        @agent.tool_plain
        def temperature(city: str) -> str:
            """Current temperature in a city."""
            return f'21 degrees C in {city}'

        result = await agent.run('How warm is it in Walldorf?')
        calls = [p.tool_name for m in result.all_messages() for p in m.parts if p.part_kind == 'tool-call']
        print(f'  tool calls {calls}  {result.output!r}')
        assert calls, 'model did not call the tool'

    async def output() -> None:
        result = await Agent(model, output_type=CityInfo).run('Where is SAP headquartered?')
        print(f'  {result.output!r}')

    await step('run', run())
    if full:
        await step('stream', stream())
        await step('tools', tools())
        await step('output', output())
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(asyncio.run(main()))
