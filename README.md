# pydantic-ai-sap-ai-core

[SAP AI Core](https://help.sap.com/docs/sap-ai-core) (Generative AI Hub) provider for
[Pydantic AI](https://pydantic.dev/docs/ai/).

Pydantic AI's OpenAI-based models work against SAP AI Core's foundation-model deployments once
three SAP-specific things are taken care of: the OAuth2 client-credentials token that replaces a
static API key, the `AI-Resource-Group` header, and the per-tenant deployment URL that stands in
for a model name. This package does those three things in a `Provider`, so the rest of Pydantic AI
(tools, structured output, streaming, message history) is used unchanged.

```python
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel

from pydantic_ai_sap_ai_core import SapAiCoreProvider

model = OpenAIChatModel('gpt-4o', provider=SapAiCoreProvider())
agent = Agent(model, instructions='Be concise.')
print(agent.run_sync('What is SAP BTP?').output)
```

> **Status: alpha, not yet verified against a live tenant.** The wire format, deployment
> discovery and token flow are implemented from SAP's SDK sources and recorded responses, and
> tested offline against those recordings. See [Verification status](#verification-status) before
> relying on it in production, and please report what you find.

## Installation

```bash
pip install pydantic-ai-sap-ai-core
# or
uv add pydantic-ai-sap-ai-core
```

It depends on `pydantic-ai-slim[openai]` and nothing else from SAP: the token flow and deployment
lookup are a few hundred lines of `httpx2`, so neither `sap-ai-sdk-gen` nor LangChain is pulled in.

## Configuration

Create a service key for your SAP AI Core instance (BTP cockpit, *Instances and Subscriptions*,
*SAP AI Core*, *Create service key*). The provider reads it from the environment when nothing is
passed, using the names the SAP Cloud SDK for AI and LiteLLM use:

| Variable | Meaning |
| --- | --- |
| `AICORE_SERVICE_KEY` | The whole service key JSON. Easiest. |
| `AICORE_CLIENT_ID`, `AICORE_CLIENT_SECRET`, `AICORE_AUTH_URL`, `AICORE_BASE_URL` | The four fields individually (`clientid`, `clientsecret`, `url`, `serviceurls.AI_API_URL` in the key). |
| `AICORE_RESOURCE_GROUP` | The resource group whose deployments are used. Default `default`. |
| `AICORE_CONFIG`, `AICORE_HOME`, `AICORE_PROFILE` | A config file holding the same `AICORE_*` keys, default `~/.aicore/config.json`. |
| `VCAP_SERVICES` | On Cloud Foundry, the `aicore` binding is read automatically. |

The first source that provides any credential wins and must provide all of them; the resource
group is resolved on its own. Everything can also be passed explicitly:

```python
from pydantic_ai_sap_ai_core import SapAiCoreProvider

provider = SapAiCoreProvider(
    service_key=open('service-key.json').read(),  # or client_id=..., client_secret=..., auth_url=..., base_url=...
    resource_group='my-team',
    api_version='2024-10-21',  # the Azure OpenAI api-version; default 2025-03-01-preview
)
```

## How a request is routed

1. A bearer token is fetched once from the XSUAA endpoint in the service key and cached until a
   minute before it expires. A `401` refreshes it and replays the request once.
2. The running deployments of the resource group are listed from
   `GET /v2/lm/deployments?scenarioId=foundation-models&status=RUNNING`, matched by
   `details.resources.backend_details.model.name` and cached for five minutes
   (`deployment_cache_ttl`), the way SAP's own SDKs do it. When several deployments serve one model
   the newest is used.
3. The request goes to `<deploymentUrl>/chat/completions?api-version=...` with `Authorization`,
   `AI-Resource-Group` and `AI-Client-Type` headers.

If listing deployments is not possible, or a deployment carries no model details, map model names
to deployment ids yourself and no listing happens for them:

```python
provider = SapAiCoreProvider(deployment_ids={'gpt-4o': 'd1a82575de60e02a'})
```

Model names are the ones SAP AI Launchpad shows: `gpt-4o`, `gpt-4.1`, `o3-mini`,
`mistralai--mistral-large-instruct`, `meta--llama3.1-70b-instruct`. The provider picks the
matching Pydantic AI model profile by that name and turns off strict tool definitions, which SAP AI
Core rejects on several models.

## What is covered

- Deployments of the `azure-openai` executable, which is every OpenAI model and the Meta, Mistral
  and DeepSeek models SAP serves through the OpenAI dialect. `OpenAIChatModel` is the tested path;
  `OpenAIResponsesModel` uses the same provider and SAP supports `/responses` since SDK 6.10, but it
  is untested.
- Streaming, tool calling and structured output through the normal Pydantic AI APIs.
- Any `httpx2.AsyncClient` you pass as `http_client`, so proxies, timeouts and instrumentation
  apply to the token and listing calls too.

Not covered yet, in rough order of plan:

- Anthropic and Amazon models (`anthropic--claude-*`, `amazon--nova-*`): SAP serves them through
  the Bedrock Converse API. The plan is `BedrockConverseModel` behind an unsigned boto3 client with
  the same token and URL handling. Until then the provider raises a clear error when it finds such
  a deployment.
- Gemini (`gemini-*`): the Vertex `generateContent` dialect, planned as `GoogleModel` with a custom
  transport.
- The Orchestration Service (`/v2/completion`): a different request format with SAP's masking,
  filtering and grounding modules. LiteLLM's `sap/` provider covers it today, and Pydantic AI can
  talk to a LiteLLM proxy.
- X.509 certificate authentication for the token endpoint.
- Embeddings (`/embeddings`) should route the same way but are untested.

## Verification status

This package was written without access to a Generative AI Hub tenant (it is not in the BTP trial
or the free tier). Everything is implemented from:

- the [SAP Cloud SDK for AI (Python)](https://github.com/SAP/ai-sdk-python) and
  [(JavaScript)](https://github.com/SAP/ai-sdk-js) sources, for the token flow, deployment lookup
  and the `api-version` and header conventions;
- SAP's own recorded Azure OpenAI responses from the JavaScript SDK's test utilities, vendored
  under `tests/fixtures/sap_ai_sdk_js/` (Apache-2.0, see `NOTICE`), which the test suite plays
  back through a mock transport;
- [LiteLLM's SAP provider](https://docs.litellm.ai/docs/providers/sap), for the credential
  resolution order and environment variable names;
- Wouter Lemaire's [Agentic AI on BTP](https://community.sap.com/t5/technology-blog-posts-by-members/agentic-ai-on-btp-integrating-pydantic-ai-with-sap-ai-core/ba-p/14460448)
  series and its [repository](https://github.com/lemaiwo/btp-dynamic-multiagent-app), which run
  Pydantic AI on SAP AI Core via the SAP SDK's OpenAI client and document the quirks below.

Open questions a live tenant will answer, and where to look if something breaks:

- `stream_options: {include_usage: true}`, which Pydantic AI always sends when streaming. SAP's
  JavaScript SDK sends it on `api-version=2024-10-21` and gets usage on the last chunk; one
  integrator reports it rejected on another version. If streaming fails with a 400 naming
  `stream_options`, try `api_version='2024-10-21'` and open an issue with the error.
- Replaying an assistant turn with `content: null` and no `tool_calls` has been rejected by at
  least one model (`gpt-5.6-terra`). Pydantic AI does not produce such turns on its own.
- Tool schemas with `$schema` or typeless properties (common from MCP servers) have been rejected.
  Pydantic AI's OpenAI schema transformer already strips `$schema`.

If you have a tenant and can record a few cassettes, that is the most valuable contribution right
now; see [Contributing](#contributing).

## Using an SAP SDK client instead

If you already have an `AsyncOpenAI` client from `sap-ai-sdk-gen`, it can be used as-is and the
provider only contributes the model profiles:

```python
from gen_ai_hub.proxy import get_proxy_client
from gen_ai_hub.proxy.native.openai import AsyncOpenAI

from pydantic_ai_sap_ai_core import SapAiCoreProvider

client = AsyncOpenAI(proxy_client=get_proxy_client('gen-ai-hub'))
provider = SapAiCoreProvider(openai_client=client)
```

## Contributing

```bash
uv sync --dev
uv run pytest --cov
uv run ruff check . && uv run ruff format --check . && uv run pyright
```

Tests run entirely offline against the vendored SAP recordings and an `httpx2.MockTransport`. A
recorded end-to-end suite against a real tenant is the next step; recordings must scrub the
`Authorization` header, the token response's `access_token`, and deployment URLs.

## License

MIT. The recorded SAP responses under `tests/fixtures/sap_ai_sdk_js/` are Apache-2.0, copyright
SAP SE; see `NOTICE`.
