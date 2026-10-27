"""End to end through `Agent` and `OpenAIChatModel`, against SAP's recorded Azure OpenAI responses.

Every request the OpenAI SDK produces goes through the fake tenant: token, deployment listing, then
the chat completion at the resolved deployment URL with the `api-version` query parameter.
"""

from __future__ import annotations

import httpx2
import pytest
from inline_snapshot import snapshot
from pydantic_ai import Agent, ModelHTTPError
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.openai import OpenAIChatModel

from pydantic_ai_sap_ai_core import SapAiCoreProvider

from .conftest import API_URL, MockSap, ProviderFactory, chat_completion, sse

pytestmark = pytest.mark.usefixtures('clean_env')


async def test_run(provider: SapAiCoreProvider, sap: MockSap):
    agent = Agent(OpenAIChatModel('gpt-4o', provider=provider))
    result = await agent.run('hello')

    assert result.output == 'Hello! I’m here and ready to help. How can I assist you today?'
    assert result.usage.input_tokens == 13
    assert result.usage.output_tokens == 17
    assert result.response.provider_name == 'sap-ai-core'
    assert result.response.model_name == 'gpt-4o-2024-08-06'

    assert [str(r.url) for r in sap.requests] == [
        'https://example.authentication.eu12.hana.ondemand.com/oauth/token',
        f'{API_URL}/lm/deployments?scenarioId=foundation-models&status=RUNNING',
        f'{API_URL}/inference/deployments/d768512472eae8f4/chat/completions?api-version=2025-03-01-preview',
    ]
    chat_request = sap.requests[-1]
    assert chat_request.headers['authorization'] == 'Bearer token-1'
    assert chat_request.headers['ai-resource-group'] == 'default'
    assert chat_request.headers['ai-client-type'] == 'pydantic-ai-sap-ai-core'
    assert chat_request.headers['user-agent'].startswith('pydantic-ai/')
    body = sap.bodies_to('/chat/completions')[0]
    assert body['model'] == 'gpt-4o'
    assert body['messages'] == [{'role': 'user', 'content': 'hello'}]
    assert body['stream'] is False


async def test_second_run_reuses_token_and_listing(provider: SapAiCoreProvider, sap: MockSap):
    agent = Agent(OpenAIChatModel('gpt-4o', provider=provider))
    await agent.run('hello')
    await agent.run('hello again')
    assert len(sap.requests_to('/oauth/token')) == 1
    assert len(sap.requests_to('/lm/deployments')) == 1
    assert len(sap.requests_to('/chat/completions')) == 2


async def test_stream(provider: SapAiCoreProvider, sap: MockSap):
    sap.inference_responses = [sse('azure-openai-chat-completion-stream-chunks.txt')]
    agent = Agent(OpenAIChatModel('gpt-4o', provider=provider))

    async with agent.run_stream('What is the capital of France?') as result:
        chunks = [chunk async for chunk in result.stream_text(delta=True)]
        output = await result.get_output()

    assert ''.join(chunks) == output == 'The capital of France is Paris.'
    # Usage arrives on the last chunk, which SAP sends because `stream_options.include_usage` was requested.
    assert result.usage.input_tokens == 14
    assert result.usage.output_tokens == 7
    body = sap.bodies_to('/chat/completions')[0]
    assert body['stream'] is True
    assert body['stream_options'] == {'include_usage': True}


async def test_tool_call_round_trip(provider: SapAiCoreProvider, sap: MockSap):
    """The recorded tool-call stream calls `add(a=1, b=2)`; the second request carries the result."""
    sap.inference_responses = [
        sse('azure-openai-chat-completion-stream-tools-chunks.txt'),
        sse('azure-openai-chat-completion-stream-chunks.txt'),
    ]
    agent = Agent(OpenAIChatModel('gpt-4o', provider=provider))

    @agent.tool_plain
    def add(a: int, b: int) -> int:
        return a + b

    async with agent.run_stream('add 1 and 2') as result:
        output = await result.get_output()
    assert output == 'The capital of France is Paris.'

    messages = result.all_messages()
    assert [[type(part).__name__ for part in m.parts] for m in messages] == [
        ['UserPromptPart'],
        ['ToolCallPart'],
        ['ToolReturnPart'],
        ['TextPart'],
    ]
    tool_call = messages[1].parts[0]
    assert isinstance(tool_call, ToolCallPart)
    assert (tool_call.tool_name, tool_call.args, tool_call.tool_call_id) == (
        'add',
        '{"a":1,"b":2}',
        'call_De0ejo2G1gknErC39DDH2JpS',
    )
    tool_return = messages[2].parts[0]
    assert isinstance(tool_return, ToolReturnPart)
    assert (tool_return.tool_name, tool_return.content, tool_return.tool_call_id) == (
        'add',
        3,
        'call_De0ejo2G1gknErC39DDH2JpS',
    )
    assert messages[3].parts == [TextPart(content='The capital of France is Paris.')]
    responses = [m for m in messages if isinstance(m, ModelResponse)]
    assert [
        (r.model_name, r.provider_name, r.provider_url, r.finish_reason, r.provider_response_id) for r in responses
    ] == [
        ('gpt-4o-2024-08-06', 'sap-ai-core', API_URL, 'tool_call', 'chatcmpl-BlYPCAKjvl7PBJxuZb352G12WIQK5'),
        ('gpt-4o', 'sap-ai-core', API_URL, 'stop', 'chatcmpl-ANKsHIdjvozwuOGpGI6rygvwSJH0I'),
    ]
    assert [(r.usage.input_tokens, r.usage.output_tokens) for r in responses] == [(52, 18), (14, 7)]

    first, second = sap.bodies_to('/chat/completions')
    # Strict tool definitions are off for SAP AI Core; the schema itself is the plain OpenAI shape.
    assert first['tools'] == snapshot(
        [
            {
                'type': 'function',
                'function': {
                    'name': 'add',
                    'description': '',
                    'parameters': {
                        'additionalProperties': False,
                        'properties': {'a': {'type': 'integer'}, 'b': {'type': 'integer'}},
                        'required': ['a', 'b'],
                        'type': 'object',
                    },
                },
            }
        ]
    )
    assert 'strict' not in first['tools'][0]['function']
    # Pydantic AI replays the tool-call turn with `content: None`, the standard OpenAI shape when `tool_calls`
    # is set; the null-content rejection reported for `gpt-5.6-terra` concerns turns with no tool calls.
    assert second['messages'][1:] == snapshot(
        [
            {
                'role': 'assistant',
                'content': None,
                'tool_calls': [
                    {
                        'id': 'call_De0ejo2G1gknErC39DDH2JpS',
                        'type': 'function',
                        'function': {'name': 'add', 'arguments': '{"a":1,"b":2}'},
                    }
                ],
            },
            {'role': 'tool', 'tool_call_id': 'call_De0ejo2G1gknErC39DDH2JpS', 'content': '3'},
        ]
    )


async def test_model_error_is_a_model_http_error(provider: SapAiCoreProvider, sap: MockSap):
    sap.inference_responses = [
        httpx2.Response(400, content=chat_completion('azure-openai-error-response.json').content)
    ]
    agent = Agent(OpenAIChatModel('gpt-4o', provider=provider))
    with pytest.raises(ModelHTTPError) as exc_info:
        await agent.run('hello')
    assert exc_info.value.status_code == 400
    assert exc_info.value.model_name == 'gpt-4o'
    assert exc_info.value.body == {
        'code': None,
        'message': 'Relevant error message',
        'param': None,
        'type': 'invalid_request_error',
    }


async def test_unknown_model_surfaces_the_routing_error(provider: SapAiCoreProvider):
    agent = Agent(OpenAIChatModel('gpt-5', provider=provider))
    with pytest.raises(UserError, match="No running SAP AI Core deployment serves model 'gpt-5'"):
        await agent.run('hello')


async def test_explicit_deployment_id(provider_factory: ProviderFactory, sap: MockSap):
    provider = provider_factory(deployment_ids={'gpt-4o': 'pinned'})
    agent = Agent(OpenAIChatModel('gpt-4o', provider=provider))
    await agent.run('hello')
    assert sap.requests_to('/lm/deployments') == []
    assert sap.requests[-1].url.path == '/v2/inference/deployments/pinned/chat/completions'
