"""Model profiles for models served by SAP AI Core's OpenAI-compatible deployments."""

from __future__ import annotations

from collections.abc import Callable

from pydantic_ai.profiles import ModelProfile, merge_profile
from pydantic_ai.profiles.deepseek import deepseek_model_profile
from pydantic_ai.profiles.meta import meta_model_profile
from pydantic_ai.profiles.mistral import mistral_model_profile
from pydantic_ai.profiles.openai import OpenAIJsonSchemaTransformer, OpenAIModelProfile, openai_model_profile

__all__ = ('sap_ai_core_model_profile',)

_FAMILIES: dict[str, Callable[[str], ModelProfile | None]] = {
    # SAP lists OpenAI models under their plain names.
    'gpt-': openai_model_profile,
    'o1': openai_model_profile,
    'o3': openai_model_profile,
    'o4': openai_model_profile,
    # Vendor-prefixed names that SAP serves through the Azure OpenAI dialect all the same.
    'meta--': lambda name: meta_model_profile(name.removeprefix('meta--')),
    'mistralai--': lambda name: mistral_model_profile(name.removeprefix('mistralai--')),
    'deepseek': deepseek_model_profile,
}


def sap_ai_core_model_profile(model_name: str) -> ModelProfile:
    """The profile for a model behind an SAP AI Core `azure-openai` deployment.

    The family profile is looked up by the vendor prefix SAP puts in model names (`meta--`,
    `mistralai--`) or by the plain OpenAI name, then adjusted for the gateway: strict tool
    definitions are off, because SAP AI Core rejects `strict` on several models, and JSON schemas
    are shaped for the OpenAI dialect.
    """
    name = model_name.casefold()
    family: ModelProfile | None = None
    for prefix, profile in _FAMILIES.items():
        if name.startswith(prefix):
            family = profile(name)
            break
    return merge_profile(
        OpenAIModelProfile(json_schema_transformer=OpenAIJsonSchemaTransformer),
        family,
        OpenAIModelProfile(openai_supports_strict_tool_definition=False),
    )
