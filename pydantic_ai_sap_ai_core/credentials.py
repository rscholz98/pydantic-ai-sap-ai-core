"""Credential resolution for SAP AI Core.

SAP AI Core has no static API key. Every request needs an OAuth2 bearer token obtained with the
client-credentials grant from the XSUAA token endpoint named in the service key, plus an
``AI-Resource-Group`` header. The resolution order and the ``AICORE_*`` names below match the SAP
Cloud SDK for AI and LiteLLM, so a service key that works there works here unchanged.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, TypeAdapter, ValidationError
from pydantic_ai.exceptions import UserError

from ._errors import validation_summary

__all__ = (
    'DEFAULT_RESOURCE_GROUP',
    'SapAiCoreCredentials',
    'load_credentials',
)

DEFAULT_RESOURCE_GROUP = 'default'
"""The resource group SAP AI Core creates at onboarding, used when none is configured."""

SERVICE_KEY_ENV_VAR = 'AICORE_SERVICE_KEY'
CONFIG_ENV_VAR = 'AICORE_CONFIG'
HOME_ENV_VAR = 'AICORE_HOME'
PROFILE_ENV_VAR = 'AICORE_PROFILE'
RESOURCE_GROUP_ENV_VAR = 'AICORE_RESOURCE_GROUP'
VCAP_SERVICES_ENV_VAR = 'VCAP_SERVICES'
VCAP_SERVICE_LABEL = 'aicore'

_TOKEN_ENDPOINT_SUFFIX = '/oauth/token'
_API_VERSION_SUFFIX = '/v2'
_FIELDS = ('client_id', 'client_secret', 'auth_url', 'base_url')


@dataclass(frozen=True, kw_only=True)
class SapAiCoreCredentials:
    """What is needed to call SAP AI Core: an OAuth2 client, its token endpoint, the AI API URL and a resource group.

    URLs are normalized on construction: `auth_url` always ends in `/oauth/token` and `base_url`
    always ends in `/v2`, whichever form the service key or environment used.
    """

    client_id: str
    """The OAuth2 client id, `clientid` in the service key."""
    client_secret: str
    """The OAuth2 client secret, `clientsecret` in the service key."""
    auth_url: str
    """The XSUAA token endpoint, `url` in the service key, with `/oauth/token` appended."""
    base_url: str
    """The AI API URL, `serviceurls.AI_API_URL` in the service key, with `/v2` appended."""
    resource_group: str = DEFAULT_RESOURCE_GROUP
    """The resource group whose deployments are used, sent as the `AI-Resource-Group` header."""

    def __post_init__(self) -> None:
        for name in _FIELDS:
            if not getattr(self, name):
                raise UserError(f'SAP AI Core credentials are incomplete: `{name}` is empty.')
        object.__setattr__(self, 'auth_url', _with_suffix(self.auth_url, _TOKEN_ENDPOINT_SUFFIX))
        object.__setattr__(self, 'base_url', _with_suffix(self.base_url, _API_VERSION_SUFFIX))

    @classmethod
    def from_service_key(
        cls, service_key: str | Mapping[str, Any], *, resource_group: str | None = None
    ) -> SapAiCoreCredentials:
        """Build credentials from an SAP AI Core service key, as JSON text or already parsed.

        Both the plain service key (`clientid` at the top level) and the Cloud Foundry binding shape
        (the same fields under `credentials`) are accepted.
        """
        fields = _parse_service_key(service_key, source='service key').fields()
        return cls(**_complete(fields, source='service key'), resource_group=resource_group or DEFAULT_RESOURCE_GROUP)

    @classmethod
    def from_env(cls) -> SapAiCoreCredentials:
        """Resolve credentials from the environment alone; see [`load_credentials`][pydantic_ai_sap_ai_core.load_credentials]."""
        return load_credentials()


def load_credentials(
    *,
    client_id: str | None = None,
    client_secret: str | None = None,
    auth_url: str | None = None,
    base_url: str | None = None,
    resource_group: str | None = None,
    service_key: str | Mapping[str, Any] | None = None,
) -> SapAiCoreCredentials:
    """Resolve SAP AI Core credentials, first source with any credential wins.

    The sources, in order:

    1. The explicit keyword arguments `client_id`, `client_secret`, `auth_url` and `base_url`.
    2. `service_key`, or the `AICORE_SERVICE_KEY` environment variable, holding the service key JSON.
    3. The `AICORE_CLIENT_ID`, `AICORE_CLIENT_SECRET`, `AICORE_AUTH_URL` and `AICORE_BASE_URL`
       environment variables.
    4. A config file, `$AICORE_CONFIG` if set, else `$AICORE_HOME/config.json` (default `~/.aicore/config.json`),
       or `config_<profile>.json` when `AICORE_PROFILE` is set, holding the same `AICORE_*` keys.
    5. The `aicore` binding in `VCAP_SERVICES` on Cloud Foundry.

    Credentials are not merged across sources: the first source that provides any of the four
    values must provide all of them. The resource group is resolved on its own, from the
    `resource_group` argument, then `AICORE_RESOURCE_GROUP`, then the config file, falling back to
    `default`.

    Raises:
        UserError: if no source provides credentials, a source is incomplete, or a source is malformed.
    """
    explicit = {
        name: value
        for name, value in (
            ('client_id', client_id),
            ('client_secret', client_secret),
            ('auth_url', auth_url),
            ('base_url', base_url),
        )
        if value
    }
    config = _load_config_file()

    sources: list[tuple[str, dict[str, str]]] = [
        ('arguments', explicit),
        ('service key', _service_key_source(service_key)),
        ('environment variables', {name: v for name in _FIELDS if (v := os.getenv(f'AICORE_{name.upper()}'))}),
        ('config file', {name: v for name in _FIELDS if (v := _config_value(config, name))}),
        ('VCAP_SERVICES', _vcap_source()),
    ]
    for source, fields in sources:
        if fields:
            break
    else:
        raise UserError(
            'SAP AI Core credentials not found. Set the `AICORE_SERVICE_KEY` environment variable to your service key '
            'JSON, or set `AICORE_CLIENT_ID`, `AICORE_CLIENT_SECRET`, `AICORE_AUTH_URL` and `AICORE_BASE_URL`, '
            'or pass them to `SapAiCoreProvider(...)`.'
        )

    resource_group = (
        resource_group
        or os.getenv(RESOURCE_GROUP_ENV_VAR)
        or _config_value(config, 'resource_group')
        or DEFAULT_RESOURCE_GROUP
    )
    return SapAiCoreCredentials(**_complete(fields, source=source), resource_group=resource_group)


def _complete(fields: Mapping[str, str], *, source: str) -> dict[str, str]:
    missing = [name for name in _FIELDS if name not in fields]
    if missing:
        raise UserError(
            f'SAP AI Core credentials from {source} are incomplete: missing {", ".join(f"`{m}`" for m in missing)}.'
        )
    return {name: fields[name] for name in _FIELDS}


def _with_suffix(url: str, suffix: str) -> str:
    url = url.rstrip('/')
    return url if url.endswith(suffix) else url + suffix


class _ServiceUrls(BaseModel):
    ai_api_url: str | None = Field(default=None, alias='AI_API_URL')


class _ServiceKey(BaseModel):
    """The fields of an SAP AI Core service key, optionally nested under `credentials` as in a CF binding."""

    clientid: str | None = None
    clientsecret: str | None = None
    url: str | None = None
    serviceurls: _ServiceUrls | None = None
    certificate: str | None = None
    credentials: _ServiceKey | None = None

    def fields(self) -> dict[str, str]:
        key = self.credentials or self
        candidates = {
            'client_id': key.clientid,
            'client_secret': key.clientsecret,
            'auth_url': key.url,
            'base_url': key.serviceurls.ai_api_url if key.serviceurls else None,
        }
        fields = {name: value for name, value in candidates.items() if value}
        if fields and 'client_secret' not in fields and key.certificate:
            raise UserError(
                'The SAP AI Core service key uses X.509 certificate authentication (`certificate`/`key`), '
                'which this provider does not support yet. Create a service key with a `clientsecret` instead.'
            )
        return fields


class _VcapBinding(BaseModel):
    label: str | None = None
    credentials: _ServiceKey | None = None


_service_key_adapter = TypeAdapter(_ServiceKey)
_config_adapter = TypeAdapter(dict[str, Any])
_vcap_adapter = TypeAdapter(dict[str, list[_VcapBinding]])


def _parse_service_key(service_key: str | Mapping[str, Any], *, source: str) -> _ServiceKey:
    try:
        if isinstance(service_key, str):
            return _service_key_adapter.validate_json(service_key)
        return _service_key_adapter.validate_python(service_key)
    except ValidationError as e:
        raise UserError(f'The SAP AI Core {source} is not a valid service key: {validation_summary(e)}') from None


def _service_key_source(service_key: str | Mapping[str, Any] | None) -> dict[str, str]:
    if service_key is not None:
        return _parse_service_key(service_key, source='service key').fields()
    raw = os.getenv(SERVICE_KEY_ENV_VAR)
    if not raw:
        return {}
    return _parse_service_key(raw, source=f'`{SERVICE_KEY_ENV_VAR}` environment variable').fields()


def _load_config_file() -> dict[str, Any]:
    if explicit := os.getenv(CONFIG_ENV_VAR):
        path = Path(explicit)
    else:
        home = Path(os.getenv(HOME_ENV_VAR) or Path.home() / '.aicore')
        profile = os.getenv(PROFILE_ENV_VAR)
        path = home / (f'config_{profile}.json' if profile and profile != 'default' else 'config.json')
    if not path.is_file():
        return {}
    try:
        return _config_adapter.validate_json(path.read_bytes())
    except ValidationError as e:
        raise UserError(
            f'The SAP AI Core config file {path} must hold a JSON object: {validation_summary(e)}'
        ) from None


def _config_value(config: Mapping[str, Any], name: str) -> str | None:
    value: Any = config.get(f'AICORE_{name.upper()}', config.get(name))
    return value if isinstance(value, str) and value else None


def _vcap_source() -> dict[str, str]:
    raw = os.getenv(VCAP_SERVICES_ENV_VAR)
    if not raw:
        return {}
    try:
        services = _vcap_adapter.validate_json(raw)
    except ValidationError:
        return {}
    for bindings in services.values():
        for binding in bindings:
            if binding.label == VCAP_SERVICE_LABEL and binding.credentials is not None:
                return binding.credentials.fields()
    return {}
