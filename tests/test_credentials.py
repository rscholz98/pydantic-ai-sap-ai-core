"""Credential resolution: every source, the precedence between them, and the error for each malformed input."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic_ai.exceptions import UserError

from pydantic_ai_sap_ai_core import SapAiCoreCredentials, load_credentials

from .conftest import API_URL, AUTH_URL, SERVICE_KEY

pytestmark = pytest.mark.usefixtures('clean_env')


def test_urls_are_normalized():
    credentials = SapAiCoreCredentials(
        client_id='id',
        client_secret='secret',
        auth_url='https://auth.example.com/',
        base_url='https://api.example.com',
    )
    assert credentials.auth_url == 'https://auth.example.com/oauth/token'
    assert credentials.base_url == 'https://api.example.com/v2'
    assert credentials.resource_group == 'default'

    already = SapAiCoreCredentials(
        client_id='id',
        client_secret='secret',
        auth_url='https://auth.example.com/oauth/token',
        base_url='https://api.example.com/v2/',
        resource_group='team-a',
    )
    assert already.auth_url == 'https://auth.example.com/oauth/token'
    assert already.base_url == 'https://api.example.com/v2'
    assert already.resource_group == 'team-a'


def test_empty_field_is_rejected():
    with pytest.raises(UserError, match='`client_secret` is empty'):
        SapAiCoreCredentials(client_id='id', client_secret='', auth_url='https://a', base_url='https://b')


def test_from_service_key_parsed_and_json():
    for key in (SERVICE_KEY, json.dumps(SERVICE_KEY)):
        credentials = SapAiCoreCredentials.from_service_key(key, resource_group='team-a')
        assert credentials.client_id == 'sb-client'
        assert credentials.client_secret == 'top-secret'
        assert credentials.auth_url == AUTH_URL
        assert credentials.base_url == API_URL
        assert credentials.resource_group == 'team-a'


def test_from_service_key_cloud_foundry_binding_shape():
    credentials = SapAiCoreCredentials.from_service_key({'label': 'aicore', 'credentials': SERVICE_KEY})
    assert credentials.auth_url == AUTH_URL


def test_from_service_key_errors():
    with pytest.raises(UserError, match='not a valid service key'):
        SapAiCoreCredentials.from_service_key('{not json')
    with pytest.raises(UserError, match='not a valid service key'):
        SapAiCoreCredentials.from_service_key('[1, 2]')
    with pytest.raises(UserError, match='incomplete: missing `client_secret`, `base_url`'):
        SapAiCoreCredentials.from_service_key({'clientid': 'id', 'url': 'https://auth'})
    with pytest.raises(UserError, match='X.509 certificate authentication'):
        SapAiCoreCredentials.from_service_key(
            {'clientid': 'id', 'url': 'https://auth', 'certificate': '-----BEGIN', 'key': '-----BEGIN'}
        )


def test_nothing_configured():
    with pytest.raises(UserError, match='SAP AI Core credentials not found'):
        load_credentials()


def test_explicit_arguments_win(clean_env: pytest.MonkeyPatch):
    clean_env.setenv('AICORE_SERVICE_KEY', json.dumps(SERVICE_KEY))
    credentials = load_credentials(
        client_id='explicit', client_secret='s', auth_url='https://x', base_url='https://y', resource_group='rg'
    )
    assert credentials.client_id == 'explicit'
    assert credentials.resource_group == 'rg'


def test_explicit_arguments_are_not_merged_with_other_sources(clean_env: pytest.MonkeyPatch):
    clean_env.setenv('AICORE_SERVICE_KEY', json.dumps(SERVICE_KEY))
    with pytest.raises(
        UserError, match='from arguments are incomplete: missing `client_secret`, `auth_url`, `base_url`'
    ):
        load_credentials(client_id='explicit')


def test_service_key_env_var(clean_env: pytest.MonkeyPatch):
    clean_env.setenv('AICORE_SERVICE_KEY', json.dumps(SERVICE_KEY))
    clean_env.setenv('AICORE_CLIENT_ID', 'ignored-because-service-key-wins')
    clean_env.setenv('AICORE_RESOURCE_GROUP', 'from-env')
    credentials = load_credentials()
    assert credentials.client_id == 'sb-client'
    assert credentials.resource_group == 'from-env'


def test_service_key_env_var_malformed(clean_env: pytest.MonkeyPatch):
    clean_env.setenv('AICORE_SERVICE_KEY', '{"clientid": 1}')
    with pytest.raises(UserError, match='`AICORE_SERVICE_KEY` environment variable is not a valid service key'):
        load_credentials()


def test_service_key_argument_takes_precedence_over_env(clean_env: pytest.MonkeyPatch):
    clean_env.setenv('AICORE_SERVICE_KEY', json.dumps({**SERVICE_KEY, 'clientid': 'from-env'}))
    assert load_credentials(service_key=SERVICE_KEY).client_id == 'sb-client'


def test_individual_env_vars(clean_env: pytest.MonkeyPatch):
    clean_env.setenv('AICORE_CLIENT_ID', 'id')
    clean_env.setenv('AICORE_CLIENT_SECRET', 'secret')
    clean_env.setenv('AICORE_AUTH_URL', 'https://auth.example.com')
    clean_env.setenv('AICORE_BASE_URL', 'https://api.example.com/v2')
    credentials = load_credentials()
    assert credentials.auth_url == 'https://auth.example.com/oauth/token'
    assert credentials.base_url == 'https://api.example.com/v2'

    clean_env.delenv('AICORE_BASE_URL')
    with pytest.raises(UserError, match='from environment variables are incomplete: missing `base_url`'):
        load_credentials()


def write_config(path: Path, **values: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(values), encoding='utf-8')


CONFIG = dict(
    AICORE_CLIENT_ID='cfg-id',
    AICORE_CLIENT_SECRET='cfg-secret',
    AICORE_AUTH_URL='https://cfg-auth.example.com',
    AICORE_BASE_URL='https://cfg-api.example.com',
)


def test_config_file_in_home(clean_env: pytest.MonkeyPatch, tmp_path: Path):
    write_config(tmp_path / 'aicore-home' / 'config.json', **CONFIG, AICORE_RESOURCE_GROUP='cfg-rg')
    credentials = load_credentials()
    assert credentials.client_id == 'cfg-id'
    assert credentials.resource_group == 'cfg-rg'
    # Environment variables beat the config file for credentials and for the resource group alike.
    clean_env.setenv('AICORE_RESOURCE_GROUP', 'env-rg')
    assert load_credentials().resource_group == 'env-rg'


def test_config_file_profile_and_lowercase_keys(clean_env: pytest.MonkeyPatch, tmp_path: Path):
    write_config(
        tmp_path / 'aicore-home' / 'config_staging.json',
        **{k.removeprefix('AICORE_').lower(): v for k, v in CONFIG.items()},
    )
    clean_env.setenv('AICORE_PROFILE', 'staging')
    assert load_credentials().client_id == 'cfg-id'
    clean_env.setenv('AICORE_PROFILE', 'default')
    with pytest.raises(UserError, match='not found'):
        load_credentials()


def test_config_file_explicit_path(clean_env: pytest.MonkeyPatch, tmp_path: Path):
    write_config(tmp_path / 'elsewhere.json', **CONFIG)
    clean_env.setenv('AICORE_CONFIG', str(tmp_path / 'elsewhere.json'))
    assert load_credentials().auth_url == 'https://cfg-auth.example.com/oauth/token'


def test_config_file_malformed(clean_env: pytest.MonkeyPatch, tmp_path: Path):
    path = tmp_path / 'broken.json'
    path.write_text('[]', encoding='utf-8')
    clean_env.setenv('AICORE_CONFIG', str(path))
    with pytest.raises(UserError, match='must hold a JSON object'):
        load_credentials()


def test_vcap_services(clean_env: pytest.MonkeyPatch):
    vcap = {
        'hana': [{'label': 'hana', 'credentials': {'host': 'db'}}],
        'aicore': [{'label': 'aicore', 'credentials': SERVICE_KEY}],
    }
    clean_env.setenv('VCAP_SERVICES', json.dumps(vcap))
    assert load_credentials().client_id == 'sb-client'


@pytest.mark.parametrize(
    'vcap', ['not json', '{"aicore": "nope"}', '{"other": [{"label": "other"}]}', '{"aicore": [{"label": "aicore"}]}']
)
def test_vcap_services_without_binding(clean_env: pytest.MonkeyPatch, vcap: str):
    clean_env.setenv('VCAP_SERVICES', vcap)
    with pytest.raises(UserError, match='not found'):
        load_credentials()


def test_from_env_is_load_credentials(clean_env: pytest.MonkeyPatch):
    clean_env.setenv('AICORE_SERVICE_KEY', json.dumps(SERVICE_KEY))
    assert SapAiCoreCredentials.from_env() == load_credentials()
