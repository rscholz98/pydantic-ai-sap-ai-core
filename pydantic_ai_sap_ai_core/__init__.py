"""SAP AI Core (Generative AI Hub) provider for Pydantic AI."""

from .auth import SapAiCoreAuth
from .credentials import SapAiCoreCredentials, load_credentials
from .profiles import sap_ai_core_model_profile
from .provider import DEFAULT_API_VERSION, SapAiCoreProvider

__all__ = (
    'DEFAULT_API_VERSION',
    'SapAiCoreAuth',
    'SapAiCoreCredentials',
    'SapAiCoreProvider',
    'load_credentials',
    'sap_ai_core_model_profile',
)

__version__ = '0.1.0'
