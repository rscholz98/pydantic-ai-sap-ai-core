"""Error-message helpers."""

from __future__ import annotations

from pydantic import ValidationError


def validation_summary(error: ValidationError) -> str:
    """Describe a validation error by location and message only.

    `str(error)` would include the offending input, which for a service key or a token response
    is a secret, so it must never reach an exception message or a log. For the same reason callers
    raise `from None`: a chained cause is printed with its inputs in a logged traceback.
    """
    return '; '.join(
        f'{".".join(str(part) for part in err["loc"]) or "<root>"}: {err["msg"]}'
        for err in error.errors(include_url=False, include_input=False)
    )
