"""Stable public exception hierarchy for high-level Congress queries."""


class CongressError(RuntimeError):
    """Base error raised by the high-level API."""


class QueryValidationError(CongressError, ValueError):
    """The requested filters are inconsistent or unsupported."""


class EntityNotFoundError(CongressError, LookupError):
    """An official person or entity could not be resolved."""


class AmbiguousEntityError(CongressError, LookupError):
    """A human-friendly identifier resolves to several official entities."""


class SourceUnavailableError(CongressError):
    """The official source could not be reached after bounded retries."""


class SourceContractError(CongressError, ValueError):
    """The official source returned a payload with an unexpected shape."""


class IncompleteResultError(CongressError):
    """A fail-closed query could not reconcile all planned records."""


class OptionalDependencyError(CongressError, ImportError):
    """An explicitly selected parser or model dependency is not installed."""


class ResultConsumedError(CongressError):
    """A streaming query result was consumed more than once."""
