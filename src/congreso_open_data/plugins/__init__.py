"""Public extension contracts for custom OCR, NLP and language models."""

from congreso_open_data.models import ExtractionSpec
from congreso_open_data.plugins.models import (
    CANDIDATE_ENVELOPE_SCHEMA,
    CallableModelBackend,
    CandidateEnvelope,
    CandidateValue,
    ExtractionLimits,
    ExtractionTask,
    ModelBackend,
    ModelDescriptor,
    ModelPluginError,
    ModelRequest,
    ModelResponse,
    ModelResponseError,
    ModelTimeoutError,
    ModelUnavailableError,
    StructuredModelExtractor,
    model_fingerprint,
)
from congreso_open_data.plugins.registry import (
    MODEL_ENTRY_POINT_GROUP,
    ModelFactory,
    ModelRegistry,
)
from congreso_open_data.protocols import ExtractionContext, ExtractionResult, ExtractorBackend

__all__ = [
    "CANDIDATE_ENVELOPE_SCHEMA",
    "MODEL_ENTRY_POINT_GROUP",
    "CallableModelBackend",
    "CandidateEnvelope",
    "CandidateValue",
    "ExtractionContext",
    "ExtractionLimits",
    "ExtractionResult",
    "ExtractionSpec",
    "ExtractionTask",
    "ExtractorBackend",
    "ModelBackend",
    "ModelDescriptor",
    "ModelFactory",
    "ModelPluginError",
    "ModelRegistry",
    "ModelRequest",
    "ModelResponse",
    "ModelResponseError",
    "ModelTimeoutError",
    "ModelUnavailableError",
    "StructuredModelExtractor",
    "model_fingerprint",
]
