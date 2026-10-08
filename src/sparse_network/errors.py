"""Domain exceptions used by the controller."""


class SparseNetworkError(Exception):
    """Base class for expected Sparse Network failures."""


class ConfigurationError(SparseNetworkError):
    """Configuration is missing or invalid."""


class ArtifactValidationError(SparseNetworkError):
    """A model artifact failed validation."""


class ModelDownloadError(SparseNetworkError):
    """A pinned model snapshot could not be downloaded or verified."""


class EndpointStateError(SparseNetworkError):
    """An operation is invalid for the endpoint's current state."""


class RuntimeUnavailableError(SparseNetworkError):
    """A configured serving runtime cannot be located or started."""


class RequestCancelledError(SparseNetworkError):
    """The controller cancelled an active request."""


class RequestFailedError(SparseNetworkError):
    """The serving runtime failed a request."""


class ResourceAdmissionError(SparseNetworkError):
    """A request or endpoint would violate a configured resource reserve."""


class QueueCapacityError(SparseNetworkError):
    """A bounded scheduler queue cannot accept more work."""
