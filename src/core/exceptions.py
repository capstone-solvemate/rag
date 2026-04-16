from __future__ import annotations


class GenerationError(Exception):
    """Raised when the LLM answer generation step fails.

    Wraps underlying API or runtime errors so the API layer
    can catch a single exception type and return a clean 503
    without leaking internal error details to the client.

    Attributes:
        message:  Human-readable description of the failure.
        cause:    Original exception that triggered this error, if any.
    """

    def __init__(self, message: str, cause: BaseException | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.cause = cause

    def __str__(self) -> str:
        if self.cause:
            return f"{self.message} | caused by: {type(self.cause).__name__}: {self.cause}"
        return self.message