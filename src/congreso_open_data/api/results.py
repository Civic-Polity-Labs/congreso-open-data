"""Streaming high-level results with durable reconciliation metadata."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from datetime import UTC, datetime
from typing import Generic, TypeVar

from pydantic import BaseModel, ConfigDict, Field

from congreso_open_data.api.errors import ResultConsumedError
from congreso_open_data.api.queries import CongressQuery
from congreso_open_data.models import ArtifactManifest

T = TypeVar("T")


class QueryRun(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: str
    query_fingerprint: str
    started_at: datetime
    finished_at: datetime | None = None
    planned_resources: int = 0
    succeeded_resources: int = 0
    reused_resources: int = 0
    failed_resources: int = 0
    raw_records: int = 0
    normalized_records: int = 0
    duplicate_records: int = 0
    unmatched_text_records: int = 0
    complete: bool = False
    checkpoint_path: str | None = None
    event_log_path: str | None = None
    failures: tuple[str, ...] = ()
    resolved_entities: dict[str, str] = Field(default_factory=dict)


class SearchResult(Generic[T], Iterable[T]):
    """A bounded, single-pass iterable; call ``collect`` only for small results."""

    def __init__(
        self,
        *,
        query: CongressQuery,
        records: Callable[[], Iterator[T]],
        manifests: tuple[ArtifactManifest, ...] = (),
        run: QueryRun,
        on_finish: Callable[[int, bool, BaseException | None], QueryRun] | None = None,
    ) -> None:
        self.query = query
        self.manifests = manifests
        self._records = records
        self._run = run
        self._on_finish = on_finish
        self._consumed = False

    @property
    def run(self) -> QueryRun:
        return self._run

    def __iter__(self) -> Iterator[T]:
        if self._consumed:
            raise ResultConsumedError("SearchResult is streaming and can only be consumed once")
        self._consumed = True
        count = 0
        completed = False
        failure: BaseException | None = None
        try:
            for item in self._records():
                count += 1
                yield item
            completed = True
        except BaseException as exc:
            failure = exc
            raise
        finally:
            if self._on_finish is not None:
                self._run = self._on_finish(count, completed, failure)
            else:
                self._run = self._run.model_copy(
                    update={
                        "finished_at": datetime.now(UTC),
                        "normalized_records": count,
                        "complete": completed and failure is None,
                        "failures": (
                            self._run.failures
                            if failure is None
                            else (*self._run.failures, f"{type(failure).__name__}: {failure}")
                        ),
                    }
                )

    def collect(self, *, max_items: int = 10_000) -> list[T]:
        if max_items < 1:
            raise ValueError("max_items must be positive")
        output: list[T] = []
        for item in self:
            if len(output) >= max_items:
                raise ValueError(
                    f"Search result exceeds collect(max_items={max_items}); iterate it instead"
                )
            output.append(item)
        return output
