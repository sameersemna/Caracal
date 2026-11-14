"""DDL shape of the Core tables (docs/05 §1, card T0.2 requirement 8)."""

from __future__ import annotations

import pytest
import sqlalchemy as sa
from sqlalchemy import Table
from sqlalchemy.dialects import postgresql
from sqlalchemy.schema import CreateIndex, CreateTable

from caracal.db.tables import (
    JOB_SOURCE_KINDS,
    JOB_STATUSES,
    STAGE_STATUSES,
    job_stages,
    jobs,
    metadata,
    reports,
    segments,
    uploads,
)

DIALECT = postgresql.dialect()


def _ddl(table: Table) -> str:
    return str(CreateTable(table).compile(dialect=DIALECT))


def _index_ddl(table: Table, name: str) -> str:
    index = next(index for index in table.indexes if index.name == name)
    return str(CreateIndex(index).compile(dialect=DIALECT))


def test_only_the_five_tables_exist_and_metadata_is_schema_less() -> None:
    assert set(metadata.tables) == {"jobs", "job_stages", "uploads", "reports", "segments"}
    assert "voiceprints" not in metadata.tables
    assert metadata.schema is None


def test_no_column_is_a_sql_reserved_word() -> None:
    reserved = {"start", "end"}
    for table in metadata.tables.values():
        assert reserved.isdisjoint(table.c.keys()), table.name
    assert {"start_s", "end_s"} <= set(segments.c.keys())


def test_jobs_table_shape() -> None:
    ddl = _ddl(jobs)
    assert "CREATE TABLE jobs" in ddl
    for fragment in (
        "id UUID NOT NULL",
        "status TEXT DEFAULT 'queued' NOT NULL",
        "source_kind TEXT NOT NULL",
        "source_url TEXT",
        "upload_id UUID",
        "title TEXT",
        "options JSONB DEFAULT '{}'::jsonb NOT NULL",
        "progress JSONB DEFAULT '{}'::jsonb NOT NULL",
        "error JSONB",
        "created_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL",
        "updated_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL",
        "finished_at TIMESTAMP WITH TIME ZONE",
        "sha256 TEXT",
        "CONSTRAINT pk_jobs PRIMARY KEY (id)",
    ):
        assert fragment in ddl, fragment
    assert isinstance(jobs.c.created_at.type, sa.DateTime)
    assert jobs.c.created_at.type.timezone is True
    assert isinstance(jobs.c.options.type, postgresql.JSONB)


def test_jobs_check_constraints_cover_every_enum_value() -> None:
    ddl = _ddl(jobs)
    status_check = next(constraint for constraint in jobs.constraints if constraint.name == "ck_jobs_status")
    rendered = str(status_check.sqltext)
    for status in JOB_STATUSES:
        assert f"'{status}'" in rendered
    assert "ck_jobs_status" in ddl
    assert "ck_jobs_source_kind" in ddl
    for kind in JOB_SOURCE_KINDS:
        assert f"'{kind}'" in str(
            next(c for c in jobs.constraints if c.name == "ck_jobs_source_kind").sqltext
        )


def test_jobs_indexes() -> None:
    assert _index_ddl(jobs, "ix_jobs_status_created_at") == (
        "CREATE INDEX ix_jobs_status_created_at ON jobs (status, created_at DESC)"
    )
    assert _index_ddl(jobs, "ix_jobs_sha256") == "CREATE INDEX ix_jobs_sha256 ON jobs (sha256)"


def test_job_stages_shape_and_cascade() -> None:
    ddl = _ddl(job_stages)
    assert "CONSTRAINT pk_job_stages PRIMARY KEY (job_id, name)" in ddl
    assert "CONSTRAINT ck_job_stages_status CHECK" in ddl
    assert "FOREIGN KEY(job_id) REFERENCES jobs (id) ON DELETE CASCADE" in ddl
    assert "attempts INTEGER DEFAULT 0 NOT NULL" in ddl
    status_check = next(
        constraint for constraint in job_stages.constraints if constraint.name == "ck_job_stages_status"
    )
    for status in STAGE_STATUSES:
        assert f"'{status}'" in str(status_check.sqltext)


def test_uploads_shape() -> None:
    ddl = _ddl(uploads)
    for fragment in (
        "id UUID NOT NULL",
        "filename TEXT NOT NULL",
        "size_bytes BIGINT NOT NULL",
        "sha256 TEXT NOT NULL",
        "path TEXT NOT NULL",
        "created_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL",
        "consumed_by UUID",
        "CONSTRAINT pk_uploads PRIMARY KEY (id)",
    ):
        assert fragment in ddl, fragment


def test_reports_shape_and_cascade() -> None:
    ddl = _ddl(reports)
    for fragment in (
        "job_id UUID NOT NULL",
        "schema_version TEXT NOT NULL",
        "report JSONB NOT NULL",
        "created_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL",
        "CONSTRAINT pk_reports PRIMARY KEY (job_id)",
        "FOREIGN KEY(job_id) REFERENCES jobs (id) ON DELETE CASCADE",
    ):
        assert fragment in ddl, fragment


def test_segments_generated_tsv_and_gin_index() -> None:
    ddl = _ddl(segments)
    assert "CONSTRAINT pk_segments PRIMARY KEY (job_id, seg_id)" in ddl
    assert "FOREIGN KEY(job_id) REFERENCES jobs (id) ON DELETE CASCADE" in ddl
    assert "tsv TSVECTOR GENERATED ALWAYS AS (to_tsvector('simple', text)) STORED" in ddl
    assert isinstance(segments.c.text.type, sa.Text)
    assert "CREATE INDEX ix_segments_tsv ON segments USING gin (tsv)" == _index_ddl(
        segments, "ix_segments_tsv"
    )


@pytest.mark.parametrize("table", [job_stages, reports, segments])
def test_every_foreign_key_cascades(table: Table) -> None:
    constraints = list(table.foreign_key_constraints)
    assert constraints
    assert all(constraint.ondelete == "CASCADE" for constraint in constraints)
