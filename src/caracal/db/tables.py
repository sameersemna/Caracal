"""SQLAlchemy Core table definitions (docs/05 §1).

Tables are deliberately schema-less: every connection sets ``search_path`` to
``CARACAL_DB_SCHEMA`` (see :mod:`caracal.db.engine`), so unqualified statements resolve to the
configured schema and the same objects work in the throwaway schemas used by live tests.
"""

from __future__ import annotations

from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    Column,
    Computed,
    DateTime,
    Float,
    ForeignKeyConstraint,
    Index,
    Integer,
    MetaData,
    PrimaryKeyConstraint,
    Table,
    Text,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB, TSVECTOR, UUID

metadata = MetaData(schema=None)

JOB_STATUSES: tuple[str, ...] = ("queued", "running", "done", "failed", "cancelled")
JOB_TERMINAL_STATUSES: frozenset[str] = frozenset({"done", "failed", "cancelled"})
JOB_SOURCE_KINDS: tuple[str, ...] = ("upload", "direct", "platform")
STAGE_STATUSES: tuple[str, ...] = ("pending", "running", "done", "skipped", "failed")


def _in_list(column: str, values: tuple[str, ...]) -> str:
    quoted = ", ".join(f"'{value}'" for value in values)
    return f"{column} IN ({quoted})"


jobs = Table(
    "jobs",
    metadata,
    Column("id", UUID(as_uuid=True), nullable=False),
    Column("status", Text, nullable=False, server_default="queued"),
    Column("source_kind", Text, nullable=False),
    Column("source_url", Text, nullable=True),
    Column("upload_id", UUID(as_uuid=True), nullable=True),
    Column("title", Text, nullable=True),
    Column("options", JSONB, nullable=False, server_default=text("'{}'::jsonb")),
    Column("progress", JSONB, nullable=False, server_default=text("'{}'::jsonb")),
    Column("error", JSONB, nullable=True),
    Column("created_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
    Column("updated_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
    Column("finished_at", DateTime(timezone=True), nullable=True),
    Column("sha256", Text, nullable=True),
    PrimaryKeyConstraint("id", name="pk_jobs"),
    CheckConstraint(_in_list("status", JOB_STATUSES), name="ck_jobs_status"),
    CheckConstraint(_in_list("source_kind", JOB_SOURCE_KINDS), name="ck_jobs_source_kind"),
)

Index("ix_jobs_status_created_at", jobs.c.status, jobs.c.created_at.desc())
Index("ix_jobs_sha256", jobs.c.sha256)

job_stages = Table(
    "job_stages",
    metadata,
    Column("job_id", UUID(as_uuid=True), nullable=False),
    Column("name", Text, nullable=False),
    Column("status", Text, nullable=False),
    Column("reason", Text, nullable=True),
    Column("attempts", Integer, nullable=False, server_default=text("0")),
    Column("started_at", DateTime(timezone=True), nullable=True),
    Column("ended_at", DateTime(timezone=True), nullable=True),
    PrimaryKeyConstraint("job_id", "name", name="pk_job_stages"),
    CheckConstraint(_in_list("status", STAGE_STATUSES), name="ck_job_stages_status"),
    ForeignKeyConstraint(
        ["job_id"],
        ["jobs.id"],
        name="fk_job_stages_job_id_jobs",
        ondelete="CASCADE",
    ),
)

uploads = Table(
    "uploads",
    metadata,
    Column("id", UUID(as_uuid=True), nullable=False),
    Column("filename", Text, nullable=False),
    Column("size_bytes", BigInteger, nullable=False),
    Column("sha256", Text, nullable=False),
    Column("path", Text, nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
    Column("consumed_by", UUID(as_uuid=True), nullable=True),
    PrimaryKeyConstraint("id", name="pk_uploads"),
)

reports = Table(
    "reports",
    metadata,
    Column("job_id", UUID(as_uuid=True), nullable=False),
    Column("schema_version", Text, nullable=False),
    Column("report", JSONB, nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
    PrimaryKeyConstraint("job_id", name="pk_reports"),
    ForeignKeyConstraint(
        ["job_id"],
        ["jobs.id"],
        name="fk_reports_job_id_jobs",
        ondelete="CASCADE",
    ),
)

segments = Table(
    "segments",
    metadata,
    Column("job_id", UUID(as_uuid=True), nullable=False),
    Column("seg_id", Integer, nullable=False),
    Column("start_s", Float, nullable=False),
    Column("end_s", Float, nullable=False),
    Column("speaker", Text, nullable=True),
    Column("lang", Text, nullable=True),
    Column("text", Text, nullable=False),
    Column(
        "tsv",
        TSVECTOR,
        Computed("to_tsvector('simple', text)", persisted=True),
    ),
    PrimaryKeyConstraint("job_id", "seg_id", name="pk_segments"),
    ForeignKeyConstraint(
        ["job_id"],
        ["jobs.id"],
        name="fk_segments_job_id_jobs",
        ondelete="CASCADE",
    ),
)

Index("ix_segments_tsv", segments.c.tsv, postgresql_using="gin")

__all__ = [
    "JOB_SOURCE_KINDS",
    "JOB_STATUSES",
    "JOB_TERMINAL_STATUSES",
    "STAGE_STATUSES",
    "job_stages",
    "jobs",
    "metadata",
    "reports",
    "segments",
    "uploads",
]
