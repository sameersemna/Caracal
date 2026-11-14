"""SQLAlchemy Core table definitions (docs/05 §1).

Tables are deliberately schema-less: every connection sets ``search_path`` to
``CARACAL_DB_SCHEMA`` (see :mod:`caracal.db.engine`), so unqualified statements resolve to the
configured schema and the same objects work in the throwaway schemas used by live tests.
"""

from __future__ import annotations

from pgvector.sqlalchemy import Vector
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

EMB_DIM = 256
"""Speaker embedding dimension (docs/05 §1; measured community-1 ``speaker_embeddings``)."""

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

known_speakers = Table(
    "known_speakers",
    metadata,
    Column("id", UUID(as_uuid=True), nullable=False),
    Column("name", Text, nullable=False),
    Column("name_norm", Text, nullable=False),
    Column("notes", Text, nullable=True),
    Column("created_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
    Column("updated_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
    PrimaryKeyConstraint("id", name="pk_known_speakers"),
)

Index("uq_known_speakers_name_norm", known_speakers.c.name_norm, unique=True)

voiceprints = Table(
    "voiceprints",
    metadata,
    Column("id", UUID(as_uuid=True), nullable=False),
    Column("known_speaker_id", UUID(as_uuid=True), nullable=False),
    Column("job_id", UUID(as_uuid=True), nullable=True),
    Column("speaker_label", Text, nullable=False),
    Column("embedding", Vector(EMB_DIM), nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
    PrimaryKeyConstraint("id", name="pk_voiceprints"),
    ForeignKeyConstraint(
        ["known_speaker_id"],
        ["known_speakers.id"],
        name="fk_voiceprints_known_speaker_id_known_speakers",
        ondelete="CASCADE",
    ),
    # Biometric embeddings outlive a job: deleting the job nulls the provenance link only.
    ForeignKeyConstraint(
        ["job_id"],
        ["jobs.id"],
        name="fk_voiceprints_job_id_jobs",
        ondelete="SET NULL",
    ),
)

Index(
    "ix_voiceprints_embedding_hnsw",
    voiceprints.c.embedding,
    postgresql_using="hnsw",
    postgresql_ops={"embedding": "vector_cosine_ops"},
)

__all__ = [
    "EMB_DIM",
    "JOB_SOURCE_KINDS",
    "JOB_STATUSES",
    "JOB_TERMINAL_STATUSES",
    "STAGE_STATUSES",
    "job_stages",
    "jobs",
    "known_speakers",
    "metadata",
    "reports",
    "segments",
    "uploads",
    "voiceprints",
]
