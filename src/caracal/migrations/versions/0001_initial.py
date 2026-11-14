"""initial schema: jobs, job_stages, uploads, reports, segments

Revision ID: 0001_initial
Revises:
Create Date: 2026-11-14

Handwritten from docs/05 §1. Tables are created unqualified: the connection sets
``search_path`` to ``CARACAL_DB_SCHEMA`` (see ``env.py``), so the same migration works in the
default schema and in the throwaway schemas used by tests.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0001_initial"
down_revision: str | None = None
branch_labels: str | None = None
depends_on: str | None = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table(
        "jobs",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("status", sa.Text(), nullable=False, server_default="queued"),
        sa.Column("source_kind", sa.Text(), nullable=False),
        sa.Column("source_url", sa.Text(), nullable=True),
        sa.Column("upload_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("title", sa.Text(), nullable=True),
        sa.Column("options", postgresql.JSONB(), nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("progress", postgresql.JSONB(), nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("error", postgresql.JSONB(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("sha256", sa.Text(), nullable=True),
        sa.PrimaryKeyConstraint("id", name="pk_jobs"),
        sa.CheckConstraint(
            "status IN ('queued', 'running', 'done', 'failed', 'cancelled')",
            name="ck_jobs_status",
        ),
        sa.CheckConstraint(
            "source_kind IN ('upload', 'direct', 'platform')",
            name="ck_jobs_source_kind",
        ),
    )
    op.create_index(
        "ix_jobs_status_created_at",
        "jobs",
        ["status", sa.text("created_at DESC")],
    )
    op.create_index("ix_jobs_sha256", "jobs", ["sha256"])

    op.create_table(
        "job_stages",
        sa.Column("job_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("status", sa.Text(), nullable=False),
        sa.Column("reason", sa.Text(), nullable=True),
        sa.Column("attempts", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("ended_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("job_id", "name", name="pk_job_stages"),
        sa.CheckConstraint(
            "status IN ('pending', 'running', 'done', 'skipped', 'failed')",
            name="ck_job_stages_status",
        ),
        sa.ForeignKeyConstraint(
            ["job_id"],
            ["jobs.id"],
            name="fk_job_stages_job_id_jobs",
            ondelete="CASCADE",
        ),
    )

    op.create_table(
        "uploads",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("filename", sa.Text(), nullable=False),
        sa.Column("size_bytes", sa.BigInteger(), nullable=False),
        sa.Column("sha256", sa.Text(), nullable=False),
        sa.Column("path", sa.Text(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column("consumed_by", postgresql.UUID(as_uuid=True), nullable=True),
        sa.PrimaryKeyConstraint("id", name="pk_uploads"),
    )

    op.create_table(
        "reports",
        sa.Column("job_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("schema_version", sa.Text(), nullable=False),
        sa.Column("report", postgresql.JSONB(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.PrimaryKeyConstraint("job_id", name="pk_reports"),
        sa.ForeignKeyConstraint(
            ["job_id"],
            ["jobs.id"],
            name="fk_reports_job_id_jobs",
            ondelete="CASCADE",
        ),
    )

    op.create_table(
        "segments",
        sa.Column("job_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("seg_id", sa.Integer(), nullable=False),
        sa.Column("start_s", sa.Float(), nullable=False),
        sa.Column("end_s", sa.Float(), nullable=False),
        sa.Column("speaker", sa.Text(), nullable=True),
        sa.Column("lang", sa.Text(), nullable=True),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column(
            "tsv",
            postgresql.TSVECTOR(),
            sa.Computed("to_tsvector('simple', text)", persisted=True),
        ),
        sa.PrimaryKeyConstraint("job_id", "seg_id", name="pk_segments"),
        sa.ForeignKeyConstraint(
            ["job_id"],
            ["jobs.id"],
            name="fk_segments_job_id_jobs",
            ondelete="CASCADE",
        ),
    )
    op.create_index("ix_segments_tsv", "segments", ["tsv"], postgresql_using="gin")


def downgrade() -> None:
    op.drop_table("segments")
    op.drop_table("reports")
    op.drop_table("job_stages")
    op.drop_table("uploads")
    op.drop_table("jobs")
