"""voiceprints: known_speakers and voiceprints (docs/05 §1, docs/04 §5)

Revision ID: 0002_voiceprints
Revises: 0001_initial
Create Date: 2026-11-14

The vector extension is created by 0001 in the same schema (``search_path``), so the
``vector(256)`` column and its HNSW cosine index resolve here. Embeddings are biometric
data: a job delete only nulls ``voiceprints.job_id`` (``ON DELETE SET NULL``) while deleting
a known speaker cascades to every embedding it owns.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects import postgresql

revision: str = "0002_voiceprints"
down_revision: str | None = "0001_initial"
branch_labels: str | None = None
depends_on: str | None = None

EMB_DIM = 256


def upgrade() -> None:
    op.create_table(
        "known_speakers",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("name_norm", sa.Text(), nullable=False),
        sa.Column("notes", sa.Text(), nullable=True),
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
        sa.PrimaryKeyConstraint("id", name="pk_known_speakers"),
    )
    op.create_index(
        "uq_known_speakers_name_norm",
        "known_speakers",
        ["name_norm"],
        unique=True,
    )

    op.create_table(
        "voiceprints",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("known_speaker_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("job_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("speaker_label", sa.Text(), nullable=False),
        sa.Column("embedding", Vector(EMB_DIM), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.PrimaryKeyConstraint("id", name="pk_voiceprints"),
        sa.ForeignKeyConstraint(
            ["known_speaker_id"],
            ["known_speakers.id"],
            name="fk_voiceprints_known_speaker_id_known_speakers",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["job_id"],
            ["jobs.id"],
            name="fk_voiceprints_job_id_jobs",
            ondelete="SET NULL",
        ),
    )
    op.create_index(
        "ix_voiceprints_embedding_hnsw",
        "voiceprints",
        ["embedding"],
        postgresql_using="hnsw",
        postgresql_ops={"embedding": "vector_cosine_ops"},
    )


def downgrade() -> None:
    op.drop_table("voiceprints")
    op.drop_table("known_speakers")
