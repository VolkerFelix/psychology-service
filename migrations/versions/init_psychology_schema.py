"""Initial psychology profile schema.

Revision ID: 001
Revises:
Create Date: 2025-04-15 10:00:00.000000

"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSON

# revision identifiers, used by Alembic.
revision = "001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Create initial schema for psychology profiling service."""
    # Create psychological_profiles table
    op.create_table(
        "psychological_profiles",
        sa.Column("profile_id", sa.String(), nullable=False),
        sa.Column("user_id", sa.String(), nullable=False),
        sa.Column("personality_traits", JSON, nullable=True),
        sa.Column("sleep_preferences", JSON, nullable=True),
        sa.Column("behavioral_patterns", JSON, nullable=True),
        sa.Column("cluster_info", JSON, nullable=True),
        sa.Column("profile_metadata", JSON, nullable=False),
        sa.Column("raw_scores", JSON, nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("profile_id"),
    )

    # Create indexes
    op.create_index(
        op.f("ix_psychological_profiles_user_id"),
        "psychological_profiles",
        ["user_id"],
        unique=True,
    )

    # Create questionnaires table
    op.create_table(
        "questionnaires",
        sa.Column("questionnaire_id", sa.String(), nullable=False),
        sa.Column("title", sa.String(), nullable=False),
        sa.Column("description", sa.Text(), nullable=False),
        sa.Column("type", sa.String(), nullable=False),
        sa.Column("questions", JSON, nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.Column("version", sa.String(), nullable=False),
        sa.Column("is_active", sa.Boolean(), nullable=False),
        sa.Column("estimated_duration_minutes", sa.Integer(), nullable=True),
        sa.Column("tags", JSON, nullable=True),
        sa.PrimaryKeyConstraint("questionnaire_id"),
    )

    # Create indexes
    op.create_index(
        op.f("ix_questionnaires_type"), "questionnaires", ["type"], unique=False
    )
    op.create_index(
        op.f("ix_questionnaires_is_active"),
        "questionnaires",
        ["is_active"],
        unique=False,
    )

    # Create user_questionnaires table
    op.create_table(
        "user_questionnaires",
        sa.Column("user_questionnaire_id", sa.String(), nullable=False),
        sa.Column("user_id", sa.String(), nullable=False),
        sa.Column("questionnaire_id", sa.String(), nullable=False),
        sa.Column("profile_id", sa.String(), nullable=True),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("answers", JSON, nullable=True),
        sa.Column("started_at", sa.DateTime(), nullable=True),
        sa.Column("completed_at", sa.DateTime(), nullable=True),
        sa.Column("current_page", sa.Integer(), nullable=False),
        sa.Column("total_pages", sa.Integer(), nullable=False),
        sa.Column("scored", sa.Boolean(), nullable=False),
        sa.Column("score_results", JSON, nullable=True),
        sa.ForeignKeyConstraint(
            ["questionnaire_id"],
            ["questionnaires.questionnaire_id"],
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["profile_id"], ["psychological_profiles.profile_id"], ondelete="SET NULL"
        ),
        sa.PrimaryKeyConstraint("user_questionnaire_id"),
    )

    # Create indexes
    op.create_index(
        op.f("ix_user_questionnaires_user_id"),
        "user_questionnaires",
        ["user_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_user_questionnaires_questionnaire_id"),
        "user_questionnaires",
        ["questionnaire_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_user_questionnaires_status"),
        "user_questionnaires",
        ["status"],
        unique=False,
    )

    # Create clustering_models table
    op.create_table(
        "clustering_models",
        sa.Column("model_id", sa.String(), nullable=False),
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("description", sa.Text(), nullable=False),
        sa.Column("algorithm", sa.String(), nullable=False),
        sa.Column("parameters", JSON, nullable=False),
        sa.Column("num_clusters", sa.Integer(), nullable=False),
        sa.Column("features_used", JSON, nullable=False),
        sa.Column("clusters", JSON, nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.Column("version", sa.String(), nullable=False),
        sa.Column("is_active", sa.Boolean(), nullable=False),
        sa.Column("total_users_clustered", sa.Integer(), nullable=False),
        sa.Column("quality_metrics", JSON, nullable=True),
        sa.PrimaryKeyConstraint("model_id"),
    )

    # Create indexes
    op.create_index(
        op.f("ix_clustering_models_is_active"),
        "clustering_models",
        ["is_active"],
        unique=False,
    )

    # Create user_cluster_assignments table
    op.create_table(
        "user_cluster_assignments",
        sa.Column("assignment_id", sa.String(), nullable=False),
        sa.Column("user_id", sa.String(), nullable=False),
        sa.Column("cluster_id", sa.String(), nullable=False),
        sa.Column("model_id", sa.String(), nullable=False),
        sa.Column("confidence_score", sa.Float(), nullable=False),
        sa.Column("features", JSON, nullable=False),
        sa.Column("distance_to_centroid", sa.Float(), nullable=False),
        sa.Column(
            "assigned_at",
            sa.DateTime(),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.Column("is_current", sa.Boolean(), nullable=False),
        sa.ForeignKeyConstraint(
            ["model_id"], ["clustering_models.model_id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("assignment_id"),
    )

    # Create indexes
    op.create_index(
        op.f("ix_user_cluster_assignments_user_id"),
        "user_cluster_assignments",
        ["user_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_user_cluster_assignments_cluster_id"),
        "user_cluster_assignments",
        ["cluster_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_user_cluster_assignments_is_current"),
        "user_cluster_assignments",
        ["is_current"],
        unique=False,
    )
    # Create a composite index for efficient user+current lookups
    op.create_index(
        op.f("ix_user_cluster_assignments_user_current"),
        "user_cluster_assignments",
        ["user_id", "is_current"],
        unique=False,
    )

    # Create clustering_jobs table
    op.create_table(
        "clustering_jobs",
        sa.Column("job_id", sa.String(), nullable=False),
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("algorithm", sa.String(), nullable=False),
        sa.Column("parameters", JSON, nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.Column("started_at", sa.DateTime(), nullable=True),
        sa.Column("completed_at", sa.DateTime(), nullable=True),
        sa.Column("user_count", sa.Integer(), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("result_model_id", sa.String(), nullable=True),
        sa.ForeignKeyConstraint(
            ["result_model_id"], ["clustering_models.model_id"], ondelete="SET NULL"
        ),
        sa.PrimaryKeyConstraint("job_id"),
    )

    # Create indexes
    op.create_index(
        op.f("ix_clustering_jobs_status"),
        "clustering_jobs",
        ["status"],
        unique=False,
    )


def downgrade() -> None:
    """Drop all created tables."""
    op.drop_table("clustering_jobs")
    op.drop_table("user_cluster_assignments")
    op.drop_table("clustering_models")
    op.drop_table("user_questionnaires")
    op.drop_table("questionnaires")
    op.drop_table("psychological_profiles")
