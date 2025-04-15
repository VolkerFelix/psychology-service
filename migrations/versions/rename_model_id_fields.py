"""Rename model_id fields to clustering_model_id.

Revision ID: 002
Revises: 001
Create Date: 2025-04-16 10:00:00.000000

"""

from alembic import op

# revision identifiers, used by Alembic.
revision = "002"
down_revision = "001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Rename model_id fields to clustering_model_id."""
    # Rename column in clustering_models table
    op.alter_column(
        "clustering_models", "model_id", new_column_name="clustering_model_id"
    )

    # Rename column in user_cluster_assignments table
    op.alter_column(
        "user_cluster_assignments", "model_id", new_column_name="clustering_model_id"
    )

    # Rename column in clustering_jobs table
    op.alter_column(
        "clustering_jobs",
        "result_model_id",
        new_column_name="result_clustering_model_id",
    )

    # Update foreign key constraints
    op.drop_constraint(
        "user_cluster_assignments_model_id_fkey",
        "user_cluster_assignments",
        type_="foreignkey",
    )
    op.create_foreign_key(
        "user_cluster_assignments_clustering_model_id_fkey",
        "user_cluster_assignments",
        "clustering_models",
        ["clustering_model_id"],
        ["clustering_model_id"],
        ondelete="CASCADE",
    )

    op.drop_constraint(
        "clustering_jobs_result_model_id_fkey", "clustering_jobs", type_="foreignkey"
    )
    op.create_foreign_key(
        "clustering_jobs_result_clustering_model_id_fkey",
        "clustering_jobs",
        "clustering_models",
        ["result_clustering_model_id"],
        ["clustering_model_id"],
        ondelete="SET NULL",
    )


def downgrade() -> None:
    """Revert the renaming of model_id fields."""
    # Revert foreign key constraints
    op.drop_constraint(
        "user_cluster_assignments_clustering_model_id_fkey",
        "user_cluster_assignments",
        type_="foreignkey",
    )
    op.create_foreign_key(
        "user_cluster_assignments_model_id_fkey",
        "user_cluster_assignments",
        "clustering_models",
        ["clustering_model_id"],
        ["clustering_model_id"],
        ondelete="CASCADE",
    )

    op.drop_constraint(
        "clustering_jobs_result_clustering_model_id_fkey",
        "clustering_jobs",
        type_="foreignkey",
    )
    op.create_foreign_key(
        "clustering_jobs_result_model_id_fkey",
        "clustering_jobs",
        "clustering_models",
        ["result_clustering_model_id"],
        ["clustering_model_id"],
        ondelete="SET NULL",
    )

    # Revert column names
    op.alter_column(
        "clustering_models", "clustering_model_id", new_column_name="model_id"
    )
    op.alter_column(
        "user_cluster_assignments", "clustering_model_id", new_column_name="model_id"
    )
    op.alter_column(
        "clustering_jobs",
        "result_clustering_model_id",
        new_column_name="result_model_id",
    )
