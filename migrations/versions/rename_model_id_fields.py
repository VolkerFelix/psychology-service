"""Rename model_id fields to clustering_model_id.

Revision ID: 002
Revises: 001
Create Date: 2025-04-16 10:00:00.000000

"""

from alembic import op
from sqlalchemy.engine.reflection import Inspector

# revision identifiers, used by Alembic.
revision = "002"
down_revision = "001"
branch_labels = None
depends_on = None


def is_sqlite():
    """Check if we're using SQLite."""
    conn = op.get_bind()
    return conn.dialect.name == "sqlite"


def upgrade() -> None:
    """Rename model_id fields to clustering_model_id."""
    # Different approach needed for SQLite (using batch mode)
    if is_sqlite():
        upgrade_sqlite()
    else:
        upgrade_postgresql()


def upgrade_postgresql() -> None:
    """Upgrade using PostgreSQL-specific operations."""
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


def upgrade_sqlite() -> None:
    """Upgrade using SQLite batch mode."""
    # For SQLite, we need to use batch mode to recreate tables with new column names

    conn = op.get_bind()
    inspector = Inspector.from_engine(conn)

    # 1. Handle clustering_models table
    with op.batch_alter_table("clustering_models") as batch_op:
        batch_op.alter_column("model_id", new_column_name="clustering_model_id")

    # 2. Handle user_cluster_assignments table with foreign keys
    existing_columns = [
        c["name"] for c in inspector.get_columns("user_cluster_assignments")
    ]
    with op.batch_alter_table("user_cluster_assignments") as batch_op:
        # Rename model_id column
        if "model_id" in existing_columns:
            batch_op.alter_column("model_id", new_column_name="clustering_model_id")

        # The batch operation will automatically handle recreating foreign keys

    # 3. Handle clustering_jobs table with foreign keys
    existing_columns = [c["name"] for c in inspector.get_columns("clustering_jobs")]
    with op.batch_alter_table("clustering_jobs") as batch_op:
        # Rename result_model_id column
        if "result_model_id" in existing_columns:
            batch_op.alter_column(
                "result_model_id", new_column_name="result_clustering_model_id"
            )

        # The batch operation will automatically handle recreating foreign keys


def downgrade() -> None:
    """Revert the renaming of model_id fields."""
    # Different approach needed for SQLite (using batch mode)
    if is_sqlite():
        downgrade_sqlite()
    else:
        downgrade_postgresql()


def downgrade_postgresql() -> None:
    """Downgrade using PostgreSQL-specific operations."""
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


def downgrade_sqlite() -> None:
    """Downgrade using SQLite batch mode."""
    # For SQLite, we need to use batch mode to
    # recreate tables with original column names

    conn = op.get_bind()
    inspector = Inspector.from_engine(conn)

    # 1. Handle clustering_models table
    with op.batch_alter_table("clustering_models") as batch_op:
        batch_op.alter_column("clustering_model_id", new_column_name="model_id")

    # 2. Handle user_cluster_assignments table
    existing_columns = [
        c["name"] for c in inspector.get_columns("user_cluster_assignments")
    ]
    with op.batch_alter_table("user_cluster_assignments") as batch_op:
        if "clustering_model_id" in existing_columns:
            batch_op.alter_column("clustering_model_id", new_column_name="model_id")

    # 3. Handle clustering_jobs table
    existing_columns = [c["name"] for c in inspector.get_columns("clustering_jobs")]
    with op.batch_alter_table("clustering_jobs") as batch_op:
        if "result_clustering_model_id" in existing_columns:
            batch_op.alter_column(
                "result_clustering_model_id", new_column_name="result_model_id"
            )
