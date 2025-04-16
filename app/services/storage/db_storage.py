"""Database storage service for psychology profile data."""

import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from loguru import logger
from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    create_engine,
    desc,
    text,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker
from sqlalchemy.pool import StaticPool

from app.config.settings import settings

Base = declarative_base()


class PsychologicalProfileDB(Base):  # type: ignore
    """SQLAlchemy model for psychological profiles."""

    __tablename__ = "psychological_profiles"

    profile_id = Column(String, primary_key=True)
    user_id = Column(String, index=True, nullable=False)
    personality_traits = Column(JSON, nullable=True)
    sleep_preferences = Column(JSON, nullable=True)
    behavioral_patterns = Column(JSON, nullable=True)
    cluster_info = Column(JSON, nullable=True)
    profile_metadata = Column(JSON, nullable=False)
    raw_scores = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    questionnaires = relationship(
        "UserQuestionnaireDB", back_populates="profile", cascade="all, delete-orphan"
    )

    def to_dict(self) -> Dict:
        """Convert the profile to a dictionary."""
        return {
            "profile_id": self.profile_id,
            "user_id": self.user_id,
            "personality_traits": self.personality_traits,
            "sleep_preferences": self.sleep_preferences,
            "behavioral_patterns": self.behavioral_patterns,
            "cluster_info": self.cluster_info,
            "profile_metadata": self.profile_metadata,
            "raw_scores": self.raw_scores,
        }


class QuestionnaireDB(Base):  # type: ignore
    """SQLAlchemy model for questionnaires."""

    __tablename__ = "questionnaires"

    questionnaire_id = Column(String, primary_key=True)
    title = Column(String, nullable=False)
    description = Column(Text, nullable=False)
    questionnaire_type = Column(String, nullable=False)
    questions = Column(JSON, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    version = Column(String, default="1.0.0")
    is_active = Column(Boolean, default=True)
    estimated_duration_minutes = Column(Integer, nullable=True)
    tags = Column(JSON, nullable=True)

    # Relationships
    user_questionnaires = relationship(
        "UserQuestionnaireDB",
        back_populates="questionnaire",
        cascade="all, delete-orphan",
    )

    def to_dict(self) -> Dict:
        """Convert the questionnaire to a dictionary."""
        return {
            "questionnaire_id": self.questionnaire_id,
            "title": self.title,
            "description": self.description,
            "questionnaire_type": self.questionnaire_type,
            "questions": self.questions,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "version": self.version,
            "is_active": self.is_active,
            "estimated_duration_minutes": self.estimated_duration_minutes,
            "tags": self.tags,
        }


class UserQuestionnaireDB(Base):  # type: ignore
    """SQLAlchemy model for user questionnaire progress."""

    __tablename__ = "user_questionnaires"

    user_questionnaire_id = Column(String, primary_key=True)
    user_id = Column(String, index=True, nullable=False)
    questionnaire_id = Column(
        String, ForeignKey("questionnaires.questionnaire_id"), nullable=False
    )
    profile_id = Column(
        String, ForeignKey("psychological_profiles.profile_id"), nullable=True
    )
    status = Column(String, nullable=False)
    answers = Column(JSON, nullable=True)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    current_page = Column(Integer, default=1)
    total_pages = Column(Integer, default=1)
    scored = Column(Boolean, default=False)
    score_results = Column(JSON, nullable=True)

    # Relationships
    questionnaire = relationship(
        "QuestionnaireDB", back_populates="user_questionnaires"
    )
    profile = relationship("PsychologicalProfileDB", back_populates="questionnaires")

    def to_dict(self) -> Dict:
        """Convert the user questionnaire to a dictionary."""
        return {
            "user_questionnaire_id": self.user_questionnaire_id,
            "user_id": self.user_id,
            "questionnaire_id": self.questionnaire_id,
            "profile_id": self.profile_id,
            "status": self.status,
            "answers": self.answers,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat()
            if self.completed_at
            else None,
            "current_page": self.current_page,
            "total_pages": self.total_pages,
            "scored": self.scored,
            "score_results": self.score_results,
        }


class ClusteringModelDB(Base):  # type: ignore
    """Database model for clustering models."""

    __tablename__ = "clustering_models"

    clustering_model_id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    description = Column(Text, nullable=False)
    algorithm = Column(String, nullable=False)
    parameters = Column(JSON, nullable=False)
    num_clusters = Column(Integer, nullable=False)
    features_used = Column(JSON, nullable=False)
    clusters = Column(JSON, nullable=False)
    created_at = Column(
        DateTime, server_default=text("CURRENT_TIMESTAMP"), nullable=False
    )
    updated_at = Column(
        DateTime, server_default=text("CURRENT_TIMESTAMP"), nullable=False
    )
    version = Column(String, nullable=False)
    is_active = Column(Boolean, nullable=False)
    total_users_clustered = Column(Integer, nullable=False)
    quality_metrics = Column(JSON, nullable=True)

    # Relationships
    user_assignments = relationship(
        "UserClusterAssignmentDB", back_populates="model", cascade="all, delete-orphan"
    )
    # Fix this relationship to match the new column name
    jobs = relationship(
        "ClusteringJobDB",
        back_populates="result_model",
        foreign_keys="ClusteringJobDB.result_clustering_model_id",
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "clustering_model_id": self.clustering_model_id,
            "name": self.name,
            "description": self.description,
            "algorithm": self.algorithm,
            "parameters": self.parameters,
            "num_clusters": self.num_clusters,
            "features_used": self.features_used,
            "clusters": self.clusters,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "version": self.version,
            "is_active": self.is_active,
            "total_users_clustered": self.total_users_clustered,
            "quality_metrics": self.quality_metrics,
        }


class UserClusterAssignmentDB(Base):  # type: ignore
    """Database model for user cluster assignments."""

    __tablename__ = "user_cluster_assignments"

    assignment_id = Column(String, primary_key=True)
    user_id = Column(String, nullable=False)
    cluster_id = Column(String, nullable=False)
    clustering_model_id = Column(
        String, ForeignKey("clustering_models.clustering_model_id"), nullable=False
    )
    confidence_score = Column(Float, nullable=False)
    features = Column(JSON, nullable=False)
    distance_to_centroid = Column(Float, nullable=False)
    assigned_at = Column(
        DateTime, server_default=text("CURRENT_TIMESTAMP"), nullable=False
    )
    is_current = Column(Boolean, nullable=False)

    # Relationships
    model = relationship("ClusteringModelDB", back_populates="user_assignments")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "assignment_id": self.assignment_id,
            "user_id": self.user_id,
            "cluster_id": self.cluster_id,
            "clustering_model_id": self.clustering_model_id,
            "confidence_score": self.confidence_score,
            "features": self.features,
            "distance_to_centroid": self.distance_to_centroid,
            "assigned_at": self.assigned_at,
            "is_current": self.is_current,
        }


class ClusteringJobDB(Base):  # type: ignore
    """Database model for clustering jobs."""

    __tablename__ = "clustering_jobs"

    job_id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    description = Column(Text, nullable=True)
    status = Column(String, nullable=False)
    algorithm = Column(String, nullable=False)
    parameters = Column(JSON, nullable=False)
    created_at = Column(
        DateTime, server_default=text("CURRENT_TIMESTAMP"), nullable=False
    )
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    user_count = Column(Integer, nullable=True)
    error_message = Column(Text, nullable=True)
    # Fix this column name to match the migration
    result_clustering_model_id = Column(
        String, ForeignKey("clustering_models.clustering_model_id"), nullable=True
    )

    # Fix this relationship to match the new column names
    result_model = relationship(
        "ClusteringModelDB",
        back_populates="jobs",
        foreign_keys=[result_clustering_model_id],
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "job_id": self.job_id,
            "name": self.name,
            "description": self.description,
            "status": self.status,
            "algorithm": self.algorithm,
            "parameters": self.parameters,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "user_count": self.user_count,
            "error_message": self.error_message,
            "result_clustering_model_id": self.result_clustering_model_id,
        }


class DatabaseStorage:
    """PostgreSQL and SQLite database storage service for psychology profile data."""

    def __init__(self, db_url: Optional[str] = None):
        """
        Initialize the database storage service.

        Args:
            db_url: Optional database connection URL.
                    If not provided, uses settings.DATABASE_URL.
        """
        # Use provided URL or fall back to settings
        self.db_url = db_url or settings.DATABASE_URL

        # For testing with SQLite, use a static connection pool
        if self.db_url.startswith("sqlite"):
            self.engine = create_engine(
                self.db_url,
                connect_args={"check_same_thread": False},
                poolclass=StaticPool,
            )
        # For PostgreSQL, regular connection
        elif self.db_url.startswith("postgresql"):
            self.engine = create_engine(self.db_url)
        else:
            raise ValueError(
                "Database URL must start with 'sqlite://' or 'postgresql://'"
            )

        self.Session = sessionmaker(bind=self.engine)

        # Create tables if they don't exist
        Base.metadata.create_all(self.engine)

    # Profile methods
    def save_profile(self, profile_data: Dict[str, Any]) -> bool:
        """
        Save a psychological profile to the database.

        Args:
            profile_data: Profile data dictionary

        Returns:
            Success status
        """
        session = self.Session()
        try:
            # Check if profile exists
            profile_id = profile_data.get("profile_id")
            if not profile_id:
                profile_id = str(uuid.uuid4())
                profile_data["profile_id"] = profile_id

            existing = (
                session.query(PsychologicalProfileDB)
                .filter(PsychologicalProfileDB.profile_id == profile_id)
                .first()
            )

            if existing:
                # Update existing profile
                for key, value in profile_data.items():
                    if key != "profile_id":
                        setattr(existing, key, value)
                existing.updated_at = datetime.utcnow()
            else:
                # Create new profile
                profile = PsychologicalProfileDB(**profile_data)
                session.add(profile)

            session.commit()
            return True
        except Exception as e:
            session.rollback()
            logger.error(f"Error saving profile to database: {str(e)}")
            return False
        finally:
            session.close()

    def get_profile(self, profile_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a psychological profile by ID.

        Args:
            profile_id: Profile ID

        Returns:
            Profile dictionary or None if not found
        """
        session = self.Session()
        try:
            profile = (
                session.query(PsychologicalProfileDB)
                .filter(PsychologicalProfileDB.profile_id == profile_id)
                .first()
            )

            if profile:
                return profile.to_dict()
            return None
        except Exception as e:
            logger.error(f"Error getting profile from database: {str(e)}")
            return None
        finally:
            session.close()

    def get_profile_by_user(self, user_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a psychological profile by user ID.

        Args:
            user_id: User ID

        Returns:
            Profile dictionary or None if not found
        """
        session = self.Session()
        try:
            profile = (
                session.query(PsychologicalProfileDB)
                .filter(PsychologicalProfileDB.user_id == user_id)
                .first()
            )

            if profile:
                return profile.to_dict()
            return None
        except Exception as e:
            logger.error(f"Error getting profile by user from database: {str(e)}")
            return None
        finally:
            session.close()

    def delete_profile(self, profile_id: str) -> bool:
        """
        Delete a psychological profile.

        Args:
            profile_id: Profile ID

        Returns:
            Success status
        """
        session = self.Session()
        try:
            profile = (
                session.query(PsychologicalProfileDB)
                .filter(PsychologicalProfileDB.profile_id == profile_id)
                .first()
            )

            if profile:
                session.delete(profile)
                session.commit()
                return True
            return False
        except Exception as e:
            session.rollback()
            logger.error(f"Error deleting profile from database: {str(e)}")
            return False
        finally:
            session.close()

    def get_profiles(
        self,
        limit: int = 100,
        offset: int = 0,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Get multiple psychological profiles.

        Args:
            limit: Maximum number of profiles to return
            offset: Number of profiles to skip
            filters: Optional filters to apply

        Returns:
            List of profile dictionaries
        """
        session = self.Session()
        try:
            query = session.query(PsychologicalProfileDB)

            # Apply filters if provided
            if filters:
                if "user_id" in filters:
                    query = query.filter(
                        PsychologicalProfileDB.user_id == filters["user_id"]
                    )
                if "cluster_id" in filters:
                    query = query.filter(
                        PsychologicalProfileDB.cluster_info.contains(
                            {"cluster_id": filters["cluster_id"]}
                        )
                    )

            # Apply pagination
            profiles = query.limit(limit).offset(offset).all()

            return [profile.to_dict() for profile in profiles]
        except Exception as e:
            logger.error(f"Error getting profiles from database: {str(e)}")
            return []
        finally:
            session.close()

    # Questionnaire methods
    def save_questionnaire(self, questionnaire_data: Dict[str, Any]) -> bool:
        """
        Save a questionnaire to the database.

        Args:
            questionnaire_data: Questionnaire data dictionary

        Returns:
            Success status
        """
        session = self.Session()
        try:
            # Check if questionnaire exists
            questionnaire_id = questionnaire_data.get("questionnaire_id")
            if not questionnaire_id:
                questionnaire_id = str(uuid.uuid4())
                questionnaire_data["questionnaire_id"] = questionnaire_id

            existing = (
                session.query(QuestionnaireDB)
                .filter(QuestionnaireDB.questionnaire_id == questionnaire_id)
                .first()
            )

            if existing:
                # Update existing questionnaire
                for key, value in questionnaire_data.items():
                    if key != "questionnaire_id":
                        setattr(existing, key, value)
                existing.updated_at = datetime.utcnow()
            else:
                # Create new questionnaire
                questionnaire = QuestionnaireDB(**questionnaire_data)
                session.add(questionnaire)

            session.commit()
            return True
        except Exception as e:
            session.rollback()
            logger.error(f"Error saving questionnaire to database: {str(e)}")
            return False
        finally:
            session.close()

    def get_questionnaire(self, questionnaire_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a questionnaire by ID.

        Args:
            questionnaire_id: Questionnaire ID

        Returns:
            Questionnaire dictionary or None if not found
        """
        session = self.Session()
        try:
            questionnaire = (
                session.query(QuestionnaireDB)
                .filter(QuestionnaireDB.questionnaire_id == questionnaire_id)
                .first()
            )

            if questionnaire:
                return questionnaire.to_dict()
            return None
        except Exception as e:
            logger.error(f"Error getting questionnaire from database: {str(e)}")
            return None
        finally:
            session.close()

    def get_questionnaires(
        self,
        limit: int = 100,
        offset: int = 0,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Get multiple questionnaires.

        Args:
            limit: Maximum number of questionnaires to return
            offset: Number of questionnaires to skip
            filters: Optional filters to apply

        Returns:
            List of questionnaire dictionaries
        """
        session = self.Session()
        try:
            query = session.query(QuestionnaireDB)

            # Apply filters if provided
            if filters:
                if "questionnaire_type" in filters:
                    query = query.filter(
                        QuestionnaireDB.questionnaire_type
                        == filters["questionnaire_type"]
                    )
                if "is_active" in filters:
                    query = query.filter(
                        QuestionnaireDB.is_active == filters["is_active"]
                    )
                if "tags" in filters and filters["tags"]:
                    for tag in filters["tags"]:
                        query = query.filter(QuestionnaireDB.tags.contains([tag]))

            # Apply pagination
            questionnaires = query.limit(limit).offset(offset).all()

            return [questionnaire.to_dict() for questionnaire in questionnaires]
        except Exception as e:
            logger.error(f"Error getting questionnaires from database: {str(e)}")
            return []
        finally:
            session.close()

    def delete_questionnaire(self, questionnaire_id: str) -> bool:
        """
        Delete a questionnaire.

        Args:
            questionnaire_id: Questionnaire ID

        Returns:
            Success status
        """
        session = self.Session()
        try:
            questionnaire = (
                session.query(QuestionnaireDB)
                .filter(QuestionnaireDB.questionnaire_id == questionnaire_id)
                .first()
            )

            if questionnaire:
                session.delete(questionnaire)
                session.commit()
                return True
            return False
        except Exception as e:
            session.rollback()
            logger.error(f"Error deleting questionnaire from database: {str(e)}")
            return False
        finally:
            session.close()

    # User questionnaire methods
    def save_user_questionnaire(self, user_questionnaire_data: Dict[str, Any]) -> bool:
        """
        Save a user questionnaire to the database.

        Args:
            user_questionnaire_data: User questionnaire data dictionary

        Returns:
            Success status
        """
        session = self.Session()
        try:
            # Check if user questionnaire exists
            user_questionnaire_id = user_questionnaire_data.get("user_questionnaire_id")
            if not user_questionnaire_id:
                user_questionnaire_id = str(uuid.uuid4())
                user_questionnaire_data["user_questionnaire_id"] = user_questionnaire_id

            # Serialize datetime objects in answers
            if "answers" in user_questionnaire_data:
                for answer in user_questionnaire_data["answers"]:
                    if "answered_at" in answer and isinstance(
                        answer["answered_at"], datetime
                    ):
                        answer["answered_at"] = answer["answered_at"].isoformat()

            # Convert string dates to datetime objects for database fields
            if "started_at" in user_questionnaire_data and isinstance(
                user_questionnaire_data["started_at"], str
            ):
                try:
                    user_questionnaire_data["started_at"] = datetime.fromisoformat(
                        user_questionnaire_data["started_at"]
                    )
                except (ValueError, TypeError):
                    # If conversion fails, use current time
                    user_questionnaire_data["started_at"] = datetime.now()

            if "completed_at" in user_questionnaire_data and isinstance(
                user_questionnaire_data["completed_at"], str
            ):
                try:
                    user_questionnaire_data["completed_at"] = datetime.fromisoformat(
                        user_questionnaire_data["completed_at"]
                    )
                except (ValueError, TypeError):
                    # If conversion fails, use current time
                    user_questionnaire_data["completed_at"] = datetime.now()

            existing = (
                session.query(UserQuestionnaireDB)
                .filter(
                    UserQuestionnaireDB.user_questionnaire_id == user_questionnaire_id
                )
                .first()
            )

            if existing:
                # Update existing user questionnaire
                for key, value in user_questionnaire_data.items():
                    if key != "user_questionnaire_id":
                        setattr(existing, key, value)
            else:
                # Create new user questionnaire
                user_questionnaire = UserQuestionnaireDB(**user_questionnaire_data)
                session.add(user_questionnaire)

            session.commit()
            return True
        except Exception as e:
            session.rollback()
            logger.error(f"Error saving user questionnaire to database: {str(e)}")
            return False
        finally:
            session.close()

    def get_user_questionnaire(
        self, user_questionnaire_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Get a user questionnaire by ID.

        Args:
            user_questionnaire_id: User questionnaire ID

        Returns:
            User questionnaire dictionary or None if not found
        """
        session = self.Session()
        try:
            user_questionnaire = (
                session.query(UserQuestionnaireDB)
                .filter(
                    UserQuestionnaireDB.user_questionnaire_id == user_questionnaire_id
                )
                .first()
            )

            if user_questionnaire:
                return user_questionnaire.to_dict()
            return None
        except Exception as e:
            logger.error(f"Error getting user questionnaire from database: {str(e)}")
            return None
        finally:
            session.close()

    def get_user_questionnaires(
        self,
        user_id: str,
        questionnaire_id: Optional[str] = None,
        status: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Get questionnaires for a specific user.

        Args:
            user_id: User ID
            questionnaire_id: Optional questionnaire ID to filter by
            status: Optional status to filter by

        Returns:
            List of user questionnaire dictionaries
        """
        session = self.Session()
        try:
            query = session.query(UserQuestionnaireDB).filter(
                UserQuestionnaireDB.user_id == user_id
            )

            if questionnaire_id:
                query = query.filter(
                    UserQuestionnaireDB.questionnaire_id == questionnaire_id
                )

            if status:
                query = query.filter(UserQuestionnaireDB.status == status)

            user_questionnaires = query.all()

            return [uq.to_dict() for uq in user_questionnaires]
        except Exception as e:
            logger.error(f"Error getting user questionnaires from database: {str(e)}")
            return []
        finally:
            session.close()

    # Clustering model methods
    def save_clustering_model(self, model_data: Dict[str, Any]) -> bool:
        """
        Save a clustering model to the database.

        Args:
            model_data: Clustering model data dictionary

        Returns:
            Success status
        """
        session = self.Session()
        try:
            # Check if model exists
            model_id = model_data.get("clustering_model_id")
            if not model_id:
                model_id = str(uuid.uuid4())
                model_data["clustering_model_id"] = model_id

            # Convert datetime objects in clusters
            if "clusters" in model_data:
                for cluster in model_data["clusters"]:
                    if isinstance(cluster.get("created_at"), datetime):
                        cluster["created_at"] = cluster["created_at"].isoformat()
                    if isinstance(cluster.get("updated_at"), datetime):
                        cluster["updated_at"] = cluster["updated_at"].isoformat()

            existing = (
                session.query(ClusteringModelDB)
                .filter(ClusteringModelDB.clustering_model_id == model_id)
                .first()
            )

            if existing:
                # Update existing model
                for key, value in model_data.items():
                    if key != "clustering_model_id":
                        setattr(existing, key, value)
                existing.updated_at = datetime.utcnow()
            else:
                # Create new model
                model = ClusteringModelDB(**model_data)
                session.add(model)

            session.commit()
            return True
        except Exception as e:
            session.rollback()
            logger.error(f"Error saving clustering model to database: {str(e)}")
            return False
        finally:
            session.close()

    def get_clustering_model(self, model_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a clustering model by ID.

        Args:
            model_id: Model ID

        Returns:
            Clustering model dictionary or None if not found
        """
        session = self.Session()
        try:
            model = (
                session.query(ClusteringModelDB)
                .filter(ClusteringModelDB.clustering_model_id == model_id)
                .first()
            )

            if model:
                return model.to_dict()
            return None
        except Exception as e:
            logger.error(f"Error getting clustering model from database: {str(e)}")
            return None
        finally:
            session.close()

    def get_active_clustering_model(self) -> Optional[Dict[str, Any]]:
        """
        Get the current active clustering model.

        Returns:
            Active clustering model dictionary or None if not found
        """
        session = self.Session()
        try:
            model = (
                session.query(ClusteringModelDB)
                .filter(ClusteringModelDB.is_active is True)
                .order_by(desc(ClusteringModelDB.created_at))
                .first()
            )

            if model:
                return model.to_dict()
            return None
        except Exception as e:
            logger.error(
                f"Error getting active clustering model from database: {str(e)}"
            )
            return None
        finally:
            session.close()

    def get_clustering_models(
        self, limit: int = 100, offset: int = 0, include_inactive: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Get multiple clustering models.

        Args:
            limit: Maximum number of models to return
            offset: Number of models to skip
            include_inactive: Whether to include inactive models

        Returns:
            List of clustering model dictionaries
        """
        session = self.Session()
        try:
            query = session.query(ClusteringModelDB)

            if not include_inactive:
                query = query.filter(ClusteringModelDB.is_active is True)

            # Sort by creation date, newest first
            query = query.order_by(desc(ClusteringModelDB.created_at))

            # Apply pagination
            models = query.limit(limit).offset(offset).all()

            return [model.to_dict() for model in models]
        except Exception as e:
            logger.error(f"Error getting clustering models from database: {str(e)}")
            return []
        finally:
            session.close()

    # User cluster assignment methods
    def save_user_cluster_assignment(self, assignment_data: Dict[str, Any]) -> bool:
        """
        Save a user cluster assignment to the database.

        Args:
            assignment_data: User cluster assignment data dictionary

        Returns:
            Success status
        """
        session = self.Session()
        try:
            # Check if assignment exists
            assignment_id = assignment_data.get("assignment_id")
            if not assignment_id:
                assignment_id = str(uuid.uuid4())
                assignment_data["assignment_id"] = assignment_id

            # Mark previous assignments for this user as not current
            if assignment_data.get("is_current", True):
                previous_assignments = (
                    session.query(UserClusterAssignmentDB)
                    .filter(
                        UserClusterAssignmentDB.user_id == assignment_data["user_id"],
                        UserClusterAssignmentDB.is_current is True,
                        UserClusterAssignmentDB.assignment_id != assignment_id,
                    )
                    .all()
                )

                for prev in previous_assignments:
                    prev.is_current = False

            existing = (
                session.query(UserClusterAssignmentDB)
                .filter(UserClusterAssignmentDB.assignment_id == assignment_id)
                .first()
            )

            if existing:
                # Update existing assignment
                for key, value in assignment_data.items():
                    if key != "assignment_id":
                        setattr(existing, key, value)
            else:
                # Create new assignment
                assignment = UserClusterAssignmentDB(**assignment_data)
                session.add(assignment)

            session.commit()
            return True
        except Exception as e:
            session.rollback()
            logger.error(f"Error saving user cluster assignment to database: {str(e)}")
            return False
        finally:
            session.close()

    def get_user_cluster_assignment(
        self, user_id: str, current_only: bool = True
    ) -> Optional[Dict[str, Any]]:
        """
        Get a user's cluster assignment.

        Args:
            user_id: User ID
            current_only: Whether to return only the current assignment

        Returns:
            User cluster assignment dictionary or None if not found
        """
        session = self.Session()
        try:
            query = session.query(UserClusterAssignmentDB).filter(
                UserClusterAssignmentDB.user_id == user_id
            )

            if current_only:
                query = query.filter(UserClusterAssignmentDB.is_current is True)

            # Get the most recent assignment
            assignment = query.order_by(
                desc(UserClusterAssignmentDB.assigned_at)
            ).first()

            if assignment:
                return assignment.to_dict()
            return None
        except Exception as e:
            logger.error(
                f"Error getting user cluster assignment from database: {str(e)}"
            )
            return None
        finally:
            session.close()

    def get_cluster_users(
        self,
        cluster_id: str,
        model_id: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """
        Get users assigned to a specific cluster.

        Args:
            cluster_id: Cluster ID
            model_id: Optional model ID to filter by
            limit: Maximum number of assignments to return
            offset: Number of assignments to skip

        Returns:
            List of user cluster assignment dictionaries
        """
        session = self.Session()
        try:
            query = session.query(UserClusterAssignmentDB).filter(
                UserClusterAssignmentDB.cluster_id == cluster_id,
                UserClusterAssignmentDB.is_current is True,
            )

            if model_id:
                query = query.filter(
                    UserClusterAssignmentDB.clustering_model_id == model_id
                )

            # Apply pagination
            assignments = query.limit(limit).offset(offset).all()

            return [assignment.to_dict() for assignment in assignments]
        except Exception as e:
            logger.error(f"Error getting cluster users from database: {str(e)}")
            return []
        finally:
            session.close()

    # Clustering job methods
    def save_clustering_job(self, job_data: Dict[str, Any]) -> bool:
        """
        Save a clustering job to the database.

        Args:
            job_data: Clustering job data dictionary

        Returns:
            Success status
        """
        session = self.Session()
        try:
            # Check if job exists
            job_id = job_data.get("job_id")
            if not job_id:
                job_id = str(uuid.uuid4())
                job_data["job_id"] = job_id

            existing = (
                session.query(ClusteringJobDB)
                .filter(ClusteringJobDB.job_id == job_id)
                .first()
            )

            if existing:
                # Update existing job
                for key, value in job_data.items():
                    if key != "job_id":
                        setattr(existing, key, value)
            else:
                # Create new job
                job = ClusteringJobDB(**job_data)
                session.add(job)

            session.commit()
            return True
        except Exception as e:
            session.rollback()
            logger.error(f"Error saving clustering job to database: {str(e)}")
            return False
        finally:
            session.close()

    def get_clustering_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a clustering job by ID.

        Args:
            job_id: Job ID

        Returns:
            Clustering job dictionary or None if not found
        """
        session = self.Session()
        try:
            job = (
                session.query(ClusteringJobDB)
                .filter(ClusteringJobDB.job_id == job_id)
                .first()
            )

            if job:
                return job.to_dict()
            return None
        except Exception as e:
            logger.error(f"Error getting clustering job from database: {str(e)}")
            return None
        finally:
            session.close()

    def get_clustering_jobs(
        self, limit: int = 100, offset: int = 0, status: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get multiple clustering jobs.

        Args:
            limit: Maximum number of jobs to return
            offset: Number of jobs to skip
            status: Optional status to filter by

        Returns:
            List of clustering job dictionaries
        """
        session = self.Session()
        try:
            query = session.query(ClusteringJobDB)

            if status:
                query = query.filter(ClusteringJobDB.status == status)

            # Sort by creation date, newest first
            query = query.order_by(desc(ClusteringJobDB.created_at))

            # Apply pagination
            jobs = query.limit(limit).offset(offset).all()

            return [job.to_dict() for job in jobs]
        except Exception as e:
            logger.error(f"Error getting clustering jobs from database: {str(e)}")
            return []
        finally:
            session.close()
