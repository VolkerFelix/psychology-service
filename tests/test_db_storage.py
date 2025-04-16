"""Tests for the database storage service."""
import os
import sys
import uuid
from datetime import datetime

from app.services.storage.db_storage import (
    ClusteringJobDB,
    ClusteringModelDB,
    DatabaseStorage,
    PsychologicalProfileDB,
    QuestionnaireDB,
    UserClusterAssignmentDB,
    UserQuestionnaireDB,
)

# Add the application to the python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


class TestDatabaseStorage:
    """Tests for the DatabaseStorage class."""

    def setup_method(self):
        """Set up test database."""
        # Use in-memory SQLite for testing
        self.db_url = "sqlite:///:memory:"
        self.storage = DatabaseStorage(db_url=self.db_url)

        # Create test data
        self.user_id = f"test_user_{uuid.uuid4()}"
        self.profile_id = str(uuid.uuid4())
        self.questionnaire_id = str(uuid.uuid4())
        self.model_id = str(uuid.uuid4())
        self.cluster_id = str(uuid.uuid4())
        self.job_id = str(uuid.uuid4())

    def test_save_profile(self):
        """Test saving a psychological profile to the database."""
        # Create test profile data
        profile_data = {
            "profile_id": self.profile_id,
            "user_id": self.user_id,
            "personality_traits": {
                "openness": 75.0,
                "conscientiousness": 85.0,
                "extraversion": 60.0,
                "agreeableness": 70.0,
                "neuroticism": 45.0,
            },
            "sleep_preferences": {
                "chronotype": "morning_person",
                "ideal_bedtime": "22:30",
                "ideal_waketime": "06:30",
            },
            "behavioral_patterns": {
                "routine_consistency": 8,
                "exercise_frequency": 4,
            },
            "profile_metadata": {
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat(),
                "completeness": "partial",
                "completion_percentage": 60,
                "questions_answered": 15,
                "source": "questionnaire",
                "valid": True,
            },
            "raw_scores": {},
        }

        # Save profile
        result = self.storage.save_profile(profile_data)

        # Verify
        assert result is True

        # Check if profile exists in database
        session = self.storage.Session()
        profile = (
            session.query(PsychologicalProfileDB)
            .filter_by(profile_id=self.profile_id)
            .first()
        )
        session.close()

        assert profile is not None
        assert profile.user_id == self.user_id
        assert profile.personality_traits["openness"] == 75.0
        assert profile.sleep_preferences["chronotype"] == "morning_person"
        assert profile.behavioral_patterns["routine_consistency"] == 8

    def test_get_profile(self):
        """Test retrieving a profile by ID."""
        # Create and save a profile
        profile_data = {
            "profile_id": self.profile_id,
            "user_id": self.user_id,
            "personality_traits": {"openness": 80.0},
            "sleep_preferences": {},
            "behavioral_patterns": {},
            "profile_metadata": {
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat(),
                "completeness": "partial",
                "completion_percentage": 20,
                "questions_answered": 5,
                "source": "manual",
                "valid": False,
            },
            "raw_scores": {},
        }
        self.storage.save_profile(profile_data)

        # Retrieve profile
        result = self.storage.get_profile(self.profile_id)

        # Verify
        assert result is not None
        assert result["profile_id"] == self.profile_id
        assert result["user_id"] == self.user_id
        assert result["personality_traits"]["openness"] == 80.0

    def test_get_profile_by_user(self):
        """Test retrieving a profile by user ID."""
        # Create and save a profile
        profile_data = {
            "profile_id": self.profile_id,
            "user_id": self.user_id,
            "personality_traits": {"extraversion": 65.0},
            "sleep_preferences": {},
            "behavioral_patterns": {},
            "profile_metadata": {
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat(),
                "completeness": "partial",
                "completion_percentage": 20,
                "questions_answered": 5,
                "source": "manual",
                "valid": False,
            },
            "raw_scores": {},
        }
        self.storage.save_profile(profile_data)

        # Retrieve profile by user
        result = self.storage.get_profile_by_user(self.user_id)

        # Verify
        assert result is not None
        assert result["profile_id"] == self.profile_id
        assert result["user_id"] == self.user_id
        assert result["personality_traits"]["extraversion"] == 65.0

    def test_delete_profile(self):
        """Test deleting a profile."""
        # Create and save a profile
        profile_data = {
            "profile_id": self.profile_id,
            "user_id": self.user_id,
            "personality_traits": {},
            "sleep_preferences": {},
            "behavioral_patterns": {},
            "profile_metadata": {
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat(),
                "completeness": "not_started",
                "completion_percentage": 0,
                "questions_answered": 0,
                "source": "manual",
                "valid": False,
            },
            "raw_scores": {},
        }
        self.storage.save_profile(profile_data)

        # Delete profile
        result = self.storage.delete_profile(self.profile_id)

        # Verify
        assert result is True

        # Check profile doesn't exist
        session = self.storage.Session()
        profile = (
            session.query(PsychologicalProfileDB)
            .filter_by(profile_id=self.profile_id)
            .first()
        )
        session.close()

        assert profile is None

    def test_get_profiles(self):
        """Test retrieving multiple profiles with filters."""
        # Create and save multiple profiles
        for i in range(3):
            profile_id = str(uuid.uuid4())
            user_id = f"user_{i}_{uuid.uuid4()}"
            profile_data = {
                "profile_id": profile_id,
                "user_id": user_id,
                "personality_traits": {"openness": 70.0 + i * 5},
                "sleep_preferences": {},
                "behavioral_patterns": {},
                "profile_metadata": {
                    "created_at": datetime.now().isoformat(),
                    "updated_at": datetime.now().isoformat(),
                    "completeness": "partial",
                    "completion_percentage": 20,
                    "questions_answered": 5,
                    "source": "manual",
                    "valid": False,
                },
                "raw_scores": {},
            }
            self.storage.save_profile(profile_data)

        # Retrieve profiles
        results = self.storage.get_profiles(limit=10, offset=0)

        # Verify
        assert len(results) == 3
        assert all("profile_id" in profile for profile in results)
        assert all("user_id" in profile for profile in results)

    def test_save_questionnaire(self):
        """Test saving a questionnaire to the database."""
        # Create test questionnaire data
        questionnaire_data = {
            "questionnaire_id": self.questionnaire_id,
            "title": "Test Questionnaire",
            "description": "Test description",
            "questionnaire_type": "personality",
            "questions": [
                {
                    "question_id": "q1",
                    "text": "Test question",
                    "description": "Test description",
                    "question_type": "likert_5",
                    "category": "personality",
                    "dimensions": ["openness"],
                    "required": True,
                }
            ],
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "version": "1.0.0",
            "is_active": True,
            "estimated_duration_minutes": 5,
            "tags": ["test"],
        }

        # Save questionnaire
        result = self.storage.save_questionnaire(questionnaire_data)

        # Verify
        assert result is True

        # Check if questionnaire exists in database
        session = self.storage.Session()
        questionnaire = (
            session.query(QuestionnaireDB)
            .filter_by(questionnaire_id=self.questionnaire_id)
            .first()
        )
        session.close()

        assert questionnaire is not None
        assert questionnaire.title == "Test Questionnaire"
        assert questionnaire.questionnaire_type == "personality"
        assert len(questionnaire.questions) == 1

    def test_get_questionnaire(self):
        """Test retrieving a questionnaire by ID."""
        # Create and save a questionnaire
        questionnaire_data = {
            "questionnaire_id": self.questionnaire_id,
            "title": "Test Questionnaire",
            "description": "Test description",
            "questionnaire_type": "personality",
            "questions": [
                {
                    "question_id": "q1",
                    "text": "Test question",
                    "description": "Test description",
                    "question_type": "likert_5",
                    "category": "personality",
                    "dimensions": ["openness"],
                    "required": True,
                }
            ],
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "version": "1.0.0",
            "is_active": True,
            "estimated_duration_minutes": 5,
            "tags": ["test"],
        }
        self.storage.save_questionnaire(questionnaire_data)

        # Retrieve questionnaire
        result = self.storage.get_questionnaire(self.questionnaire_id)

        # Verify
        assert result is not None
        assert result["questionnaire_id"] == self.questionnaire_id
        assert result["title"] == "Test Questionnaire"
        assert result["questionnaire_type"] == "personality"
        assert len(result["questions"]) == 1

    def test_save_user_questionnaire(self):
        """Test saving a user questionnaire to the database."""
        # First save a questionnaire
        questionnaire_data = {
            "questionnaire_id": self.questionnaire_id,
            "title": "Test Questionnaire",
            "description": "Test description",
            "questionnaire_type": "personality",
            "questions": [],
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "version": "1.0.0",
            "is_active": True,
            "estimated_duration_minutes": 5,
            "tags": ["test"],
        }
        self.storage.save_questionnaire(questionnaire_data)

        # Create test user questionnaire data
        user_questionnaire_id = str(uuid.uuid4())
        user_questionnaire_data = {
            "user_questionnaire_id": user_questionnaire_id,
            "user_id": self.user_id,
            "questionnaire_id": self.questionnaire_id,
            "status": "in_progress",
            "answers": [
                {
                    "question_id": "q1",
                    "value": 4,
                    "answered_at": datetime.now().isoformat(),
                }
            ],
            "started_at": datetime.now().isoformat(),
            "current_page": 1,
            "total_pages": 2,
            "scored": False,
        }

        # Save user questionnaire
        result = self.storage.save_user_questionnaire(user_questionnaire_data)

        # Verify
        assert result is True

        # Check if user questionnaire exists in database
        session = self.storage.Session()
        user_questionnaire = (
            session.query(UserQuestionnaireDB)
            .filter_by(user_questionnaire_id=user_questionnaire_id)
            .first()
        )
        session.close()

        assert user_questionnaire is not None
        assert user_questionnaire.user_id == self.user_id
        assert user_questionnaire.questionnaire_id == self.questionnaire_id
        assert user_questionnaire.status == "in_progress"
        assert len(user_questionnaire.answers) == 1

    def test_get_user_questionnaire(self):
        """Test retrieving a user questionnaire by ID."""
        # First save a questionnaire
        questionnaire_data = {
            "questionnaire_id": self.questionnaire_id,
            "title": "Test Questionnaire",
            "description": "Test description",
            "questionnaire_type": "personality",
            "questions": [],
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "version": "1.0.0",
            "is_active": True,
            "estimated_duration_minutes": 5,
            "tags": ["test"],
        }
        self.storage.save_questionnaire(questionnaire_data)

        # Create and save a user questionnaire
        user_questionnaire_id = str(uuid.uuid4())
        user_questionnaire_data = {
            "user_questionnaire_id": user_questionnaire_id,
            "user_id": self.user_id,
            "questionnaire_id": self.questionnaire_id,
            "status": "in_progress",
            "answers": [
                {
                    "question_id": "q1",
                    "value": 4,
                    "answered_at": datetime.now().isoformat(),
                }
            ],
            "started_at": datetime.now().isoformat(),
            "current_page": 1,
            "total_pages": 2,
            "scored": False,
        }
        self.storage.save_user_questionnaire(user_questionnaire_data)

        # Retrieve user questionnaire
        result = self.storage.get_user_questionnaire(user_questionnaire_id)

        # Verify
        assert result is not None
        assert result["user_questionnaire_id"] == user_questionnaire_id
        assert result["user_id"] == self.user_id
        assert result["questionnaire_id"] == self.questionnaire_id
        assert result["status"] == "in_progress"
        assert len(result["answers"]) == 1

    def test_save_clustering_model(self):
        """Test saving a clustering model to the database."""
        # Create test model data
        model_data = {
            "clustering_model_id": self.model_id,
            "name": "Test Model",
            "description": "Test description",
            "algorithm": "kmeans",
            "parameters": {"n_clusters": 3},
            "num_clusters": 3,
            "features_used": ["openness", "conscientiousness"],
            "clusters": [
                {
                    "cluster_id": self.cluster_id,
                    "name": "Test Cluster",
                    "description": "Test description",
                    "size": 10,
                    "centroid": {"dim_0": 0.7, "dim_1": 0.3},
                    "key_features": [],
                    "created_at": datetime.now().isoformat(),
                    "updated_at": datetime.now().isoformat(),
                    "version": "1.0.0",
                    "is_active": True,
                    "tags": [],
                }
            ],
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "version": "1.0.0",
            "is_active": True,
            "total_users_clustered": 50,
            "quality_metrics": {"inertia": 800.5},
        }

        # Save model
        result = self.storage.save_clustering_model(model_data)

        # Verify
        assert result is True

        # Check if model exists in database
        session = self.storage.Session()
        model = (
            session.query(ClusteringModelDB)
            .filter_by(clustering_model_id=self.model_id)
            .first()
        )
        session.close()

        assert model is not None
        assert model.name == "Test Model"
        assert model.algorithm == "kmeans"
        assert model.num_clusters == 3
        assert len(model.clusters) == 1
        assert model.is_active is True

    def test_save_user_cluster_assignment(self):
        """Test saving a user cluster assignment to the database."""
        # First save a clustering model
        model_data = {
            "clustering_model_id": self.model_id,
            "name": "Test Model",
            "description": "Test description",
            "algorithm": "kmeans",
            "parameters": {"n_clusters": 3},
            "num_clusters": 3,
            "features_used": ["openness", "conscientiousness"],
            "clusters": [
                {
                    "cluster_id": self.cluster_id,
                    "name": "Test Cluster",
                    "description": "Test description",
                    "size": 10,
                    "centroid": {"dim_0": 0.7, "dim_1": 0.3},
                    "key_features": [],
                    "created_at": datetime.now().isoformat(),
                    "updated_at": datetime.now().isoformat(),
                    "version": "1.0.0",
                    "is_active": True,
                    "tags": [],
                }
            ],
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "version": "1.0.0",
            "is_active": True,
            "total_users_clustered": 50,
            "quality_metrics": {"inertia": 800.5},
        }
        self.storage.save_clustering_model(model_data)

        # Create assignment data
        assignment_id = str(uuid.uuid4())
        assignment_data = {
            "assignment_id": assignment_id,
            "user_id": self.user_id,
            "cluster_id": self.cluster_id,
            "clustering_model_id": self.model_id,
            "confidence_score": 0.85,
            "features": {"openness": 0.7, "conscientiousness": 0.8},
            "distance_to_centroid": 0.25,
            "assigned_at": datetime.now().isoformat(),
            "is_current": True,
        }

        # Save assignment
        result = self.storage.save_user_cluster_assignment(assignment_data)

        # Verify
        assert result is True

        # Check if assignment exists in database
        session = self.storage.Session()
        assignment = (
            session.query(UserClusterAssignmentDB)
            .filter_by(assignment_id=assignment_id)
            .first()
        )
        session.close()

        assert assignment is not None
        assert assignment.user_id == self.user_id
        assert assignment.cluster_id == self.cluster_id
        assert assignment.clustering_model_id == self.model_id
        assert assignment.confidence_score == 0.85
        assert assignment.is_current is True

    def test_save_clustering_job(self):
        """Test saving a clustering job to the database."""
        # Create test job data
        job_data = {
            "job_id": self.job_id,
            "name": "Test Job",
            "description": "Test description",
            "status": "pending",
            "algorithm": "kmeans",
            "parameters": {"n_clusters": 3},
            "created_at": datetime.now().isoformat(),
        }

        # Save job
        result = self.storage.save_clustering_job(job_data)

        # Verify
        assert result is True

        # Check if job exists in database
        session = self.storage.Session()
        job = session.query(ClusteringJobDB).filter_by(job_id=self.job_id).first()
        session.close()

        assert job is not None
        assert job.name == "Test Job"
        assert job.status == "pending"
        assert job.algorithm == "kmeans"

    def test_get_clustering_job(self):
        """Test retrieving a clustering job by ID."""
        # Create and save a job
        job_data = {
            "job_id": self.job_id,
            "name": "Test Job",
            "description": "Test description",
            "status": "pending",
            "algorithm": "kmeans",
            "parameters": {"n_clusters": 3},
            "created_at": datetime.now().isoformat(),
        }
        self.storage.save_clustering_job(job_data)

        # Retrieve job
        result = self.storage.get_clustering_job(self.job_id)

        # Verify
        assert result is not None
        assert result["job_id"] == self.job_id
        assert result["name"] == "Test Job"
        assert result["status"] == "pending"
        assert result["algorithm"] == "kmeans"
