"""Fixtures for the tests."""
import os
from datetime import timedelta
from unittest.mock import patch

import pytest

# Define a fixed path for the test database
TEST_DB_PATH = os.path.join(os.path.dirname(__file__), "test_db.db")
TEST_DB_URL = f"sqlite:///{TEST_DB_PATH}"


@pytest.fixture(scope="session", autouse=True)
def use_sqlite_db():
    """Use a fixed SQLite database for all tests."""
    # Store original database URL
    original_db_url = os.environ.get("DATABASE_URL")

    # Set our test database URL as an environment variable
    os.environ["DATABASE_URL"] = TEST_DB_URL

    # Also patch the settings module
    with patch("app.config.settings.DATABASE_URL", TEST_DB_URL):
        yield

    # Clean up
    if original_db_url:
        os.environ["DATABASE_URL"] = original_db_url
    else:
        os.environ.pop("DATABASE_URL", None)

    # Optionally: Clean up the database file after tests
    if os.path.exists(TEST_DB_PATH):
        os.remove(TEST_DB_PATH)


@pytest.fixture
def mock_storage():
    """Create a mock storage service for testing."""
    from unittest.mock import MagicMock

    storage = MagicMock()
    return storage


@pytest.fixture
def mock_profile_service(mock_storage):
    """Create a mock profile service for testing."""
    from app.services.profile_service import ProfileService

    return ProfileService(storage_service=mock_storage)


@pytest.fixture
def mock_questionnaire_service(mock_storage):
    """Create a mock questionnaire service for testing."""
    from app.services.questionnaire_service import QuestionnaireService

    return QuestionnaireService(storage_service=mock_storage)


@pytest.fixture
def mock_clustering_service(mock_storage):
    """Create a mock clustering service for testing."""
    from app.services.clustering_service import ClusteringService

    return ClusteringService(storage_service=mock_storage)


@pytest.fixture
def sample_profile_data():
    """Sample profile data for testing."""
    import uuid
    from datetime import datetime

    user_id = f"test_user_{uuid.uuid4()}"
    profile_id = str(uuid.uuid4())

    return {
        "profile_id": profile_id,
        "user_id": user_id,
        "personality_traits": {
            "openness": 75.0,
            "conscientiousness": 85.0,
            "extraversion": 60.0,
            "agreeableness": 70.0,
            "neuroticism": 45.0,
            "dominant_traits": ["openness", "conscientiousness"],
        },
        "sleep_preferences": {
            "chronotype": "morning_person",
            "ideal_bedtime": "22:30",
            "ideal_waketime": "06:30",
            "environment_preference": "dark_quiet",
            "sleep_anxiety_level": 3,
            "relaxation_techniques": ["reading", "meditation"],
        },
        "behavioral_patterns": {
            "stress_response": "problem_focused",
            "routine_consistency": 8,
            "exercise_frequency": 4,
            "social_activity_preference": 6,
            "screen_time_before_bed": 30,
            "typical_stress_level": 4,
        },
        "profile_metadata": {
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "completeness": "complete",
            "completion_percentage": 90,
            "questions_answered": 25,
            "source": "questionnaire",
            "valid": True,
        },
        "raw_scores": {
            "q1": 5,
            "q2": 4,
            "q3": 2,
        },
    }


@pytest.fixture
def sample_questionnaire_data():
    """Sample questionnaire data for testing."""
    import uuid
    from datetime import datetime

    questionnaire_id = str(uuid.uuid4())

    return {
        "questionnaire_id": questionnaire_id,
        "title": "Personality Assessment",
        "description": "A comprehensive personality assessment",
        "questionnaire_type": "personality",
        "questions": [
            {
                "question_id": "q1",
                "text": "I enjoy trying new experiences",
                "description": "Rate how much you agree with this statement",
                "question_type": "likert_5",
                "category": "personality",
                "dimensions": ["openness"],
                "required": True,
            },
            {
                "question_id": "q2",
                "text": "I prefer a consistent daily routine",
                "description": "Rate how much you agree with this statement",
                "question_type": "likert_5",
                "category": "behavior",
                "dimensions": ["routine_consistency"],
                "required": True,
            },
            {
                "question_id": "q3",
                "text": "What is your ideal bedtime?",
                "description": "Select the time you prefer to go to sleep",
                "question_type": "multiple_choice",
                "category": "sleep",
                "dimensions": ["chronotype"],
                "options": [
                    {
                        "option_id": "opt1",
                        "text": "Before 10 PM",
                        "value": "early",
                    },
                    {
                        "option_id": "opt2",
                        "text": "10 PM - Midnight",
                        "value": "medium",
                    },
                    {
                        "option_id": "opt3",
                        "text": "After Midnight",
                        "value": "late",
                    },
                ],
                "required": True,
            },
        ],
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat(),
        "version": "1.0.0",
        "is_active": True,
        "estimated_duration_minutes": 10,
        "tags": ["personality", "onboarding"],
    }


@pytest.fixture
def sample_clustering_model_data():
    """Sample clustering model data for testing."""
    import uuid
    from datetime import datetime

    model_id = str(uuid.uuid4())
    cluster_id = str(uuid.uuid4())

    return {
        "clustering_model_id": model_id,
        "name": "Personality Profile Clusters",
        "description": "Clustering model based on personality and sleep traits",
        "algorithm": "kmeans",
        "parameters": {"n_clusters": 5, "random_state": 42},
        "num_clusters": 5,
        "features_used": [
            "openness",
            "conscientiousness",
            "extraversion",
            "agreeableness",
            "neuroticism",
            "chronotype",
            "sleep_environment",
            "sleep_anxiety",
            "stress_response",
            "routine_consistency",
        ],
        "clusters": [
            {
                "cluster_id": cluster_id,
                "name": "Disciplined Early Bird",
                "description": "People who are organized and wake up early",
                "size": 25,
                "centroid": {"dim_0": 0.7, "dim_1": 0.2, "dim_2": 0.8, "dim_3": 0.4},
                "key_features": [
                    {
                        "feature_name": "conscientiousness",
                        "feature_value": 85.0,
                        "significance": 0.9,
                        "comparison_to_mean": 0.7,
                    },
                    {
                        "feature_name": "chronotype",
                        "feature_value": 0.2,
                        "significance": 0.8,
                        "comparison_to_mean": -0.8,
                    },
                ],
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat(),
                "version": "1.0.0",
                "is_active": True,
                "tags": ["organized", "early_bird"],
            }
        ],
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat(),
        "version": "1.0.0",
        "is_active": True,
        "total_users_clustered": 120,
        "quality_metrics": {"inertia": 850.5, "silhouette_score": 0.65},
    }


@pytest.fixture
def sample_user_questionnaire_data(sample_questionnaire_data):
    """Sample user questionnaire data for testing."""
    import uuid
    from datetime import datetime

    user_id = f"test_user_{uuid.uuid4()}"
    user_questionnaire_id = str(uuid.uuid4())

    return {
        "user_questionnaire_id": user_questionnaire_id,
        "user_id": user_id,
        "questionnaire_id": sample_questionnaire_data["questionnaire_id"],
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


@pytest.fixture
def sample_clustering_job_data(sample_clustering_model_data):
    """Sample clustering job data for testing."""
    import uuid
    from datetime import datetime

    job_id = str(uuid.uuid4())

    return {
        "job_id": job_id,
        "name": "Test Clustering Job",
        "description": "Testing the clustering algorithm",
        "status": "completed",
        "algorithm": "kmeans",
        "parameters": {"n_clusters": 5, "random_state": 42},
        "created_at": (datetime.now() - timedelta(hours=1)).isoformat(),
        "started_at": (datetime.now() - timedelta(minutes=55)).isoformat(),
        "completed_at": datetime.now().isoformat(),
        "user_count": 120,
        "error_message": None,
        "result_clustering_model_id": sample_clustering_model_data[
            "clustering_model_id"
        ],
    }


@pytest.fixture
def sample_user_cluster_assignment_data(sample_clustering_model_data):
    """Sample user cluster assignment data for testing."""
    import uuid
    from datetime import datetime

    assignment_id = str(uuid.uuid4())
    user_id = f"test_user_{uuid.uuid4()}"
    cluster_id = sample_clustering_model_data["clusters"][0]["cluster_id"]

    return {
        "assignment_id": assignment_id,
        "user_id": user_id,
        "cluster_id": cluster_id,
        "clustering_model_id": sample_clustering_model_data["clustering_model_id"],
        "confidence_score": 0.85,
        "features": {
            "openness": 0.75,
            "conscientiousness": 0.85,
            "extraversion": 0.60,
            "agreeableness": 0.70,
            "neuroticism": 0.45,
            "chronotype": 0.20,
            "sleep_environment": 0.10,
            "sleep_anxiety": 0.30,
            "stress_response": 0.25,
            "routine_consistency": 0.80,
        },
        "distance_to_centroid": 0.25,
        "assigned_at": datetime.now().isoformat(),
        "is_current": True,
    }
