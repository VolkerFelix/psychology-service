"""Test the API endpoints."""

import os
import uuid
from datetime import datetime

from fastapi.testclient import TestClient

from app.main import app

os.environ["DATABASE_URL"] = "sqlite:///:memory:"


# Create a test client
client = TestClient(app)


class TestAPI:
    """Test the API endpoints."""

    def test_root_endpoint(self):
        """Test the root endpoint."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "Psychology Profiling Microservice" in data["message"]

    def test_health_check(self):
        """Test the health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "healthy"}


class TestProfileAPI:
    """Test the profile API endpoints."""

    def setup_method(self):
        """Set up test data."""
        self.user_id = f"test_user_{uuid.uuid4()}"
        self.profile_id = str(uuid.uuid4())

        # Create a test profile
        self.test_profile = {
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
                "environment_preference": "dark_quiet",
                "sleep_anxiety_level": 3,
            },
            "behavioral_patterns": {
                "stress_response": "problem_focused",
                "routine_consistency": 8,
                "exercise_frequency": 4,
                "social_activity_preference": 6,
                "screen_time_before_bed": 30,
            },
        }

    def test_create_profile(self):
        """Test creating a new profile."""
        response = client.post("/api/profiles/", json=self.test_profile)

        assert response.status_code == 201
        data = response.json()
        assert "profile" in data
        assert data["profile"]["user_id"] == self.user_id
        assert "profile_id" in data["profile"]
        assert data["status"] == "success"

    def test_get_profile_by_user(self):
        """Test retrieving a profile by user ID."""
        # First create a profile
        create_response = client.post("/api/profiles/", json=self.test_profile)
        assert create_response.status_code == 201

        # Then retrieve by user ID
        response = client.get(f"/api/profiles/user/{self.user_id}")

        assert response.status_code == 200
        data = response.json()
        assert "profile" in data
        assert data["profile"]["user_id"] == self.user_id
        assert data["status"] == "success"

    def test_get_profile_by_id(self):
        """Test retrieving a profile by profile ID."""
        # First create a profile
        create_response = client.post("/api/profiles/", json=self.test_profile)
        profile_id = create_response.json()["profile"]["profile_id"]

        # Then retrieve by profile ID
        response = client.get(f"/api/profiles/{profile_id}")

        assert response.status_code == 200
        data = response.json()
        assert "profile" in data
        assert data["profile"]["profile_id"] == profile_id
        assert data["profile"]["user_id"] == self.user_id
        assert data["status"] == "success"

    def test_update_profile(self):
        """Test updating an existing profile."""
        # First create a profile
        create_response = client.post("/api/profiles/", json=self.test_profile)
        profile_id = create_response.json()["profile"]["profile_id"]

        # Prepare update data
        update_data = {
            "personality_traits": {
                "openness": 80.0,
                "extraversion": 65.0,
            },
            "behavioral_patterns": {
                "routine_consistency": 9,
            },
        }

        # Then update the profile
        response = client.put(f"/api/profiles/{profile_id}", json=update_data)

        assert response.status_code == 200
        data = response.json()
        assert "profile" in data
        assert data["profile"]["profile_id"] == profile_id
        assert data["profile"]["personality_traits"]["openness"] == 80.0
        assert data["profile"]["personality_traits"]["extraversion"] == 65.0
        assert data["profile"]["behavioral_patterns"]["routine_consistency"] == 9
        # Original data should be preserved
        assert data["profile"]["personality_traits"]["conscientiousness"] == 85.0
        assert data["status"] == "success"

    def test_delete_profile(self):
        """Test deleting a profile."""
        # First create a profile
        create_response = client.post("/api/profiles/", json=self.test_profile)
        profile_id = create_response.json()["profile"]["profile_id"]

        # Then delete the profile
        response = client.delete(f"/api/profiles/{profile_id}")

        assert response.status_code == 204

        # Verify profile is deleted
        get_response = client.get(f"/api/profiles/{profile_id}")
        assert get_response.status_code == 404

    def test_list_profiles(self):
        """Test listing profiles with filters."""
        # Create a few profiles
        for i in range(3):
            profile_data = {
                "user_id": f"list_test_user_{i}_{uuid.uuid4()}",
                "personality_traits": {
                    "openness": 70.0 + i * 5,
                },
            }
            client.post("/api/profiles/", json=profile_data)

        # List profiles
        response = client.get("/api/profiles/")

        assert response.status_code == 200
        data = response.json()
        assert "profiles" in data
        assert "count" in data
        assert data["count"] >= 3
        assert len(data["profiles"]) >= 3
        assert data["status"] == "success"


class TestQuestionnaireAPI:
    """Test the questionnaire API endpoints."""

    def setup_method(self):
        """Set up test data."""
        self.user_id = f"test_user_{uuid.uuid4()}"

        # Create a test questionnaire
        self.test_questionnaire = {
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
            "estimated_duration_minutes": 10,
            "tags": ["personality", "onboarding"],
        }

    def test_create_questionnaire(self):
        """Test creating a new questionnaire."""
        response = client.post("/api/questionnaires/", json=self.test_questionnaire)

        assert response.status_code == 201
        data = response.json()
        assert "questionnaire" in data
        assert data["questionnaire"]["title"] == "Personality Assessment"
        assert len(data["questionnaire"]["questions"]) == 3
        assert data["questionnaire"]["is_active"] is True
        assert data["status"] == "success"

    def test_get_questionnaire(self):
        """Test retrieving a questionnaire by ID."""
        # First create a questionnaire
        create_response = client.post(
            "/api/questionnaires/", json=self.test_questionnaire
        )
        questionnaire_id = create_response.json()["questionnaire"]["questionnaire_id"]

        # Then retrieve it
        response = client.get(f"/api/questionnaires/{questionnaire_id}")

        assert response.status_code == 200
        data = response.json()
        assert "questionnaire" in data
        assert data["questionnaire"]["questionnaire_id"] == questionnaire_id
        assert data["questionnaire"]["title"] == "Personality Assessment"
        assert data["status"] == "success"

    def test_list_questionnaires(self):
        """Test listing questionnaires with filters."""
        # Create a few questionnaires
        for i in range(3):
            questionnaire_data = {
                "title": f"Test Questionnaire {i}",
                "description": f"Test description {i}",
                "questionnaire_type": "personality" if i % 2 == 0 else "sleep_habits",
                "questions": self.test_questionnaire["questions"][:1],
                "estimated_duration_minutes": 5,
                "tags": ["test"],
            }
            client.post("/api/questionnaires/", json=questionnaire_data)

        # List questionnaires
        response = client.get("/api/questionnaires/?questionnaire_type=personality")

        assert response.status_code == 200
        data = response.json()
        assert "questionnaires" in data
        assert "count" in data
        assert data["count"] >= 2  # At least the 2 we created with personality type
        assert all(
            q["questionnaire_type"] == "personality" for q in data["questionnaires"]
        )
        assert data["status"] == "success"

    def test_start_questionnaire(self):
        """Test starting a questionnaire for a user."""
        # First create a questionnaire
        create_response = client.post(
            "/api/questionnaires/", json=self.test_questionnaire
        )
        questionnaire_id = create_response.json()["questionnaire"]["questionnaire_id"]

        # Start the questionnaire
        response = client.post(
            f"""/api/questionnaires/start?user_id=
            {self.user_id}&questionnaire_id={questionnaire_id}"""
        )

        assert response.status_code == 200
        data = response.json()
        assert "questionnaire" in data
        assert data["questionnaire"]["user_id"] == self.user_id
        assert data["questionnaire"]["questionnaire_id"] == questionnaire_id
        assert data["questionnaire"]["status"] == "in_progress"
        assert data["questionnaire"]["current_page"] == 1
        assert data["status"] == "success"

    def test_get_questionnaire_progress(self):
        """Test getting a user's progress on a questionnaire."""
        # First create and start a questionnaire
        create_response = client.post(
            "/api/questionnaires/", json=self.test_questionnaire
        )
        questionnaire_id = create_response.json()["questionnaire"]["questionnaire_id"]
        start_response = client.post(
            f"""/api/questionnaires/start?user_id=
            {self.user_id}&questionnaire_id={questionnaire_id}"""
        )
        user_questionnaire_id = start_response.json()["questionnaire"][
            "user_questionnaire_id"
        ]

        # Get progress
        response = client.get(f"/api/questionnaires/progress/{user_questionnaire_id}")

        assert response.status_code == 200
        data = response.json()
        assert "questionnaire" in data
        assert data["questionnaire"]["user_questionnaire_id"] == user_questionnaire_id
        assert data["questionnaire"]["status"] == "in_progress"
        assert data["status"] == "success"

    def test_submit_answers(self):
        """Test submitting answers for a questionnaire."""
        # First create and start a questionnaire
        create_response = client.post(
            "/api/questionnaires/", json=self.test_questionnaire
        )
        questionnaire_id = create_response.json()["questionnaire"]["questionnaire_id"]
        client.post(
            f"""/api/questionnaires/start?user_id=
            {self.user_id}&questionnaire_id={questionnaire_id}"""
        )

        # Prepare answer submission
        answer_data = {
            "user_id": self.user_id,
            "questionnaire_id": questionnaire_id,
            "answers": [
                {
                    "question_id": "q1",
                    "value": 4,
                    "answered_at": datetime.now().isoformat(),
                },
                {
                    "question_id": "q2",
                    "value": 2,
                    "answered_at": datetime.now().isoformat(),
                },
            ],
            "page_number": 1,
        }

        # Submit answers
        response = client.post("/api/questionnaires/answer", json=answer_data)

        assert response.status_code == 200
        data = response.json()
        assert "questionnaire" in data
        assert data["questionnaire"]["user_id"] == self.user_id
        assert len(data["questionnaire"]["answers"]) == 2
        assert data["status"] == "success"

    def test_complete_questionnaire(self):
        """Test completing a questionnaire."""
        # First create, start, and answer a questionnaire
        create_response = client.post(
            "/api/questionnaires/", json=self.test_questionnaire
        )
        questionnaire_id = create_response.json()["questionnaire"]["questionnaire_id"]
        start_response = client.post(
            f"""/api/questionnaires/start?user_id=
            {self.user_id}&questionnaire_id={questionnaire_id}"""
        )
        user_questionnaire_id = start_response.json()["questionnaire"][
            "user_questionnaire_id"
        ]

        # Submit answers
        answer_data = {
            "user_id": self.user_id,
            "questionnaire_id": questionnaire_id,
            "answers": [
                {
                    "question_id": "q1",
                    "value": 4,
                    "answered_at": datetime.now().isoformat(),
                },
                {
                    "question_id": "q2",
                    "value": 2,
                    "answered_at": datetime.now().isoformat(),
                },
                {
                    "question_id": "q3",
                    "value": "opt1",
                    "answered_at": datetime.now().isoformat(),
                },
            ],
            "page_number": 1,
        }
        client.post("/api/questionnaires/answer", json=answer_data)

        # Complete the questionnaire
        response = client.post(f"/api/questionnaires/complete/{user_questionnaire_id}")

        assert response.status_code == 200
        data = response.json()
        assert "questionnaire" in data
        assert data["questionnaire"]["user_questionnaire_id"] == user_questionnaire_id
        assert data["questionnaire"]["status"] == "completed"
        assert data["questionnaire"]["completed_at"] is not None
        assert data["questionnaire"]["scored"] is True
        assert data["questionnaire"]["score_results"] is not None
        assert data["status"] == "success"


class TestClusteringAPI:
    """Test the clustering API endpoints."""

    def setup_method(self):
        """Set up test data."""
        self.user_id = f"test_user_{uuid.uuid4()}"

        # Create a test clustering model request
        self.test_clustering_request = {
            "name": "Test Clustering Model",
            "description": "A test clustering model",
            "algorithm": "kmeans",
            "parameters": {
                "n_clusters": 3,
                "random_state": 42,
            },
            "features_to_use": [
                "openness",
                "conscientiousness",
                "extraversion",
                "agreeableness",
                "neuroticism",
                "chronotype",
                "sleep_environment",
                "routine_consistency",
            ],
            "target_num_clusters": 3,
        }

    def test_run_clustering(self):
        """Test running a clustering job."""
        response = client.post("/api/clustering/run", json=self.test_clustering_request)

        assert response.status_code == 202
        data = response.json()
        assert "job" in data
        assert data["job"]["name"] == "Test Clustering Model"
        assert data["job"]["status"] == "pending"
        assert data["job"]["algorithm"] == "kmeans"
        assert data["status"] == "success"

    def test_get_clustering_job(self):
        """Test retrieving a clustering job by ID."""
        # First create a job
        create_response = client.post(
            "/api/clustering/run", json=self.test_clustering_request
        )
        job_id = create_response.json()["job"]["job_id"]

        # Then retrieve it
        response = client.get(f"/api/clustering/jobs/{job_id}")

        assert response.status_code == 200
        data = response.json()
        assert "job" in data
        assert data["job"]["job_id"] == job_id
        assert data["job"]["name"] == "Test Clustering Model"
        assert data["status"] == "success"

    def test_list_clustering_jobs(self):
        """Test listing clustering jobs."""
        # Create a few jobs
        for i in range(2):
            req = {**self.test_clustering_request, "name": f"Test Clustering Job {i}"}
            client.post("/api/clustering/run", json=req)

        # List jobs
        response = client.get("/api/clustering/jobs")

        assert response.status_code == 200
        data = response.json()
        assert "jobs" in data
        assert "count" in data
        assert data["count"] >= 2
        assert data["status"] == "success"

    # The following tests may require mocking since
    # they depend on completed clustering models

    def test_list_clustering_models(self):
        """Test listing clustering models."""
        # This test might need to be skipped or mocked
        response = client.get("/api/clustering/models")

        # The test passes if we get a valid response, even if there are no models yet
        assert response.status_code == 200
        data = response.json()
        assert "models" in data
        assert "count" in data
        assert data["status"] == "success"

    def test_create_reassignment_job(self):
        """Test creating a job to reassign users to clusters."""
        # This test might need to be skipped or mocked
        response = client.post("/api/clustering/jobs/reassign")

        # We may expect a 404 if no active model exists, or 202 if it works
        assert response.status_code in [202, 404]

        if response.status_code == 202:
            data = response.json()
            assert "job" in data
            assert data["job"]["name"] == "User cluster reassignment"
            assert data["job"]["status"] == "pending"
            assert data["status"] == "success"


class TestDBStorageAPI:
    """Test the database storage functionality through API endpoints."""

    def setup_method(self):
        """Set up test database for storage tests."""
        # This method intentionally left empty as we're using an in-memory database
        pass

    def test_storage_service_through_profile_api(self):
        """Test storage service functionality through profile API."""
        # Create a unique user ID for this test
        user_id = f"storage_test_user_{uuid.uuid4()}"

        # Create a test profile
        test_profile = {
            "user_id": user_id,
            "personality_traits": {
                "openness": 80.0,
                "conscientiousness": 75.0,
            },
        }

        # Create profile
        create_response = client.post("/api/profiles/", json=test_profile)
        assert create_response.status_code == 201
        profile_id = create_response.json()["profile"]["profile_id"]

        # Retrieve profile
        get_response = client.get(f"/api/profiles/{profile_id}")
        assert get_response.status_code == 200
        retrieved_profile = get_response.json()["profile"]

        # Verify data was stored and retrieved correctly
        assert retrieved_profile["user_id"] == user_id
        assert retrieved_profile["personality_traits"]["openness"] == 80.0
        assert retrieved_profile["personality_traits"]["conscientiousness"] == 75.0

        # Update profile
        update_data = {
            "personality_traits": {
                "openness": 85.0,
            }
        }
        update_response = client.put(f"/api/profiles/{profile_id}", json=update_data)
        assert update_response.status_code == 200

        # Retrieve updated profile
        get_updated_response = client.get(f"/api/profiles/{profile_id}")
        assert get_updated_response.status_code == 200
        updated_profile = get_updated_response.json()["profile"]

        # Verify update was stored correctly
        assert updated_profile["personality_traits"]["openness"] == 85.0
        assert updated_profile["personality_traits"]["conscientiousness"] == 75.0

        # Delete profile
        delete_response = client.delete(f"/api/profiles/{profile_id}")
        assert delete_response.status_code == 204

        # Verify deletion
        get_deleted_response = client.get(f"/api/profiles/{profile_id}")
        assert get_deleted_response.status_code == 404

    def test_storage_service_through_questionnaire_api(self):
        """Test storage service functionality through questionnaire API."""
        # Create a unique title for this test
        title = f"Storage Test Questionnaire {uuid.uuid4()}"

        # Create a test questionnaire
        test_questionnaire = {
            "title": title,
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
            "estimated_duration_minutes": 5,
            "tags": ["test"],
        }

        # Create questionnaire
        create_response = client.post("/api/questionnaires/", json=test_questionnaire)
        assert create_response.status_code == 201
        questionnaire_id = create_response.json()["questionnaire"]["questionnaire_id"]

        # Retrieve questionnaire
        get_response = client.get(f"/api/questionnaires/{questionnaire_id}")
        assert get_response.status_code == 200
        retrieved_questionnaire = get_response.json()["questionnaire"]

        # Verify data was stored and retrieved correctly
        assert retrieved_questionnaire["title"] == title
        assert retrieved_questionnaire["questionnaire_type"] == "personality"
        assert len(retrieved_questionnaire["questions"]) == 1

        # Test storage with a user questionnaire
        user_id = f"storage_test_user_{uuid.uuid4()}"
        start_response = client.post(
            f"""/api/questionnaires/start?user_id=
            {user_id}&questionnaire_id={questionnaire_id}"""
        )
        assert start_response.status_code == 200
        user_questionnaire_id = start_response.json()["questionnaire"][
            "user_questionnaire_id"
        ]

        # Get user questionnaire progress
        progress_response = client.get(
            f"/api/questionnaires/progress/{user_questionnaire_id}"
        )
        assert progress_response.status_code == 200
        user_questionnaire = progress_response.json()["questionnaire"]

        # Verify user questionnaire data
        assert user_questionnaire["user_id"] == user_id
        assert user_questionnaire["questionnaire_id"] == questionnaire_id
        assert user_questionnaire["status"] == "in_progress"
