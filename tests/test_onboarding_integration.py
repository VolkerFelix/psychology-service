"""Integration tests for the onboarding questionnaire flow."""

import os
import uuid

from fastapi.testclient import TestClient

from app.main import app
from app.models.questionnaire_models import (
    QuestionCategory,
    QuestionnaireType,
    QuestionType,
)

# Set SQLite database for testing
os.environ["DATABASE_URL"] = "sqlite:///:memory:"

# Create a test client
client = TestClient(app)


class TestOnboardingQuestionnaire:
    """Integration tests for the onboarding questionnaire flow."""

    def setup_method(self):
        """Set up test data - create our test questionnaire directly in the database."""
        self.user_id = "test_onboarding_user"
        self.questionnaire_id = str(uuid.uuid4())

        # Create sample questions for our test questionnaire
        personality_questions = self._create_personality_questions()
        sleep_questions = self._create_sleep_questions()
        behavioral_questions = self._create_behavioral_questions()

        # Combine all questions
        questions = personality_questions + sleep_questions + behavioral_questions

        # Create questionnaire data
        questionnaire_data = {
            "title": "Test Onboarding Questionnaire",
            "description": "Test questionnaire for integration tests",
            "questionnaire_type": QuestionnaireType.ONBOARDING.value,
            "questions": questions,
            "estimated_duration_minutes": 10,
            "tags": ["test", "onboarding"],
        }

        # Call API to create the questionnaire
        response = client.post("/api/questionnaires/", json=questionnaire_data)
        assert (
            response.status_code == 201
        ), f"Failed to create questionnaire: {response.text}"

        data = response.json()
        self.questionnaire = data["questionnaire"]
        self.questionnaire_id = self.questionnaire["questionnaire_id"]

        # Make sure we have the expected questions
        assert (
            len(self.questionnaire["questions"]) >= 15
        ), "Onboarding questionnaire is missing expected questions"

    def _create_personality_questions(self):
        """Create personality trait questions for testing."""
        dimensions = [
            "openness",
            "conscientiousness",
            "extraversion",
            "agreeableness",
            "neuroticism",
        ]
        questions = []

        for i, dimension in enumerate(dimensions):
            # Create two questions for each dimension
            for j in range(2):
                questions.append(
                    {
                        "question_id": f"personality_{dimension}_{j}",
                        "text": f"Test question for {dimension} #{j+1}",
                        "description": f"""Rate how much you agree with
                        this statement about {dimension}""",
                        "question_type": QuestionType.LIKERT_5.value,
                        "category": QuestionCategory.PERSONALITY.value,
                        "dimensions": [dimension],
                        "required": True,
                        "weight": 1.0,
                    }
                )

        return questions

    def _create_sleep_questions(self):
        """Create sleep preference questions for testing."""
        return [
            {
                "question_id": "sleep_chronotype",
                "text": "When do you naturally prefer to go to sleep?",
                "description": "Select your preference",
                "question_type": QuestionType.MULTIPLE_CHOICE.value,
                "category": QuestionCategory.SLEEP.value,
                "dimensions": ["chronotype"],
                "options": [
                    {
                        "option_id": "early",
                        "text": "Early (before 10 PM)",
                        "value": "morning_person",
                    },
                    {
                        "option_id": "medium",
                        "text": "Medium (10 PM - midnight)",
                        "value": "intermediate",
                    },
                    {
                        "option_id": "late",
                        "text": "Late (after midnight)",
                        "value": "evening_person",
                    },
                ],
                "required": True,
                "weight": 1.0,
            },
            {
                "question_id": "sleep_environment",
                "text": "What is your preferred sleep environment?",
                "description": "Select your preference",
                "question_type": QuestionType.MULTIPLE_CHOICE.value,
                "category": QuestionCategory.SLEEP.value,
                "dimensions": ["sleep_environment"],
                "options": [
                    {
                        "option_id": "dark_quiet",
                        "text": "Dark and quiet",
                        "value": "dark_quiet",
                    },
                    {
                        "option_id": "some_light",
                        "text": "Some light",
                        "value": "some_light",
                    },
                    {
                        "option_id": "some_noise",
                        "text": "Some noise",
                        "value": "some_noise",
                    },
                ],
                "required": True,
                "weight": 1.0,
            },
            {
                "question_id": "sleep_anxiety",
                "text": "How often do you worry about sleep?",
                "description": "Rate on a scale",
                "question_type": QuestionType.SLIDER.value,
                "category": QuestionCategory.SLEEP.value,
                "dimensions": ["sleep_anxiety"],
                "min_value": 0,
                "max_value": 10,
                "step": 1,
                "required": True,
                "weight": 1.0,
            },
        ]

    def _create_behavioral_questions(self):
        """Create behavioral pattern questions for testing."""
        return [
            {
                "question_id": "stress_response",
                "text": "How do you typically respond to stress?",
                "description": "Select your typical response",
                "question_type": QuestionType.MULTIPLE_CHOICE.value,
                "category": QuestionCategory.BEHAVIOR.value,
                "dimensions": ["stress_response"],
                "options": [
                    {
                        "option_id": "problem",
                        "text": "Problem solving",
                        "value": "problem_focused",
                    },
                    {
                        "option_id": "emotion",
                        "text": "Emotional coping",
                        "value": "emotion_focused",
                    },
                    {"option_id": "avoidant", "text": "Avoidance", "value": "avoidant"},
                ],
                "required": True,
                "weight": 1.0,
            },
            {
                "question_id": "routine_consistency",
                "text": "How consistent is your daily routine?",
                "description": "Rate consistency",
                "question_type": QuestionType.SLIDER.value,
                "category": QuestionCategory.BEHAVIOR.value,
                "dimensions": ["routine_consistency"],
                "min_value": 0,
                "max_value": 10,
                "step": 1,
                "required": True,
                "weight": 1.0,
            },
            {
                "question_id": "screen_time",
                "text": "Screen time before bed",
                "description": "Minutes of screen time",
                "question_type": QuestionType.SLIDER.value,
                "category": QuestionCategory.BEHAVIOR.value,
                "dimensions": ["screen_time"],
                "min_value": 0,
                "max_value": 120,
                "step": 5,
                "required": True,
                "weight": 1.0,
            },
        ]

    def test_get_onboarding_questionnaire(self):
        """Test retrieving the onboarding questionnaire for a new user."""
        # For a new user, get_next_onboarding_questionnaire
        # should return our test questionnaire
        response = client.get(f"/api/questionnaires/onboarding/{self.user_id}")

        assert response.status_code == 200
        data = response.json()

        assert data["status"] == "success"
        assert data["questionnaire"] is not None
        assert data["questionnaire"]["questionnaire_type"] == "onboarding"
        assert data["questionnaire"]["questionnaire_id"] == self.questionnaire_id

    def test_start_onboarding_questionnaire(self):
        """Test starting the onboarding questionnaire."""
        response = client.post(
            f"/api/questionnaires/start?user_id={self.user_id}&questionnaire_id={self.questionnaire_id}"  # noqa: E501
        )

        assert response.status_code == 200
        data = response.json()

        assert data["status"] == "success"
        assert data["questionnaire"]["user_id"] == self.user_id
        assert data["questionnaire"]["questionnaire_id"] == self.questionnaire_id
        assert data["questionnaire"]["status"] == "in_progress"

        # Save the user_questionnaire_id for subsequent tests
        self.user_questionnaire_id = data["questionnaire"]["user_questionnaire_id"]
        return self.user_questionnaire_id

    def test_submit_personality_answers(self):
        """Test submitting answers to personality questions."""
        # Start the questionnaire if not already started
        user_questionnaire_id = getattr(self, "user_questionnaire_id", None)
        if not user_questionnaire_id:
            user_questionnaire_id = self.test_start_onboarding_questionnaire()

        # Get the actual question IDs from our created questionnaire
        personality_questions = [
            q for q in self.questionnaire["questions"] if q["category"] == "personality"
        ]

        # Submit answers to personality questions
        personality_answers = []
        for question in personality_questions:
            personality_answers.append(
                {
                    "question_id": question["question_id"],
                    "value": 4,  # "Agree" on Likert 5 scale
                    "answered_at": "2025-04-17T10:00:00.000Z",
                }
            )

        response = client.post(
            "/api/questionnaires/answer",
            json={
                "user_id": self.user_id,
                "questionnaire_id": self.questionnaire_id,
                "answers": personality_answers,
                "page_number": 1,
            },
        )

        assert response.status_code == 200
        data = response.json()

        assert data["status"] == "success"
        assert len(data["questionnaire"]["answers"]) == len(personality_questions)

        # Verify we can get the current page
        response = client.get(
            f"/api/questionnaires/progress/{user_questionnaire_id}/page/1"
        )
        assert response.status_code == 200

    def test_submit_sleep_answers(self):
        """Test submitting answers to sleep preference questions."""
        # Start the questionnaire if not already started
        user_questionnaire_id = getattr(self, "user_questionnaire_id", None)
        if not user_questionnaire_id:
            user_questionnaire_id = self.test_start_onboarding_questionnaire()
            self.test_submit_personality_answers()  # Submit personality answers first

        # Get the actual sleep questions from our created questionnaire
        sleep_questions = [
            q for q in self.questionnaire["questions"] if q["category"] == "sleep"
        ]

        # Submit answers to sleep questions
        sleep_answers = []

        for question in sleep_questions:
            answer = {
                "question_id": question["question_id"],
                "answered_at": "2025-04-17T10:01:00.000Z",
            }

            # Add appropriate value based on question type
            if question["question_type"] == "multiple_choice":
                # Take the first option for multiple choice questions
                answer["value"] = question["options"][0]["option_id"]
            # Use mid-range value for sliders
            if question["question_type"] == "slider":
                # Use mid-range value for sliders, but ensure we respect max bounds
                min_val = question.get("min_value", 0)
                max_val = question.get("max_value", 10)
                # For sleep_anxiety, ensure value is between 0-10
                if "sleep_anxiety" in question["dimensions"]:
                    answer["value"] = min(5, max_val)  # Use 5 or max allowed
                else:
                    answer["value"] = min((min_val + max_val) // 2, max_val)

            sleep_answers.append(answer)

        response = client.post(
            "/api/questionnaires/answer",
            json={
                "user_id": self.user_id,
                "questionnaire_id": self.questionnaire_id,
                "answers": sleep_answers,
                "page_number": 2,
            },
        )

        assert response.status_code == 200
        data = response.json()

        assert data["status"] == "success"
        # Should now have more answers (personality + sleep)
        personality_questions = [
            q for q in self.questionnaire["questions"] if q["category"] == "personality"
        ]
        expected_answer_count = len(personality_questions) + len(sleep_questions)
        assert len(data["questionnaire"]["answers"]) == expected_answer_count

    def test_submit_behavioral_answers(self):
        """Test submitting answers to behavioral pattern questions."""
        # Start the questionnaire if not already started
        user_questionnaire_id = getattr(self, "user_questionnaire_id", None)
        if not user_questionnaire_id:
            user_questionnaire_id = self.test_start_onboarding_questionnaire()
            self.test_submit_personality_answers()  # Submit personality answers first
            self.test_submit_sleep_answers()  # Submit sleep answers second

        # Get the actual behavioral questions from our created questionnaire
        behavioral_questions = [
            q for q in self.questionnaire["questions"] if q["category"] == "behavior"
        ]

        # Submit answers to behavioral questions
        behavioral_answers = []

        for question in behavioral_questions:
            answer = {
                "question_id": question["question_id"],
                "answered_at": "2025-04-17T10:02:00.000Z",
            }

            # Add appropriate value based on question type
            if question["question_type"] == "multiple_choice":
                # Take the first option for multiple choice questions
                answer["value"] = question["options"][0]["option_id"]
            # Use mid-range value for sliders
            elif question["question_type"] == "slider":
                # Use mid-range value for sliders, but ensure we respect max bounds
                min_val = question.get("min_value", 0)
                max_val = question.get("max_value", 10)
                # For sleep_anxiety, ensure value is between 0-10
                if "sleep_anxiety" in question["dimensions"]:
                    answer["value"] = min(5, max_val)  # Use 5 or max allowed
                else:
                    answer["value"] = min((min_val + max_val) // 2, max_val)

            behavioral_answers.append(answer)

        response = client.post(
            "/api/questionnaires/answer",
            json={
                "user_id": self.user_id,
                "questionnaire_id": self.questionnaire_id,
                "answers": behavioral_answers,
                "page_number": 3,
            },
        )

        assert response.status_code == 200
        data = response.json()

        assert data["status"] == "success"
        # Should now have all answers (personality + sleep + behavioral)
        personality_questions = [
            q for q in self.questionnaire["questions"] if q["category"] == "personality"
        ]
        sleep_questions = [
            q for q in self.questionnaire["questions"] if q["category"] == "sleep"
        ]
        expected_answer_count = (
            len(personality_questions)
            + len(sleep_questions)
            + len(behavioral_questions)
        )
        assert len(data["questionnaire"]["answers"]) == expected_answer_count

    def test_complete_questionnaire(self):
        """Test completing the questionnaire and verifying profile creation."""
        # Start the questionnaire and submit all answers if not already done
        user_questionnaire_id = getattr(self, "user_questionnaire_id", None)
        if not user_questionnaire_id:
            self.test_start_onboarding_questionnaire()
            self.test_submit_personality_answers()
            self.test_submit_sleep_answers()
            self.test_submit_behavioral_answers()

        # Complete the questionnaire
        response = client.post(
            f"/api/questionnaires/complete/{self.user_questionnaire_id}"
        )

        if response.status_code != 200:
            # Print detailed error information for debugging
            print(f"Failed to complete questionnaire: {response.status_code}")
            print(f"Response body: {response.text}")

            # Get the current answers to see what might be wrong
            progress_response = client.get(
                f"/api/questionnaires/progress/{self.user_questionnaire_id}"
            )
            if progress_response.status_code == 200:
                progress_data = progress_response.json()
                if (
                    "questionnaire" in progress_data
                    and "answers" in progress_data["questionnaire"]
                ):
                    answers = progress_data["questionnaire"]["answers"]
                    for answer in answers:
                        if answer.get("question_id") == "sleep_anxiety":
                            print(f"Current sleep_anxiety answer: {answer}")

        assert (
            response.status_code == 200
        ), f"Failed to complete questionnaire: {response.text}"
        data = response.json()

        assert data["status"] == "success"
        assert data["questionnaire"]["status"] == "completed"
        assert data["questionnaire"]["completed_at"] is not None
        assert data["questionnaire"]["scored"] is True

        # Verify a psychological profile was created
        response = client.get(f"/api/profiles/user/{self.user_id}")

        assert response.status_code == 200, f"Failed to get profile: {response.text}"
        profile_data = response.json()

        assert profile_data["status"] == "success"
        assert profile_data["profile"]["user_id"] == self.user_id

        # Verify profile contains data from our questionnaire
        profile = profile_data["profile"]
        assert profile["personality_traits"] is not None
        assert profile["sleep_preferences"] is not None
        assert profile["behavioral_patterns"] is not None

        # Verify profile metadata
        assert profile["profile_metadata"]["completeness"] != "not_started"
        assert profile["profile_metadata"]["completion_percentage"] > 0
        assert profile["profile_metadata"]["questions_answered"] > 0
