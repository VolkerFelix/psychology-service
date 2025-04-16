"""Tests for the profile service."""
import os
import sys
import uuid
from datetime import datetime
from unittest.mock import MagicMock

from app.models.profile_models import (
    BehavioralPatterns,
    PersonalityDimension,
    PersonalityTraits,
    ProfileCompleteness,
    ProfileCreate,
    ProfileUpdate,
    PsychologicalProfile,
    SleepPreferences,
)
from app.services.profile_service import ProfileService

# Add the application to the python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


class TestProfileService:
    """Tests for the ProfileService class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_storage = MagicMock()
        self.service = ProfileService(storage_service=self.mock_storage)

        # Test data
        self.user_id = "test_user"
        self.profile_id = str(uuid.uuid4())

    def test_create_profile(self):
        """Test creating a new profile."""
        # Setup mock
        self.mock_storage.save_profile.return_value = True

        # Create profile
        profile_data = ProfileCreate(user_id=self.user_id)
        result = self.service.create_profile(profile_data)

        # Verify
        assert self.mock_storage.save_profile.called
        assert result.user_id == self.user_id
        assert result.profile_id is not None
        assert result.profile_metadata.completeness == ProfileCompleteness.NOT_STARTED
        assert result.profile_metadata.completion_percentage == 0

    def test_create_profile_with_traits(self):
        """Test creating a profile with personality traits."""
        # Setup mock
        self.mock_storage.save_profile.return_value = True

        # Create personality traits
        traits = PersonalityTraits(
            openness=85.0,
            conscientiousness=70.0,
            extraversion=60.0,
            agreeableness=75.0,
            neuroticism=45.0,
        )

        # Create profile
        profile_data = ProfileCreate(user_id=self.user_id, personality_traits=traits)
        result = self.service.create_profile(profile_data)

        # Verify
        assert self.mock_storage.save_profile.called
        assert result.user_id == self.user_id
        assert result.personality_traits.openness == 85.0
        assert result.personality_traits.conscientiousness == 70.0
        assert result.personality_traits.extraversion == 60.0
        assert result.personality_traits.agreeableness == 75.0
        assert result.personality_traits.neuroticism == 45.0
        assert len(result.personality_traits.dominant_traits) > 0
        assert (
            PersonalityDimension.OPENNESS in result.personality_traits.dominant_traits
        )

    def test_create_profile_with_sleep_preferences(self):
        """Test creating a profile with sleep preferences."""
        # Setup mock
        self.mock_storage.save_profile.return_value = True

        # Create sleep preferences
        sleep_prefs = SleepPreferences(
            ideal_bedtime="22:30",
            ideal_waketime="06:30",
            sleep_anxiety_level=3,
        )

        # Create profile
        profile_data = ProfileCreate(
            user_id=self.user_id, sleep_preferences=sleep_prefs
        )
        result = self.service.create_profile(profile_data)

        # Verify
        assert self.mock_storage.save_profile.called
        assert result.user_id == self.user_id
        assert result.sleep_preferences.ideal_bedtime == "22:30"
        assert result.sleep_preferences.ideal_waketime == "06:30"
        assert result.sleep_preferences.sleep_anxiety_level == 3

    def test_create_profile_with_behavioral_patterns(self):
        """Test creating a profile with behavioral patterns."""
        # Setup mock
        self.mock_storage.save_profile.return_value = True

        # Create behavioral patterns
        patterns = BehavioralPatterns(
            routine_consistency=8,
            exercise_frequency=4,
            social_activity_preference=6,
            screen_time_before_bed=45,
        )

        # Create profile
        profile_data = ProfileCreate(user_id=self.user_id, behavioral_patterns=patterns)
        result = self.service.create_profile(profile_data)

        # Verify
        assert self.mock_storage.save_profile.called
        assert result.user_id == self.user_id
        assert result.behavioral_patterns.routine_consistency == 8
        assert result.behavioral_patterns.exercise_frequency == 4
        assert result.behavioral_patterns.social_activity_preference == 6
        assert result.behavioral_patterns.screen_time_before_bed == 45

    def test_get_profile(self):
        """Test retrieving a profile by ID."""
        # Setup mock
        mock_profile = {
            "profile_id": self.profile_id,
            "user_id": self.user_id,
            "personality_traits": {"openness": 80.0},
            "sleep_preferences": {"ideal_bedtime": "23:00"},
            "behavioral_patterns": {"routine_consistency": 7},
            "profile_metadata": {
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat(),
                "completeness": "partial",
                "completion_percentage": 40,
                "questions_answered": 10,
                "source": "questionnaire",
                "valid": True,
            },
            "raw_scores": {},
        }
        self.mock_storage.get_profile.return_value = mock_profile

        # Call method
        result = self.service.get_profile(self.profile_id)

        # Verify
        self.mock_storage.get_profile.assert_called_once_with(self.profile_id)
        assert result is not None
        assert result.profile_id == self.profile_id
        assert result.user_id == self.user_id
        assert result.personality_traits.openness == 80.0
        assert result.sleep_preferences.ideal_bedtime == "23:00"
        assert result.behavioral_patterns.routine_consistency == 7
        assert result.profile_metadata.completion_percentage == 40
        assert result.profile_metadata.questions_answered == 10

    def test_get_profile_by_user(self):
        """Test retrieving a profile by user ID."""
        # Setup mock
        mock_profile = {
            "profile_id": self.profile_id,
            "user_id": self.user_id,
            "personality_traits": {"openness": 75.0},
            "sleep_preferences": {},
            "behavioral_patterns": {},
            "profile_metadata": {
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat(),
                "completeness": "not_started",
                "completion_percentage": 10,
                "questions_answered": 3,
                "source": "manual",
                "valid": False,
            },
            "raw_scores": {},
        }
        self.mock_storage.get_profile_by_user.return_value = mock_profile

        # Call method
        result = self.service.get_profile_by_user(self.user_id)

        # Verify
        self.mock_storage.get_profile_by_user.assert_called_once_with(self.user_id)
        assert result is not None
        assert result.profile_id == self.profile_id
        assert result.user_id == self.user_id
        assert result.personality_traits.openness == 75.0

    def test_update_profile(self):
        """Test updating a profile."""
        # Setup mock
        existing_profile = {
            "profile_id": self.profile_id,
            "user_id": self.user_id,
            "personality_traits": {"openness": 70.0},
            "sleep_preferences": {"ideal_bedtime": "22:00"},
            "behavioral_patterns": {"routine_consistency": 5},
            "profile_metadata": {
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat(),
                "completeness": "partial",
                "completion_percentage": 30,
                "questions_answered": 8,
                "source": "questionnaire",
                "valid": False,
            },
            "raw_scores": {},
        }
        self.mock_storage.get_profile.return_value = existing_profile
        self.mock_storage.save_profile.return_value = True

        # Create update data
        new_traits = PersonalityTraits(
            openness=80.0,
            conscientiousness=75.0,
            extraversion=65.0,
        )
        update_data = ProfileUpdate(personality_traits=new_traits)

        # Call method
        result = self.service.update_profile(self.profile_id, update_data)

        # Verify
        assert result is not None
        assert result.profile_id == self.profile_id
        assert result.personality_traits.openness == 80.0
        assert result.personality_traits.conscientiousness == 75.0
        assert result.personality_traits.extraversion == 65.0
        # Existing data should be preserved
        assert result.sleep_preferences.ideal_bedtime == "22:00"
        assert result.behavioral_patterns.routine_consistency == 5

    def test_delete_profile(self):
        """Test deleting a profile."""
        # Setup mock
        self.mock_storage.delete_profile.return_value = True

        # Call method
        result = self.service.delete_profile(self.profile_id)

        # Verify
        self.mock_storage.delete_profile.assert_called_once_with(self.profile_id)
        assert result is True

    def test_list_profiles(self):
        """Test listing profiles with filters."""
        # Setup mock
        mock_profiles = [
            {
                "profile_id": str(uuid.uuid4()),
                "user_id": f"user_{i}",
                "personality_traits": {"openness": 70.0 + i},
                "sleep_preferences": {},
                "behavioral_patterns": {},
                "profile_metadata": {
                    "created_at": datetime.now().isoformat(),
                    "updated_at": datetime.now().isoformat(),
                    "completeness": "partial",
                    "completion_percentage": 30,
                    "questions_answered": 8,
                    "source": "questionnaire",
                    "valid": False,
                },
                "raw_scores": {},
            }
            for i in range(3)
        ]
        self.mock_storage.get_profiles.return_value = mock_profiles

        # Call method
        filters = {"cluster_id": "test_cluster"}
        result = self.service.list_profiles(limit=10, offset=0, filters=filters)

        # Verify
        self.mock_storage.get_profiles.assert_called_once_with(
            limit=10, offset=0, filters=filters
        )
        assert len(result) == 3
        assert all(isinstance(profile, PsychologicalProfile) for profile in result)

    def test_update_profile_from_questionnaire(self):
        """Test updating a profile from questionnaire results."""
        # Setup mock
        existing_profile = {
            "profile_id": self.profile_id,
            "user_id": self.user_id,
            "personality_traits": {"openness": 70.0},
            "sleep_preferences": {},
            "behavioral_patterns": {},
            "metadata": {  # Note: using 'metadata' instead of 'profile_metadata'
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat(),
                "completeness": "partial",
                "completion_percentage": 30,
                "questions_answered": 8,
                "source": "questionnaire",
                "valid": False,
            },
            "raw_scores": {},
        }
        self.mock_storage.get_profile_by_user.return_value = existing_profile
        self.mock_storage.save_profile.return_value = True

        # Create questionnaire results
        questionnaire_results = {
            "scores": {"q1": 5, "q2": 3, "q3": 4},
            "dimension_scores": {
                "openness": 85.0,
                "conscientiousness": 75.0,
                "sleep_anxiety": 4.0,
            },
            "questions_answered": 12,
        }

        # Call method
        result = self.service.update_profile_from_questionnaire(
            self.user_id, questionnaire_results
        )

        # Verify
        assert self.mock_storage.save_profile.called
        # A profile should be returned, but since the mock returns the old data
        # we can't easily verify the content was updated
        assert result is not None

    def test_calculate_dominant_traits(self):
        """Test calculation of dominant personality traits."""
        # Setup profile with traits
        profile = {
            "personality_traits": {
                "openness": 85.0,  # High
                "conscientiousness": 75.0,  # High
                "extraversion": 60.0,  # Medium
                "agreeableness": 55.0,  # Medium
                "neuroticism": 40.0,  # Low
            }
        }

        # Call method
        self.service._calculate_dominant_traits(profile)

        # Verify
        traits = profile["personality_traits"]
        assert "dominant_traits" in traits
        assert len(traits["dominant_traits"]) <= 2  # Max 2 dominant traits
        assert "openness" in traits["dominant_traits"]  # Should be the highest
        assert "conscientiousness" in traits["dominant_traits"]  # Second highest

    def test_check_section_completeness(self):
        """Test checking completeness of a profile section."""
        # Test complete section
        complete_section = {
            "openness": 85.0,
            "conscientiousness": 75.0,
            "extraversion": 60.0,
            "agreeableness": 55.0,
            "neuroticism": 40.0,
        }
        required_fields = [
            "openness",
            "conscientiousness",
            "extraversion",
            "agreeableness",
            "neuroticism",
        ]
        completeness = self.service._check_section_completeness(
            complete_section, required_fields
        )
        assert completeness == 1.0  # Fully complete

        # Test partially complete section
        partial_section = {
            "openness": 85.0,
            "conscientiousness": 75.0,
            "extraversion": None,
            "agreeableness": None,
            "neuroticism": 40.0,
        }
        completeness = self.service._check_section_completeness(
            partial_section, required_fields
        )
        assert completeness == 0.6  # 3 out of 5 fields

        # Test empty section
        empty_section = {}
        completeness = self.service._check_section_completeness(
            empty_section, required_fields
        )
        assert completeness == 0.0  # No fields

    def test_update_profile_completeness(self):
        """Test updating the completeness status of a profile."""
        # Test with relatively empty profile
        profile = {
            "personality_traits": {"openness": 70.0},
            "sleep_preferences": {},
            "behavioral_patterns": {},
            "profile_metadata": {
                "questions_answered": 3,
            },
        }
        updates = {}
        self.service._update_profile_completeness(profile, updates)

        # Verify
        assert "profile_metadata" in updates
        assert "completeness" in updates["profile_metadata"]
        assert "completion_percentage" in updates["profile_metadata"]
        assert "valid" in updates["profile_metadata"]
        assert (
            updates["profile_metadata"]["completion_percentage"] < 50
        )  # Low completion
        assert not updates["profile_metadata"]["valid"]  # Not valid with so little data

        # Test with more complete profile
        profile = {
            "personality_traits": {
                "openness": 85.0,
                "conscientiousness": 75.0,
                "extraversion": 60.0,
                "agreeableness": 55.0,
                "neuroticism": 40.0,
            },
            "sleep_preferences": {
                "chronotype": "morning_person",
                "environment_preference": "dark_quiet",
                "sleep_anxiety_level": 3,
            },
            "behavioral_patterns": {
                "stress_response": "problem_focused",
                "routine_consistency": 8,
                "social_activity_preference": 6,
            },
            "profile_metadata": {
                "questions_answered": 20,
            },
        }
        updates = {}
        self.service._update_profile_completeness(profile, updates)

        # Verify
        assert (
            updates["profile_metadata"]["completion_percentage"] > 80
        )  # High completion
        assert updates["profile_metadata"]["valid"]  # Valid with comprehensive data
