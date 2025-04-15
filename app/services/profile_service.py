"""Service for managing psychological profiles."""

import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from app.models.profile_models import (
    BehavioralPatterns,
    PersonalityTraits,
    ProfileCompleteness,
    ProfileCreate,
    ProfileMetadata,
    ProfileUpdate,
    PsychologicalProfile,
    SleepPreferences,
)


class ProfileService:
    """Service for managing psychological profiles."""

    def __init__(self, storage_service):
        """Initialize with a storage service."""
        self.storage = storage_service

    def create_profile(self, profile_data: ProfileCreate) -> PsychologicalProfile:
        """
        Create a new psychological profile.

        Args:
            profile_data: Profile creation data

        Returns:
            Created profile
        """
        # Generate ID and set initial values
        profile_id = str(uuid.uuid4())
        now = datetime.now()

        # Create metadata
        metadata = ProfileMetadata(
            created_at=now,
            updated_at=now,
            completeness=ProfileCompleteness.NOT_STARTED,
            completion_percentage=0,
            questions_answered=0,
            source="manual",
            valid=False,
            last_questionnaire_id=None,
        )

        # Build complete profile data
        profile = PsychologicalProfile(
            profile_id=profile_id,
            user_id=profile_data.user_id,
            personality_traits=profile_data.personality_traits
            or PersonalityTraits(
                openness=None,
                conscientiousness=None,
                extraversion=None,
                agreeableness=None,
                neuroticism=None,
                dominant_traits=None,
            ),
            sleep_preferences=profile_data.sleep_preferences
            or SleepPreferences(
                chronotype=None,
                ideal_bedtime=None,
                ideal_waketime=None,
                environment_preference=None,
                sleep_anxiety_level=None,
                relaxation_techniques=None,
            ),
            behavioral_patterns=profile_data.behavioral_patterns
            or BehavioralPatterns(
                stress_response=None,
                routine_consistency=None,
                exercise_frequency=None,
                social_activity_preference=None,
                screen_time_before_bed=None,
                typical_stress_level=None,
            ),
            raw_scores=profile_data.raw_scores or {},
            profile_metadata=metadata,
            cluster_info=None,
        )

        # Save to storage
        success = self.storage.save_profile(profile.dict())
        if not success:
            raise Exception("Failed to save profile to storage")

        # Return complete profile object
        return profile

    def get_profile(self, profile_id: str) -> Optional[PsychologicalProfile]:
        """
        Get a profile by ID.

        Args:
            profile_id: Profile ID

        Returns:
            Profile if found, None otherwise
        """
        profile_data = self.storage.get_profile(profile_id)
        if not profile_data:
            return None

        return PsychologicalProfile(**profile_data)

    def get_profile_by_user(self, user_id: str) -> Optional[PsychologicalProfile]:
        """
        Get a profile by user ID.

        Args:
            user_id: User ID

        Returns:
            Profile if found, None otherwise
        """
        profile_data = self.storage.get_profile_by_user(user_id)
        if not profile_data:
            return None

        return PsychologicalProfile(**profile_data)

    def update_profile(
        self, profile_id: str, profile_update: ProfileUpdate
    ) -> Optional[PsychologicalProfile]:
        """
        Update an existing profile.

        Args:
            profile_id: Profile ID to update
            profile_update: Profile update data

        Returns:
            Updated profile if successful, None otherwise
        """
        # Get existing profile
        existing_profile = self.storage.get_profile(profile_id)
        if not existing_profile:
            return None

        # Build update dictionary (only non-None fields)
        update_dict = {}
        if profile_update.personality_traits:
            update_dict["personality_traits"] = profile_update.personality_traits.dict()
        if profile_update.sleep_preferences:
            update_dict["sleep_preferences"] = profile_update.sleep_preferences.dict()
        if profile_update.behavioral_patterns:
            update_dict[
                "behavioral_patterns"
            ] = profile_update.behavioral_patterns.dict()
        if profile_update.raw_scores:
            update_dict["raw_scores"] = profile_update.raw_scores
        if profile_update.profile_metadata:
            update_dict["profile_metadata"] = profile_update.profile_metadata.dict()

        # Update timestamps
        if "profile_metadata" not in update_dict:
            update_dict["profile_metadata"] = existing_profile.get(
                "profile_metadata", {}
            )
        update_dict["profile_metadata"]["updated_at"] = datetime.now().isoformat()

        # Update completeness if not already set
        self._update_profile_completeness(existing_profile, update_dict)

        # Merge with existing profile
        updated_profile = {**existing_profile, **update_dict}

        # Save to storage
        success = self.storage.save_profile(updated_profile)
        if not success:
            raise Exception("Failed to save updated profile to storage")

        # Return updated profile
        return PsychologicalProfile(**updated_profile)

    def delete_profile(self, profile_id: str) -> bool:
        """
        Delete a profile by ID.

        Args:
            profile_id: Profile ID

        Returns:
            True if deleted, False otherwise
        """
        return self.storage.delete_profile(profile_id)

    def list_profiles(
        self,
        limit: int = 100,
        offset: int = 0,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[PsychologicalProfile]:
        """
        List profiles with optional filtering.

        Args:
            limit: Maximum number of profiles to return
            offset: Number of profiles to skip
            filters: Optional filters to apply

        Returns:
            List of profiles
        """
        profiles_data = self.storage.get_profiles(
            limit=limit, offset=offset, filters=filters
        )
        return [PsychologicalProfile(**profile) for profile in profiles_data]

    def update_profile_from_questionnaire(
        self, user_id: str, questionnaire_results: Dict[str, Any]
    ) -> Optional[PsychologicalProfile]:
        """
        Update a profile based on questionnaire results.

        Args:
            user_id: User ID
            questionnaire_results: Results from a completed questionnaire

        Returns:
            Updated profile if successful, None otherwise
        """
        # Get existing profile or create new one
        profile = self.storage.get_profile_by_user(user_id)

        if not profile:
            # Create new profile
            profile = {
                "profile_id": str(uuid.uuid4()),
                "user_id": user_id,
                "personality_traits": {},
                "sleep_preferences": {},
                "behavioral_patterns": {},
                "metadata": {
                    "created_at": datetime.now().isoformat(),
                    "updated_at": datetime.now().isoformat(),
                    "completeness": ProfileCompleteness.PARTIAL,
                    "completion_percentage": 0,
                    "questions_answered": 0,
                    "source": "questionnaire",
                    "valid": False,
                },
                "raw_scores": {},
            }

        # Update raw scores
        if "raw_scores" not in profile:
            profile["raw_scores"] = {}

        if "scores" in questionnaire_results:
            profile["raw_scores"].update(questionnaire_results["scores"])

        # Update specific trait components based on dimensions in results
        self._update_profile_from_scores(profile, questionnaire_results)

        # Update metadata
        if "metadata" not in profile:
            profile["metadata"] = {}

        profile["metadata"]["updated_at"] = datetime.now().isoformat()
        profile["metadata"]["source"] = "questionnaire"

        if "questions_answered" in questionnaire_results:
            profile["metadata"]["questions_answered"] = profile["metadata"].get(
                "questions_answered", 0
            ) + questionnaire_results.get("questions_answered", 0)

        # Update completeness
        self._update_profile_completeness(profile, {})

        # Save to storage
        success = self.storage.save_profile(profile)
        if not success:
            raise Exception("Failed to save profile updated from questionnaire")

        # Return updated profile
        return PsychologicalProfile(**profile)

    def _update_profile_from_scores(
        self, profile: Dict[str, Any], questionnaire_results: Dict[str, Any]
    ) -> None:
        """
        Update profile traits based on questionnaire scores.

        Args:
            profile: Existing profile dictionary to update
            questionnaire_results: Results from a completed questionnaire
        """
        dimension_mapping = {
            # Personality traits
            "openness": ("personality_traits", "openness"),
            "conscientiousness": ("personality_traits", "conscientiousness"),
            "extraversion": ("personality_traits", "extraversion"),
            "agreeableness": ("personality_traits", "agreeableness"),
            "neuroticism": ("personality_traits", "neuroticism"),
            # Sleep preferences
            "chronotype": ("sleep_preferences", "chronotype"),
            "sleep_environment": ("sleep_preferences", "environment_preference"),
            "sleep_anxiety": ("sleep_preferences", "sleep_anxiety_level"),
            # Behavioral patterns
            "stress_response": ("behavioral_patterns", "stress_response"),
            "routine_consistency": ("behavioral_patterns", "routine_consistency"),
            "activity_preference": (
                "behavioral_patterns",
                "social_activity_preference",
            ),
            "screen_time": ("behavioral_patterns", "screen_time_before_bed"),
        }

        # Get dimension scores from results
        dimension_scores = questionnaire_results.get("dimension_scores", {})

        # Update profile dimensions
        for dimension, score in dimension_scores.items():
            if dimension in dimension_mapping:
                category, field = dimension_mapping[dimension]

                # Ensure category exists
                if category not in profile:
                    profile[category] = {}

                # Update field with score
                profile[category][field] = score

        # Calculate dominant traits for personality if we have enough data
        self._calculate_dominant_traits(profile)

    def _calculate_dominant_traits(self, profile: Dict[str, Any]) -> None:
        """
        Calculate and set dominant personality traits.

        Args:
            profile: Profile dictionary to update
        """
        if "personality_traits" not in profile:
            return

        traits = profile["personality_traits"]
        if not traits:
            return

        # Check which traits have scores
        trait_scores = {}
        for trait in [
            "openness",
            "conscientiousness",
            "extraversion",
            "agreeableness",
            "neuroticism",
        ]:
            if trait in traits and traits[trait] is not None:
                trait_scores[trait] = traits[trait]

        # If we have at least 3 traits scored, calculate dominant traits
        if len(trait_scores) >= 3:
            # Sort by score (descending)
            sorted_traits = sorted(
                trait_scores.items(), key=lambda item: item[1], reverse=True
            )

            # Take top 2 traits with scores >= 70
            dominant_traits = [trait for trait, score in sorted_traits if score >= 70]

            # If less than 2 traits with high scores, take top 2 regardless
            if len(dominant_traits) < 2:
                dominant_traits = [trait for trait, _ in sorted_traits[:2]]

            # Set dominant traits
            traits["dominant_traits"] = dominant_traits

    def _update_profile_completeness(
        self, profile: Dict[str, Any], updates: Dict[str, Any]
    ) -> None:
        """
        Update profile completeness status and percentage.

        Args:
            profile: Existing profile dictionary
            updates: Dictionary of updates being applied
        """
        # Combine profile with updates for calculation
        combined = {**profile}
        if "metadata" in updates:
            combined["metadata"] = {
                **(combined.get("metadata") or {}),
                **updates["metadata"],
            }

        # Get metadata
        metadata = combined.get("metadata", {})

        # Check completeness of each section
        completeness_checks = {
            "personality_traits": self._check_section_completeness(
                combined.get("personality_traits", {}),
                [
                    "openness",
                    "conscientiousness",
                    "extraversion",
                    "agreeableness",
                    "neuroticism",
                ],
            ),
            "sleep_preferences": self._check_section_completeness(
                combined.get("sleep_preferences", {}),
                ["chronotype", "environment_preference", "sleep_anxiety_level"],
            ),
            "behavioral_patterns": self._check_section_completeness(
                combined.get("behavioral_patterns", {}),
                [
                    "stress_response",
                    "routine_consistency",
                    "social_activity_preference",
                ],
            ),
        }

        # Calculate overall completion percentage
        section_weights = {
            "personality_traits": 0.4,
            "sleep_preferences": 0.3,
            "behavioral_patterns": 0.3,
        }
        completion_percentage = sum(
            completeness_checks[section] * section_weights[section]
            for section in completeness_checks
        )
        completion_percentage = min(round(completion_percentage * 100), 100)

        # Determine completeness status
        if completion_percentage < 10:
            completeness = ProfileCompleteness.NOT_STARTED
        elif completion_percentage < 80:
            completeness = ProfileCompleteness.PARTIAL
        else:
            completeness = ProfileCompleteness.COMPLETE

        # Set validity
        min_questions = 15  # Minimum questions for a valid profile
        questions_answered = metadata.get("questions_answered", 0)
        valid = questions_answered >= min_questions and completion_percentage >= 60

        # Update metadata
        if "metadata" not in updates:
            updates["metadata"] = {}

        updates["metadata"]["completion_percentage"] = completion_percentage
        updates["metadata"]["completeness"] = completeness
        updates["metadata"]["valid"] = valid

    @staticmethod
    def _check_section_completeness(
        section: Dict[str, Any], required_fields: List[str]
    ) -> float:
        """
        Check how complete a section of the profile is.

        Args:
            section: Section dictionary
            required_fields: List of fields that should be present

        Returns:
            Completeness as a float from 0.0 to 1.0
        """
        if not section:
            return 0.0

        # Count how many required fields have values
        filled_fields = sum(
            1
            for field in required_fields
            if field in section and section[field] is not None
        )

        # Return percentage complete
        return filled_fields / len(required_fields) if required_fields else 0.0
