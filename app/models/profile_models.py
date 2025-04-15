"""Models for psychological profiles."""
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, validator


class PersonalityDimension(str, Enum):
    """Common personality dimensions based on Big Five model."""

    OPENNESS = "openness"
    CONSCIENTIOUSNESS = "conscientiousness"
    EXTRAVERSION = "extraversion"
    AGREEABLENESS = "agreeableness"
    NEUROTICISM = "neuroticism"


class Chronotype(str, Enum):
    """Sleep chronotype preferences."""

    MORNING_PERSON = "morning_person"
    EVENING_PERSON = "evening_person"
    INTERMEDIATE = "intermediate"
    VARIABLE = "variable"


class SleepEnvironmentPreference(str, Enum):
    """Preferred sleep environment."""

    DARK_QUIET = "dark_quiet"
    SOME_LIGHT = "some_light"
    SOME_NOISE = "some_noise"
    NOISE_AND_LIGHT = "noise_and_light"


class StressResponseStyle(str, Enum):
    """How people typically respond to stress."""

    PROBLEM_FOCUSED = "problem_focused"
    EMOTION_FOCUSED = "emotion_focused"
    AVOIDANT = "avoidant"
    SOCIAL_SUPPORT = "social_support"
    MIXED = "mixed"


class ProfileCompleteness(str, Enum):
    """Profile completeness status."""

    NOT_STARTED = "not_started"
    PARTIAL = "partial"
    COMPLETE = "complete"


class PersonalityTraits(BaseModel):
    """Personality traits based on psychological models."""

    openness: Optional[float] = Field(
        None, ge=0, le=100, description="Openness to experience score (0-100)"
    )
    conscientiousness: Optional[float] = Field(
        None, ge=0, le=100, description="Conscientiousness score (0-100)"
    )
    extraversion: Optional[float] = Field(
        None, ge=0, le=100, description="Extraversion score (0-100)"
    )
    agreeableness: Optional[float] = Field(
        None, ge=0, le=100, description="Agreeableness score (0-100)"
    )
    neuroticism: Optional[float] = Field(
        None, ge=0, le=100, description="Neuroticism score (0-100)"
    )
    dominant_traits: Optional[List[PersonalityDimension]] = Field(
        None, description="Dominant personality traits (highest scores)"
    )

    @validator("dominant_traits", always=True)
    def calculate_dominant_traits(cls, v, values):
        """Calculate dominant traits based on scores if not provided."""
        if v is not None:
            return v

        traits = {}
        for trait in PersonalityDimension:
            score = values.get(trait.value)
            if score is not None and score >= 70:  # Threshold for dominant trait
                traits[trait] = score

        # Return top 2 traits or fewer if not enough high scores
        return sorted(traits.keys(), key=lambda t: traits[t], reverse=True)[:2]


class SleepPreferences(BaseModel):
    """User's sleep preferences and patterns."""

    chronotype: Optional[Chronotype] = Field(
        None, description="Morning/evening preference"
    )
    ideal_bedtime: Optional[str] = Field(
        None, description="Ideal bedtime (HH:MM format)"
    )
    ideal_waketime: Optional[str] = Field(
        None, description="Ideal waketime (HH:MM format)"
    )
    environment_preference: Optional[SleepEnvironmentPreference] = Field(
        None, description="Preferred sleep environment"
    )
    sleep_anxiety_level: Optional[int] = Field(
        None, ge=0, le=10, description="Sleep anxiety level (0-10)"
    )
    relaxation_techniques: Optional[List[str]] = Field(
        None, description="Relaxation techniques that help with sleep"
    )


class BehavioralPatterns(BaseModel):
    """User's behavioral patterns and habits."""

    stress_response: Optional[StressResponseStyle] = Field(
        None, description="Typical stress response style"
    )
    routine_consistency: Optional[int] = Field(
        None, ge=0, le=10, description="Consistency of daily routine (0-10)"
    )
    exercise_frequency: Optional[int] = Field(
        None, ge=0, le=7, description="Exercise days per week (0-7)"
    )
    social_activity_preference: Optional[int] = Field(
        None, ge=0, le=10, description="Preference for social activities (0-10)"
    )
    screen_time_before_bed: Optional[int] = Field(
        None, ge=0, description="Average screen time before bed (minutes)"
    )
    typical_stress_level: Optional[int] = Field(
        None, ge=0, le=10, description="Typical daily stress level (0-10)"
    )


class ClusterInfo(BaseModel):
    """Information about the user's assigned psychological cluster."""

    cluster_id: str = Field(..., description="Unique identifier for the cluster")
    cluster_name: str = Field(..., description="Descriptive name for the cluster")
    cluster_description: str = Field(
        ..., description="Description of the cluster's characteristics"
    )
    confidence_score: float = Field(
        ..., ge=0, le=1, description="Confidence in cluster assignment (0-1)"
    )
    assigned_at: datetime = Field(
        ..., description="When the user was assigned to this cluster"
    )
    similar_users_count: Optional[int] = Field(
        None, description="Number of users in the same cluster"
    )


class ProfileMetadata(BaseModel):
    """Metadata about the psychological profile."""

    created_at: datetime = Field(
        default_factory=datetime.now, description="When the profile was created"
    )
    updated_at: datetime = Field(
        default_factory=datetime.now, description="When the profile was last updated"
    )
    completeness: ProfileCompleteness = Field(
        default=ProfileCompleteness.NOT_STARTED,
        description="Profile completeness status",
    )
    completion_percentage: int = Field(
        default=0, ge=0, le=100, description="Profile completion percentage"
    )
    questions_answered: int = Field(
        default=0, description="Number of questions answered"
    )
    last_questionnaire_id: Optional[str] = Field(
        None, description="ID of the last questionnaire completed or started"
    )
    source: str = Field(default="onboarding", description="Source of the profile data")
    valid: bool = Field(
        default=False, description="Whether the profile is considered valid"
    )


class PsychologicalProfile(BaseModel):
    """Complete psychological profile for a user."""

    profile_id: str = Field(..., description="Unique identifier for the profile")
    user_id: str = Field(..., description="User identifier")
    personality_traits: PersonalityTraits = Field(
        default_factory=PersonalityTraits, description="Personality trait scores"
    )
    sleep_preferences: SleepPreferences = Field(
        default_factory=SleepPreferences, description="Sleep preferences and patterns"
    )
    behavioral_patterns: BehavioralPatterns = Field(
        default_factory=BehavioralPatterns, description="Behavioral patterns and habits"
    )
    cluster_info: Optional[ClusterInfo] = Field(
        None, description="Information about assigned psychological cluster"
    )
    profile_metadata: ProfileMetadata = Field(
        default_factory=ProfileMetadata, description="Profile metadata"
    )
    raw_scores: Optional[Dict[str, Any]] = Field(
        None, description="Raw scores from questionnaires"
    )


class ProfileCreate(BaseModel):
    """Model for creating a new psychological profile."""

    user_id: str = Field(..., description="User identifier")
    personality_traits: Optional[PersonalityTraits] = None
    sleep_preferences: Optional[SleepPreferences] = None
    behavioral_patterns: Optional[BehavioralPatterns] = None
    raw_scores: Optional[Dict[str, Any]] = None


class ProfileUpdate(BaseModel):
    """Model for updating an existing psychological profile."""

    personality_traits: Optional[PersonalityTraits] = None
    sleep_preferences: Optional[SleepPreferences] = None
    behavioral_patterns: Optional[BehavioralPatterns] = None
    raw_scores: Optional[Dict[str, Any]] = None
    profile_metadata: Optional[ProfileMetadata] = None


class ProfileResponse(BaseModel):
    """Response model for profile operations."""

    profile: PsychologicalProfile
    status: str = "success"
    message: Optional[str] = None


class ProfilesResponse(BaseModel):
    """Response model for multiple profiles."""

    profiles: List[PsychologicalProfile]
    count: int
    status: str = "success"
