"""Add questionnaires for onboarding and visualization.

Revision ID: 003
Revises: 002
Create Date: 2025-04-17 10:00:00.000000

This migration adds:
1. A combined onboarding questionnaire
2. Individual questionnaires for each type (personality, sleep_habits, behavioral)
to support both the onboarding process and visualization requirements.
"""

import json
import uuid
from datetime import datetime

from alembic import op
from sqlalchemy import text

# revision identifiers, used by Alembic.
revision = "003"
down_revision = "002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Add questionnaires for onboarding and visualization."""
    # Get current time for timestamps
    now = datetime.now().isoformat()

    # Create question sets
    personality_questions = get_personality_questions()
    sleep_questions = get_sleep_questions()
    behavioral_questions = get_behavioral_questions()

    # Combined questions for the comprehensive onboarding questionnaire
    all_questions = personality_questions + sleep_questions + behavioral_questions

    # ---------- Create the onboarding questionnaire (combined) ----------
    onboarding_id = str(uuid.uuid4())
    op.get_bind().execute(
        text(
            """
            INSERT INTO questionnaires (
                questionnaire_id, title, description, questionnaire_type,
                questions, created_at, updated_at, version, is_active,
                estimated_duration_minutes, tags
            ) VALUES (
                :questionnaire_id, :title, :description, :questionnaire_type,
                :questions, :created_at, :updated_at, :version, :is_active,
                :estimated_duration_minutes, :tags
            )
            """
        ).bindparams(
            questionnaire_id=onboarding_id,
            title="Onboarding Psychological Profile Questionnaire",
            description="""This questionnaire helps us understand your
            personality traits, sleep preferences, and
            behavioral patterns to provide personalized
            recommendations.""",
            questionnaire_type="onboarding",
            questions=json.dumps(all_questions),
            created_at=now,
            updated_at=now,
            version="1.0.0",
            is_active=True,
            estimated_duration_minutes=15,
            tags=json.dumps(["onboarding", "personality", "sleep", "behavior"]),
        )
    )

    # ---------- Create individual questionnaires for visualization app ----------

    # Personality questionnaire
    personality_id = str(uuid.uuid4())
    op.get_bind().execute(
        text(
            """
            INSERT INTO questionnaires (
                questionnaire_id, title, description, questionnaire_type,
                questions, created_at, updated_at, version, is_active,
                estimated_duration_minutes, tags
            ) VALUES (
                :questionnaire_id, :title, :description, :questionnaire_type,
                :questions, :created_at, :updated_at, :version, :is_active,
                :estimated_duration_minutes, :tags
            )
            """
        ).bindparams(
            questionnaire_id=personality_id,
            title="Personality Assessment",
            description="""This questionnaire evaluates your
            personality traits based on the Big Five model.""",
            questionnaire_type="personality",
            questions=json.dumps(personality_questions),
            created_at=now,
            updated_at=now,
            version="1.0.0",
            is_active=True,
            estimated_duration_minutes=10,
            tags=json.dumps(["personality", "big five"]),
        )
    )

    # Sleep questionnaire
    sleep_id = str(uuid.uuid4())
    op.get_bind().execute(
        text(
            """
            INSERT INTO questionnaires (
                questionnaire_id, title, description, questionnaire_type,
                questions, created_at, updated_at, version, is_active,
                estimated_duration_minutes, tags
            ) VALUES (
                :questionnaire_id, :title, :description, :questionnaire_type,
                :questions, :created_at, :updated_at, :version, :is_active,
                :estimated_duration_minutes, :tags
            )
            """
        ).bindparams(
            questionnaire_id=sleep_id,
            title="Sleep Habits Assessment",
            description="""This questionnaire evaluates your
            sleep preferences and habits.""",
            questionnaire_type="sleep_habits",
            questions=json.dumps(sleep_questions),
            created_at=now,
            updated_at=now,
            version="1.0.0",
            is_active=True,
            estimated_duration_minutes=8,
            tags=json.dumps(["sleep", "chronotype"]),
        )
    )

    # Behavioral questionnaire
    behavioral_id = str(uuid.uuid4())
    op.get_bind().execute(
        text(
            """
            INSERT INTO questionnaires (
                questionnaire_id, title, description, questionnaire_type,
                questions, created_at, updated_at, version, is_active,
                estimated_duration_minutes, tags
            ) VALUES (
                :questionnaire_id, :title, :description, :questionnaire_type,
                :questions, :created_at, :updated_at, :version, :is_active,
                :estimated_duration_minutes, :tags
            )
            """
        ).bindparams(
            questionnaire_id=behavioral_id,
            title="Behavioral Patterns Assessment",
            description="This questionnaire evaluates your daily behavioral patterns.",
            questionnaire_type="behavioral",
            questions=json.dumps(behavioral_questions),
            created_at=now,
            updated_at=now,
            version="1.0.0",
            is_active=True,
            estimated_duration_minutes=7,
            tags=json.dumps(["behavior", "habits"]),
        )
    )

    # Log the created questionnaire IDs for reference
    print(f"Created onboarding questionnaire with ID: {onboarding_id}")
    print(f"Created personality questionnaire with ID: {personality_id}")
    print(f"Created sleep questionnaire with ID: {sleep_id}")
    print(f"Created behavioral questionnaire with ID: {behavioral_id}")


def downgrade() -> None:
    """Remove the added questionnaires."""
    # Delete all onboarding questionnaires
    op.execute(
        """
        DELETE FROM questionnaires
        WHERE questionnaire_type = 'onboarding' AND
        title = 'Onboarding Psychological Profile Questionnaire'
        """
    )

    # Delete individual questionnaires
    op.get_bind().execute(
        text(
            """
            DELETE FROM questionnaires
            WHERE questionnaire_type IN ('personality', 'sleep_habits', 'behavioral')
            AND title IN ('Personality Assessment', 'Sleep Habits Assessment', 'Behavioral Patterns Assessment')
            """  # noqa: E501
        )
    )


def get_personality_questions():
    """Get questions for personality assessment."""
    return [
        {
            "question_id": f"personality_{i}",
            "text": question["text"],
            "description": question["description"],
            "question_type": "likert_5",
            "category": "personality",
            "dimensions": [question["dimension"]],
            "required": True,
            "weight": 1.0,
        }
        for i, question in enumerate(
            [
                {
                    "dimension": "openness",
                    "text": "I enjoy trying new experiences and activities",
                    "description": "Rate how much you agree with this statement",
                },
                {
                    "dimension": "openness",
                    "text": "I am curious about many different things",
                    "description": "Rate how much you agree with this statement",
                },
                {
                    "dimension": "conscientiousness",
                    "text": "I am organized and keep things neat",
                    "description": "Rate how much you agree with this statement",
                },
                {
                    "dimension": "conscientiousness",
                    "text": "I plan ahead and follow through with my plans",
                    "description": "Rate how much you agree with this statement",
                },
                {
                    "dimension": "extraversion",
                    "text": "I enjoy being around people and socializing",
                    "description": "Rate how much you agree with this statement",
                },
                {
                    "dimension": "extraversion",
                    "text": "I am talkative and express my opinions freely",
                    "description": "Rate how much you agree with this statement",
                },
                {
                    "dimension": "agreeableness",
                    "text": "I am considerate and kind to almost everyone",
                    "description": "Rate how much you agree with this statement",
                },
                {
                    "dimension": "agreeableness",
                    "text": "I like to cooperate with others",
                    "description": "Rate how much you agree with this statement",
                },
                {
                    "dimension": "neuroticism",
                    "text": "I worry about things frequently",
                    "description": "Rate how much you agree with this statement",
                },
                {
                    "dimension": "neuroticism",
                    "text": "I am easily stressed or anxious",
                    "description": "Rate how much you agree with this statement",
                },
            ]
        )
    ]


def get_sleep_questions():
    """Get questions for sleep assessment."""
    return [
        {
            "question_id": "sleep_chronotype",
            "text": "When do you naturally prefer to go to sleep?",
            "description": """Select the time that best matches your
            preference, not when you actually go to bed""",
            "question_type": "multiple_choice",
            "category": "sleep",
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
                {
                    "option_id": "variable",
                    "text": "It varies significantly day to day",
                    "value": "variable",
                },
            ],
            "required": True,
            "weight": 1.0,
        },
        {
            "question_id": "sleep_environment",
            "text": "What is your preferred sleep environment?",
            "description": "Select the environment that helps you sleep best",
            "question_type": "multiple_choice",
            "category": "sleep",
            "dimensions": ["sleep_environment"],
            "options": [
                {
                    "option_id": "dark_quiet",
                    "text": "Completely dark and quiet",
                    "value": "dark_quiet",
                },
                {
                    "option_id": "some_light",
                    "text": "Some light (night light, dim lamp)",
                    "value": "some_light",
                },
                {
                    "option_id": "some_noise",
                    "text": "Some noise (fan, white noise)",
                    "value": "some_noise",
                },
                {
                    "option_id": "noise_light",
                    "text": "Both some noise and light",
                    "value": "noise_and_light",
                },
            ],
            "required": True,
            "weight": 1.0,
        },
        {
            "question_id": "sleep_anxiety",
            "text": "How often do you worry about not getting enough sleep?",
            "description": "Rate on a scale from 0 (never) to 10 (constantly)",
            "question_type": "slider",
            "category": "sleep",
            "dimensions": ["sleep_anxiety"],
            "min_value": 0,
            "max_value": 10,
            "step": 1,
            "required": True,
            "weight": 1.0,
        },
        {
            "question_id": "bedtime_routine",
            "text": "What relaxation techniques help you fall asleep?",
            "description": "Select all that apply",
            "question_type": "checkbox",
            "category": "sleep",
            "dimensions": ["relaxation_techniques"],
            "options": [
                {"option_id": "reading", "text": "Reading", "value": "reading"},
                {
                    "option_id": "meditation",
                    "text": "Meditation or breathing exercises",
                    "value": "meditation",
                },
                {
                    "option_id": "music",
                    "text": "Listening to calming music or sounds",
                    "value": "music",
                },
                {
                    "option_id": "stretching",
                    "text": "Stretching or gentle yoga",
                    "value": "stretching",
                },
                {
                    "option_id": "bath",
                    "text": "Taking a warm bath or shower",
                    "value": "bath",
                },
                {"option_id": "none", "text": "None of these", "value": "none"},
            ],
            "required": True,
            "weight": 1.0,
        },
    ]


def get_behavioral_questions():
    """Get questions for behavioral assessment."""
    return [
        {
            "question_id": "stress_response",
            "text": "How do you typically respond to stress?",
            "description": "Select the option that best describes your usual response",
            "question_type": "multiple_choice",
            "category": "behavior",
            "dimensions": ["stress_response"],
            "options": [
                {
                    "option_id": "problem",
                    "text": "I try to solve the problem directly",
                    "value": "problem_focused",
                },
                {
                    "option_id": "emotion",
                    "text": "I focus on managing my emotions",
                    "value": "emotion_focused",
                },
                {
                    "option_id": "social",
                    "text": "I seek support from others",
                    "value": "social_support",
                },
                {
                    "option_id": "avoidant",
                    "text": "I try to avoid thinking about it",
                    "value": "avoidant",
                },
                {
                    "option_id": "mixed",
                    "text": "A mix of several approaches",
                    "value": "mixed",
                },
            ],
            "required": True,
            "weight": 1.0,
        },
        {
            "question_id": "routine_consistency",
            "text": "How consistent is your daily routine?",
            "description": """Rate on a scale from 0 (completely variable)
            to 10 (highly consistent)""",
            "question_type": "slider",
            "category": "behavior",
            "dimensions": ["routine_consistency"],
            "min_value": 0,
            "max_value": 10,
            "step": 1,
            "required": True,
            "weight": 1.0,
        },
        {
            "question_id": "exercise_frequency",
            "text": "How many days per week do you typically exercise?",
            "description": "Select the number that best represents your usual habits",
            "question_type": "slider",
            "category": "behavior",
            "dimensions": ["exercise_frequency"],
            "min_value": 0,
            "max_value": 7,
            "step": 1,
            "required": True,
            "weight": 1.0,
        },
        {
            "question_id": "social_preference",
            "text": "How much do you prefer social activities before bedtime?",
            "description": """Rate on a scale from 0 (prefer solitude)
            to 10 (prefer socializing)""",
            "question_type": "slider",
            "category": "behavior",
            "dimensions": ["social_activity_preference"],
            "min_value": 0,
            "max_value": 10,
            "step": 1,
            "required": True,
            "weight": 1.0,
        },
        {
            "question_id": "screen_time",
            "text": """How many minutes do you typically spend
            on screens right before bed?""",
            "description": "Enter approximate time in minutes",
            "question_type": "slider",
            "category": "behavior",
            "dimensions": ["screen_time_before_bed"],
            "min_value": 0,
            "max_value": 120,
            "step": 5,
            "required": True,
            "weight": 1.0,
        },
    ]
