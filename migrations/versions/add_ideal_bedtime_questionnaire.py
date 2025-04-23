"""Add missing ideal_bedtime field to questionnaires.

Revision ID: 004
Revises: 003
Create Date: 2025-04-23 11:00:00.000000

This migration updates existing questionnaires to add the missing ideal_bedtime
field and ensures relaxation techniques are properly handled.
"""

import json
from datetime import datetime

from alembic import op
from sqlalchemy import text

# revision identifiers, used by Alembic.
revision = "004"
down_revision = "003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Add missing fields to questionnaires."""
    # Get current time for timestamps
    now = datetime.now().isoformat()

    # Get all active questionnaires first
    questionnaires = (
        op.get_bind()
        .execute(
            text(
                "SELECT questionnaire_id, questions, updated_at FROM questionnaires WHERE is_active = true"  # noqa: E501
            )
        )
        .fetchall()
    )

    for questionnaire in questionnaires:
        questionnaire_id = questionnaire[0]

        # Check if questions is already a list or if it's a JSON string
        if isinstance(questionnaire[1], str):
            questions = json.loads(questionnaire[1])
        else:
            # It's already a list or dict
            questions = questionnaire[1]

        questionnaire[2]

        # Check if we have sleep section questions
        has_sleep_section = any("sleep_" in q.get("question_id", "") for q in questions)

        if has_sleep_section:
            # Check if ideal_bedtime field already exists
            has_ideal_bedtime = any(
                "ideal_bedtime" in q.get("question_id", "") for q in questions
            )

            if not has_ideal_bedtime:
                # Add ideal_bedtime question
                ideal_bedtime_question = {
                    "question_id": "ideal_bedtime",
                    "text": "What is your ideal bedtime?",
                    "description": "Select your preferred time to go to bed",
                    "question_type": "time",
                    "category": "sleep",
                    "dimensions": ["sleep_preferences"],
                    "required": False,
                    "weight": 1.0,
                }

                questions.append(ideal_bedtime_question)
                print(
                    f"Added ideal_bedtime question to questionnaire {questionnaire_id}"
                )

            # Check if bedtime_routine handling is proper
            for question in questions:
                if question.get("question_id") == "bedtime_routine":
                    # Ensure it's properly set up as checkbox
                    question["question_type"] = "checkbox"
                    # Ensure options exist and have proper values
                    options = question.get("options", [])
                    if not options:
                        options = [
                            {
                                "option_id": "reading",
                                "text": "Reading",
                                "value": "reading",
                            },
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
                            {
                                "option_id": "none",
                                "text": "None of these",
                                "value": "none",
                            },
                        ]
                        question["options"] = options
                        print(
                            f"Updated bedtime_routine options in questionnaire {questionnaire_id}"  # noqa: E501
                        )

            # Update the questionnaire with the new questions
            # Convert questions back to JSON for storage
            questions_json = json.dumps(questions)

            op.get_bind().execute(
                text(
                    """
                    UPDATE questionnaires
                    SET questions = :questions, updated_at = :updated_at
                    WHERE questionnaire_id = :questionnaire_id
                    """
                ).bindparams(
                    questions=questions_json,
                    updated_at=now,
                    questionnaire_id=questionnaire_id,
                )
            )

            print(f"Updated questionnaire {questionnaire_id}")


def downgrade() -> None:
    """Remove added fields from questionnaires."""
    # Get all active questionnaires first
    questionnaires = (
        op.get_bind()
        .execute(
            text(
                "SELECT questionnaire_id, questions, updated_at FROM questionnaires WHERE is_active = true"  # noqa: E501
            )
        )
        .fetchall()
    )

    now = datetime.now().isoformat()

    for questionnaire in questionnaires:
        questionnaire_id = questionnaire[0]

        # Check if questions is already a list or if it's a JSON string
        if isinstance(questionnaire[1], str):
            questions = json.loads(questionnaire[1])
        else:
            # It's already a list or dict
            questions = questionnaire[1]

        questionnaire[2]

        # Remove ideal_bedtime question if it exists
        questions = [q for q in questions if q.get("question_id") != "ideal_bedtime"]

        # Convert questions back to JSON for storage
        questions_json = json.dumps(questions)

        # Update the questionnaire
        op.get_bind().execute(
            text(
                """
                UPDATE questionnaires
                SET questions = :questions, updated_at = :updated_at
                WHERE questionnaire_id = :questionnaire_id
                """
            ).bindparams(
                questions=questions_json,
                updated_at=now,
                questionnaire_id=questionnaire_id,
            )
        )

        print(f"Reverted questionnaire {questionnaire_id}")
