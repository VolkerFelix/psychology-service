"""Service for managing questionnaires and processing answers."""

import ast
import json
import math
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from app.config.settings import settings
from app.models.questionnaire_models import (
    Answer,
    AnswerSubmission,
    Question,
    Questionnaire,
    QuestionnaireCreate,
    QuestionnairePage,
    QuestionnairePageResponse,
    QuestionnaireStatus,
    QuestionnaireType,
    QuestionnaireUpdate,
    UserQuestionnaire,
)
from app.services.profile_service import ProfileService


class QuestionnaireService:
    """Service for managing questionnaires and processing answers."""

    def __init__(self, storage_service):
        """
        Initialize with a storage service.

        Args:
            storage_service: Storage service for persistence
        """
        self.storage = storage_service
        self.profile_service = ProfileService(storage_service)

    def _convert_is_active_to_str(self, is_active: Any) -> str:
        """Convert is_active to str."""
        if isinstance(is_active, bool):
            return str(is_active)
        if isinstance(is_active, str):
            return is_active
        return "false"

    def _convert_tags_to_list(self, tags: Any) -> Optional[List[str]]:
        """Convert tags to list representation."""
        if tags is None:
            return None
        try:
            # If it's already a list, return it
            if isinstance(tags, list):
                return [str(tag) for tag in tags]
            # If it's a string, try to parse it as a list
            if isinstance(tags, str):
                # Check if it looks like a list
                if tags.startswith("[") and tags.endswith("]"):
                    # Use ast.literal_eval to safely evaluate the string as a list
                    import ast

                    tags_list = ast.literal_eval(tags)
                    return [str(tag) for tag in tags_list]
                else:
                    # Treat as a single tag
                    return [tags]
            # For other types, convert to string and treat as a single tag
            return [str(tags)]
        except (ValueError, SyntaxError):
            # If conversion fails, treat as a single tag
            return [str(tags)]

    def create_questionnaire(
        self, questionnaire_data: QuestionnaireCreate
    ) -> Questionnaire:
        """
        Create a new questionnaire.

        Args:
            questionnaire_data: Questionnaire creation data

        Returns:
            Created questionnaire
        """
        # Generate ID and set initial values
        questionnaire_id = str(uuid.uuid4())
        now = datetime.now()

        # Build complete questionnaire data
        questionnaire_dict = {
            "questionnaire_id": questionnaire_id,
            "title": questionnaire_data.title,
            "description": questionnaire_data.description,
            "questionnaire_type": questionnaire_data.questionnaire_type,
            "questions": [
                question.model_dump() for question in questionnaire_data.questions
            ],
            "created_at": now.isoformat(),
            "updated_at": now.isoformat(),
            "version": "1.0.0",
            "is_active": True,
            "estimated_duration_minutes": questionnaire_data.estimated_duration_minutes,
            "tags": self._convert_tags_to_list(questionnaire_data.tags),
        }

        # Save to storage
        success = self.storage.save_questionnaire(questionnaire_dict)
        if not success:
            raise Exception("Failed to save questionnaire to storage")

        # Return complete questionnaire object with proper type conversion
        return Questionnaire(
            questionnaire_id=str(questionnaire_dict["questionnaire_id"]),
            title=str(questionnaire_dict["title"]),
            description=str(questionnaire_dict["description"]),
            questionnaire_type=QuestionnaireType(
                questionnaire_dict["questionnaire_type"]
            ),
            questions=questionnaire_data.questions,
            created_at=datetime.fromisoformat(str(questionnaire_dict["created_at"])),
            updated_at=datetime.fromisoformat(str(questionnaire_dict["updated_at"])),
            version=str(questionnaire_dict["version"]),
            is_active=bool(questionnaire_dict["is_active"]),
            estimated_duration_minutes=int(
                str(questionnaire_dict["estimated_duration_minutes"])
            )
            if questionnaire_dict["estimated_duration_minutes"] is not None
            else None,
            tags=self._convert_tags_to_list(questionnaire_data.tags),
        )

    def get_questionnaire(self, questionnaire_id: str) -> Optional[Questionnaire]:
        """
        Get a questionnaire by ID.

        Args:
            questionnaire_id: Questionnaire ID

        Returns:
            Questionnaire if found, None otherwise
        """
        questionnaire_data = self.storage.get_questionnaire(questionnaire_id)
        if not questionnaire_data:
            return None

        return Questionnaire(**questionnaire_data)

    def list_questionnaires(
        self,
        limit: int = 100,
        offset: int = 0,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Questionnaire]:
        """
        List questionnaires with optional filtering.

        Args:
            limit: Maximum number of questionnaires to return
            offset: Number of questionnaires to skip
            filters: Optional filters to apply

        Returns:
            List of questionnaires
        """
        questionnaires_data = self.storage.get_questionnaires(
            limit=limit, offset=offset, filters=filters
        )
        return [Questionnaire(**q) for q in questionnaires_data]

    def update_questionnaire(
        self, questionnaire_id: str, questionnaire_update: QuestionnaireUpdate
    ) -> Optional[Questionnaire]:
        """
        Update an existing questionnaire.

        Args:
            questionnaire_id: Questionnaire ID
            questionnaire_update: Questionnaire update data

        Returns:
            Updated questionnaire if successful, None otherwise
        """
        # Get existing questionnaire
        existing_questionnaire = self.storage.get_questionnaire(questionnaire_id)
        if not existing_questionnaire:
            return None

        # Build update dictionary
        update_dict = {}
        if questionnaire_update.title is not None:
            update_dict["title"] = str(questionnaire_update.title)
        if questionnaire_update.description is not None:
            update_dict["description"] = str(questionnaire_update.description)
        if questionnaire_update.questions is not None:
            update_dict["questions"] = json.dumps(
                [q.model_dump() for q in questionnaire_update.questions]
            )
        update_dict["is_active"] = self._convert_is_active_to_str(
            questionnaire_update.is_active
        )
        if questionnaire_update.estimated_duration_minutes is not None:
            update_dict["estimated_duration_minutes"] = str(
                questionnaire_update.estimated_duration_minutes
            )
        if questionnaire_update.tags is not None:
            update_dict["tags"] = str(
                self._convert_tags_to_list(questionnaire_update.tags)
            )
        if questionnaire_update.version is not None:
            update_dict["version"] = str(questionnaire_update.version)

        # Update timestamp
        update_dict["updated_at"] = datetime.now().isoformat()

        # Merge with existing questionnaire
        updated_questionnaire = {**existing_questionnaire, **update_dict}

        # Save to storage
        success = self.storage.save_questionnaire(updated_questionnaire)
        if not success:
            raise Exception("Failed to save updated questionnaire to storage")

        # Return updated questionnaire
        try:
            # Try to parse the questions as JSON
            questions = json.loads(updated_questionnaire["questions"])
        except (json.JSONDecodeError, TypeError):
            # If parsing fails, use an empty list
            questions = []

        # Try to parse the tags
        try:
            # If tags is a string, try to parse it as a list
            if isinstance(updated_questionnaire.get("tags"), str):
                tags = ast.literal_eval(updated_questionnaire["tags"])
            else:
                tags = updated_questionnaire.get("tags")
        except (ValueError, SyntaxError):
            # If parsing fails, use None
            tags = None

        # Convert tags to list of strings if it's a list
        if tags is not None and isinstance(tags, list):
            tags = [str(tag) for tag in tags]
        elif tags is not None:
            # If tags is not a list, convert it to a list with a single element
            tags = [str(tags)]

        return Questionnaire(
            questionnaire_id=str(updated_questionnaire["questionnaire_id"]),
            title=str(updated_questionnaire["title"]),
            description=str(updated_questionnaire["description"]),
            questionnaire_type=QuestionnaireType(
                updated_questionnaire["questionnaire_type"]
            ),
            questions=questions,
            created_at=datetime.fromisoformat(str(updated_questionnaire["created_at"])),
            updated_at=datetime.fromisoformat(str(updated_questionnaire["updated_at"])),
            version=str(updated_questionnaire["version"]),
            is_active=bool(updated_questionnaire["is_active"]),
            estimated_duration_minutes=int(
                str(updated_questionnaire["estimated_duration_minutes"])
            )
            if updated_questionnaire["estimated_duration_minutes"] is not None
            else None,
            tags=tags,
        )

    def delete_questionnaire(self, questionnaire_id: str) -> bool:
        """
        Delete a questionnaire.

        Args:
            questionnaire_id: Questionnaire ID

        Returns:
            True if deleted, False if not found
        """
        return self.storage.delete_questionnaire(questionnaire_id)

    def start_questionnaire(
        self, user_id: str, questionnaire_id: str
    ) -> Optional[UserQuestionnaire]:
        """
        Start a questionnaire for a user.

        Args:
            user_id: User ID
            questionnaire_id: Questionnaire ID

        Returns:
            Created user questionnaire if successful, None otherwise
        """
        # Get questionnaire
        questionnaire_data = self.storage.get_questionnaire(questionnaire_id)
        if not questionnaire_data:
            return None

        # Check if user already has this questionnaire in progress
        existing_user_questionnaires = self.storage.get_user_questionnaires(
            user_id=user_id, questionnaire_id=questionnaire_id
        )

        for uq in existing_user_questionnaires:
            if uq["status"] == QuestionnaireStatus.IN_PROGRESS:
                # Return the existing in-progress questionnaire
                return UserQuestionnaire(**uq)

        # Create new user questionnaire
        user_questionnaire_id = str(uuid.uuid4())
        now = datetime.now()

        # Calculate total pages
        questions_per_page = settings.DEFAULT_QUESTIONS_PER_PAGE
        total_questions = len(questionnaire_data.get("questions", []))
        total_pages = (
            math.ceil(total_questions / questions_per_page)
            if total_questions > 0
            else 1
        )

        user_questionnaire_dict = {
            "user_questionnaire_id": user_questionnaire_id,
            "user_id": user_id,
            "questionnaire_id": questionnaire_id,
            "status": QuestionnaireStatus.IN_PROGRESS,
            "answers": [],
            "started_at": now.isoformat(),
            "current_page": 1,
            "total_pages": total_pages,
            "scored": False,
        }

        # Save to storage
        success = self.storage.save_user_questionnaire(user_questionnaire_dict)
        if not success:
            raise Exception("Failed to save user questionnaire to storage")

        answers = [
            Answer(**a)
            for a in user_questionnaire_dict.get(  # type: ignore
                "answers", []
            )  # type: ignore
        ]

        # Return complete user questionnaire object
        return UserQuestionnaire(
            user_questionnaire_id=str(user_questionnaire_dict["user_questionnaire_id"]),
            user_id=str(user_questionnaire_dict["user_id"]),
            questionnaire_id=str(user_questionnaire_dict["questionnaire_id"]),
            status=QuestionnaireStatus(user_questionnaire_dict["status"]),
            answers=answers,
            started_at=datetime.fromisoformat(
                str(user_questionnaire_dict["started_at"])
            )
            if user_questionnaire_dict.get("started_at")
            else None,
            completed_at=datetime.fromisoformat(
                str(user_questionnaire_dict["completed_at"])
            )
            if user_questionnaire_dict.get("completed_at")
            else None,
            current_page=int(user_questionnaire_dict["current_page"]),  # type: ignore
            total_pages=int(user_questionnaire_dict["total_pages"]),  # type: ignore
            scored=bool(user_questionnaire_dict["scored"]),
            score_results=user_questionnaire_dict.get("score_results"),  # type: ignore
        )

    def get_questionnaire_progress(
        self, user_questionnaire_id: str
    ) -> Optional[UserQuestionnaire]:
        """
        Get a user's progress on a questionnaire.

        Args:
            user_questionnaire_id: User questionnaire ID

        Returns:
            User questionnaire if found, None otherwise
        """
        user_questionnaire_data = self.storage.get_user_questionnaire(
            user_questionnaire_id
        )
        if not user_questionnaire_data:
            return None

        return UserQuestionnaire(**user_questionnaire_data)

    def get_user_questionnaires(
        self, user_id: str, status: Optional[QuestionnaireStatus] = None
    ) -> List[UserQuestionnaire]:
        """
        Get all questionnaires for a user.

        Args:
            user_id: User ID
            status: Optional status filter

        Returns:
            List of user questionnaires
        """
        user_questionnaires_data = self.storage.get_user_questionnaires(
            user_id=user_id, status=status.value if status else None
        )
        return [UserQuestionnaire(**uq) for uq in user_questionnaires_data]

    def get_questionnaire_page(
        self, user_questionnaire_id: str, page_number: int
    ) -> Optional[QuestionnairePageResponse]:
        """
        Get a specific page of a questionnaire with the user's existing answers.

        Args:
            user_questionnaire_id: User questionnaire ID
            page_number: Page number to retrieve

        Returns:
            Page data if found, None otherwise
        """
        # Get user questionnaire
        user_questionnaire_data = self.storage.get_user_questionnaire(
            user_questionnaire_id
        )
        if not user_questionnaire_data:
            return None

        # Get questionnaire
        questionnaire_id = user_questionnaire_data.get("questionnaire_id")
        questionnaire_data = self.storage.get_questionnaire(questionnaire_id)
        if not questionnaire_data:
            return None

        # Calculate page information
        questions = questionnaire_data.get("questions", [])
        questions_per_page = settings.DEFAULT_QUESTIONS_PER_PAGE
        total_questions = len(questions)
        total_pages = (
            math.ceil(total_questions / questions_per_page)
            if total_questions > 0
            else 1
        )

        if page_number < 1 or page_number > total_pages:
            return None

        # Get questions for this page
        start_idx = (page_number - 1) * questions_per_page
        end_idx = min(start_idx + questions_per_page, total_questions)
        page_questions = [Question(**q) for q in questions[start_idx:end_idx]]

        # Calculate progress
        progress_percentage = (
            ((page_number - 1) / total_pages) * 100 if total_pages > 0 else 0
        )

        # Create page object
        page = QuestionnairePage(
            page_number=page_number,
            questions=page_questions,
            progress_percentage=progress_percentage,
            next_page=page_number + 1 if page_number < total_pages else None,
            prev_page=page_number - 1 if page_number > 1 else None,
            total_pages=total_pages,
        )

        # Extract user's existing answers for this page
        user_answers = {}
        for answer in user_questionnaire_data.get("answers", []):
            question_id = answer.get("question_id")
            # Check if this question is on the current page
            if any(q.question_id == question_id for q in page_questions):
                user_answers[question_id] = answer.get("value")

        # Return page with user answers
        return QuestionnairePageResponse(
            page=page,
            user_answers=user_answers,
            status="success",
        )

    def submit_answers(
        self, answer_submission: AnswerSubmission
    ) -> Optional[UserQuestionnaire]:
        """
        Submit answers for a questionnaire.

        Args:
            answer_submission: Answer submission data

        Returns:
            Updated user questionnaire if successful, None otherwise
        """
        # Get user questionnaires for this user and questionnaire
        user_questionnaires = self.storage.get_user_questionnaires(
            user_id=answer_submission.user_id,
            questionnaire_id=answer_submission.questionnaire_id,
        )

        # Find in-progress questionnaire
        user_questionnaire = None
        for uq in user_questionnaires:
            if uq["status"] == QuestionnaireStatus.IN_PROGRESS:
                user_questionnaire = uq
                break

        if not user_questionnaire:
            # Start a new questionnaire
            new_uq = self.start_questionnaire(
                user_id=answer_submission.user_id,
                questionnaire_id=answer_submission.questionnaire_id,
            )
            if not new_uq:
                return None
            user_questionnaire = new_uq.dict()

        # Update answers
        current_answers = user_questionnaire.get("answers", [])
        question_ids = [a.get("question_id") for a in current_answers]

        for answer in answer_submission.answers:
            answer_dict = {
                "question_id": answer.question_id,
                "value": answer.value,
                "answered_at": answer.answered_at.isoformat()
                if isinstance(answer.answered_at, datetime)
                else answer.answered_at,
            }

            # Replace existing answer or add new one
            if answer.question_id in question_ids:
                idx = question_ids.index(answer.question_id)
                current_answers[idx] = answer_dict
            else:
                current_answers.append(answer_dict)
                question_ids.append(answer.question_id)

        user_questionnaire["answers"] = current_answers

        # Update page if provided
        if answer_submission.page_number:
            user_questionnaire["current_page"] = answer_submission.page_number

        # Save to storage
        success = self.storage.save_user_questionnaire(user_questionnaire)
        if not success:
            raise Exception("Failed to save user questionnaire with answers")

        # Return updated user questionnaire
        return UserQuestionnaire(**user_questionnaire)

    def complete_questionnaire(
        self, user_questionnaire_id: str
    ) -> Optional[UserQuestionnaire]:
        """
        Mark a questionnaire as completed and process the results.

        Args:
            user_questionnaire_id: User questionnaire ID

        Returns:
            Completed user questionnaire if successful, None otherwise
        """
        # Get user questionnaire
        user_questionnaire = self.storage.get_user_questionnaire(user_questionnaire_id)
        if not user_questionnaire:
            return None

        # Get questionnaire
        questionnaire_id = user_questionnaire.get("questionnaire_id")
        questionnaire = self.storage.get_questionnaire(questionnaire_id)
        if not questionnaire:
            return None

        # Score the questionnaire
        score_results = self._score_questionnaire(user_questionnaire, questionnaire)

        # Update user questionnaire
        user_questionnaire["status"] = QuestionnaireStatus.COMPLETED
        user_questionnaire["completed_at"] = datetime.now().isoformat()
        user_questionnaire["scored"] = True
        user_questionnaire["score_results"] = score_results

        # Save to storage
        success = self.storage.save_user_questionnaire(user_questionnaire)
        if not success:
            raise Exception("Failed to save completed user questionnaire")

        # Update profile with results
        self.profile_service.update_profile_from_questionnaire(
            user_id=user_questionnaire["user_id"],
            questionnaire_results={
                "scores": score_results.get("raw_scores", {}),
                "dimension_scores": score_results.get("dimension_scores", {}),
                "questions_answered": len(user_questionnaire.get("answers", [])),
            },
        )

        # Return completed user questionnaire
        return UserQuestionnaire(**user_questionnaire)

    def get_next_onboarding_questionnaire(
        self, user_id: str
    ) -> Optional[Union[Questionnaire, UserQuestionnaire]]:
        """
        Get the next onboarding questionnaire for a user.

        Args:
            user_id: User ID

        Returns:
            Next questionnaire or user questionnaire in progress,
            or None if onboarding is complete
        """
        # Get user's questionnaires
        user_questionnaires = self.storage.get_user_questionnaires(user_id=user_id)

        # Check for in-progress questionnaires
        for uq in user_questionnaires:
            if uq["status"] == QuestionnaireStatus.IN_PROGRESS:
                return UserQuestionnaire(**uq)

        # Check which questionnaire types have been completed
        completed_types = set()
        for uq in user_questionnaires:
            if uq["status"] == QuestionnaireStatus.COMPLETED:
                # Get questionnaire type
                q_data = self.storage.get_questionnaire(uq["questionnaire_id"])
                if q_data:
                    completed_types.add(q_data["questionnaire_type"])

        # Define onboarding sequence
        onboarding_sequence = [
            QuestionnaireType.ONBOARDING,
            QuestionnaireType.PERSONALITY,
            QuestionnaireType.SLEEP_HABITS,
            QuestionnaireType.BEHAVIORAL,
        ]

        # Find next type to complete
        next_type = None
        for q_type in onboarding_sequence:
            if q_type not in completed_types:
                next_type = q_type
                break

        if not next_type:
            # All types completed
            return None

        # Get active questionnaire of this type
        questionnaires = self.storage.get_questionnaires(
            filters={"questionnaire_type": next_type, "is_active": True}, limit=1
        )

        if not questionnaires:
            # No questionnaire available for this type
            return None

        return Questionnaire(**questionnaires[0])

    def _score_questionnaire(
        self, user_questionnaire: Dict[str, Any], questionnaire: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Score a completed questionnaire.

        Args:
            user_questionnaire: User questionnaire data
            questionnaire: Questionnaire data

        Returns:
            Dictionary with scoring results
        """
        # Extract answers and questions
        answers = {
            a["question_id"]: a["value"] for a in user_questionnaire.get("answers", [])
        }
        questions = {q["question_id"]: q for q in questionnaire.get("questions", [])}

        # Initialize results
        raw_scores = {}
        dimension_scores = {}
        dimension_counts = {}

        # Process each answered question
        for question_id, value in answers.items():
            if question_id in questions:
                question = questions[question_id]

                # Store raw score
                raw_scores[question_id] = value

                # Process dimensions
                for dimension in question.get("dimensions", []):
                    # Skip if not a valid dimension
                    if not dimension:
                        continue

                    # Get score based on question type
                    score = self._calculate_question_score(question, value)

                    # Apply question weight
                    weight = question.get("weight", 1.0)
                    weighted_score = score * weight

                    # Add to dimension totals
                    if dimension not in dimension_scores:
                        dimension_scores[dimension] = 0
                        dimension_counts[dimension] = 0

                    dimension_scores[dimension] += weighted_score
                    dimension_counts[dimension] += weight

        # Calculate average for each dimension
        processed_dimensions = {}
        for dimension, total in dimension_scores.items():
            count = dimension_counts.get(dimension, 0)
            if count > 0:
                # Calculate normalized score (0-100)
                avg_score = total / count
                processed_dimensions[dimension] = self._normalize_dimension_score(
                    dimension, avg_score
                )

        # Return results
        return {
            "raw_scores": raw_scores,
            "dimension_scores": processed_dimensions,
            "question_count": len(answers),
        }

    def _calculate_question_score(self, question: Dict[str, Any], value: Any) -> float:
        """
        Calculate a score for a question based on its type.

        Args:
            question: Question data
            value: Answer value

        Returns:
            Score as a float (typically 0-1 range)
        """
        question_type = question.get("type")

        if question_type == "likert_5":
            # Assuming values 1-5, normalize to 0-1
            return (float(value) - 1) / 4

        elif question_type == "likert_7":
            # Assuming values 1-7, normalize to 0-1
            return (float(value) - 1) / 6

        elif question_type == "multiple_choice":
            # Look up the selected option
            options = question.get("options", [])
            for option in options:
                if option.get("id") == value:
                    # Return the option's value or default to 0
                    option_value = option.get("value")
                    if isinstance(option_value, (int, float)):
                        return float(option_value)
            return 0.0

        elif question_type == "slider":
            # Normalize slider value to 0-1 range
            min_value = question.get("min_value", 0)
            max_value = question.get("max_value", 100)
            range_size = max_value - min_value
            if range_size <= 0:
                return 0.0
            return (float(value) - min_value) / range_size

        elif question_type == "yes_no":
            # Assuming Yes=1, No=0
            return 1.0 if value else 0.0

        elif question_type == "checkbox":
            # For multiple selection, calculate average of selected values
            options = question.get("options", [])
            selected_ids = value if isinstance(value, list) else [value]

            total = 0.0
            count = 0

            for option in options:
                if option.get("id") in selected_ids:
                    option_value = option.get("value")
                    if isinstance(option_value, (int, float)):
                        total += float(option_value)
                        count += 1

            return total / count if count > 0 else 0.0

        # Default case
        return 0.0

    def _normalize_dimension_score(self, dimension: str, score: float) -> float:
        """
        Normalize a dimension score to a standard scale (0-100).

        Args:
            dimension: Dimension identifier
            score: Raw score (typically 0-1)

        Returns:
            Normalized score (0-100)
        """
        # Basic normalization to 0-100 scale
        normalized = score * 100

        # Apply dimension-specific transformations if needed
        if dimension in [
            "openness",
            "conscientiousness",
            "extraversion",
            "agreeableness",
            "neuroticism",
        ]:
            # For personality traits, ensure good distribution across the scale
            # Add some scaling to avoid clustering in the middle
            normalized = max(0, min(100, normalized))

        elif dimension == "sleep_anxiety":
            # For anxiety, higher raw score means higher anxiety (0-10 scale)
            if 0 <= normalized <= 100:
                # Keep as is - already on 0-100 scale
                pass

        elif dimension in ["routine_consistency", "sleep_environment"]:
            # These might use different scales, normalize to 0-100
            normalized = max(0, min(100, normalized))

        # Round to one decimal place
        return round(normalized, 1)
