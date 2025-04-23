"""Service for managing questionnaires and processing answers."""

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

    def _prepare_for_db(self, obj):
        """
        Convert objects to database-friendly format.

        Args:
            obj: Object to prepare

        Returns:
            Database-friendly object
        """
        if isinstance(obj, dict):
            return {k: self._prepare_for_db(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._prepare_for_db(item) for item in obj]
        elif isinstance(obj, str) and self._is_iso_date(obj):
            # Try to convert ISO format string to datetime for DB
            try:
                return datetime.fromisoformat(obj)
            except (ValueError, TypeError):
                return obj
        elif hasattr(obj, "dict") and callable(getattr(obj, "dict")):
            # Handle Pydantic models
            return self._prepare_for_db(obj.dict())
        else:
            return obj

    def _is_iso_date(self, date_str):
        """
        Check if a string looks like an ISO format date.

        Args:
            date_str: String to check

        Returns:
            True if it looks like an ISO date, False otherwise
        """
        if not isinstance(date_str, str):
            return False

        # Simple check for ISO format (YYYY-MM-DDThh:mm:ss)
        parts = date_str.split("T")
        if len(parts) != 2:
            return False

        date_part = parts[0]
        time_part = parts[1]

        if len(date_part.split("-")) != 3:
            return False

        if not time_part or ":" not in time_part:
            return False

        return True

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
            "questions": [question.dict() for question in questionnaire_data.questions],
            "created_at": now,
            "updated_at": now,
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
            created_at=questionnaire_dict["created_at"],  # type: ignore
            updated_at=questionnaire_dict["updated_at"],  # type: ignore
            version=str(questionnaire_dict["version"]),
            is_active=bool(questionnaire_dict["is_active"]),
            estimated_duration_minutes=int(  # type: ignore
                questionnaire_dict["estimated_duration_minutes"]
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
            update_dict["questions"] = [
                q.dict() for q in questionnaire_update.questions  # type: ignore
            ]
        if questionnaire_update.is_active is not None:
            update_dict["is_active"] = questionnaire_update.is_active  # type: ignore
        if questionnaire_update.estimated_duration_minutes is not None:
            update_dict[
                "estimated_duration_minutes"
            ] = questionnaire_update.estimated_duration_minutes  # type: ignore
        if questionnaire_update.tags is not None:
            update_dict["tags"] = self._convert_tags_to_list(
                questionnaire_update.tags
            )  # type: ignore
        if questionnaire_update.version is not None:
            update_dict["version"] = str(questionnaire_update.version)

        # Update timestamp
        update_dict["updated_at"] = datetime.now()  # type: ignore

        # Merge with existing questionnaire
        updated_questionnaire = {**existing_questionnaire, **update_dict}

        # Save to storage
        success = self.storage.save_questionnaire(updated_questionnaire)
        if not success:
            raise Exception("Failed to save updated questionnaire to storage")

        # Return updated questionnaire
        return Questionnaire(**updated_questionnaire)

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
            "started_at": now,
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
            started_at=user_questionnaire_dict["started_at"],  # type: ignore
            completed_at=user_questionnaire_dict.get("completed_at"),  # type: ignore
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
            # Ensure answered_at is a string
            answered_at = answer.answered_at
            if isinstance(answered_at, datetime):
                answered_at = answered_at.isoformat()  # type: ignore
            elif isinstance(answered_at, str) and self._is_iso_date(answered_at):
                # Already in ISO format, keep as is
                pass
            else:
                # Default to current time if not a valid datetime
                answered_at = datetime.now().isoformat()

            answer_dict = {
                "question_id": answer.question_id,
                "value": answer.value,
                "answered_at": answered_at,
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
        user_questionnaire["completed_at"] = datetime.now()
        user_questionnaire["scored"] = True
        user_questionnaire["score_results"] = score_results

        # Save to storage
        success = self.storage.save_user_questionnaire(user_questionnaire)
        if not success:
            raise Exception("Failed to save completed user questionnaire")

        # Create profile update data
        # Extract specific fields for direct profile updates
        profile_updates = {
            "scores": score_results.get("raw_scores", {}),
            "dimension_scores": score_results.get("dimension_scores", {}),
            "questions_answered": len(user_questionnaire.get("answers", [])),
            "sleep_preferences": {},
        }

        # Handle ideal_bedtime field
        raw_scores = score_results.get("raw_scores", {})
        if "ideal_bedtime" in raw_scores:
            profile_updates["sleep_preferences"]["ideal_bedtime"] = raw_scores[
                "ideal_bedtime"
            ]

        # Handle relaxation_techniques field
        if "relaxation_techniques" in raw_scores:
            profile_updates["sleep_preferences"]["relaxation_techniques"] = raw_scores[
                "relaxation_techniques"
            ]

        # Update profile with results
        self.profile_service.update_profile_from_questionnaire(
            user_id=user_questionnaire["user_id"], questionnaire_results=profile_updates
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

        # Special field handling
        # Check for ideal_bedtime field
        if "ideal_bedtime" in answers:
            raw_scores["ideal_bedtime"] = answers["ideal_bedtime"]

        # Check for bedtime_routine (relaxation techniques)
        if "bedtime_routine" in answers:
            bedtime_value = answers["bedtime_routine"]
            if isinstance(bedtime_value, list):
                raw_scores["relaxation_techniques"] = bedtime_value
            elif isinstance(bedtime_value, str) and "," in bedtime_value:
                raw_scores["relaxation_techniques"] = bedtime_value.split(",")
            elif bedtime_value:  # Single value
                raw_scores["relaxation_techniques"] = [bedtime_value]
            else:  # Empty value
                raw_scores["relaxation_techniques"] = []

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

                    # Convert dimension to string if it's an enum,
                    # using only the value name
                    dimension_str = (
                        dimension.value.lower()
                        if hasattr(dimension, "value")
                        else str(dimension).lower()
                    )

                    # Get score based on question type
                    score = self._calculate_question_score(question, value)

                    # Apply question weight
                    weight = question.get("weight", 1.0)
                    weighted_score = score * weight

                    # Add to dimension totals
                    if dimension_str not in dimension_scores:
                        dimension_scores[dimension_str] = 0
                        dimension_counts[dimension_str] = 0

                    dimension_scores[dimension_str] += weighted_score
                    dimension_counts[dimension_str] += weight

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
        # Get question type, handling both enum and string values
        question_type = question.get("question_type", question.get("type", "")).lower()
        if hasattr(question_type, "value"):
            question_type = question_type.value.lower()

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
                # Check if the value matches either the option_id or the option itself
                if option.get("id") == value or option.get("option_id") == value:
                    # Return the option's value or default to 0
                    option_value = option.get("value")
                    if isinstance(option_value, (int, float)):
                        return float(option_value)
                    # If the value is a string like "early",
                    # "medium", "late", map to numeric values
                    elif isinstance(option_value, str):
                        if option_value == "early":
                            return 0.2  # Low value for early bedtime
                        elif option_value == "medium":
                            return 0.5  # Medium value for medium bedtime
                        elif option_value == "late":
                            return 0.8  # High value for late bedtime
                return 0.0

        elif question_type == "slider":
            # Special handling for sleep_anxiety to ensure it doesn't exceed bounds
            if any(dim == "sleep_anxiety" for dim in question.get("dimensions", [])):
                # For sleep_anxiety, make sure we return a value between 0-1
                # which will be scaled to 0-100 by _normalize_dimension_score
                return min(float(value) / 10, 1.0)

            # Normal handling for other slider questions
            # Normalize slider value to 0-1 range
            min_value = question.get("min_value", 0)
            max_value = question.get("max_value", 100)
            range_size = max_value - min_value
            if range_size <= 0:
                return 0.0
            return min((float(value) - min_value) / range_size, 1.0)

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
            # For sleep_anxiety, ensure the value is in 0-10 range
            # since the model expects sleep_anxiety_level to be 0-10
            normalized = max(0, min(10, normalized * 10))

        elif dimension == "routine_consistency":
            # For routine consistency, use 0-10 scale and round to integer
            normalized = round(max(0, min(10, normalized * 10)))

        elif dimension in ["sleep_environment"]:
            # These might use different scales, normalize to 0-100
            normalized = max(0, min(100, normalized))

        # Round to one decimal place for non-integer fields
        if dimension != "routine_consistency":
            normalized = round(normalized, 1)
        return normalized
