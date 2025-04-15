"""Models for questionnaires and questions."""
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, validator


class QuestionType(str, Enum):
    """Types of questions that can be asked in questionnaires."""

    LIKERT_5 = "likert_5"  # 5-point Likert scale (Strongly Disagree to Strongly Agree)
    LIKERT_7 = "likert_7"  # 7-point Likert scale
    MULTIPLE_CHOICE = "multiple_choice"  # Select one from multiple options
    CHECKBOX = "checkbox"  # Select multiple from options
    SLIDER = "slider"  # Slider with range
    TEXT = "text"  # Free text response
    YES_NO = "yes_no"  # Yes/No question


class ProfileDimension(str, Enum):
    """Dimensions of the psychological profile that questions can measure."""

    OPENNESS = "openness"
    CONSCIENTIOUSNESS = "conscientiousness"
    EXTRAVERSION = "extraversion"
    AGREEABLENESS = "agreeableness"
    NEUROTICISM = "neuroticism"
    CHRONOTYPE = "chronotype"
    SLEEP_ENVIRONMENT = "sleep_environment"
    SLEEP_ANXIETY = "sleep_anxiety"
    STRESS_RESPONSE = "stress_response"
    ROUTINE_CONSISTENCY = "routine_consistency"
    ACTIVITY_PREFERENCE = "activity_preference"
    SCREEN_TIME = "screen_time"
    GENERAL = "general"


class QuestionCategory(str, Enum):
    """Categories of questions for organization and filtering."""

    PERSONALITY = "personality"
    SLEEP = "sleep"
    BEHAVIOR = "behavior"
    LIFESTYLE = "lifestyle"
    STRESS = "stress"
    MISC = "misc"


class QuestionOption(BaseModel):
    """Option for multiple choice or checkbox questions."""

    option_id: str = Field(..., description="Unique identifier for this option")
    text: str = Field(..., description="Text of the option")
    value: Any = Field(..., description="Value to be recorded if selected")
    image_url: Optional[str] = Field(
        None, description="URL to an image for this option"
    )


class Question(BaseModel):
    """Model for a single question in a questionnaire."""

    question_id: str = Field(..., description="Unique identifier for the question")
    text: str = Field(..., description="Text of the question")
    description: Optional[str] = Field(
        None, description="Additional description or clarification"
    )
    question_type: QuestionType = Field(..., description="Type of question")
    category: QuestionCategory = Field(
        ..., description="Category the question belongs to"
    )
    dimensions: List[ProfileDimension] = Field(
        ..., description="Dimensions this question measures"
    )
    options: Optional[List[QuestionOption]] = Field(
        None, description="Options for multiple choice or checkbox questions"
    )
    min_value: Optional[int] = Field(
        None, description="Minimum value for slider questions"
    )
    max_value: Optional[int] = Field(
        None, description="Maximum value for slider questions"
    )
    step: Optional[float] = Field(None, description="Step size for slider questions")
    required: bool = Field(
        True, description="Whether an answer is required to continue"
    )
    weight: float = Field(
        1.0, description="Weight of this question in scoring (default 1.0)"
    )
    scoring_logic: Optional[str] = Field(
        None, description="Logic used to score this question"
    )

    @validator("options")
    def validate_options(cls, v, values):
        """Validate options for multiple choice and checkbox questions."""
        question_type = values.get("question_type")
        if question_type in [QuestionType.MULTIPLE_CHOICE, QuestionType.CHECKBOX] and (
            not v or len(v) < 2
        ):
            raise ValueError(f"{question_type} questions must have at least 2 options")
        return v

    @validator("min_value", "max_value", "step")
    def validate_slider_values(cls, v, values):
        """Validate that slider questions have min, max, and step values."""
        question_type = values.get("question_type")
        if question_type == QuestionType.SLIDER and v is None:
            field_name = next(name for name, value in values.items() if value is v)
            raise ValueError(f"Slider questions must have {field_name}")
        return v


class Answer(BaseModel):
    """Model for an answer to a question."""

    question_id: str = Field(..., description="ID of the question being answered")
    value: Any = Field(..., description="Answer value")
    answered_at: datetime = Field(
        default_factory=datetime.now, description="When the question was answered"
    )


class QuestionnaireType(str, Enum):
    """Types of questionnaires that can be administered."""

    ONBOARDING = "onboarding"
    PERSONALITY = "personality"
    SLEEP_HABITS = "sleep_habits"
    BEHAVIORAL = "behavioral"
    FOLLOWUP = "followup"
    CUSTOM = "custom"


class QuestionnaireStatus(str, Enum):
    """Status of a questionnaire for a user."""

    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    EXPIRED = "expired"


class Questionnaire(BaseModel):
    """Model for a questionnaire containing multiple questions."""

    questionnaire_id: str = Field(
        ..., description="Unique identifier for the questionnaire"
    )
    title: str = Field(..., description="Title of the questionnaire")
    description: str = Field(..., description="Description of the questionnaire")
    questionnaire_type: QuestionnaireType = Field(
        ..., description="Type of questionnaire"
    )
    questions: List[Question] = Field(
        ..., description="Questions in this questionnaire"
    )
    created_at: datetime = Field(
        default_factory=datetime.now, description="When the questionnaire was created"
    )
    updated_at: datetime = Field(
        default_factory=datetime.now,
        description="When the questionnaire was last updated",
    )
    version: str = Field(default="1.0.0", description="Version of the questionnaire")
    is_active: bool = Field(
        default=True, description="Whether the questionnaire is active"
    )
    estimated_duration_minutes: Optional[int] = Field(
        None, description="Estimated time to complete in minutes"
    )
    tags: Optional[List[str]] = Field(None, description="Tags for categorization")


class UserQuestionnaire(BaseModel):
    """Model for tracking a user's progress through a questionnaire."""

    user_questionnaire_id: str = Field(
        ..., description="Unique identifier for this user's questionnaire instance"
    )
    user_id: str = Field(..., description="User identifier")
    questionnaire_id: str = Field(..., description="Questionnaire identifier")
    status: QuestionnaireStatus = Field(
        default=QuestionnaireStatus.NOT_STARTED, description="Status of completion"
    )
    answers: List[Answer] = Field(
        default_factory=list, description="User's answers to questions"
    )
    started_at: Optional[datetime] = Field(
        None, description="When the user started the questionnaire"
    )
    completed_at: Optional[datetime] = Field(
        None, description="When the user completed the questionnaire"
    )
    current_page: int = Field(default=1, description="Current page the user is on")
    total_pages: int = Field(
        default=1, description="Total number of pages in the questionnaire"
    )
    scored: bool = Field(
        default=False, description="Whether the questionnaire has been scored"
    )
    score_results: Optional[Dict[str, Any]] = Field(
        None, description="Results from scoring the questionnaire"
    )


class QuestionnaireCreate(BaseModel):
    """Model for creating a new questionnaire."""

    title: str = Field(..., description="Title of the questionnaire")
    description: str = Field(..., description="Description of the questionnaire")
    questionnaire_type: QuestionnaireType = Field(
        ..., description="Type of questionnaire"
    )
    questions: List[Question] = Field(
        ..., description="Questions in this questionnaire"
    )
    estimated_duration_minutes: Optional[int] = Field(
        None, description="Estimated time to complete in minutes"
    )
    tags: Optional[List[str]] = Field(None, description="Tags for categorization")


class QuestionnaireUpdate(BaseModel):
    """Model for updating an existing questionnaire."""

    title: Optional[str] = Field(None, description="Title of the questionnaire")
    description: Optional[str] = Field(
        None, description="Description of the questionnaire"
    )
    questions: Optional[List[Question]] = Field(
        None, description="Questions in this questionnaire"
    )
    is_active: Optional[bool] = Field(
        None, description="Whether the questionnaire is active"
    )
    estimated_duration_minutes: Optional[int] = Field(
        None, description="Estimated time to complete in minutes"
    )
    tags: Optional[List[str]] = Field(None, description="Tags for categorization")
    version: Optional[str] = Field(None, description="Version of the questionnaire")


class AnswerSubmission(BaseModel):
    """Model for submitting answers to a questionnaire."""

    questionnaire_id: str = Field(..., description="Questionnaire identifier")
    user_id: str = Field(..., description="User identifier")
    answers: List[Answer] = Field(..., description="Answers to submit")
    page_number: Optional[int] = Field(None, description="Page number being submitted")


class QuestionnaireResponse(BaseModel):
    """Response model for questionnaire operations."""

    questionnaire: Optional[Union[Questionnaire, UserQuestionnaire]] = None
    status: str = "success"
    message: Optional[str] = None


class QuestionnairesResponse(BaseModel):
    """Response model for multiple questionnaires."""

    questionnaires: List[Questionnaire]
    count: int
    status: str = "success"


class UserQuestionnairesResponse(BaseModel):
    """Response model for multiple user questionnaires."""

    questionnaires: List[UserQuestionnaire]
    count: int
    status: str = "success"


class QuestionnairePage(BaseModel):
    """Model for a page of questions in a questionnaire."""

    page_number: int = Field(..., description="Page number")
    questions: List[Question] = Field(..., description="Questions on this page")
    progress_percentage: float = Field(
        ..., description="Percentage of questionnaire completed"
    )
    next_page: Optional[int] = Field(None, description="Next page number")
    prev_page: Optional[int] = Field(None, description="Previous page number")
    total_pages: int = Field(..., description="Total number of pages")


class QuestionnairePageResponse(BaseModel):
    """Response model for a page of questions."""

    page: QuestionnairePage
    user_answers: Dict[str, Any] = Field(
        default_factory=dict, description="User's existing answers for this page"
    )
    status: str = "success"
