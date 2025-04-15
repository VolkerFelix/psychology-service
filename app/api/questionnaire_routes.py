"""API routes for questionnaires."""

from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Path, Query
from fastapi import status as http_status
from fastapi.responses import JSONResponse
from loguru import logger

from app.models.questionnaire_models import (
    AnswerSubmission,
    QuestionnaireCreate,
    QuestionnairePageResponse,
    QuestionnaireResponse,
    QuestionnairesResponse,
    QuestionnaireStatus,
    QuestionnaireType,
    QuestionnaireUpdate,
    UserQuestionnairesResponse,
)
from app.services.questionnaire_service import QuestionnaireService
from app.services.storage.factory import StorageFactory

router = APIRouter(prefix="/questionnaires", tags=["questionnaires"])


def get_storage_service():
    """Get the storage service."""
    return StorageFactory.create_storage_service()


def get_questionnaire_service(storage_service=Depends(get_storage_service)):
    """Get the questionnaire service."""
    return QuestionnaireService(storage_service)


@router.post(
    "/", response_model=QuestionnaireResponse, status_code=http_status.HTTP_201_CREATED
)
async def create_questionnaire(
    questionnaire: QuestionnaireCreate,
    questionnaire_service: QuestionnaireService = Depends(get_questionnaire_service),
):
    """Create a new questionnaire."""
    try:
        created_questionnaire = questionnaire_service.create_questionnaire(
            questionnaire
        )
        return QuestionnaireResponse(
            questionnaire=created_questionnaire,
            message="Questionnaire created successfully",
        )
    except Exception as e:
        logger.error(f"Error creating questionnaire: {str(e)}")
        raise HTTPException(
            status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error creating questionnaire: {str(e)}",
        )


@router.get("/{questionnaire_id}", response_model=QuestionnaireResponse)
async def get_questionnaire(
    questionnaire_id: str = Path(..., description="Questionnaire ID"),
    questionnaire_service: QuestionnaireService = Depends(get_questionnaire_service),
):
    """Get a specific questionnaire by ID."""
    try:
        questionnaire = questionnaire_service.get_questionnaire(questionnaire_id)
        if not questionnaire:
            raise HTTPException(
                status_code=http_status.HTTP_404_NOT_FOUND,
                detail=f"Questionnaire with ID {questionnaire_id} not found",
            )
        return QuestionnaireResponse(questionnaire=questionnaire)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving questionnaire: {str(e)}")
        raise HTTPException(
            status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving questionnaire: {str(e)}",
        )


@router.get("/", response_model=QuestionnairesResponse)
async def list_questionnaires(
    limit: int = Query(100, description="Maximum number of questionnaires to return"),
    offset: int = Query(0, description="Number of questionnaires to skip"),
    questionnaire_type: Optional[QuestionnaireType] = Query(
        None, description="Filter by questionnaire type"
    ),
    is_active: bool = Query(True, description="Filter by active status"),
    tags: List[str] = Query([], description="Filter by tags"),
    questionnaire_service: QuestionnaireService = Depends(get_questionnaire_service),
):
    """List questionnaires with optional filtering."""
    try:
        filters = {"is_active": is_active}
        if questionnaire_type:
            filters["questionnaire_type"] = questionnaire_type  # type: ignore
        if tags:
            filters["tags"] = tags  # type: ignore

        questionnaires = questionnaire_service.list_questionnaires(
            limit=limit, offset=offset, filters=filters
        )
        return QuestionnairesResponse(
            questionnaires=questionnaires, count=len(questionnaires)
        )
    except Exception as e:
        logger.error(f"Error listing questionnaires: {str(e)}")
        raise HTTPException(
            status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error listing questionnaires: {str(e)}",
        )


@router.put("/{questionnaire_id}", response_model=QuestionnaireResponse)
async def update_questionnaire(
    questionnaire_update: QuestionnaireUpdate,
    questionnaire_id: str = Path(..., description="Questionnaire ID"),
    questionnaire_service: QuestionnaireService = Depends(get_questionnaire_service),
):
    """Update an existing questionnaire."""
    try:
        updated_questionnaire = questionnaire_service.update_questionnaire(
            questionnaire_id, questionnaire_update
        )
        if not updated_questionnaire:
            raise HTTPException(
                status_code=http_status.HTTP_404_NOT_FOUND,
                detail=f"Questionnaire with ID {questionnaire_id} not found",
            )
        return QuestionnaireResponse(
            questionnaire=updated_questionnaire,
            message="Questionnaire updated successfully",
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating questionnaire: {str(e)}")
        raise HTTPException(
            status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error updating questionnaire: {str(e)}",
        )


@router.delete("/{questionnaire_id}", status_code=http_status.HTTP_204_NO_CONTENT)
async def delete_questionnaire(
    questionnaire_id: str = Path(..., description="Questionnaire ID"),
    questionnaire_service: QuestionnaireService = Depends(get_questionnaire_service),
):
    """Delete a questionnaire."""
    try:
        success = questionnaire_service.delete_questionnaire(questionnaire_id)
        if not success:
            raise HTTPException(
                status_code=http_status.HTTP_404_NOT_FOUND,
                detail=f"Questionnaire with ID {questionnaire_id} not found",
            )
        return JSONResponse(status_code=http_status.HTTP_204_NO_CONTENT, content={})
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting questionnaire: {str(e)}")
        raise HTTPException(
            status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error deleting questionnaire: {str(e)}",
        )


@router.post("/start", response_model=QuestionnaireResponse)
async def start_questionnaire(
    user_id: str = Query(..., description="User ID"),
    questionnaire_id: str = Query(..., description="Questionnaire ID"),
    questionnaire_service: QuestionnaireService = Depends(get_questionnaire_service),
):
    """Start a questionnaire for a user."""
    try:
        user_questionnaire = questionnaire_service.start_questionnaire(
            user_id, questionnaire_id
        )
        if not user_questionnaire:
            raise HTTPException(
                status_code=http_status.HTTP_404_NOT_FOUND,
                detail=f"Questionnaire with ID {questionnaire_id} not found",
            )
        return QuestionnaireResponse(
            questionnaire=user_questionnaire,
            message="Questionnaire started successfully",
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting questionnaire: {str(e)}")
        raise HTTPException(
            status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error starting questionnaire: {str(e)}",
        )


@router.get("/progress/{user_questionnaire_id}", response_model=QuestionnaireResponse)
async def get_questionnaire_progress(
    user_questionnaire_id: str = Path(..., description="User Questionnaire ID"),
    questionnaire_service: QuestionnaireService = Depends(get_questionnaire_service),
):
    """Get a user's progress on a questionnaire."""
    try:
        progress = questionnaire_service.get_questionnaire_progress(
            user_questionnaire_id
        )
        if not progress:
            raise HTTPException(
                status_code=http_status.HTTP_404_NOT_FOUND,
                detail=f"User questionnaire with ID {user_questionnaire_id} not found",
            )
        return QuestionnaireResponse(questionnaire=progress)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving questionnaire progress: {str(e)}")
        raise HTTPException(
            status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving questionnaire progress: {str(e)}",
        )


@router.get("/user/{user_id}", response_model=UserQuestionnairesResponse)
async def get_user_questionnaires(
    user_id: str = Path(..., description="User ID"),
    status: Optional[QuestionnaireStatus] = Query(
        None, description="Filter by questionnaire status"
    ),
    questionnaire_service: QuestionnaireService = Depends(get_questionnaire_service),
):
    """Get all questionnaires for a user."""
    try:
        user_questionnaires = questionnaire_service.get_user_questionnaires(
            user_id, status
        )
        return UserQuestionnairesResponse(
            questionnaires=user_questionnaires, count=len(user_questionnaires)
        )
    except Exception as e:
        logger.error(f"Error retrieving user questionnaires: {str(e)}")
        raise HTTPException(
            status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving user questionnaires: {str(e)}",
        )


@router.get(
    "/progress/{user_questionnaire_id}/page/{page_number}",
    response_model=QuestionnairePageResponse,
)
async def get_questionnaire_page(
    user_questionnaire_id: str = Path(..., description="User Questionnaire ID"),
    page_number: int = Path(..., description="Page number"),
    questionnaire_service: QuestionnaireService = Depends(get_questionnaire_service),
):
    """Get a specific page of a questionnaire with the user's existing answers."""
    try:
        page_data = questionnaire_service.get_questionnaire_page(
            user_questionnaire_id, page_number
        )
        if not page_data:
            raise HTTPException(
                status_code=http_status.HTTP_404_NOT_FOUND,
                detail=f"""Page {page_number} not found
                for user questionnaire {user_questionnaire_id}""",
            )
        return page_data
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving questionnaire page: {str(e)}")
        raise HTTPException(
            status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving questionnaire page: {str(e)}",
        )


@router.post("/answer", response_model=QuestionnaireResponse)
async def submit_answers(
    answer_submission: AnswerSubmission,
    questionnaire_service: QuestionnaireService = Depends(get_questionnaire_service),
):
    """Submit answers for a questionnaire."""
    try:
        updated_questionnaire = questionnaire_service.submit_answers(answer_submission)
        if not updated_questionnaire:
            raise HTTPException(
                status_code=http_status.HTTP_404_NOT_FOUND,
                detail="User questionnaire not found",
            )
        return QuestionnaireResponse(
            questionnaire=updated_questionnaire,
            message="Answers submitted successfully",
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error submitting answers: {str(e)}")
        raise HTTPException(
            status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error submitting answers: {str(e)}",
        )


@router.post("/complete/{user_questionnaire_id}", response_model=QuestionnaireResponse)
async def complete_questionnaire(
    user_questionnaire_id: str = Path(..., description="User Questionnaire ID"),
    questionnaire_service: QuestionnaireService = Depends(get_questionnaire_service),
):
    """Mark a questionnaire as completed and process the results."""
    try:
        completed_questionnaire = questionnaire_service.complete_questionnaire(
            user_questionnaire_id
        )
        if not completed_questionnaire:
            raise HTTPException(
                status_code=http_status.HTTP_404_NOT_FOUND,
                detail=f"User questionnaire with ID {user_questionnaire_id} not found",
            )
        return QuestionnaireResponse(
            questionnaire=completed_questionnaire,
            message="Questionnaire completed and processed successfully",
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error completing questionnaire: {str(e)}")
        raise HTTPException(
            status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error completing questionnaire: {str(e)}",
        )


@router.get("/onboarding/{user_id}", response_model=QuestionnaireResponse)
async def get_next_onboarding_questionnaire(
    user_id: str = Path(..., description="User ID"),
    questionnaire_service: QuestionnaireService = Depends(get_questionnaire_service),
):
    """Get the next onboarding questionnaire for a user."""
    try:
        next_questionnaire = questionnaire_service.get_next_onboarding_questionnaire(
            user_id
        )
        if not next_questionnaire:
            return QuestionnaireResponse(
                questionnaire=None,
                message="Onboarding completed or no questionnaires available",
            )
        return QuestionnaireResponse(questionnaire=next_questionnaire)
    except Exception as e:
        logger.error(f"Error getting next onboarding questionnaire: {str(e)}")
        raise HTTPException(
            status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting next onboarding questionnaire: {str(e)}",
        )
