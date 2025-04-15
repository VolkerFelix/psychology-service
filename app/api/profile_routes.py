"""API routes for psychological profiles."""

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Path, Query, status
from fastapi.responses import JSONResponse
from loguru import logger

from app.models.profile_models import (
    ProfileCreate,
    ProfileResponse,
    ProfilesResponse,
    ProfileUpdate,
)
from app.services.profile_service import ProfileService
from app.services.storage.factory import StorageFactory

router = APIRouter(prefix="/profiles", tags=["profiles"])


def get_storage_service():
    """Get the storage service."""
    return StorageFactory.create_storage_service()


def get_profile_service(storage_service=Depends(get_storage_service)):
    """Get the profile service."""
    return ProfileService(storage_service)


@router.post("/", response_model=ProfileResponse, status_code=status.HTTP_201_CREATED)
async def create_profile(
    profile: ProfileCreate,
    profile_service: ProfileService = Depends(get_profile_service),
):
    """Create a new psychological profile."""
    try:
        created_profile = profile_service.create_profile(profile)
        return ProfileResponse(
            profile=created_profile, message="Profile created successfully"
        )
    except Exception as e:
        logger.error(f"Error creating profile: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error creating profile: {str(e)}",
        )


@router.get("/{profile_id}", response_model=ProfileResponse)
async def get_profile(
    profile_id: str = Path(..., description="Profile ID"),
    profile_service: ProfileService = Depends(get_profile_service),
):
    """Get a specific psychological profile by ID."""
    try:
        profile = profile_service.get_profile(profile_id)
        if not profile:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Profile with ID {profile_id} not found",
            )
        return ProfileResponse(profile=profile)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving profile: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving profile: {str(e)}",
        )


@router.get("/user/{user_id}", response_model=ProfileResponse)
async def get_profile_by_user(
    user_id: str = Path(..., description="User ID"),
    profile_service: ProfileService = Depends(get_profile_service),
):
    """Get a user's psychological profile."""
    try:
        profile = profile_service.get_profile_by_user(user_id)
        if not profile:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Profile for user {user_id} not found",
            )
        return ProfileResponse(profile=profile)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving profile: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving profile: {str(e)}",
        )


@router.put("/{profile_id}", response_model=ProfileResponse)
async def update_profile(
    profile_update: ProfileUpdate,
    profile_id: str = Path(..., description="Profile ID"),
    profile_service: ProfileService = Depends(get_profile_service),
):
    """Update an existing psychological profile."""
    try:
        updated_profile = profile_service.update_profile(profile_id, profile_update)
        if not updated_profile:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Profile with ID {profile_id} not found",
            )
        return ProfileResponse(
            profile=updated_profile, message="Profile updated successfully"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating profile: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error updating profile: {str(e)}",
        )


@router.delete("/{profile_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_profile(
    profile_id: str = Path(..., description="Profile ID"),
    profile_service: ProfileService = Depends(get_profile_service),
):
    """Delete a psychological profile."""
    try:
        success = profile_service.delete_profile(profile_id)
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Profile with ID {profile_id} not found",
            )
        return JSONResponse(status_code=status.HTTP_204_NO_CONTENT, content={})
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting profile: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error deleting profile: {str(e)}",
        )


@router.get("/", response_model=ProfilesResponse)
async def list_profiles(
    limit: int = Query(100, description="Maximum number of profiles to return"),
    offset: int = Query(0, description="Number of profiles to skip"),
    cluster_id: Optional[str] = Query(None, description="Filter by cluster ID"),
    profile_service: ProfileService = Depends(get_profile_service),
):
    """List psychological profiles with optional filtering."""
    try:
        filters = {}
        if cluster_id:
            filters["cluster_id"] = cluster_id

        profiles = profile_service.list_profiles(
            limit=limit, offset=offset, filters=filters
        )
        return ProfilesResponse(profiles=profiles, count=len(profiles))
    except Exception as e:
        logger.error(f"Error listing profiles: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error listing profiles: {str(e)}",
        )
