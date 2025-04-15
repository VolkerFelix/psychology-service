"""API routes module for the psychology profiling microservice.

This module defines the main API router and includes
all sub-routers for different API endpoints.
It serves as the central point for organizing all API routes.
"""
from fastapi import APIRouter

from app.api.clustering_routes import router as clustering_router
from app.api.profile_routes import router as profile_router
from app.api.questionnaire_routes import router as questionnaire_router

# Main API router
router = APIRouter(prefix="/api")

# Include sub-routers
router.include_router(profile_router)
router.include_router(questionnaire_router)
router.include_router(clustering_router)
