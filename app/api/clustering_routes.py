"""API routes for clustering operations."""

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Path, Query, status
from loguru import logger

from app.models.clustering_models import (
    Cluster,
    ClusteringJobResponse,
    ClusteringJobsResponse,
    ClusteringModelResponse,
    ClusteringModelsResponse,
    ClusteringRequest,
    ClusterResponse,
    ClustersResponse,
    UserClusterResponse,
)
from app.services.clustering_service import ClusteringService
from app.services.storage.factory import StorageFactory

router = APIRouter(prefix="/clustering", tags=["clustering"])


def get_storage_service():
    """Get the storage service."""
    return StorageFactory.create_storage_service()


def get_clustering_service(storage_service=Depends(get_storage_service)):
    """Get the clustering service."""
    return ClusteringService(storage_service)


@router.post(
    "/run", response_model=ClusteringJobResponse, status_code=status.HTTP_202_ACCEPTED
)
async def run_clustering(
    request: ClusteringRequest,
    clustering_service: ClusteringService = Depends(get_clustering_service),
):
    """Run a clustering job to create a new clustering model."""
    try:
        job = clustering_service.create_clustering_job(request)
        return ClusteringJobResponse(
            job=job, message="Clustering job started successfully"
        )
    except Exception as e:
        logger.error(f"Error starting clustering job: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error starting clustering job: {str(e)}",
        )


@router.get("/jobs/{job_id}", response_model=ClusteringJobResponse)
async def get_clustering_job(
    job_id: str = Path(..., description="Job ID"),
    clustering_service: ClusteringService = Depends(get_clustering_service),
):
    """Get information about a clustering job."""
    try:
        job = clustering_service.get_clustering_job(job_id)
        if not job:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Clustering job with ID {job_id} not found",
            )
        return ClusteringJobResponse(job=job)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving clustering job: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving clustering job: {str(e)}",
        )


@router.get("/jobs", response_model=ClusteringJobsResponse)
async def list_clustering_jobs(
    limit: int = Query(100, description="Maximum number of jobs to return"),
    offset: int = Query(0, description="Number of jobs to skip"),
    clustering_service: ClusteringService = Depends(get_clustering_service),
):
    """List clustering jobs."""
    try:
        jobs = clustering_service.list_clustering_jobs(limit=limit, offset=offset)
        return ClusteringJobsResponse(jobs=jobs, count=len(jobs))
    except Exception as e:
        logger.error(f"Error listing clustering jobs: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error listing clustering jobs: {str(e)}",
        )


@router.get("/models/{clustering_model_id}", response_model=ClusteringModelResponse)
async def get_clustering_model(
    clustering_model_id: str = Path(..., description="Model ID"),
    clustering_service: ClusteringService = Depends(get_clustering_service),
) -> ClusteringModelResponse:
    """Get a clustering model by ID."""
    model = clustering_service.get_clustering_model(clustering_model_id)
    if not model:
        raise HTTPException(
            status_code=404,
            detail=f"Clustering model with ID {clustering_model_id} not found",
        )

    return ClusteringModelResponse(model=model)


@router.get("/models", response_model=ClusteringModelsResponse)
async def list_clustering_models(
    limit: int = Query(100, description="Maximum number of models to return"),
    offset: int = Query(0, description="Number of models to skip"),
    include_inactive: bool = Query(
        False, description="Whether to include inactive models"
    ),
    clustering_service: ClusteringService = Depends(get_clustering_service),
):
    """List clustering models."""
    try:
        models = clustering_service.list_clustering_models(
            limit=limit, offset=offset, include_inactive=include_inactive
        )
        return ClusteringModelsResponse(models=models, count=len(models))
    except Exception as e:
        logger.error(f"Error listing clustering models: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error listing clustering models: {str(e)}",
        )


@router.post(
    "/models/{clustering_model_id}/activate", response_model=ClusteringModelResponse
)
async def activate_clustering_model(
    clustering_model_id: str = Path(..., description="Model ID"),
    clustering_service: ClusteringService = Depends(get_clustering_service),
) -> ClusteringModelResponse:
    """Activate a clustering model."""
    activated_model = clustering_service.activate_clustering_model(clustering_model_id)
    if not activated_model:
        raise HTTPException(
            status_code=404,
            detail=f"Clustering model with ID {clustering_model_id} not found",
        )

    return ClusteringModelResponse(
        model=activated_model,
        message=f"Model {clustering_model_id} activated successfully",
    )


@router.get("/models/active", response_model=ClusteringModelResponse)
async def get_active_clustering_model(
    clustering_service: ClusteringService = Depends(get_clustering_service),
):
    """Get the current active clustering model."""
    try:
        model = clustering_service.get_active_clustering_model()
        if not model:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No active clustering model found",
            )
        return ClusteringModelResponse(model=model)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving active clustering model: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving active clustering model: {str(e)}",
        )


@router.get("/clusters/{cluster_id}", response_model=ClusterResponse)
async def get_cluster(
    cluster_id: str = Path(..., description="Cluster ID"),
    clustering_model_id: Optional[str] = Query(None, description="Model ID (optional)"),
    clustering_service: ClusteringService = Depends(get_clustering_service),
) -> ClusterResponse:
    """Get a cluster by ID."""
    cluster = clustering_service.get_cluster(cluster_id, clustering_model_id)
    if not cluster:
        raise HTTPException(
            status_code=404,
            detail=f"Cluster with ID {cluster_id} not found",
        )

    return ClusterResponse(cluster=cluster)


@router.get("/clusters", response_model=ClustersResponse)
async def list_clusters(
    clustering_model_id: Optional[str] = Query(None, description="Model ID (optional)"),
    clustering_service: ClusteringService = Depends(get_clustering_service),
) -> ClustersResponse:
    """List all clusters."""
    clusters = clustering_service.list_clusters(clustering_model_id)

    return ClustersResponse(clusters=clusters, count=len(clusters))


@router.get("/users/{user_id}/cluster", response_model=UserClusterResponse)
async def get_user_cluster(
    user_id: str = Path(..., description="User ID"),
    clustering_model_id: Optional[str] = Query(None, description="Model ID (optional)"),
    clustering_service: ClusteringService = Depends(get_clustering_service),
) -> UserClusterResponse:
    """Get a user's cluster assignment."""
    assignment = clustering_service.get_user_cluster_assignment(user_id)
    if not assignment:
        raise HTTPException(
            status_code=404,
            detail=f"No cluster assignment found for user {user_id}",
        )

    # Get the cluster
    cluster = clustering_service.get_cluster(
        assignment["assignment"].cluster_id,
        assignment["assignment"].clustering_model_id,
    )
    if not cluster:
        raise HTTPException(
            status_code=404,
            detail=f"Cluster with ID {assignment['assignment'].cluster_id} not found",
        )

    return UserClusterResponse(assignment=assignment["assignment"], cluster=cluster)


@router.post("/user/{user_id}/assign", response_model=UserClusterResponse)
async def assign_user_to_cluster(
    user_id: str = Path(..., description="User ID"),
    clustering_service: ClusteringService = Depends(get_clustering_service),
):
    """Assign a user to a cluster based on their profile."""
    try:
        result = clustering_service.assign_user_to_cluster(user_id)
        if not result:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Could not assign user {user_id} to a cluster",
            )
        return UserClusterResponse(
            assignment=result["assignment"],
            cluster=result["cluster"],
            message="User assigned to cluster successfully",
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error assigning user to cluster: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error assigning user to cluster: {str(e)}",
        )


@router.get("/clusters/{cluster_id}/users", response_model=ClustersResponse)
async def get_cluster_users(
    cluster_id: str = Path(..., description="Cluster ID"),
    limit: int = Query(100, description="Maximum number of users to return"),
    offset: int = Query(0, description="Number of users to skip"),
    model_id: Optional[str] = Query(None, description="Model ID (optional)"),
    clustering_service: ClusteringService = Depends(get_clustering_service),
):
    """Get users assigned to a specific cluster."""
    try:
        user_assignments = clustering_service.get_cluster_users(
            cluster_id, model_id, limit=limit, offset=offset
        )

        # Convert dictionary results to Cluster objects
        clusters = [Cluster(**user) for user in user_assignments]

        return ClustersResponse(count=len(clusters), clusters=clusters)
    except Exception as e:
        logger.error(f"Error retrieving cluster users: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving cluster users: {str(e)}",
        )


@router.post("/jobs/reassign", response_model=ClusteringJobResponse)
async def create_reassignment_job(
    clustering_model_id: Optional[str] = Query(None, description="Model ID (optional)"),
    clustering_service: ClusteringService = Depends(get_clustering_service),
) -> ClusteringJobResponse:
    """Create a job to reassign users to clusters."""
    job = clustering_service.create_reassignment_job(clustering_model_id)

    return ClusteringJobResponse(
        job=job,
        message="Reassignment job created successfully",
    )
