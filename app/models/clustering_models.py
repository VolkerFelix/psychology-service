"""Models for user clustering."""
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ClusteringAlgorithm(str, Enum):
    """Algorithms used for clustering user profiles."""

    KMEANS = "kmeans"
    HIERARCHICAL = "hierarchical"
    DBSCAN = "dbscan"
    SPECTRAL = "spectral"
    CUSTOM = "custom"


class ClusteringStatus(str, Enum):
    """Status of a clustering operation."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class ClusterIdentifier(BaseModel):
    """Basic identifier for a cluster."""

    cluster_id: str = Field(..., description="Unique identifier for the cluster")
    name: str = Field(..., description="Name of the cluster")
    size: int = Field(..., description="Number of users in the cluster")


class ClusterFeature(BaseModel):
    """Significant feature that characterizes a cluster."""

    feature_name: str = Field(..., description="Name of the feature")
    feature_value: Any = Field(..., description="Value or score of the feature")
    significance: float = Field(
        ..., ge=0, le=1, description="Significance of this feature (0-1)"
    )
    comparison_to_mean: Optional[float] = Field(
        None, description="Difference from overall population mean"
    )


class Cluster(BaseModel):
    """Complete definition of a psychological cluster."""

    cluster_id: str = Field(..., description="Unique identifier for the cluster")
    name: str = Field(..., description="Name of the cluster")
    description: str = Field(..., description="Description of the cluster")
    size: int = Field(..., description="Number of users in the cluster")
    centroid: Dict[str, float] = Field(
        ..., description="Centroid values for each dimension"
    )
    key_features: List[ClusterFeature] = Field(
        ..., description="Key features that define this cluster"
    )
    created_at: datetime = Field(
        default_factory=datetime.now, description="When the cluster was created"
    )
    updated_at: datetime = Field(
        default_factory=datetime.now, description="When the cluster was last updated"
    )
    version: str = Field(default="1.0.0", description="Version of the cluster model")
    is_active: bool = Field(default=True, description="Whether the cluster is active")
    tags: Optional[List[str]] = Field(None, description="Tags for categorization")


class ClusteringModel(BaseModel):
    """A clustering model for psychological profiles."""

    clustering_model_id: str = Field(
        ..., description="Unique identifier for the clustering model"
    )
    name: str = Field(..., description="Name of the clustering model")
    description: str = Field(..., description="Description of the clustering model")
    algorithm: ClusteringAlgorithm = Field(
        ..., description="Algorithm used for clustering"
    )
    parameters: Dict[str, Any] = Field(
        ..., description="Parameters used for the clustering algorithm"
    )
    num_clusters: int = Field(..., description="Number of clusters in the model")
    features_used: List[str] = Field(
        ..., description="Profile features used for clustering"
    )
    clusters: List[Cluster] = Field(..., description="Clusters defined in this model")
    created_at: datetime = Field(
        default_factory=datetime.now, description="When the model was created"
    )
    updated_at: datetime = Field(
        default_factory=datetime.now, description="When the model was last updated"
    )
    version: str = Field(default="1.0.0", description="Version of the clustering model")
    is_active: bool = Field(default=True, description="Whether the model is active")
    total_users_clustered: int = Field(
        ..., description="Total number of users clustered with this model"
    )
    quality_metrics: Dict[str, float] = Field(
        ..., description="Quality metrics for the clustering model"
    )


class UserClusterAssignment(BaseModel):
    """Assignment of a user to a psychological cluster."""

    assignment_id: str = Field(..., description="Unique identifier for this assignment")
    user_id: str = Field(..., description="User identifier")
    cluster_id: str = Field(..., description="Cluster identifier")
    clustering_model_id: str = Field(..., description="Clustering model identifier")
    confidence_score: float = Field(
        ..., ge=0, le=1, description="Confidence in cluster assignment (0-1)"
    )
    features: Dict[str, float] = Field(
        ..., description="User's features used for clustering"
    )
    distance_to_centroid: float = Field(
        ..., description="Distance to the cluster centroid"
    )
    assigned_at: datetime = Field(
        default_factory=datetime.now, description="When the assignment was made"
    )
    is_current: bool = Field(
        default=True, description="Whether this is the current assignment"
    )


class ClusteringRequest(BaseModel):
    """Request to create a new clustering model."""

    name: str = Field(..., description="Name of the clustering model")
    description: str = Field(..., description="Description of the clustering model")
    algorithm: ClusteringAlgorithm = Field(
        ..., description="Algorithm to use for clustering"
    )
    parameters: Dict[str, Any] = Field(
        ..., description="Parameters for the clustering algorithm"
    )
    features_to_use: List[str] = Field(
        ..., description="Profile features to use for clustering"
    )
    target_num_clusters: Optional[int] = Field(
        None, description="Target number of clusters (if applicable)"
    )


class ClusteringJob(BaseModel):
    """A job for running the clustering algorithm."""

    job_id: str = Field(..., description="Unique identifier for the clustering job")
    name: str = Field(..., description="Name of the clustering job")
    description: Optional[str] = Field(None, description="Description of the job")
    status: ClusteringStatus = Field(
        default=ClusteringStatus.PENDING, description="Status of the job"
    )
    algorithm: ClusteringAlgorithm = Field(
        ..., description="Algorithm used for clustering"
    )
    parameters: Dict[str, Any] = Field(
        ..., description="Parameters for the clustering algorithm"
    )
    created_at: datetime = Field(
        default_factory=datetime.now, description="When the job was created"
    )
    started_at: Optional[datetime] = Field(None, description="When the job was started")
    completed_at: Optional[datetime] = Field(
        None, description="When the job was completed"
    )
    user_count: Optional[int] = Field(
        None, description="Number of users included in clustering"
    )
    error_message: Optional[str] = Field(
        None, description="Error message if the job failed"
    )
    result_clustering_model_id: Optional[str] = Field(
        None, description="ID of the resulting clustering model"
    )


class ClusteringModelResponse(BaseModel):
    """Response model for clustering model operations."""

    model: ClusteringModel
    status: str = "success"
    message: Optional[str] = None


class ClusteringModelsResponse(BaseModel):
    """Response model for multiple clustering models."""

    models: List[ClusteringModel]
    count: int
    status: str = "success"


class ClusterResponse(BaseModel):
    """Response model for cluster operations."""

    cluster: Cluster
    status: str = "success"
    message: Optional[str] = None


class ClustersResponse(BaseModel):
    """Response model for multiple clusters."""

    clusters: List[Cluster]
    count: int
    status: str = "success"


class UserClusterResponse(BaseModel):
    """Response model for user cluster assignment operations."""

    assignment: UserClusterAssignment
    cluster: Cluster
    status: str = "success"
    message: Optional[str] = None


class ClusteringJobResponse(BaseModel):
    """Response model for clustering job operations."""

    job: ClusteringJob
    status: str = "success"
    message: Optional[str] = None


class ClusteringJobsResponse(BaseModel):
    """Response model for multiple clustering jobs."""

    jobs: List[ClusteringJob]
    count: int
    status: str = "success"
