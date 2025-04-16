"""Service for clustering psychological profiles."""

import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from loguru import logger
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from app.config.settings import settings
from app.models.clustering_models import (
    Cluster,
    ClusteringAlgorithm,
    ClusteringJob,
    ClusteringModel,
    ClusteringRequest,
    ClusteringStatus,
    UserClusterAssignment,
)


class ClusteringService:
    """Service for clustering psychological profiles."""

    def __init__(self, storage_service):
        """Initialize with a storage service."""
        self.storage = storage_service

    def _convert_status_to_enum(self, status: Any) -> ClusteringStatus:
        """Convert status to ClusteringStatus enum."""
        if isinstance(status, ClusteringStatus):
            return status

        if isinstance(status, str):
            try:
                return ClusteringStatus(status)
            except ValueError:
                # Fallback to a default status if conversion fails
                logger.warning(f"Invalid status '{status}', defaulting to PENDING")
                return ClusteringStatus.PENDING

        # Fallback for any other type
        logger.warning(f"Unexpected status type {type(status)}, defaulting to PENDING")
        return ClusteringStatus.PENDING

    def _convert_algorithm_to_enum(self, algorithm: Any) -> ClusteringAlgorithm:
        """Convert algorithm to ClusteringAlgorithm enum."""
        if isinstance(algorithm, ClusteringAlgorithm):
            return algorithm

        if isinstance(algorithm, str):
            try:
                return ClusteringAlgorithm(algorithm)
            except ValueError:
                # Fallback to a default status if conversion fails
                logger.warning(f"Invalid algorithm '{algorithm}', defaulting to KMEANS")
                return ClusteringAlgorithm.KMEANS

        # Fallback for any other type
        logger.warning(
            f"Unexpected algorithm type {type(algorithm)}, defaulting to KMEANS"
        )
        return ClusteringAlgorithm.KMEANS

    def _convert_parameters_to_dict(self, parameters: Any) -> Dict[str, Any]:
        """Convert parameters to a dictionary."""
        if isinstance(parameters, dict):
            return parameters

        # If it's a collection but not a dict, try to convert it
        if hasattr(parameters, "__iter__") and not isinstance(parameters, (str, bytes)):
            try:
                # Try to convert to dict if it's a collection of key-value pairs
                return dict(parameters)
            except (TypeError, ValueError):
                # If conversion fails, create a default dict
                logger.warning("Could not convert parameters to dict, using empty dict")
                return {}

        # For any other type, return an empty dict
        logger.warning(
            f"Unexpected parameters type {type(parameters)}, using empty dict"
        )
        return {}

    def create_clustering_job(self, request: ClusteringRequest) -> ClusteringJob:
        """
        Create a new clustering job.

        Args:
            request: Clustering request data

        Returns:
            Created clustering job
        """
        # Generate job ID
        job_id = str(uuid.uuid4())
        now = datetime.now()

        # Create job data
        job_data = {
            "job_id": job_id,
            "name": request.name,
            "description": request.description,
            "status": ClusteringStatus.PENDING,
            "algorithm": request.algorithm,
            "parameters": self._convert_parameters_to_dict(request.parameters),
            "created_at": now.isoformat(),
        }

        # Save job to storage
        success = self.storage.save_clustering_job(job_data)
        if not success:
            raise Exception("Failed to save clustering job to storage")

        # Start job processing (in a background task or queue in a real implementation)
        # For this example, we'll process synchronously
        self._process_clustering_job(job_id)

        # Return job data
        return ClusteringJob(
            job_id=job_id,
            name=request.name,
            description=request.description,
            status=ClusteringStatus.PENDING,
            algorithm=request.algorithm,
            parameters=self._convert_parameters_to_dict(request.parameters),
            created_at=now,
            started_at=None,
            completed_at=None,
            user_count=None,
            error_message=None,
            result_clustering_model_id=None,
        )

    def get_clustering_job(self, job_id: str) -> Optional[ClusteringJob]:
        """
        Get a clustering job by ID.

        Args:
            job_id: Job ID

        Returns:
            Job if found, None otherwise
        """
        job_data = self.storage.get_clustering_job(job_id)
        if not job_data:
            return None

        # Convert string status to enum
        status = self._convert_status_to_enum(job_data["status"])

        # Convert string algorithm to enum
        algorithm = self._convert_algorithm_to_enum(job_data["algorithm"])

        return ClusteringJob(
            job_id=str(job_data["job_id"]),  # Ensure job_id is a string
            name=str(job_data["name"]),
            description=str(job_data["description"]),
            status=status,  # Use the enum directly
            algorithm=algorithm,  # Use the enum directly
            parameters=self._convert_parameters_to_dict(job_data["parameters"]),
            created_at=datetime.fromisoformat(job_data["created_at"])
            if isinstance(job_data["created_at"], str)
            else job_data["created_at"],
            started_at=datetime.fromisoformat(job_data["started_at"])
            if job_data.get("started_at") and isinstance(job_data["started_at"], str)
            else job_data.get("started_at"),
            completed_at=datetime.fromisoformat(job_data["completed_at"])
            if job_data.get("completed_at")
            and isinstance(job_data["completed_at"], str)
            else job_data.get("completed_at"),
            user_count=job_data.get("user_count"),
            error_message=job_data.get("error_message"),
            result_clustering_model_id=job_data.get("result_clustering_model_id"),
        )

    def list_clustering_jobs(
        self,
        limit: int = 100,
        offset: int = 0,
        status: Optional[ClusteringStatus] = None,
    ) -> List[ClusteringJob]:
        """
        List clustering jobs.

        Args:
            limit: Maximum number of jobs to return
            offset: Number of jobs to skip
            status: Optional status filter

        Returns:
            List of jobs
        """
        jobs_data = self.storage.get_clustering_jobs(
            limit=limit, offset=offset, status=status.value if status else None
        )
        return [
            ClusteringJob(
                job_id=str(job["job_id"]),  # Ensure job_id is a string
                name=str(job["name"]),
                description=str(job["description"]),
                status=ClusteringStatus(job["status"])
                if isinstance(job["status"], str)
                else job["status"],
                algorithm=ClusteringAlgorithm(job["algorithm"])
                if isinstance(job["algorithm"], str)
                else job["algorithm"],
                parameters=self._convert_parameters_to_dict(job["parameters"]),
                created_at=datetime.fromisoformat(job["created_at"])
                if isinstance(job["created_at"], str)
                else job["created_at"],
                started_at=datetime.fromisoformat(job["started_at"])
                if job.get("started_at") and isinstance(job["started_at"], str)
                else job.get("started_at"),
                completed_at=datetime.fromisoformat(job["completed_at"])
                if job.get("completed_at") and isinstance(job["completed_at"], str)
                else job.get("completed_at"),
                user_count=job.get("user_count"),
                error_message=job.get("error_message"),
                result_clustering_model_id=job.get("result_clustering_model_id"),
            )
            for job in jobs_data
        ]

    def get_clustering_model(
        self, clustering_model_id: str
    ) -> Optional[ClusteringModel]:
        """
        Get a clustering model by ID.

        Args:
            clustering_model_id: Model ID

        Returns:
            ClusteringModel if found, None otherwise
        """
        model_data = self.storage.get_clustering_model(clustering_model_id)
        if not model_data:
            return None

        return ClusteringModel(**model_data)

    def get_active_clustering_model(self) -> Optional[ClusteringModel]:
        """
        Get the current active clustering model.

        Returns:
            Active model if found, None otherwise
        """
        model_data = self.storage.get_active_clustering_model()
        if not model_data:
            return None

        return ClusteringModel(**model_data)

    def list_clustering_models(
        self, limit: int = 100, offset: int = 0, include_inactive: bool = False
    ) -> List[ClusteringModel]:
        """
        List clustering models.

        Args:
            limit: Maximum number of models to return
            offset: Number of models to skip
            include_inactive: Whether to include inactive models

        Returns:
            List of models
        """
        models_data = self.storage.get_clustering_models(
            limit=limit, offset=offset, include_inactive=include_inactive
        )
        return [ClusteringModel(**model) for model in models_data]

    def activate_clustering_model(
        self, clustering_model_id: str
    ) -> Optional[ClusteringModel]:
        """
        Activate a clustering model and deactivate all others.

        Args:
            clustering_model_id: Model ID

        Returns:
            Activated ClusteringModel if found, None otherwise
        """
        model_data = self.storage.get_clustering_model(clustering_model_id)
        if not model_data:
            return None

        # Deactivate all other models
        all_models = self.storage.list_clustering_models()
        for other_model in all_models:
            if other_model[
                "clustering_model_id"
            ] != clustering_model_id and other_model.get("is_active"):
                self.storage.update_clustering_model(
                    other_model["clustering_model_id"], {"is_active": False}
                )

        # Activate the requested model
        updated_model = self.storage.update_clustering_model(
            clustering_model_id, {"is_active": True}
        )

        return ClusteringModel(**updated_model) if updated_model else None

    def get_cluster(
        self, cluster_id: str, clustering_model_id: Optional[str] = None
    ) -> Optional[Cluster]:
        """
        Get a cluster by ID.

        Args:
            cluster_id: Cluster ID
            clustering_model_id: Optional model ID

        Returns:
            Cluster if found, None otherwise
        """
        if clustering_model_id:
            model_data = self.storage.get_clustering_model(clustering_model_id)
            if not model_data:
                return None

            # Find the cluster in the model
            for cluster_data in model_data.get("clusters", []):
                if cluster_data.get("cluster_id") == cluster_id:
                    return Cluster(**cluster_data)
        else:
            # Search all active models
            active_models = self.storage.list_clustering_models(is_active=True)
            for model_data in active_models:
                for cluster_data in model_data.get("clusters", []):
                    if cluster_data.get("cluster_id") == cluster_id:
                        return Cluster(**cluster_data)

        return None

    def list_clusters(self, clustering_model_id: Optional[str] = None) -> List[Cluster]:
        """
        List all clusters.

        Args:
            clustering_model_id: Optional model ID

        Returns:
            List of clusters
        """
        clusters = []

        if clustering_model_id:
            model_data = self.storage.get_clustering_model(clustering_model_id)
            if not model_data:
                return []

            # Get clusters from the specified model
            for cluster_data in model_data.get("clusters", []):
                clusters.append(Cluster(**cluster_data))
        else:
            # Get clusters from all active models
            active_models = self.storage.list_clustering_models(is_active=True)
            for model_data in active_models:
                for cluster_data in model_data.get("clusters", []):
                    clusters.append(Cluster(**cluster_data))

        return clusters

    def get_user_cluster_assignment(self, user_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a user's cluster assignment.

        Args:
            user_id: User ID

        Returns:
            Dictionary with assignment and cluster data if found, None otherwise
        """
        # Get user's current assignment
        assignment_data = self.storage.get_user_cluster_assignment(
            user_id, current_only=True
        )
        if not assignment_data:
            return None

        # Get the cluster
        cluster = self.get_cluster(
            assignment_data["cluster_id"], assignment_data["clustering_model_id"]
        )
        if not cluster:
            return None

        return {
            "assignment": UserClusterAssignment(**assignment_data),
            "cluster": cluster,
        }

    def assign_user_to_cluster(self, user_id: str) -> Optional[Dict[str, Any]]:
        """
        Assign a user to a cluster based on their profile.

        Args:
            user_id: User ID

        Returns:
            Dictionary with assignment and cluster data if successful, None otherwise
        """
        # Get active clustering model
        model_data = self.storage.get_active_clustering_model()
        if not model_data:
            return None

        # Get user's profile
        profile = self.storage.get_profile_by_user(user_id)
        if not profile:
            return None

        # Extract features for clustering
        features = self._extract_profile_features(
            profile, model_data.get("features_used", [])
        )

        # Find nearest cluster
        cluster_id, cluster_data, distance, confidence = self._find_nearest_cluster(
            features, model_data
        )

        if not cluster_id or not cluster_data:
            return None

        # Create assignment
        assignment_id = str(uuid.uuid4())
        assignment_data = {
            "assignment_id": assignment_id,
            "user_id": user_id,
            "cluster_id": cluster_id,
            "clustering_model_id": model_data["clustering_model_id"],
            "confidence_score": confidence,
            "features": features,
            "distance_to_centroid": distance,
            "assigned_at": datetime.now().isoformat(),
            "is_current": True,
        }

        # Save assignment
        success = self.storage.save_user_cluster_assignment(assignment_data)
        if not success:
            raise Exception("Failed to save user cluster assignment")

        # Increment user count in model
        model_data["total_users_clustered"] = (
            model_data.get("total_users_clustered", 0) + 1
        )
        self.storage.save_clustering_model(model_data)

        return {
            "assignment": UserClusterAssignment(**assignment_data),
            "cluster": Cluster(**cluster_data),
        }

    def get_cluster_users(
        self,
        cluster_id: str,
        clustering_model_id: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """
        Get users assigned to a specific cluster.

        Args:
            cluster_id: Cluster ID
            clustering_model_id: Optional model ID
            limit: Maximum number of assignments to return
            offset: Number of assignments to skip

        Returns:
            List of user assignments
        """
        assignments = self.storage.get_cluster_users(
            cluster_id=cluster_id,
            clustering_model_id=clustering_model_id,
            limit=limit,
            offset=offset,
        )

        return assignments

    def create_reassignment_job(
        self, clustering_model_id: Optional[str] = None
    ) -> ClusteringJob:
        """
        Create a job to reassign users to clusters.

        Args:
            clustering_model_id: Optional model ID to use

        Returns:
            Created ClusteringJob
        """
        # Get the target model
        target_model_id = clustering_model_id
        if not target_model_id:
            # Use the active model if no specific model is provided
            active_models = self.storage.list_clustering_models(is_active=True)
            if not active_models:
                raise Exception("No active clustering model found")

            active_model = active_models[0]
            target_model_id = active_model["clustering_model_id"]

        # Create the job data
        job_id = str(uuid.uuid4())
        now = datetime.now()
        job_data = {
            "job_id": job_id,
            "name": "User cluster reassignment",
            "description": f"""Reassign all users to clusters
            using model {target_model_id}""",
            "status": ClusteringStatus.PENDING,
            "algorithm": ClusteringAlgorithm.CUSTOM,
            "parameters": {
                "clustering_model_id": target_model_id,
                "job_type": "reassignment",
            },
            "created_at": now.isoformat(),
        }

        # Save the job
        saved_job = self.storage.save_clustering_job(job_data)
        if not saved_job:
            raise Exception("Failed to save clustering job")

        status = self._convert_status_to_enum(job_data["status"])
        algorithm = self._convert_algorithm_to_enum(job_data["algorithm"])

        # Return job data
        return ClusteringJob(
            job_id=str(job_data["job_id"]),
            name=str(job_data["name"]),
            description=str(job_data["description"]),
            status=status,
            algorithm=algorithm,
            parameters=self._convert_parameters_to_dict(job_data["parameters"]),
            created_at=now,
            started_at=None,
            completed_at=None,
            user_count=None,
            error_message=None,
            result_clustering_model_id=None,
        )

    def _process_clustering_job(self, job_id: str) -> None:
        """
        Process a clustering job.

        In a real implementation, this would be handled by a background task or queue.

        Args:
            job_id: Job ID
        """
        # Get job data
        job_data = self.storage.get_clustering_job(job_id)
        if not job_data:
            logger.error(f"Clustering job {job_id} not found")
            return

        try:
            # Update job status
            job_data["status"] = ClusteringStatus.IN_PROGRESS
            job_data["started_at"] = datetime.now().isoformat()
            self.storage.save_clustering_job(job_data)

            # Get profiles for clustering
            profiles = self.storage.get_profiles(
                limit=1000
            )  # In real app, handle pagination

            if len(profiles) < settings.MIN_USERS_FOR_CLUSTERING:
                raise Exception(
                    f"""Not enough users for clustering
                    (minimum {settings.MIN_USERS_FOR_CLUSTERING})"""
                )

            # Extract features for clustering
            feature_names = job_data["parameters"].get("features_to_use", [])
            if not feature_names:
                # Default features
                feature_names = [
                    "openness",
                    "conscientiousness",
                    "extraversion",
                    "agreeableness",
                    "neuroticism",
                    "chronotype",
                    "sleep_environment",
                    "sleep_anxiety",
                    "stress_response",
                    "routine_consistency",
                ]

            # Create feature matrix
            features = []
            user_ids = []

            for profile in profiles:
                profile_features = self._extract_profile_features(
                    profile, feature_names
                )

                # Skip profiles with missing features
                if not all(f is not None for f in profile_features.values()):
                    continue

                features.append(list(profile_features.values()))
                user_ids.append(profile["user_id"])

            if not features:
                raise Exception("No profiles with complete feature data")

            # Convert to numpy array
            X = np.array(features)

            # Update job with user count
            job_data["user_count"] = len(X)
            self.storage.save_clustering_job(job_data)

            # Create a clustering model
            clusters, centroids, labels, inertia = self._create_clusters(
                X, job_data["algorithm"], job_data["parameters"]
            )

            # Generate model ID
            clustering_model_id = str(uuid.uuid4())

            # Create model data
            model_data = {
                "clustering_model_id": clustering_model_id,
                "name": job_data["name"],
                "description": job_data["description"],
                "algorithm": job_data["algorithm"],
                "parameters": job_data["parameters"],
                "num_clusters": len(clusters),
                "features_used": feature_names,
                "clusters": clusters,
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat(),
                "version": "1.0.0",
                "is_active": False,  # New models are inactive by default
                "total_users_clustered": 0,
                "quality_metrics": {
                    "inertia": float(inertia),
                    "silhouette_score": 0.0,  # Calculate with sklearn if needed
                },
            }

            # Save model to storage
            success = self.storage.save_clustering_model(model_data)
            if not success:
                raise Exception("Failed to save clustering model to storage")

            # Update job with result
            job_data["status"] = ClusteringStatus.COMPLETED
            job_data["completed_at"] = datetime.now().isoformat()
            job_data["result_clustering_model_id"] = clustering_model_id
            self.storage.save_clustering_job(job_data)

        except Exception as e:
            logger.error(f"Error processing clustering job {job_id}: {str(e)}")

            # Update job with error
            job_data["status"] = ClusteringStatus.FAILED
            job_data["completed_at"] = datetime.now().isoformat()
            job_data["error_message"] = str(e)
            self.storage.save_clustering_job(job_data)

    def _process_reassignment_job(self, job_id: str) -> None:
        """
        Process a reassignment job.

        In a real implementation, this would be handled by a background task or queue.

        Args:
            job_id: Job ID
        """
        # Get job data
        job_data = self.storage.get_clustering_job(job_id)
        if not job_data:
            logger.error(f"Reassignment job {job_id} not found")
            return

        try:
            # Update job status
            job_data["status"] = ClusteringStatus.IN_PROGRESS
            job_data["started_at"] = datetime.now().isoformat()
            self.storage.save_clustering_job(job_data)

            # Get model
            clustering_model_id = job_data["parameters"].get("clustering_model_id")
            if not clustering_model_id:
                raise Exception("No clustering model ID specified for reassignment")

            model_data = self.storage.get_clustering_model(clustering_model_id)
            if not model_data:
                raise Exception(f"Model {clustering_model_id} not found")

            # Get profiles for reassignment
            profiles = self.storage.get_profiles(
                limit=1000
            )  # In real app, handle pagination

            # Reset user count in model
            model_data["total_users_clustered"] = 0

            # Process each profile
            processed_count = 0

            for profile in profiles:
                user_id = profile["user_id"]

                try:
                    # Extract features
                    features = self._extract_profile_features(
                        profile, model_data.get("features_used", [])
                    )

                    # Find nearest cluster
                    cluster_id, _, distance, confidence = self._find_nearest_cluster(
                        features, model_data
                    )

                    if cluster_id:
                        # Create assignment
                        assignment_id = str(uuid.uuid4())
                        assignment_data = {
                            "assignment_id": assignment_id,
                            "user_id": user_id,
                            "cluster_id": cluster_id,
                            "clustering_model_id": clustering_model_id,
                            "confidence_score": confidence,
                            "features": features,
                            "distance_to_centroid": distance,
                            "assigned_at": datetime.now().isoformat(),
                            "is_current": True,
                        }

                        # Save assignment
                        self.storage.save_user_cluster_assignment(assignment_data)
                        processed_count += 1

                except Exception as e:
                    logger.warning(f"Error processing user {user_id}: {str(e)}")

            # Update model with new user count
            model_data["total_users_clustered"] = processed_count
            model_data["updated_at"] = datetime.now().isoformat()
            self.storage.save_clustering_model(model_data)

            # Update job with result
            job_data["status"] = ClusteringStatus.COMPLETED
            job_data["completed_at"] = datetime.now().isoformat()
            job_data["user_count"] = processed_count
            self.storage.save_clustering_job(job_data)

        except Exception as e:
            logger.error(f"Error processing reassignment job {job_id}: {str(e)}")

            # Update job with error
            job_data["status"] = ClusteringStatus.FAILED
            job_data["completed_at"] = datetime.now().isoformat()
            job_data["error_message"] = str(e)
            self.storage.save_clustering_job(job_data)

    def _extract_profile_features(
        self, profile: Dict[str, Any], feature_names: List[str]
    ) -> Dict[str, float]:
        """
        Extract features from a profile for clustering.

        Args:
            profile: Profile data dictionary
            feature_names: List of feature names to extract

        Returns:
            Dictionary mapping feature names to values
        """
        features = {}

        # Extract personality trait features
        personality_traits = profile.get("personality_traits", {})
        if personality_traits:
            for trait in [
                "openness",
                "conscientiousness",
                "extraversion",
                "agreeableness",
                "neuroticism",
            ]:
                if trait in feature_names and trait in personality_traits:
                    features[trait] = (
                        float(personality_traits[trait]) / 100
                        if personality_traits[trait] is not None
                        else 0.5
                    )

        # Extract sleep preference features
        sleep_preferences = profile.get("sleep_preferences", {})
        if sleep_preferences:
            # Chronotype (convert categorical to numeric)
            if "chronotype" in feature_names and "chronotype" in sleep_preferences:
                chronotype_map = {
                    "morning_person": 0.0,
                    "intermediate": 0.5,
                    "evening_person": 1.0,
                    "variable": 0.5,
                }
                features["chronotype"] = chronotype_map.get(
                    sleep_preferences["chronotype"], 0.5
                )

            # Sleep environment preference
            if (
                "sleep_environment" in feature_names
                and "environment_preference" in sleep_preferences
            ):
                environment_map = {
                    "dark_quiet": 0.0,
                    "some_light": 0.33,
                    "some_noise": 0.67,
                    "noise_and_light": 1.0,
                }
                features["sleep_environment"] = environment_map.get(
                    sleep_preferences["environment_preference"], 0.0
                )

            # Sleep anxiety
            if (
                "sleep_anxiety" in feature_names
                and "sleep_anxiety_level" in sleep_preferences
            ):
                anxiety = sleep_preferences["sleep_anxiety_level"]
                features["sleep_anxiety"] = (
                    float(anxiety) / 10 if anxiety is not None else 0.5
                )

        # Extract behavioral pattern features
        behavioral_patterns = profile.get("behavioral_patterns", {})
        if behavioral_patterns:
            # Stress response
            if (
                "stress_response" in feature_names
                and "stress_response" in behavioral_patterns
            ):
                stress_map = {
                    "problem_focused": 0.0,
                    "emotion_focused": 0.33,
                    "social_support": 0.67,
                    "avoidant": 1.0,
                    "mixed": 0.5,
                }
                features["stress_response"] = stress_map.get(
                    behavioral_patterns["stress_response"], 0.5
                )

            # Routine consistency
            if (
                "routine_consistency" in feature_names
                and "routine_consistency" in behavioral_patterns
            ):
                consistency = behavioral_patterns["routine_consistency"]
                features["routine_consistency"] = (
                    float(consistency) / 10 if consistency is not None else 0.5
                )

            # Social activity preference
            if (
                "activity_preference" in feature_names
                and "social_activity_preference" in behavioral_patterns
            ):
                social = behavioral_patterns["social_activity_preference"]
                features["activity_preference"] = (
                    float(social) / 10 if social is not None else 0.5
                )

            # Screen time
            if (
                "screen_time" in feature_names
                and "screen_time_before_bed" in behavioral_patterns
            ):
                screen_time = behavioral_patterns["screen_time_before_bed"]
                # Normalize to 0-1 range (assuming max is 120 minutes)
                features["screen_time"] = (
                    min(float(screen_time) / 120, 1.0)
                    if screen_time is not None
                    else 0.5
                )

        # Fill missing features with default values
        for feature in feature_names:
            if feature not in features:
                features[feature] = 0.5  # Default value for missing features

        return features

    def _create_clusters(
        self, X: np.ndarray, algorithm: str, parameters: Dict[str, Any]
    ) -> Tuple[List[Dict[str, Any]], np.ndarray, np.ndarray, float]:
        """
        Create clusters using the specified algorithm.

        Args:
            X: Feature matrix
            algorithm: Clustering algorithm
            parameters: Algorithm parameters

        Returns:
            Tuple of (clusters, centroids, labels, inertia)
        """
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        if algorithm == ClusteringAlgorithm.KMEANS:
            # Get parameters
            n_clusters = parameters.get("n_clusters", settings.DEFAULT_CLUSTER_COUNT)

            # Create KMeans model
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)

            # Fit model
            kmeans.fit(X_scaled)

            # Get results
            centroids = kmeans.cluster_centers_
            labels = kmeans.labels_
            inertia = kmeans.inertia_

        else:
            # Default to KMeans
            n_clusters = settings.DEFAULT_CLUSTER_COUNT
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            kmeans.fit(X_scaled)
            centroids = kmeans.cluster_centers_
            labels = kmeans.labels_
            inertia = kmeans.inertia_

        # Count members in each cluster
        unique_labels, counts = np.unique(labels, return_counts=True)

        # Create cluster objects
        clusters = []

        for i, (label, count) in enumerate(zip(unique_labels, counts)):
            # Get cluster index
            cluster_idx = int(label)

            # Generate cluster ID
            cluster_id = str(uuid.uuid4())

            # Get centroid for this cluster
            centroid = centroids[cluster_idx]

            # Generate cluster name based on dominant features
            cluster_name, key_features = self._generate_cluster_characteristics(
                centroid, scaler, cluster_idx
            )

            # Create cluster object
            cluster = {
                "cluster_id": cluster_id,
                "name": cluster_name,
                "description": self._generate_cluster_description(
                    cluster_name, key_features
                ),
                "size": int(count),
                "centroid": {f"dim_{i}": float(v) for i, v in enumerate(centroid)},
                "key_features": key_features,
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat(),
                "version": "1.0.0",
                "is_active": True,
                "tags": [],
            }

            clusters.append(cluster)

        return clusters, centroids, labels, inertia

    def _generate_cluster_characteristics(
        self, centroid: np.ndarray, scaler: StandardScaler, cluster_idx: int
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Generate cluster name and key features based on the centroid.

        Args:
            centroid: Cluster centroid
            scaler: Feature scaler used
            cluster_idx: Cluster index

        Returns:
            Tuple of (name, key_features)
        """
        # Transform centroid back to original scale
        original_centroid = scaler.inverse_transform([centroid])[0]

        # Default feature names
        feature_names = [
            "openness",
            "conscientiousness",
            "extraversion",
            "agreeableness",
            "neuroticism",
            "chronotype",
            "sleep_environment",
            "sleep_anxiety",
            "stress_response",
            "routine_consistency",
        ]

        # Find top features
        feature_importances = [
            (name, abs(val)) for name, val in zip(feature_names, centroid)
        ]
        top_features = sorted(feature_importances, key=lambda x: x[1], reverse=True)[:5]

        # Generate key features
        key_features = []

        for feature_name, importance in top_features:
            # Get original value
            feature_idx = feature_names.index(feature_name)
            value = original_centroid[feature_idx]

            # Determine if high or low
            is_high = centroid[feature_idx] > 0

            # Generate feature description
            feature = {
                "feature_name": feature_name,
                "feature_value": float(value),
                "significance": float(min(importance, 1.0)),
                "comparison_to_mean": float(centroid[feature_idx]),
            }

            key_features.append(feature)

        # Generate name based on top 2 features
        if len(top_features) >= 2:
            name_features = top_features[:2]
            name_parts = []

            for feature_name, importance in name_features:
                feature_idx = feature_names.index(feature_name)
                is_high = centroid[feature_idx] > 0

                if feature_name == "openness":
                    name_parts.append("Open-minded" if is_high else "Traditional")
                elif feature_name == "conscientiousness":
                    name_parts.append("Disciplined" if is_high else "Flexible")
                elif feature_name == "extraversion":
                    name_parts.append("Extraverted" if is_high else "Introverted")
                elif feature_name == "agreeableness":
                    name_parts.append("Cooperative" if is_high else "Independent")
                elif feature_name == "neuroticism":
                    name_parts.append("Sensitive" if is_high else "Resilient")
                elif feature_name == "chronotype":
                    name_parts.append("Night Owl" if is_high else "Early Bird")
                elif feature_name == "sleep_environment":
                    name_parts.append(
                        "Ambient Sleeper" if is_high else "Silent Sleeper"
                    )
                elif feature_name == "sleep_anxiety":
                    name_parts.append("Vigilant" if is_high else "Carefree")
                elif feature_name == "stress_response":
                    name_parts.append(
                        "Emotion-Focused" if is_high else "Problem-Solver"
                    )
                elif feature_name == "routine_consistency":
                    name_parts.append("Consistent" if is_high else "Spontaneous")

            if name_parts:
                cluster_name = " ".join(name_parts)
            else:
                cluster_name = f"Cluster {cluster_idx + 1}"
        else:
            cluster_name = f"Cluster {cluster_idx + 1}"

        return cluster_name, key_features

    def _generate_cluster_description(
        self, cluster_name: str, key_features: List[Dict[str, Any]]
    ) -> str:
        """
        Generate a description for a cluster based on its characteristics.

        Args:
            cluster_name: Cluster name
            key_features: Key features of the cluster

        Returns:
            Cluster description
        """
        # Start with basic description
        description = f"People in the {cluster_name} group "

        # Add details about key features
        feature_descriptions = []

        for feature in key_features[:3]:  # Use top 3 features for description
            feature_name = feature["feature_name"]
            is_high = feature["comparison_to_mean"] > 0

            if feature_name == "openness":
                if is_high:
                    feature_descriptions.append(
                        "tend to be curious and open to new experiences"
                    )
                else:
                    feature_descriptions.append(
                        "prefer familiar routines and traditional approaches"
                    )
            elif feature_name == "conscientiousness":
                if is_high:
                    feature_descriptions.append(
                        "are organized and disciplined with their sleep schedule"
                    )
                else:
                    feature_descriptions.append(
                        "have a more flexible approach to their sleep schedule"
                    )
            elif feature_name == "extraversion":
                if is_high:
                    feature_descriptions.append(
                        "are socially energetic and may have busier evenings"
                    )
                else:
                    feature_descriptions.append(
                        "value quiet time to recharge before sleep"
                    )
            elif feature_name == "agreeableness":
                if is_high:
                    feature_descriptions.append(
                        "prioritize harmony and comfortable sleep environments"
                    )
                else:
                    feature_descriptions.append(
                        "have specific personal preferences for their sleep environment"
                    )
            elif feature_name == "neuroticism":
                if is_high:
                    feature_descriptions.append(
                        "may experience sleep anxiety or worry about sleep quality"
                    )
                else:
                    feature_descriptions.append(
                        "typically approach sleep with minimal worry or stress"
                    )
            elif feature_name == "chronotype":
                if is_high:
                    feature_descriptions.append(
                        """are typically evening people who feel
                        more alert later in the day"""
                    )
                else:
                    feature_descriptions.append(
                        """are morning people who feel
                        most alert earlier in the day"""
                    )
            elif feature_name == "sleep_environment":
                if is_high:
                    feature_descriptions.append(
                        "can sleep with some background noise or light"
                    )
                else:
                    feature_descriptions.append(
                        "prefer a dark, quiet environment for optimal sleep"
                    )
            elif feature_name == "sleep_anxiety":
                if is_high:
                    feature_descriptions.append(
                        "may experience worry or anxiety around bedtime"
                    )
                else:
                    feature_descriptions.append(
                        "typically feel relaxed when preparing for sleep"
                    )
            elif feature_name == "stress_response":
                if is_high:
                    feature_descriptions.append(
                        "tend to process emotions and stress before they can sleep well"
                    )
                else:
                    feature_descriptions.append(
                        "handle stress in practical ways that don't disrupt sleep"
                    )
            elif feature_name == "routine_consistency":
                if is_high:
                    feature_descriptions.append(
                        "thrive with consistent sleep and wake times"
                    )
                else:
                    feature_descriptions.append("adapt well to varying sleep schedules")

        # Combine feature descriptions
        if feature_descriptions:
            if len(feature_descriptions) == 1:
                description += feature_descriptions[0]
            elif len(feature_descriptions) == 2:
                description += (
                    f"{feature_descriptions[0]} and {feature_descriptions[1]}"
                )
            else:
                description += f"""{feature_descriptions[0]},
                {feature_descriptions[1]}, and {feature_descriptions[2]}"""

        # Add summary
        description += ". " + self._generate_cluster_summary(key_features)

        return description

    def _generate_cluster_summary(self, key_features: List[Dict[str, Any]]) -> str:
        """
        Generate a summary statement about the cluster.

        Args:
            key_features: Key features of the cluster

        Returns:
            Summary statement
        """
        # Look at combinations of features to generate insights
        feature_dict = {
            f["feature_name"]: f["comparison_to_mean"] > 0 for f in key_features
        }

        summaries = []

        # Check for specific combinations
        if feature_dict.get("neuroticism", False) and feature_dict.get(
            "sleep_anxiety", False
        ):
            summaries.append(
                """Members of this group may benefit from
                relaxation techniques before bedtime."""
            )

        if feature_dict.get("extraversion", False) and not feature_dict.get(
            "routine_consistency", True
        ):
            summaries.append(
                """These individuals often have active
                social lives that can impact their sleep schedule."""
            )

        if feature_dict.get("conscientiousness", True) and feature_dict.get(
            "routine_consistency", True
        ):
            summaries.append(
                """This group tends to maintain consistent
                sleep habits that support good sleep quality."""
            )

        if not feature_dict.get("chronotype", True) and feature_dict.get(
            "openness", False
        ):
            summaries.append(
                "Early risers in this group prefer predictable morning routines."
            )

        # Default summary if no specific combinations match
        if not summaries:
            summaries.append(
                """Understanding these patterns can help
                develop personalized strategies for better sleep."""
            )

        return summaries[0]

    def _find_nearest_cluster(
        self, features: Dict[str, float], model_data: Dict[str, Any]
    ) -> Tuple[Optional[str], Optional[Dict[str, Any]], float, float]:
        """
        Find the nearest cluster for a set of features.

        Args:
            features: User profile features
            model_data: Clustering model data

        Returns:
            Tuple of (cluster_id, cluster_data, distance, confidence)
        """
        clusters = model_data.get("clusters", [])
        if not clusters:
            return None, None, 0.0, 0.0

        # Extract feature values in the same order as used in clustering
        feature_names = model_data.get("features_used", [])
        feature_values = [features.get(name, 0.5) for name in feature_names]

        # Find nearest cluster
        min_distance = float("inf")
        nearest_cluster_id = None
        nearest_cluster_data = None

        for cluster in clusters:
            # Get centroid
            centroid = cluster.get("centroid", {})
            if not centroid:
                continue

            # Convert centroid to list in same order as features
            centroid_values = [
                centroid.get(f"dim_{i}", 0.0) for i in range(len(feature_names))
            ]

            # Calculate Euclidean distance
            distance = np.sqrt(
                np.sum([(a - b) ** 2 for a, b in zip(feature_values, centroid_values)])
            )

            if distance < min_distance:
                min_distance = distance
                nearest_cluster_id = cluster.get("cluster_id")
                nearest_cluster_data = cluster

        # Calculate confidence score (inversely proportional to distance)
        # Normalize to 0-1 range (closer to 1 means higher confidence)
        max_possible_distance = np.sqrt(
            len(feature_names) * 4
        )  # Maximum possible distance in feature space
        confidence = max(0.0, 1.0 - (min_distance / max_possible_distance))

        return nearest_cluster_id, nearest_cluster_data, min_distance, confidence
