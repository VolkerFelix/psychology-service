"""Configuration settings for the Psychology Profiling Microservice."""

import os
from typing import List

from dotenv import load_dotenv
from pydantic_settings import BaseSettings

# Load environment variables from .env file
load_dotenv()


class Settings(BaseSettings):
    """Application settings."""

    # API settings
    API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
    API_PORT: int = int(os.getenv("API_PORT", 8002))

    # Application settings
    DEBUG: bool = os.getenv("DEBUG", "False").lower() == "true"
    SECRET_KEY: str = os.getenv("SECRET_KEY", "dev-secret-key-change-in-production")

    # CORS settings
    CORS_ORIGINS: List[str] = [
        "http://localhost:3000",  # Frontend
        "http://localhost:8000",  # Main service
        "http://localhost:8001",  # Sleep microservice
    ]

    # Database settings
    DATABASE_URL: str = os.getenv(
        "DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/psychology_data"
    )

    # Sleep Data Service
    PSYCHOLOGY_SERVICE_URL: str = os.getenv(
        "PSYCHOLOGY_SERVICE_URL", "http://localhost:8001/api"
    )

    # Authentication
    JWT_SECRET_KEY: str = os.getenv(
        "JWT_SECRET_KEY", "your-jwt-secret-key-change-in-production"
    )
    JWT_ALGORITHM: str = os.getenv("JWT_ALGORITHM", "HS256")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", 30))

    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    # API Documentation
    VERSION: str = os.getenv("VERSION", "0.1.0")
    SHOW_DOCS: bool = os.getenv("SHOW_DOCS", "True").lower() == "true"

    # Questionnaire settings
    DEFAULT_QUESTIONS_PER_PAGE: int = 5
    MIN_QUESTIONS_FOR_VALID_PROFILE: int = 15

    # Clustering settings
    DEFAULT_CLUSTER_COUNT: int = 5
    CLUSTER_REFRESH_INTERVAL_HOURS: int = 24
    MIN_USERS_FOR_CLUSTERING: int = 10

    # Pydantic v2 configuration
    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": True,
    }


# Create an instance of the settings
settings = Settings()
