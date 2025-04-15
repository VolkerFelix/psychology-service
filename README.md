# Psychology Profiling Microservice

A Python-based microservice for creating and managing psychological profiles through questionnaires and clustering algorithms. It's designed to work alongside sleep tracking applications to provide personalized insights and recommendations.

## Features

- **Psychological Profiling**: Create and manage comprehensive user psychological profiles
- **Interactive Questionnaires**: Administer personality and behavior assessment questionnaires
- **Profile Clustering**: Group users with similar psychological traits and preferences
- **RESTful API**: Easy integration with frontend applications and other services
- **PostgreSQL Storage**: Reliable data persistence with proper schema design

## Architecture

This microservice follows a clean architecture pattern:

- **API Layer**: FastAPI endpoints for accessing all functionality
- **Service Layer**: Core business logic for profiles, questionnaires, and clustering
- **Data Layer**: Database abstraction with PostgreSQL support
- **Models**: Pydantic models for data validation and documentation

## Requirements

- Python 3.9+
- PostgreSQL (for production) or SQLite (for development)
- Dependencies listed in `requirements.txt`

## Setup

### Local Development

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/psychology-profiling-service.git
   cd psychology-profiling-service
   ```

2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Configure environment variables:
   ```bash
   cp .env.example .env
   ```
   Edit the `.env` file to set your database connection and other settings.

4. Run database migrations:
   ```bash
   alembic upgrade head
   ```

5. Start the development server:
   ```bash
   uvicorn app.main:app --reload
   ```

### Using Docker

For a containerized setup:

```bash
docker-compose up -d
```

This will start:
- The psychology profiling microservice on port 8002
- PostgreSQL database on port 5433
- PgAdmin for database management on port 5051

## API Documentation

Once the service is running, API documentation is available at:
- Swagger UI: http://localhost:8002/docs
- ReDoc: http://localhost:8002/redoc

### Core Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/profiles/user/{user_id}` | Get a user's psychological profile |
| POST | `/api/profiles/` | Create a new psychological profile |
| PUT | `/api/profiles/{profile_id}` | Update an existing profile |
| POST | `/api/questionnaires/start` | Start a questionnaire for a user |
| POST | `/api/questionnaires/answer` | Submit answers to a questionnaire |
| POST | `/api/questionnaires/complete/{id}` | Complete a questionnaire and process results |
| GET | `/api/questionnaires/onboarding/{user_id}` | Get the next onboarding questionnaire |
| POST | `/api/clustering/run` | Run clustering algorithm to create personality groups |
| GET | `/api/clustering/user/{user_id}` | Get a user's cluster assignment |
| POST | `/api/clustering/user/{user_id}/assign` | Assign a user to a cluster |

## Psychological Profile Components

The service evaluates profiles based on several key psychological dimensions:

### Personality Traits
- **Openness**: Curiosity and openness to new experiences
- **Conscientiousness**: Organization and dependability
- **Extraversion**: Sociability and engagement with the external world
- **Agreeableness**: Cooperation and concern for social harmony
- **Neuroticism**: Emotional sensitivity and tendency toward negative emotions

### Sleep Preferences
- **Chronotype**: Natural inclination regarding times for sleeping (morning/evening person)
- **Environment Preferences**: Preferred sleeping environment conditions
- **Sleep Anxiety**: Tendency to worry about sleep quality or insomnia

### Behavioral Patterns
- **Stress Response**: How a person typically handles stress
- **Routine Consistency**: Preference for regular routines vs. flexibility
- **Social Activity**: Level of social engagement preferred before bedtime
- **Screen Time**: Typical screen usage before sleep

## Integration with Sleep Data Service

This microservice is designed to work alongside a sleep data tracking service. To enable full functionality:

1. Set the `SLEEP_SERVICE_URL` environment variable to point to your sleep data microservice
2. Ensure proper CORS configuration to allow communication between services
3. Use shared user IDs between services for profile correlation

## Development

### Database Migrations

This project uses Alembic for database migrations:

```bash
# Create a new migration after changing models
alembic revision --autogenerate -m "Description of changes"

# Apply migrations
alembic upgrade head
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage report
pytest --cov=app tests/
```

## License

[Apache 2.0](LICENSE)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request
