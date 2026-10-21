"""Response models for the StarCast API."""
from typing import Optional

from pydantic import BaseModel


class Components(BaseModel):
    xgboost: float
    matrix_factorization: float
    bayesian_baseline: float


class PredictResponse(BaseModel):
    user_id: str
    business_id: str
    predicted_stars: float
    components: Components
    known_user: bool
    known_business: bool


class Recommendation(BaseModel):
    rank: int
    business_id: str
    name: str
    city: str
    state: str
    categories: str
    yelp_stars: float
    predicted_stars: float


class RecommendFilters(BaseModel):
    city: Optional[str]
    state: Optional[str]
    category: Optional[str]
    include_closed: bool
    exclude_rated: bool


class RecommendResponse(BaseModel):
    user_id: str
    known_user: bool
    k: int
    filters: RecommendFilters
    recommendations: list[Recommendation]


class HealthResponse(BaseModel):
    status: str
    version: str
    model_load_seconds: float
    n_users: int
    n_businesses: int
    validation_rmse: Optional[dict[str, float]]
