"""StarCast FastAPI service."""
import time
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Path, Query, Request

from .. import __version__, config
from ..models import HybridRecommender
from .schemas import HealthResponse, PredictResponse, RecommendFilters, RecommendResponse


@asynccontextmanager
async def lifespan(app: FastAPI):
    t0 = time.time()
    app.state.model = HybridRecommender.load(config.artifact_dir())
    app.state.load_seconds = round(time.time() - t0, 2)
    yield


app = FastAPI(
    title="StarCast",
    description="Star-rating prediction and top-K business recommendations from an "
                "XGBoost + matrix factorization hybrid model.",
    version=__version__,
    lifespan=lifespan,
)


def get_model(request: Request) -> HybridRecommender:
    return request.app.state.model


@app.middleware("http")
async def timing_header(request: Request, call_next):
    t0 = time.perf_counter()
    response = await call_next(request)
    response.headers["X-Process-Time-Ms"] = f"{(time.perf_counter() - t0) * 1000:.1f}"
    return response


@app.get("/health", response_model=HealthResponse, tags=["ops"])
def health(request: Request):
    m = get_model(request)
    return HealthResponse(
        status="ok",
        version=__version__,
        model_load_seconds=request.app.state.load_seconds,
        n_users=len(m.u_stats),
        n_businesses=len(m.item_ids),
        validation_rmse=m.metrics.get("validation", {}).get("rmse"),
    )


@app.get("/predict", response_model=PredictResponse, tags=["model"])
def predict(
    request: Request,
    user_id: str = Query(..., min_length=1, description="Yelp user_id"),
    business_id: str = Query(..., min_length=1, description="Yelp business_id"),
):
    """Predicted star rating for one (user, business) pair."""
    m = get_model(request)
    known_user, known_biz = m.is_known_user(user_id), m.is_known_business(business_id)
    if not known_user and not known_biz:
        raise HTTPException(404, "Neither user_id nor business_id is known to the model.")
    return PredictResponse(user_id=user_id, business_id=business_id, known_user=known_user,
                           known_business=known_biz, **m.predict(user_id, business_id))


@app.get("/recommend/{user_id}", response_model=RecommendResponse, tags=["model"])
def recommend(
    request: Request,
    user_id: str = Path(..., min_length=1),
    k: int = Query(10, ge=1, le=config.MAX_K, description="number of businesses to return"),
    city: Optional[str] = Query(None, description="exact city match, case-insensitive"),
    state: Optional[str] = Query(None, description="two-letter state code, e.g. NV"),
    category: Optional[str] = Query(None, description="substring of the business categories, e.g. 'pizza'"),
    anywhere: bool = Query(False, description="don't default to the user's home state"),
    include_closed: bool = Query(False),
    exclude_rated: bool = Query(True, description="skip businesses the user already rated"),
):
    """Top-K businesses for a user: MF candidate retrieval, then hybrid re-ranking."""
    m = get_model(request)
    if state is None and city is None and not anywhere:
        state = m.home_state(user_id)
    recs = m.recommend(user_id, k=k, city=city, state=state, category=category,
                       include_closed=include_closed, exclude_rated=exclude_rated)
    return RecommendResponse(
        user_id=user_id, known_user=m.is_known_user(user_id), k=k,
        filters=RecommendFilters(city=city, state=state, category=category,
                                 include_closed=include_closed, exclude_rated=exclude_rated),
        recommendations=recs,
    )
