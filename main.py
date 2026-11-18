import base64
import hashlib
import hmac
import json
import os
import pickle
import re
import secrets
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path as FilePath
from typing import Any, Dict, List, Optional

import pandas as pd

import httpx

from fastapi import Body, Depends, FastAPI, Header, HTTPException, Query

from fastapi.middleware.cors import CORSMiddleware

from pydantic import BaseModel, Field

from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"), override=False)

TMDB_API_KEY = os.getenv("TMDB_API_KEY")

TMDB_BASE_URL = "https://api.themoviedb.org/3"

TMDB_IMG_500 = "https://image.tmdb.org/t/p/w500"

app = FastAPI(
    title="Movie Recommendation API",
    description="A simple movie recommendation API using FastAPI",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        origin.strip()
        for origin in os.getenv(
            "CORS_ORIGINS",
            "http://localhost:8501,http://127.0.0.1:8501",
        ).split(",")
        if origin.strip()
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DF_PATH = os.path.join(BASE_DIR, "df.pkl")

INDICES_PATH = os.path.join(BASE_DIR, "indices.pkl")

TFIDF_PATH = os.path.join(BASE_DIR, "tfidf.pkl")
TFIDF_MATRIX_PATH = os.path.join(BASE_DIR, "tfidf_matrix.pkl")

df: Optional[pd.DataFrame] = None

indices_obj: Any = None

tfidf_matrix: Any = None

tfidf_obj: Any = None
recommender_error: Optional[str] = None

def load_recommender() -> bool:
    """Load the precomputed recommendation assets once, with useful diagnostics."""
    global df, indices_obj, tfidf_obj, tfidf_matrix, recommender_error
    if all(value is not None for value in (df, indices_obj, tfidf_obj, tfidf_matrix)):
        return True
    try:
        required = {"df.pkl": DF_PATH, "indices.pkl": INDICES_PATH,
                    "tfidf.pkl": TFIDF_PATH, "tfidf_matrix.pkl": TFIDF_MATRIX_PATH}
        missing = [name for name, path in required.items() if not os.path.isfile(path)]
        if missing:
            raise FileNotFoundError("Missing recommender asset(s): " + ", ".join(missing))
        with open(DF_PATH, "rb") as file:
            loaded_df = pickle.load(file)
        with open(INDICES_PATH, "rb") as file:
            loaded_indices = pickle.load(file)
        with open(TFIDF_PATH, "rb") as file:
            loaded_vectorizer = pickle.load(file)
        with open(TFIDF_MATRIX_PATH, "rb") as file:
            loaded_matrix = pickle.load(file)
        if not isinstance(loaded_df, pd.DataFrame) or "title" not in loaded_df.columns:
            raise ValueError("df.pkl must contain a DataFrame with a 'title' column.")
        if loaded_matrix.shape[0] != len(loaded_df):
            raise ValueError("TF-IDF matrix rows do not match the movie dataset.")
        if len(loaded_indices) == 0:
            raise ValueError("indices.pkl is empty.")
        df, indices_obj, tfidf_obj, tfidf_matrix = (
            loaded_df, loaded_indices, loaded_vectorizer, loaded_matrix
        )
        recommender_error = None
        return True
    except Exception as exc:
        recommender_error = f"{type(exc).__name__}: {exc}"
        return False

def get_tfidf_recommendations(title: str, limit: int = 12) -> List[dict]:
    """Return nearest movies using cosine similarity in the precomputed sparse matrix."""
    if not load_recommender():
        raise HTTPException(status_code=503, detail=f"Recommendation engine unavailable: {recommender_error}")
    if not isinstance(indices_obj, (pd.Series, dict)):
        raise HTTPException(status_code=500, detail="indices.pkl has an unsupported format.")
    lookup = {_norm_title(key): int(value) for key, value in indices_obj.items()}
    row = lookup.get(_norm_title(title))
    if row is None:
        
        normalized = _norm_title(title)
        candidates = [i for i, value in enumerate(df["title"].astype(str))
                      if _norm_title(value) == normalized]
        row = candidates[0] if candidates else None
    if row is None or row < 0 or row >= tfidf_matrix.shape[0]:
        raise HTTPException(status_code=404, detail=f"'{title}' is not in the recommendation dataset. Try another movie title.")
    
    scores = (tfidf_matrix[row] @ tfidf_matrix.T).toarray().ravel()
    scores[row] = -1.0
    top_indices = scores.argsort()[::-1]
    output = []
    for movie_index in top_indices:
        score = float(scores[movie_index])
        if score <= 0 or len(output) >= limit:
            break
        movie_title = str(df.iloc[int(movie_index)]["title"])
        if movie_title and movie_title.lower() != title.lower():
            output.append({"title": movie_title, "score": round(score, 4), "tmdb": None})
    return output

class TMDBMovieCard(BaseModel):

    tmdb_id: int

    title: str

    poster_url: Optional[str] = None

    release_date: Optional[str] = None

    vote_average: Optional[float] = None

class TMDBMovieDetails(BaseModel):

    tmdb_id: int

    title: str

    overview: Optional[str] = None

    release_date: Optional[str] = None

    poster_url: Optional[str] = None

    backdrop_url: Optional[str] = None

    genres: List[Dict[str, Any]] = Field(default_factory=list)

class TFIDFRecItem(BaseModel):

    title: str

    score: float

    tmdb: Optional[TMDBMovieCard] = None

class SearchBundleResponse(BaseModel):

    query: str

    movie_details: TMDBMovieDetails

    tfidf_recommendations: List[TFIDFRecItem]

    genre_recommendations: List[TMDBMovieCard]

def _norm_title(t: str) -> str:

    return str(t).strip().lower()

def make_img_url(path: Optional[str]) -> Optional[str]:

    if path:

        return f"{TMDB_IMG_500}{path}"

    return None

async def tmdb_get(path: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """Send an authenticated GET request to TMDB and return its JSON response."""
    if not TMDB_API_KEY:
        raise HTTPException(
            status_code=503,
            detail="TMDB_API_KEY is missing. Add it to your .env file and restart the API."
        )

    query_params = dict(params)
    query_params["api_key"] = TMDB_API_KEY

    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(30.0)) as client:
            response = await client.get(
                f"{TMDB_BASE_URL}{path}",
                params=query_params
            )
            response.raise_for_status()

        try:
            payload = response.json()
        except ValueError:
            raise HTTPException(
                status_code=502,
                detail="TMDB returned an invalid JSON response."
            )

        if not isinstance(payload, dict):
            raise HTTPException(
                status_code=502,
                detail="TMDB returned an unexpected response format."
            )

        return payload

    except httpx.TimeoutException:
        raise HTTPException(
            status_code=504,
            detail="TMDB request timed out. Please try again."
        )
    except httpx.HTTPStatusError as exc:
        status_code = exc.response.status_code
        try:
            error_payload = exc.response.json()
            message = error_payload.get("status_message", "TMDB request failed.")
        except ValueError:
            message = "TMDB request failed."

        raise HTTPException(
            status_code=502,
            detail=f"TMDB API error ({status_code}): {message}"
        )
    except httpx.RequestError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Could not connect to TMDB: {type(exc).__name__}: {exc}"
        )

async def tmdb_cards_from_results(
    results: List[Dict[str, Any]],
    limit: Optional[int] = None
) -> List[TMDBMovieCard]:
    """Convert TMDB movie dictionaries into validated movie cards."""
    cards: List[TMDBMovieCard] = []

    for movie in results:
        if not isinstance(movie, dict):
            continue

        tmdb_id = movie.get("id")
        title = movie.get("title") or movie.get("name")

        if tmdb_id is None or not title:
            continue

        cards.append(
            TMDBMovieCard(
                tmdb_id=int(tmdb_id),
                title=str(title),
                poster_url=make_img_url(movie.get("poster_path")),
                release_date=movie.get("release_date"),
                vote_average=movie.get("vote_average")
            )
        )

        if limit is not None and len(cards) >= limit:
            break

    return cards

async def tmdb_movie_details(tmdb_id: int) -> TMDBMovieDetails:

    data = await tmdb_get(
        f"/movie/{tmdb_id}",
        {'language': "en-US"}
    )

    return TMDBMovieDetails(
        tmdb_id=data.get("id"),

        title=data.get("title") or "",

        overview=data.get("overview"),

        release_date=data.get("release_date"),

        poster_url=make_img_url(data.get("poster_path")),

        backdrop_url=make_img_url(data.get("backdrop_path")),

        genres=data.get("genres", [])

    )

async def tmdb_search_movies(
    query: str,
    page: int = 1
) -> Dict[str, Any]:

    """
    Raw TMDB response for keyword search (multiple results).

    The frontend can use this response to display:
    - Movie suggestions.
    - Search result grids.
    - Pagination information.
    """

    query = query.strip()
    if not query:
        raise HTTPException(status_code=422, detail="Search query cannot be empty.")

    data = await tmdb_get(
        "/search/movie",
        {
            "query": query,

            "page": page,

            "language": "en-US",

            "include_adult": False
            
        }
    )

    return data

async def tmdb_search_first(query: str) -> Optional[dict]:

    data = await tmdb_search_movies(
        query=query,
        page=1
    )

    results = data.get("results", [])

    if not results:

        return None

    first_result = results[0]

    tmdb_id = first_result.get("id")

    if not tmdb_id:

        return None

    return await tmdb_movie_details(tmdb_id)

@app.get("/health")
def health():

    return {"status": "ok", "recommendation_engine": "ready" if load_recommender() else "unavailable", "recommendation_error": recommender_error}

@app.get("/home", response_model=list[TMDBMovieCard])
async def home(

    category: str = Query(
        "popular",
        description="Category of movies to fetch"
    ),

    limit: int = Query(
        24,
        ge=1,
        le=50,
        description="Number of movies to fetch (1-50)"
    )
):

    try:

        if category == "trending":

            data = await tmdb_get(
                "/trending/movie/day",
                {"language": "en-US"}
            )

            return await tmdb_cards_from_results(
                data.get("results", []),
                limit=limit
            )

        if category not in {
            "popular",
            "top_rated",
            "upcoming",
            "now_playing"
        }:

            raise HTTPException(
                status_code=400,
                detail="Invalid category"
            )

        data = await tmdb_get(
            f"/movie/{category}",
            {
                "language": "en-US",
                "page": 1
            }
        )

        return await tmdb_cards_from_results(
            data.get("results", []),
            limit=limit
        )

    except HTTPException:

        raise

@app.get("/tmdb/search")
async def tmdb_search(

    query: str = Query(
        ...,
        min_length=1
    ),

    page: int = Query(
        1,
        ge=1,
        le=10
    ),
):

    """
    Returns the raw TMDB API response containing a results list.

    Streamlit uses this response for:

    1. Dropdown suggestions:
       Display matching movie titles while the user types.

    2. Grid results:
       Display multiple matching movies in a grid layout.
    """

    return await tmdb_search_movies(
        query=query,
        page=page
    )

@app.get("/movie/id/tmdb_id/{tmdb_id}", response_model=TMDBMovieDetails)
async def movie_details_route(tmdb_id: int):
    return await tmdb_movie_details(tmdb_id)

@app.get("/recommend/tfidf", response_model=List[TFIDFRecItem])
def recommend_tfidf(
    title: str = Query(..., min_length=1, max_length=200),
    limit: int = Query(12, ge=1, le=30),
):
    """Recommend dataset movies using the precomputed TF-IDF similarity matrix."""
    return get_tfidf_recommendations(title.strip(), limit=limit)

@app.get("/recommend/genre", response_model=List[TMDBMovieCard])
@app.get("/recommend/gerne", response_model=List[TMDBMovieCard], include_in_schema=False)
async def recommend_genre(
    tmdb_id: int = Query(...),
    limit: int = Query(18, ge=1, le=50)
):
    """
    -fetch details
    -pick first gerne
    -discover movie in the genre (popular)

    """
    details = await tmdb_movie_details(tmdb_id)
    if not details.genres:
        return []

    genre_ids = [str(genre.get("id")) for genre in details.genres if genre.get("id") is not None]
    if not genre_ids:
        return []

    discover = await tmdb_get(
        "/discover/movie",
        {
            "with_genres": "|".join(genre_ids),
            "sort_by": "popularity.desc",
            "language": "en-US",
            "page": 1
        },
    )
    cards = await tmdb_cards_from_results(
        discover.get("results", []),
        limit=limit + 1
    )
    
    return [card for card in cards if card.tmdb_id != tmdb_id][:limit]

DATA_DIR = FilePath(BASE_DIR) / "data"
USERS_DIR = DATA_DIR / "users"
USER_INDEX_FILE = DATA_DIR / "users.json"
TOKEN_SECRET_FILE = DATA_DIR / ".token_secret"
AUTH_LOCK = threading.RLock()
TOKEN_TTL_SECONDS = 60 * 60 * 24 * 7

def _ensure_storage() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    USERS_DIR.mkdir(parents=True, exist_ok=True)
    if not USER_INDEX_FILE.exists():
        _write_json_atomic(USER_INDEX_FILE, {})
    if not TOKEN_SECRET_FILE.exists() and not os.getenv("APP_SECRET_KEY", "").strip():
        try:
            with TOKEN_SECRET_FILE.open("x", encoding="utf-8") as file:
                file.write(secrets.token_urlsafe(64))
        except FileExistsError:
            pass

def _write_json_atomic(path: FilePath, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + f".{uuid.uuid4().hex}.tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temp, path)

def _read_json(path: FilePath, fallback: Any) -> Any:
    try:
        with path.open("r", encoding="utf-8") as file:
            return json.load(file)
    except FileNotFoundError:
        return fallback
    except (json.JSONDecodeError, UnicodeDecodeError, OSError):
        raise HTTPException(status_code=500, detail="Stored account data is unreadable.")

def _token_secret() -> bytes:
    configured = os.getenv("APP_SECRET_KEY", "").strip()
    if configured:
        if len(configured.encode("utf-8")) < 32:
            raise HTTPException(status_code=500, detail="APP_SECRET_KEY must be at least 32 bytes.")
        return configured.encode("utf-8")
    _ensure_storage()
    return TOKEN_SECRET_FILE.read_text(encoding="utf-8").strip().encode("utf-8")

def _b64encode(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")

def _b64decode(value: str) -> bytes:
    return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))

def _issue_access_token(user_id: str, email: str) -> str:
    claims = {
        "sub": user_id,
        "email": email,
        "exp": int(time.time()) + TOKEN_TTL_SECONDS,
    }
    payload = _b64encode(json.dumps(claims, separators=(",", ":")).encode("utf-8"))
    signature = _b64encode(
        hmac.new(_token_secret(), payload.encode("ascii"), hashlib.sha256).digest()
    )
    return f"{payload}.{signature}"

def _verify_access_token(token: str) -> dict:
    try:
        payload_part, signature = token.split(".", 1)
        expected = _b64encode(
            hmac.new(
                _token_secret(), payload_part.encode("ascii"), hashlib.sha256
            ).digest()
        )
        if not hmac.compare_digest(signature, expected):
            raise ValueError("Invalid signature")
        payload = json.loads(_b64decode(payload_part))
        if not isinstance(payload, dict):
            raise ValueError("Invalid token payload")
        if int(payload.get("exp", 0)) <= int(time.time()):
            raise ValueError("Expired token")
        if not isinstance(payload.get("sub"), str) or not isinstance(
            payload.get("email"), str
        ):
            raise ValueError("Invalid token claims")
        return payload
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(
            status_code=401,
            detail="Your session is invalid or expired. Please sign in again.",
        )

def _hash_password(password: str, salt: Optional[bytes] = None) -> str:
    salt = salt or secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 600_000)
    return _b64encode(salt) + "$" + _b64encode(digest)

def _verify_password(password: str, encoded: str) -> bool:
    try:
        salt_text, digest_text = encoded.split("$", 1)
        salt = _b64decode(salt_text)
        expected = _b64decode(digest_text)
        actual = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 600_000)
        return hmac.compare_digest(actual, expected)
    except Exception:
        return False

def _public_user(record: dict) -> dict:
    return {"id": record["id"], "email": record["email"], "name": record.get("name", "")}

def _new_user(email: str, name: str, password_hash: Optional[str] = None,
              google_sub: Optional[str] = None) -> dict:
    user = {
        "id": str(uuid.uuid4()),
        "email": email,
        "name": name.strip()[:100],
        "password_hash": password_hash,
        "google_sub": google_sub,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    return user

def _load_user_index() -> dict:
    _ensure_storage()
    data = _read_json(USER_INDEX_FILE, {})
    if not isinstance(data, dict):
        raise HTTPException(status_code=500, detail="User registry has an invalid format.")
    return data

def _save_user_index(index: dict) -> None:
    _write_json_atomic(USER_INDEX_FILE, index)

def _create_session_response(record: dict) -> dict:
    return {
        "access_token": _issue_access_token(record["id"], record["email"]),
        "token_type": "bearer",
        "user": _public_user(record),
    }

def _current_user(authorization: str = Header(default="")) -> dict:
    if not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Sign in is required.")
    claims = _verify_access_token(authorization.split(" ", 1)[1].strip())
    return {"id": claims["sub"], "email": claims["email"]}

class RegisterRequest(BaseModel):
    name: str = Field(default="", max_length=100)
    email: str = Field(..., max_length=254)
    password: str = Field(..., min_length=8, max_length=128)

class LoginRequest(BaseModel):
    email: str = Field(..., max_length=254)
    password: str = Field(..., min_length=1, max_length=128)

class GoogleOIDCRequest(BaseModel):
    sub: str = Field(..., min_length=1, max_length=255)
    email: str = Field(..., min_length=3, max_length=254)
    name: str = Field(default="", max_length=100)

@app.post("/auth/register")
def register(request: RegisterRequest):
    email = request.email.strip().lower()
    if not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", email):
        raise HTTPException(status_code=422, detail="Enter a valid email address.")
    if len(request.password) < 8:
        raise HTTPException(status_code=422, detail="Password must contain at least 8 characters.")

    with AUTH_LOCK:
        index = _load_user_index()
        if email in index:
            raise HTTPException(status_code=409, detail="An account with this email already exists.")
        record = _new_user(email, request.name, _hash_password(request.password))
        session = _create_session_response(record)
        _write_json_atomic(USERS_DIR / f"{record['id']}.json", {})
        index[email] = record
        _save_user_index(index)
    return session

@app.post("/auth/login")
def login(request: LoginRequest):
    email = request.email.strip().lower()
    with AUTH_LOCK:
        index = _load_user_index()
        record = index.get(email)
    if not record or not record.get("password_hash") or not _verify_password(
        request.password, record["password_hash"]
    ):
        raise HTTPException(status_code=401, detail="Incorrect email or password.")
    return _create_session_response(record)

@app.post("/auth/oidc")
def google_oidc_login(
    request: GoogleOIDCRequest,
    x_app_internal_secret: str = Header(default=""),
):

    configured_secret = os.getenv("APP_INTERNAL_SECRET", "").strip()
    if (
        len(configured_secret.encode("utf-8")) < 32
        or not hmac.compare_digest(x_app_internal_secret, configured_secret)
    ):
        raise HTTPException(status_code=403, detail="Google sign-in is not configured securely.")
    email = request.email.strip().lower()
    if not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", email):
        raise HTTPException(status_code=422, detail="Google did not return a valid email address.")

    with AUTH_LOCK:
        index = _load_user_index()
        record = index.get(email)
        if record:
            existing_sub = record.get("google_sub")
            if existing_sub and existing_sub != request.sub:
                raise HTTPException(status_code=409, detail="This email is linked to another Google account.")
            record["google_sub"] = request.sub
            if request.name and not record.get("name"):
                record["name"] = request.name[:100]
        else:
            record = _new_user(email, request.name, google_sub=request.sub)
            _write_json_atomic(USERS_DIR / f"{record['id']}.json", {})
            index[email] = record
        session = _create_session_response(record)
        _save_user_index(index)
    return session

@app.get("/auth/me")
def auth_me(user: dict = Depends(_current_user)):
    index = _load_user_index()
    record = next((item for item in index.values() if item.get("id") == user["id"]), None)
    if not record:
        raise HTTPException(status_code=401, detail="Account no longer exists.")
    return _public_user(record)

@app.get("/user/data")
def get_user_data(user: dict = Depends(_current_user)):
    _ensure_storage()
    path = USERS_DIR / f"{user['id']}.json"
    data = _read_json(path, {})
    return data if isinstance(data, dict) else {}

@app.put("/user/data")
def put_user_data(payload: dict = Body(...), user: dict = Depends(_current_user)):
    
    try:
        serialized = json.dumps(payload, ensure_ascii=False)
    except (TypeError, ValueError):
        raise HTTPException(status_code=422, detail="Account data must be valid JSON.")
    if len(serialized.encode("utf-8")) > 2_000_000:
        raise HTTPException(status_code=413, detail="Account data is too large.")
    with AUTH_LOCK:
        _ensure_storage()
        _write_json_atomic(USERS_DIR / f"{user['id']}.json", payload)
    return {"saved": True}
