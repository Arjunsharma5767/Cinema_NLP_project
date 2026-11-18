import os
import json
import html
import random
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Any, Optional
from urllib.parse import quote_plus

import pandas as pd
import requests
import streamlit as st
import streamlit.components.v1 as components

try:  # optional: loads TMDB_API_KEY / OMDB_API_KEY from a .env file
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# =========================================================
# CONFIG
# =========================================================

st.set_page_config(
    page_title="CINÉMA | Movie Intelligence",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

API_BASE = os.getenv("BACKEND_URL", "http://127.0.0.1:8000").rstrip("/")
TMDB_IMG = "https://image.tmdb.org/"
TMDB_KEY = os.getenv("TMDB_API_KEY", "").strip()          # needed for "Where to watch"
TMDB_TOKEN = os.getenv("TMDB_READ_TOKEN", "").strip()     # alternative to TMDB_API_KEY
OMDB_KEY = os.getenv("OMDB_API_KEY", "").strip()          # optional: real IMDb ratings
TMDB_ON = bool(TMDB_KEY or TMDB_TOKEN)

CATEGORIES = {
    "Popular": "popular",
    "Trending": "trending",
    "Top Rated": "top_rated",
    "Upcoming": "upcoming",
    "Now Playing": "now_playing",
}
VIEWS = ["Discover", "Watchlist", "Insights"]
SORTS = ["Default", "Rating (high → low)", "Newest first", "Oldest first",
         "Title A–Z", "Shortest first", "Longest first"]
PAGE_STEP, PAGE_MAX = 12, 60
RUNTIME_NO_LIMIT = 240

REGIONS = {
    "Netherlands": "NL", "India": "IN", "United States": "US", "United Kingdom": "GB",
    "Canada": "CA", "Australia": "AU", "Germany": "DE",
}

# Approximate starting prices. TMDB/JustWatch do NOT publish prices and plans change often,
# so edit this table whenever you like. Keys are matched against the provider name (lowercase).
PLATFORM_PRICES = {
    "NL": {"netflix": "from ~€6/mo (with ads)", "prime video": "~€6/mo",
           "disney": "from ~€6/mo (with ads)"},
    "IN": {"netflix": "from ~₹149/mo", "prime video": "~₹299/mo (or ~₹1,499/yr)",
           "hotstar": "from ~₹149 / 3 months"},
    "US": {"netflix": "from ~$8/mo (with ads)", "prime video": "~$9/mo",
           "disney": "from ~$10/mo (with ads)"},
}

DEFAULTS = {
    "watchlist": [],
    "selected_movie_id": None,
    "search_query": "",
    "category": "Popular",
    "view": "Discover",
    "light": False,
    "density": 4,
    "layout": "Grid",
    "min_rating": 0.0,
    "max_runtime": RUNTIME_NO_LIMIT,
    "sort": "Default",
    "limit": PAGE_STEP,
    "recent": [],
    "online": None,
    "wl_filter": "All",
    "region": "Netherlands",
    "scroll_top": False,
    "access_token": None,
    "auth_user": None,
    "auth_loaded_for": None,
    "_last_saved_payload": None,
    "auth_error": None,
}
for _k, _v in DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


# =========================================================
# THEME
# =========================================================

DARK = {
    "bg": "#0B0E13", "surface": "#131922", "surface2": "#1B2330",
    "ink": "#E8EDF3", "muted": "#8A96A5", "accent": "#F2B84B",
    "accent-ink": "#1A1204", "accent-hi": "#FFD27A", "data": "#4FD1C5",
    "line": "rgba(255,255,255,.08)", "shadow": "0 18px 50px rgba(0,0,0,.45)",
}
LIGHT = {
    "bg": "#F3F5F8", "surface": "#FFFFFF", "surface2": "#E8ECF1",
    "ink": "#0E1620", "muted": "#586577", "accent": "#B96F00",
    "accent-ink": "#FFFFFF", "accent-hi": "#8F5500", "data": "#0F8F86",
    "line": "rgba(14,22,32,.12)", "shadow": "0 14px 40px rgba(14,22,32,.10)",
}

FONTS = (
    "@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600"
    "&family=IBM+Plex+Sans:wght@400;500;600&family=Sora:wght@500;600;700;800&display=swap');"
)

BASE_CSS = """
html,body,.stApp,[data-testid="stAppViewContainer"],[data-testid="stMain"]{background:var(--bg)!important;color:var(--ink)!important;font-family:'IBM Plex Sans',sans-serif!important}
[data-testid="stHeader"]{background:transparent!important}
[data-testid="stAppViewBlockContainer"],.main .block-container{max-width:1480px!important;padding:1.2rem clamp(1rem,4vw,3rem) 3rem!important}
#MainMenu,footer,[data-testid="stToolbar"]{visibility:hidden}
[data-testid="stMarkdownContainer"],[data-testid="stMarkdownContainer"] p,[data-testid="stWidgetLabel"] *,[data-testid="stRadio"] label,[data-testid="stExpander"] summary,[data-testid="stMetricLabel"] *,[data-testid="stMetricValue"]{color:var(--ink)!important}
[data-testid="stCaptionContainer"],[data-testid="stCaptionContainer"] *{color:var(--muted)!important}
h1,h2,h3{font-family:'Sora',sans-serif!important;letter-spacing:-.02em}
:focus-visible{outline:2px solid var(--accent)!important;outline-offset:2px}
hr{border-color:var(--line)!important;margin:1.4rem 0!important}
iframe[height="0"]{display:none}
.mono{font-family:'IBM Plex Mono',monospace}
.brand{font-family:'Sora',sans-serif;font-weight:800;font-size:clamp(1.7rem,3.4vw,2.3rem);letter-spacing:-.04em;line-height:1}
.brand b{color:var(--accent);font-weight:800}
.tagline{color:var(--muted);font-size:.8rem;margin-top:.35rem}
.status{display:inline-flex;align-items:center;gap:.45rem;font-family:'IBM Plex Mono',monospace;font-size:.72rem;color:var(--muted);border:1px solid var(--line);border-radius:999px;padding:.25rem .7rem;background:var(--surface)}
.dot{width:8px;height:8px;border-radius:50%;background:var(--muted)}
.dot.on{background:var(--data);box-shadow:0 0 8px var(--data)}
.dot.off{background:#E5484D}
.sec{margin:1.6rem 0 .9rem}
.sec h2{font-size:clamp(1.3rem,2.6vw,1.8rem);margin:0}
.sec p{color:var(--muted)!important;font-size:.88rem;margin:.25rem 0 0}
/* hero */
.hero{position:relative;overflow:hidden;border-radius:20px;border:1px solid var(--line);background:var(--surface);box-shadow:var(--shadow);display:flex;align-items:center;justify-content:space-between;gap:2rem;padding:clamp(1.3rem,4vw,3rem);margin:1rem 0 .6rem;min-height:300px}
.hero::before{content:"";position:absolute;inset:-30px;background-image:var(--bg-img);background-size:cover;background-position:center;filter:blur(30px) saturate(1.3);opacity:.38;transform:scale(1.1)}
.hero::after{content:"";position:absolute;inset:0;background:linear-gradient(90deg,var(--surface) 8%,transparent 85%);opacity:.92}
.hero>*{position:relative;z-index:1}
.hero-copy{max-width:640px}
.hero-kicker{font-family:'IBM Plex Mono',monospace;font-size:.75rem;color:var(--data)!important}
.hero h1{font-size:clamp(2rem,5vw,3.8rem);line-height:1.04;margin:.6rem 0 .9rem;color:var(--ink)!important}
.hero p{color:var(--muted)!important;line-height:1.7;max-width:540px;display:-webkit-box;-webkit-line-clamp:3;-webkit-box-orient:vertical;overflow:hidden}
.hero-poster{height:260px;aspect-ratio:2/3;object-fit:cover;border-radius:14px;box-shadow:var(--shadow);border:1px solid var(--line);transform:rotate(2deg)}
.chip{display:inline-block;font-family:'IBM Plex Mono',monospace;font-size:.72rem;padding:.25rem .65rem;border-radius:6px;border:1px solid var(--line);background:var(--surface2);color:var(--ink)!important;margin:.15rem .3rem .15rem 0}
.chip.hot{border-color:var(--accent);color:var(--accent)!important}
/* cards */
.movie-card{position:relative;background:var(--surface);border:1px solid var(--line);border-radius:14px;overflow:hidden;margin-bottom:.5rem;transition:border-color .2s,transform .2s}
.movie-card:hover{border-color:var(--accent);transform:translateY(-3px)}
.poster{width:100%;aspect-ratio:2/3;object-fit:cover;display:block;background:var(--surface2)}
.poster-frame{position:relative;width:100%;aspect-ratio:2/3;background:var(--surface2);overflow:hidden}
.poster-frame img{width:100%;height:100%;object-fit:cover;display:block}
.poster-frame .poster-placeholder{position:absolute;inset:0;display:none;aspect-ratio:auto}
.poster-frame.detail-frame{aspect-ratio:2/3;border-radius:14px;overflow:hidden;border:1px solid var(--line)}
.poster-frame.detail-frame img{border-radius:0;border:0}
.poster-frame.hero-frame{height:260px;width:auto;aspect-ratio:2/3;flex:0 0 auto;border-radius:14px;box-shadow:var(--shadow);border:1px solid var(--line);transform:rotate(2deg)}
.poster-frame.hero-frame img{height:100%;width:100%;object-fit:cover}
.auth-shell{max-width:520px;margin:4vh auto;padding:clamp(1.2rem,4vw,2.2rem);background:var(--surface);border:1px solid var(--line);border-radius:18px;box-shadow:var(--shadow)}
.auth-subtitle{color:var(--muted);line-height:1.6;margin:.4rem 0 1.2rem}
.poster-detail{width:100%;border-radius:14px;border:1px solid var(--line);display:block;background:var(--surface2)}
.poster-placeholder{aspect-ratio:2/3;display:flex;align-items:center;justify-content:center;background:var(--surface2);color:var(--muted);font-size:.8rem;text-align:center;padding:.5rem}
.badge{position:absolute;top:.55rem;right:.55rem;font-family:'IBM Plex Mono',monospace;font-size:.72rem;font-weight:600;padding:.2rem .5rem;border-radius:6px;background:rgba(8,10,14,.82);color:var(--accent-hi)!important;backdrop-filter:blur(6px)}
.badge.saved{left:.55rem;right:auto;color:var(--data)!important}
.card-info{padding:.75rem .85rem .85rem}
.card-title{font-family:'Sora',sans-serif;font-weight:600;font-size:.92rem;line-height:1.3;min-height:2.4em;display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;overflow:hidden;color:var(--ink)!important}
.card-meta{margin-top:.45rem;font-family:'IBM Plex Mono',monospace;font-size:.72rem;color:var(--muted)!important}
.thumb{width:100%;aspect-ratio:2/3;object-fit:cover;border-radius:8px;background:var(--surface2)}
.row-title{font-family:'Sora',sans-serif;font-weight:600;color:var(--ink)!important}
.row-meta{font-family:'IBM Plex Mono',monospace;font-size:.74rem;color:var(--muted)!important}
/* buttons */
div[data-testid="stButton"]>button,div[data-testid="stFormSubmitButton"]>button,div[data-testid="stDownloadButton"]>button{background:var(--surface2)!important;color:var(--ink)!important;border:1px solid var(--line)!important;border-radius:9px!important;font-weight:600!important;min-height:2.4rem;width:100%;padding:.35rem .6rem!important;white-space:nowrap!important;transition:all .15s}
div[data-testid="stButton"]>button *,div[data-testid="stFormSubmitButton"]>button *,div[data-testid="stDownloadButton"]>button *{color:inherit!important}
div[data-testid="stButton"]>button:hover,div[data-testid="stDownloadButton"]>button:hover{border-color:var(--accent)!important;color:var(--accent)!important}
div[data-testid="stButton"]>button[kind="primary"],div[data-testid="stFormSubmitButton"]>button{background:var(--accent)!important;color:var(--accent-ink)!important;border-color:var(--accent)!important}
div[data-testid="stButton"]>button[kind="primary"]:hover,div[data-testid="stFormSubmitButton"]>button:hover{background:var(--accent-hi)!important;color:var(--accent-ink)!important}
.btn-link{display:inline-block;text-decoration:none!important;text-align:center;width:100%;box-sizing:border-box;padding:.55rem .6rem;border-radius:9px;border:1px solid var(--line);background:var(--surface2);color:var(--ink)!important;font-weight:600;font-size:.9rem}
.btn-link:hover{border-color:var(--accent);color:var(--accent)!important}
/* inputs */
div[data-testid="stTextInput"] input,[data-baseweb="select"]>div{background:var(--surface)!important;color:var(--ink)!important;border:1px solid var(--line)!important;border-radius:10px!important;min-height:2.7rem}
[data-baseweb="popover"] *,[data-baseweb="menu"]{background:var(--surface)!important;color:var(--ink)!important}
div[data-testid="stTextInput"] input::placeholder{color:var(--muted)!important;opacity:1}
div[data-testid="stTextInput"] input:focus{border-color:var(--accent)!important;box-shadow:0 0 0 1px var(--accent)!important}
div[data-testid="stRadio"]>div{gap:.5rem;flex-wrap:wrap}
div[data-testid="stRadio"] label{background:var(--surface);border:1px solid var(--line);padding:.35rem .9rem;border-radius:999px;font-size:.84rem;cursor:pointer}
div[data-testid="stRadio"] label:has(input:checked){background:var(--accent);border-color:var(--accent)}
div[data-testid="stRadio"] label:has(input:checked) *{color:var(--accent-ink)!important}
div[data-testid="stRadio"] label>div:first-child{display:none}
[data-testid="stExpander"]{background:var(--surface)!important;border:1px solid var(--line)!important;border-radius:12px!important}
[data-testid="stMetric"]{background:var(--surface);border:1px solid var(--line);border-radius:14px;padding:.9rem 1rem}
[data-testid="stMetricLabel"] *{color:var(--muted)!important;font-family:'IBM Plex Mono',monospace;font-size:.75rem}
[data-testid="stMetricValue"]{font-family:'Sora',sans-serif;font-weight:700}
[data-testid="stImage"] img{border-radius:14px}
[data-testid="stAlert"]{border-radius:12px}
/* detail */
.detail-title{font-family:'Sora',sans-serif;font-weight:800;font-size:clamp(1.9rem,4.6vw,3.2rem);line-height:1.08;margin:.4rem 0 .5rem;color:var(--ink)!important}
.tagline-quote{color:var(--accent)!important;font-style:italic;margin-bottom:1rem}
.overview{color:var(--muted)!important;line-height:1.85;max-width:72ch}
.backdrop{height:clamp(150px,26vw,300px);border-radius:18px;background-size:cover;background-position:center 25%;border:1px solid var(--line);margin-bottom:1.2rem;position:relative;overflow:hidden}
.backdrop::after{content:"";position:absolute;inset:0;background:linear-gradient(0deg,var(--bg),transparent 70%)}
.ring-wrap{text-align:center}
.ring-wrap small{display:block;margin-top:.25rem;font-family:'IBM Plex Mono',monospace;font-size:.62rem;color:var(--muted)!important}
.ring{--p:0;width:74px;height:74px;border-radius:50%;display:grid;place-items:center;background:conic-gradient(var(--accent) calc(var(--p)*1%),var(--surface2) 0)}
.ring span{width:58px;height:58px;border-radius:50%;background:var(--surface);display:grid;place-items:center;font-family:'IBM Plex Mono',monospace;font-weight:600;color:var(--ink)!important}
.facts{display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:.7rem;margin:1.1rem 0}
.fact{background:var(--surface);border:1px solid var(--line);border-radius:12px;padding:.7rem .9rem}
.fact small{display:block;color:var(--muted)!important;font-family:'IBM Plex Mono',monospace;font-size:.68rem}
.fact b{font-family:'Sora',sans-serif;font-size:1rem;color:var(--ink)!important}
.bar{height:6px;border-radius:99px;background:var(--surface2);overflow:hidden;margin-top:.45rem}
.bar i{display:block;height:100%;background:linear-gradient(90deg,var(--data),var(--accent))}
.rec-row{background:var(--surface);border:1px solid var(--line);border-radius:12px;padding:.7rem .9rem;margin-bottom:.4rem}
/* where to watch */
.prov-label{font-family:'IBM Plex Mono',monospace;font-size:.72rem;color:var(--data)!important;margin:1rem 0 .45rem;text-transform:uppercase;letter-spacing:.06em}
.prov-grid{display:flex;flex-wrap:wrap;gap:.6rem}
.prov{display:flex;align-items:center;gap:.65rem;background:var(--surface);border:1px solid var(--line);border-radius:12px;padding:.5rem .85rem .5rem .5rem}
.prov img{width:38px;height:38px;border-radius:9px;object-fit:cover}
.prov b{display:block;font-size:.88rem;color:var(--ink)!important}
.prov small{display:block;font-family:'IBM Plex Mono',monospace;font-size:.68rem;color:var(--muted)!important}
.prov.free{border-color:var(--data)}
/* responsive grid rows */
div[data-testid="stHorizontalBlock"]:has(.movie-card){flex-wrap:wrap!important;gap:1rem!important}
div[data-testid="stElementContainer"]:has(.pair)+div[data-testid="stHorizontalBlock"],div[data-testid="element-container"]:has(.pair)+div[data-testid="stHorizontalBlock"]{flex-wrap:nowrap!important;gap:.4rem!important}
div[data-testid="stElementContainer"]:has(.pair)+div[data-testid="stHorizontalBlock"]>div,div[data-testid="element-container"]:has(.pair)+div[data-testid="stHorizontalBlock"]>div{flex:1 1 0!important;min-width:0!important}
@media(max-width:900px){
div[data-testid="stHorizontalBlock"]:has(.movie-card)>div{flex:1 1 calc(50% - 1rem)!important;min-width:calc(50% - 1rem)!important}
.hero-poster{display:none}
.hero::after{background:linear-gradient(90deg,var(--surface),transparent 120%)}
}
@media(max-width:640px){
[data-testid="stAppViewBlockContainer"]{padding-top:.6rem!important}
.hero{min-height:240px;border-radius:16px}
.card-info{padding:.6rem}
.card-title{font-size:.84rem}
.ring{width:60px;height:60px}.ring span{width:46px;height:46px;font-size:.8rem}
div[data-testid="stButton"]>button{font-size:.8rem!important;padding:.3rem .3rem!important}
}
@media(max-width:340px){div[data-testid="stHorizontalBlock"]:has(.movie-card)>div{flex:1 1 100%!important;min-width:100%!important}}
/* compact cards (detail page recommendations): 6 per row, 3 on tablet, 2 on phone */
.movie-card.compact{border-radius:12px}
.movie-card.compact .card-info{padding:.5rem .6rem .65rem}
.movie-card.compact .card-title{font-size:.8rem;min-height:2.6em}
.movie-card.compact .card-meta{font-size:.66rem;margin-top:.3rem}
.movie-card.compact .badge{font-size:.64rem;padding:.12rem .38rem;top:.4rem;right:.4rem}
.movie-card.compact .badge.saved{left:.4rem;right:auto}
div[data-testid="stHorizontalBlock"]:has(.movie-card.compact){gap:.7rem!important}
div[data-testid="stHorizontalBlock"]:has(.movie-card.compact)>div{flex:1 1 calc(16.66% - .7rem)!important;min-width:calc(16.66% - .7rem)!important}
@media(max-width:900px){div[data-testid="stHorizontalBlock"]:has(.movie-card.compact)>div{flex:1 1 calc(33.33% - .7rem)!important;min-width:calc(33.33% - .7rem)!important}}
@media(max-width:560px){div[data-testid="stHorizontalBlock"]:has(.movie-card.compact)>div{flex:1 1 calc(50% - .7rem)!important;min-width:calc(50% - .7rem)!important}}
@media(prefers-reduced-motion:reduce){*{transition:none!important;animation:none!important}}
"""


def inject_css(light: bool):
    palette = LIGHT if light else DARK
    root = ":root{" + ";".join(f"--{k}:{v}" for k, v in palette.items()) + "}"
    css = "\n".join(line for line in BASE_CSS.splitlines() if line.strip())
    st.markdown(f"<style>{FONTS}{root}{css}</style>", unsafe_allow_html=True)


def compact(markup: str) -> str:
    """Strip indentation and blank lines so Markdown never treats HTML as code."""
    return "".join(line.strip() for line in markup.splitlines())


def md(markup: str):
    st.markdown(compact(markup), unsafe_allow_html=True)


inject_css(st.session_state["light"])


# =========================================================
# AUTHENTICATION + PER-USER DATA
# =========================================================

AUTH_TIMEOUT = (5, 20)


def _secret(name: str, default: str = "") -> str:
    value = os.getenv(name, "")
    try:
        value = st.secrets.get(name, value)
    except Exception:
        pass
    return str(value or default).strip()


def auth_headers() -> dict:
    token = st.session_state.get("access_token")
    return {"Authorization": f"Bearer {token}"} if token else {}


def auth_post(path: str, payload: dict, headers: Optional[dict] = None) -> dict:
    response = requests.post(f"{API_BASE}{path}", json=payload,
                             headers=headers or {}, timeout=AUTH_TIMEOUT)
    response.raise_for_status()
    data = response.json()
    if not isinstance(data, dict):
        raise ValueError("The authentication server returned an invalid response.")
    return data


def _extract_token(data: dict) -> Optional[str]:
    return data.get("access_token") or data.get("token")


def _unwrap_user(data: dict) -> dict:
    user = data.get("user") or data.get("profile") or {}
    if not isinstance(user, dict):
        user = {}
    return user


def _user_data_payload() -> dict:
    return {
        "watchlist": st.session_state.get("watchlist", []),
        "recent": st.session_state.get("recent", []),
        "preferences": {
            "light": bool(st.session_state.get("light", False)),
            "density": int(st.session_state.get("density", 4)),
            "layout": st.session_state.get("layout", "Grid"),
            "category": st.session_state.get("category", "Popular"),
            "region": st.session_state.get("region", "Netherlands"),
            "min_rating": float(st.session_state.get("min_rating", 0.0)),
            "max_runtime": int(st.session_state.get("max_runtime", RUNTIME_NO_LIMIT)),
            "sort": st.session_state.get("sort", "Default"),
        },
    }


def load_user_data(token: str) -> None:
    """Load the authenticated user's private JSON data from the FastAPI backend."""
    response = requests.get(f"{API_BASE}/user/data",
                            headers={"Authorization": f"Bearer {token}"}, timeout=AUTH_TIMEOUT)
    response.raise_for_status()
    body = response.json()
    data = body.get("data", body) if isinstance(body, dict) else {}
    if not isinstance(data, dict):
        data = {}
    watchlist = data.get("watchlist", [])
    recent = data.get("recent", [])
    prefs = data.get("preferences", {})
    st.session_state["watchlist"] = [m for m in watchlist if isinstance(m, dict)] if isinstance(watchlist, list) else []
    st.session_state["recent"] = [m for m in recent if isinstance(m, dict)] if isinstance(recent, list) else []
    if isinstance(prefs, dict):
        allowed = {"light", "density", "layout", "category", "region", "min_rating", "max_runtime", "sort"}
        for key in allowed:
            if key in prefs:
                st.session_state[key] = prefs[key]
    st.session_state["_last_saved_payload"] = json.dumps(_user_data_payload(), sort_keys=True, default=str)


def persist_user_data() -> None:
    """Save only when user-owned data changes; movie catalogue calls remain cacheable/public."""
    token = st.session_state.get("access_token")
    if not token or not st.session_state.get("auth_user"):
        return
    payload = _user_data_payload()
    fingerprint = json.dumps(payload, sort_keys=True, default=str)
    if fingerprint == st.session_state.get("_last_saved_payload"):
        return
    response = requests.put(f"{API_BASE}/user/data", json=payload,
                            headers={"Authorization": f"Bearer {token}"}, timeout=AUTH_TIMEOUT)
    response.raise_for_status()
    st.session_state["_last_saved_payload"] = fingerprint


def _accept_auth_response(data: dict) -> None:
    token = _extract_token(data)
    if not token:
        raise ValueError("The backend did not return an access token.")
    st.session_state["access_token"] = token
    st.session_state["auth_user"] = _unwrap_user(data)
    if not st.session_state["auth_user"].get("email"):
        st.session_state["auth_user"]["email"] = data.get("email", "")
    load_user_data(token)
    st.session_state["auth_error"] = None


def _exchange_google_identity() -> None:
    """Exchange Streamlit's verified Google OIDC identity for this app's API token."""
    try:
        google_user = st.user
        if not getattr(google_user, "is_logged_in", False):
            return
        claims = {key: google_user.get(key) for key in ("sub", "email", "name", "email_verified")}
        if not claims.get("sub") or not claims.get("email"):
            raise ValueError("Google did not return the required account details.")
        if claims.get("email_verified") is not True:
            raise ValueError("Your Google email must be verified before signing in.")
        internal_secret = _secret("APP_INTERNAL_SECRET")
        if len(internal_secret) < 32:
            raise ValueError("APP_INTERNAL_SECRET is missing. Configure it in .env and Streamlit secrets.")
        data = auth_post("/auth/oidc", claims, {"X-App-Internal-Secret": internal_secret})
        _accept_auth_response(data)
    except Exception as exc:
        st.session_state["auth_error"] = error_message(exc)


def render_auth_gate() -> None:
    """Render sign-in/register and stop the app until a user has authenticated."""
    if not st.session_state.get("access_token"):
        _exchange_google_identity()
    if st.session_state.get("access_token") and st.session_state.get("auth_user"):
        return

    st.markdown('<div class="auth-shell">', unsafe_allow_html=True)
    st.markdown('<div class="brand">CINÉ<b>MA</b></div><div class="auth-subtitle">Sign in to keep your watchlist, viewing history and preferences private to your account.</div>', unsafe_allow_html=True)
    if st.session_state.get("auth_error"):
        st.error(st.session_state["auth_error"])

    try:
        if hasattr(st, "login"):
            if st.button("Continue with Google", use_container_width=True, type="primary"):
                st.login("google")
        else:
            st.info("Google sign-in requires Streamlit 1.42+ with auth support.")
    except Exception as exc:
        st.warning("Google sign-in is not configured. " + str(exc))

    st.markdown("<div style='text-align:center;color:var(--muted);margin:1rem 0'>or use email and password</div>", unsafe_allow_html=True)
    login_tab, register_tab = st.tabs(["Sign in", "Create account"])
    with login_tab:
        with st.form("email_login", clear_on_submit=False):
            email = st.text_input("Email address", key="login_email")
            password = st.text_input("Password", type="password", key="login_password")
            submitted = st.form_submit_button("Sign in", use_container_width=True, type="primary")
        if submitted:
            try:
                data = auth_post("/auth/login", {"email": email.strip().lower(), "password": password})
                _accept_auth_response(data)
                st.rerun()
            except Exception as exc:
                st.error(error_message(exc))
    with register_tab:
        with st.form("email_register", clear_on_submit=False):
            name = st.text_input("Your name", key="register_name")
            email = st.text_input("Email address", key="register_email")
            password = st.text_input("Password (at least 8 characters)", type="password", key="register_password")
            confirm = st.text_input("Confirm password", type="password", key="register_confirm")
            submitted = st.form_submit_button("Create account", use_container_width=True, type="primary")
        if submitted:
            if password != confirm:
                st.error("The passwords do not match.")
            elif len(password) < 8:
                st.error("Use a password with at least 8 characters.")
            elif not name.strip() or not email.strip():
                st.error("Name and email are required.")
            else:
                try:
                    data = auth_post("/auth/register", {"name": name.strip(), "email": email.strip().lower(), "password": password})
                    _accept_auth_response(data)
                    st.rerun()
                except Exception as exc:
                    st.error(error_message(exc))
    st.markdown('</div>', unsafe_allow_html=True)
    st.stop()


# =========================================================
# API + DATA HELPERS
# =========================================================

@st.cache_data(ttl=180, show_spinner=False)
def api_get(path: str, params: Optional[dict] = None):
    response = requests.get(f"{API_BASE}{path}", params=params, timeout=(5, 25))
    response.raise_for_status()
    try:
        return response.json()
    except ValueError:
        raise ValueError("The backend returned an invalid response.")


def tmdb_raw(path: str, params: Optional[dict] = None):
    """Direct TMDB call (used for runtime, ratings and streaming providers)."""
    if not TMDB_ON:
        return None
    headers, p = {}, dict(params or {})
    if TMDB_TOKEN:
        headers["Authorization"] = f"Bearer {TMDB_TOKEN}"
    else:
        p["api_key"] = TMDB_KEY
    r = requests.get(f"https://api.themoviedb.org/3{path}", params=p, headers=headers, timeout=(3, 10))
    r.raise_for_status()
    return r.json()


def error_message(exc: Exception) -> str:
    if isinstance(exc, requests.ConnectionError):
        return f"Can't reach the backend at {API_BASE}. Start FastAPI in another terminal."
    if isinstance(exc, requests.Timeout):
        return "The server took too long to respond. Retry in a moment."
    if isinstance(exc, requests.HTTPError):
        try:
            detail = exc.response.json().get("detail")
            return str(detail or f"HTTP {exc.response.status_code}")
        except Exception:
            return f"Backend error: HTTP {exc.response.status_code}"
    return str(exc)


render_auth_gate()


def esc(value: Any, fallback: str = "") -> str:
    return html.escape(fallback if value is None else str(value))


def rnum(value: Any) -> Optional[float]:
    try:
        n = float(value)
        return n if 0 <= n <= 10 else None
    except (TypeError, ValueError):
        return None


def pick_rating(d: dict) -> Optional[float]:
    """First usable rating from the several keys a backend might use (0 = unrated)."""
    for key in ("vote_average", "tmdb_rating", "rating", "imdb_rating"):
        n = rnum(d.get(key))
        if n:
            return n
    return None


def safe_rating(value: Any) -> str:
    n = rnum(value)
    return "N/A" if not n else f"{n:.1f}"


def safe_year(date: Any) -> str:
    return str(date)[:4] if date else "TBA"


def year_num(movie: dict) -> int:
    try:
        return int(str(movie.get("release_date"))[:4])
    except (TypeError, ValueError):
        return 0


def to_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def fmt_runtime(minutes: Any) -> str:
    m = to_int(minutes)
    if m <= 0:
        return ""
    return f"{m}m" if m < 60 else f"{m // 60}h {m % 60:02d}m"


def get_list(data: Any) -> list:
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("results", "movies", "data"):
            if isinstance(data.get(key), list):
                return data[key]
    return []


def valid_img(url: Any) -> bool:
    """Allow only HTTPS TMDB image URLs; reject malformed and unsafe HTML URLs."""
    if not isinstance(url, str):
        return False
    value = url.strip()
    return value.startswith(("https://image.tmdb.org/t/p/", "https://www.themoviedb.org/t/p/")) and not any(ch in value for ch in ('"', "'", "\n", "\r"))


def normalise_movie(movie: dict) -> dict:
    raw_poster = movie.get("poster_url") or movie.get("poster_path")
    poster = None
    if isinstance(raw_poster, str) and raw_poster.strip():
        raw_poster = raw_poster.strip()
        if raw_poster.startswith("/"):
            poster = f"https://image.tmdb.org/t/p/w500{raw_poster}"
        elif valid_img(raw_poster):
            poster = raw_poster
    return {
        "tmdb_id": to_int(movie.get("tmdb_id") or movie.get("id")) or None,
        "title": movie.get("title") or movie.get("name") or "Untitled",
        "poster_url": poster,
        "release_date": movie.get("release_date"),
        "vote_average": pick_rating(movie),
        "overview": movie.get("overview"),
        "runtime": to_int(movie.get("runtime") or movie.get("duration")) or None,
    }


# ---- per-movie detail enrichment (rating + runtime for cards) ----

@st.cache_resource
def _detail_store() -> dict:
    return {}


def _pick_fields(d: dict) -> dict:
    out = {}
    rt = to_int(d.get("runtime") or d.get("duration"))
    if rt:
        out["runtime"] = rt
    r = pick_rating(d)
    if r:
        out["vote_average"] = r
    if d.get("vote_count"):
        out["vote_count"] = d["vote_count"]
    if d.get("imdb_id"):
        out["imdb_id"] = d["imdb_id"]
    return out


def _fetch_detail(mid: int):
    """Plain function (no Streamlit calls) so it is safe inside worker threads."""
    out: dict = {}
    try:
        r = requests.get(f"{API_BASE}/movie/id/tmdb_id/{mid}", timeout=(3, 10))
        r.raise_for_status()
        d = r.json()
        if isinstance(d, dict):
            out = _pick_fields(d)
    except Exception:
        pass
    if TMDB_ON and not (out.get("runtime") and out.get("vote_average") and out.get("imdb_id")):
        try:
            d2 = tmdb_raw(f"/movie/{mid}") or {}
            for k, v in _pick_fields(d2).items():
                out.setdefault(k, v)
        except Exception:
            pass
    return out or None


def batch_details(ids: tuple) -> dict:
    store, now = _detail_store(), time.time()
    todo = [i for i in ids if i not in store or now - store[i][0] > 3600]
    if todo:
        with ThreadPoolExecutor(max_workers=8) as pool:
            for mid, res in zip(todo, pool.map(_fetch_detail, todo)):
                if res:
                    store[mid] = (now, res)
    return {i: store[i][1] for i in ids if i in store}


def enrich(movies: list, need_runtime: bool = False) -> list:
    """Fetch missing metadata only when the active view actually needs it."""
    need = tuple({to_int(m.get("tmdb_id")) for m in movies
                  if to_int(m.get("tmdb_id")) and
                  (not rnum(m.get("vote_average")) or (need_runtime and not m.get("runtime")))})
    if not need:
        return movies
    info = batch_details(need)
    for m in movies:
        extra = info.get(to_int(m.get("tmdb_id")))
        if not extra:
            continue
        if not m.get("runtime"):
            m["runtime"] = extra.get("runtime")
        if not rnum(m.get("vote_average")):
            m["vote_average"] = extra.get("vote_average")
    return movies


@st.cache_data(ttl=86400, show_spinner=False)
def imdb_lookup(imdb_id: str):
    """Real IMDb rating via OMDb (optional, needs OMDB_API_KEY)."""
    if not OMDB_KEY or not imdb_id:
        return None
    try:
        r = requests.get("https://www.omdbapi.com/", params={"i": imdb_id, "apikey": OMDB_KEY},
                         timeout=(3, 8))
        data = r.json()
        n = rnum(data.get("imdbRating")) if data.get("Response") == "True" else None
        return {"rating": n, "votes": data.get("imdbVotes")} if n else None
    except Exception:
        return None


@st.cache_data(ttl=3600, show_spinner=False)
def watch_providers(mid: int):
    return tmdb_raw(f"/movie/{int(mid)}/watch/providers")


def fetch_movies(category_key: str, limit: int, enrich_data: bool = False) -> list:
    data = api_get("/home", {"category": category_key, "limit": limit})
    return [normalise_movie(m) for m in get_list(data) if isinstance(m, dict)]


def saved_ids() -> set:
    return {m.get("tmdb_id") for m in st.session_state["watchlist"]}


def apply_filters(movies: list) -> list:
    floor = st.session_state["min_rating"]
    out = [m for m in movies if (rnum(m.get("vote_average")) or 0) >= floor]
    cap = st.session_state["max_runtime"]
    if cap < RUNTIME_NO_LIMIT:
        out = [m for m in out if not m.get("runtime") or m["runtime"] <= cap]
    sort = st.session_state["sort"]
    if sort.startswith("Rating"):
        out.sort(key=lambda m: rnum(m.get("vote_average")) or 0, reverse=True)
    elif sort == "Newest first":
        out.sort(key=year_num, reverse=True)
    elif sort == "Oldest first":
        out.sort(key=lambda m: year_num(m) or 9999)
    elif sort == "Title A–Z":
        out.sort(key=lambda m: str(m.get("title")).lower())
    elif sort == "Shortest first":
        out.sort(key=lambda m: m.get("runtime") or 9999)
    elif sort == "Longest first":
        out.sort(key=lambda m: m.get("runtime") or 0, reverse=True)
    return out


# =========================================================
# CALLBACKS
# =========================================================

def request_scroll():
    st.session_state["scroll_top"] = True


def open_movie(movie: dict):
    mid = int(movie["tmdb_id"])
    st.session_state["selected_movie_id"] = mid
    recent = [r for r in st.session_state["recent"] if r["tmdb_id"] != mid]
    st.session_state["recent"] = ([{"tmdb_id": mid, "title": movie.get("title")}] + recent)[:6]
    request_scroll()


def open_by_id(mid: int, title: str):
    open_movie({"tmdb_id": mid, "title": title})


def go_back():
    st.session_state["selected_movie_id"] = None
    request_scroll()


def toggle_save(movie: dict):
    mid = movie.get("tmdb_id")
    wl = st.session_state["watchlist"]
    if mid in saved_ids():
        st.session_state["watchlist"] = [m for m in wl if m.get("tmdb_id") != mid]
        st.toast(f"Removed {movie.get('title')}")
    else:
        wl.append({**movie, "watched": False, "added": datetime.now().strftime("%Y-%m-%d")})
        st.toast(f"Saved {movie.get('title')}")


def toggle_watched(mid):
    for m in st.session_state["watchlist"]:
        if m.get("tmdb_id") == mid:
            m["watched"] = not m.get("watched", False)


def set_search(query: str):
    st.session_state["search_query"] = query
    st.session_state["selected_movie_id"] = None
    request_scroll()


def clear_search():
    st.session_state["search_query"] = ""
    request_scroll()


def on_nav():
    st.session_state["search_query"] = ""
    st.session_state["selected_movie_id"] = None
    request_scroll()


def sign_out():
    """Clear the app session from a callback, before widgets are recreated."""
    was_google = False
    try:
        was_google = bool(getattr(st.user, "is_logged_in", False))
    except Exception:
        pass

    # Clear authentication and user-specific sync state.
    for key in (
        "access_token",
        "auth_user",
        "auth_loaded_for",
        "_last_saved_payload",
        "auth_error",
    ):
        st.session_state.pop(key, None)

    # Callback execution happens before the next script run creates widgets,
    # so resetting widget-backed keys here is safe.
    for key, value in DEFAULTS.items():
        st.session_state[key] = value.copy() if isinstance(value, (list, dict)) else value

    if was_google and hasattr(st, "logout"):
        st.logout()


def on_category():
    st.session_state["limit"] = PAGE_STEP


def more():
    st.session_state["limit"] = min(st.session_state["limit"] + PAGE_STEP, PAGE_MAX)


def clear_watchlist():
    st.session_state["watchlist"] = []


def scroll_to_top_if_requested():
    """Scrolls Streamlit's main container to the top after page changes."""
    if not st.session_state.get("scroll_top"):
        return
    st.session_state["scroll_top"] = False
    components.html(
        f"""<script>/* {time.time()} */
        const go = () => {{
          const d = window.parent.document;
          [d.querySelector('[data-testid="stMain"]'), d.querySelector('section.main'),
           d.querySelector('[data-testid="stAppViewContainer"]'), d.documentElement, d.body]
            .forEach(el => {{ if (el) {{ el.scrollTop = 0; if (el.scrollTo) el.scrollTo(0, 0); }} }});
          window.parent.scrollTo(0, 0);
        }};
        go(); setTimeout(go, 60); setTimeout(go, 250); setTimeout(go, 600);
        </script>""",
        height=0,
    )


# =========================================================
# UI COMPONENTS
# =========================================================

def section(title: str, sub: str = ""):
    md(f'<div class="sec"><h2>{esc(title)}</h2>{f"<p>{esc(sub)}</p>" if sub else ""}</div>')


def poster_html(poster: Any, title: str, cls: str = "poster") -> str:
    """Render a fixed-ratio poster with an in-place fallback when the remote image fails."""
    frame_cls = "poster-frame"
    if cls == "poster-detail":
        frame_cls += " detail-frame"
    elif cls == "hero-poster":
        frame_cls += " hero-frame"
    if valid_img(poster):
        image = (f"<img class=\"{cls}\" src=\"{html.escape(poster, quote=True)}\" "
                 f"alt=\"Poster for {esc(title)}\" loading=\"lazy\" decoding=\"async\" "
                 f"onerror=\"this.style.display='none';this.nextElementSibling.style.display='flex'\">")
        fallback = '<div class="poster-placeholder">🎬 Image unavailable</div>'
        return f'<div class="{frame_cls}">{image}{fallback}</div>'
    return f'<div class="{frame_cls}"><div class="poster-placeholder" style="display:flex">🎬 No poster</div></div>'


def meta_line(movie: dict) -> str:
    parts = [safe_year(movie.get("release_date")), fmt_runtime(movie.get("runtime"))]
    return " · ".join(p for p in parts if p)


def render_grid(movies: list, prefix: str, compact_cards: bool = False):
    movies = [normalise_movie(m) for m in movies if isinstance(m, dict)]
    runtime_needed = (st.session_state.get("max_runtime", RUNTIME_NO_LIMIT) < RUNTIME_NO_LIMIT
                      or st.session_state.get("sort", "Default") in ("Shortest first", "Longest first"))
    if runtime_needed:
        movies = enrich(movies, need_runtime=True)
        movies = apply_filters(movies)
    if not movies:
        st.info("Nothing to show here. Try lowering the minimum rating or changing the category.")
        return

    ids = saved_ids()
    if st.session_state["layout"] == "List" and not compact_cards:
        for i, m in enumerate(movies):
            render_list_row(m, f"{prefix}_{i}", m.get("tmdb_id") in ids)
        return

    n = 6 if compact_cards else st.session_state["density"]
    cls = "movie-card compact" if compact_cards else "movie-card"
    for start in range(0, len(movies), n):
        cols = st.columns(n, gap="small")
        for offset, movie in enumerate(movies[start:start + n]):
            i = start + offset
            mid = movie.get("tmdb_id")
            title = str(movie.get("title"))
            saved = mid in ids
            with cols[offset]:
                md(f"""
                <div class="{cls}">
                  <span class="badge">★ {safe_rating(movie.get("vote_average"))}</span>
                  {'<span class="badge saved">✓ SAVED</span>' if saved else ''}
                  {poster_html(movie.get("poster_url"), title)}
                  <div class="card-info">
                    <div class="card-title">{esc(title)}</div>
                    <div class="card-meta">{esc(meta_line(movie))}</div>
                  </div>
                </div>""")
                md('<span class="pair"></span>')
                a, b = st.columns(2, gap="small")
                a.button("Details", key=f"{prefix}_d_{i}_{mid}", on_click=open_movie,
                         args=(movie,), disabled=not mid, use_container_width=True)
                b.button("✓ Saved" if saved else "＋ Save", key=f"{prefix}_s_{i}_{mid}",
                         on_click=toggle_save, args=(movie,), use_container_width=True,
                         type="primary" if not saved else "secondary")


def render_list_row(movie: dict, key: str, saved: bool):
    mid = movie.get("tmdb_id")
    c1, c2, c3, c4 = st.columns([1, 6, 2, 2], gap="small", vertical_alignment="center")
    c1.markdown(compact(poster_html(movie.get("poster_url"), movie.get("title"), "thumb")),
                unsafe_allow_html=True)
    c2.markdown(compact(f"""<div class="row-title">{esc(movie.get("title"))}</div>
        <div class="row-meta">{esc(meta_line(movie))} · ★ {safe_rating(movie.get("vote_average"))}</div>"""),
                unsafe_allow_html=True)
    c3.button("Details", key=f"{key}_d", on_click=open_movie, args=(movie,), disabled=not mid,
              use_container_width=True)
    c4.button("✓ Saved" if saved else "＋ Save", key=f"{key}_s", on_click=toggle_save,
              args=(movie,), use_container_width=True)


def filter_bar():
    with st.expander("Sort, filter and display", expanded=False):
        c1, c2, c3 = st.columns(3)
        c1.selectbox("Sort by", SORTS, key="sort")
        c2.slider("Minimum rating", 0.0, 9.0, step=0.5, key="min_rating")
        c3.slider("Max duration (min) · 240 = no limit", 60, RUNTIME_NO_LIMIT, step=10, key="max_runtime")
        c4, c5 = st.columns(2)
        c4.select_slider("Cards per row", options=[2, 3, 4, 5, 6], key="density")
        c5.radio("Layout", ["Grid", "List"], horizontal=True, key="layout")


def price_note(cc: str, name: str, kind: str) -> str:
    if kind == "free":
        return "Free"
    if kind == "ads":
        return "Free with ads"
    if kind == "rent":
        return "Rent · price on platform"
    if kind == "buy":
        return "Buy · price on platform"
    low = name.lower()
    for key, text in PLATFORM_PRICES.get(cc, {}).items():
        if key in low:
            return text + " *"
    return "Subscription · see platform"


def render_where_to_watch(mid: int, title: str):
    section("Where to watch", "Streaming availability for your region.")
    sel, _ = st.columns([1, 3])
    sel.selectbox("Region", list(REGIONS), key="region")
    cc = REGIONS[st.session_state["region"]]
    jw = f"https://www.justwatch.com/{cc.lower()}/search?q=" + quote_plus(title)

    if not TMDB_ON:
        st.info("Set the TMDB_API_KEY (or TMDB_READ_TOKEN) environment variable to see streaming "
                "platforms here.")
        st.markdown(f'<a class="btn-link" href="{jw}" target="_blank" rel="noopener">Search on JustWatch</a>',
                    unsafe_allow_html=True)
        return
    try:
        data = watch_providers(int(mid))
    except Exception as exc:
        st.warning("Couldn't load streaming availability. " + error_message(exc))
        return

    info = ((data or {}).get("results") or {}).get(cc)
    if not info:
        st.info(f"No streaming info for {st.session_state['region']} yet. It may be in cinemas, or "
                "try another region.")
        st.markdown(f'<a class="btn-link" href="{jw}" target="_blank" rel="noopener">Search on JustWatch</a>',
                    unsafe_allow_html=True)
        return

    groups = [("flatrate", "Stream (subscription)"), ("free", "Free"), ("ads", "Free with ads"),
              ("rent", "Rent"), ("buy", "Buy")]
    parts = []
    for kind, label in groups:
        provs = [p for p in (info.get(kind) or []) if isinstance(p, dict) and p.get("provider_name")]
        if not provs:
            continue
        tiles = []
        for p in provs:
            name = str(p["provider_name"])
            logo = f"https://image.tmdb.org/t/p/w92{p['logo_path']}" if p.get("logo_path") else ""
            img = f'<img src="{html.escape(logo, quote=True)}" alt="{esc(name)} logo" loading="lazy">' if logo else ""
            free = " free" if kind in ("free", "ads") else ""
            tiles.append(f'<div class="prov{free}">{img}<div><b>{esc(name)}</b>'
                         f'<small>{esc(price_note(cc, name, kind))}</small></div></div>')
        parts.append(f'<div class="prov-label">{esc(label)}</div><div class="prov-grid">{"".join(tiles)}</div>')
    md("".join(parts) or "<p>No providers listed.</p>")

    st.caption("* Approximate starting price and may be outdated. TMDB/JustWatch don't publish prices; "
               "check the platform before subscribing. Availability data from JustWatch via TMDB.")
    l1, l2, _ = st.columns([1, 1, 2])
    if info.get("link"):
        l1.markdown(f'<a class="btn-link" href="{html.escape(str(info["link"]), quote=True)}" '
                    f'target="_blank" rel="noopener">All options</a>', unsafe_allow_html=True)
    l2.markdown(f'<a class="btn-link" href="{jw}" target="_blank" rel="noopener">JustWatch</a>',
                unsafe_allow_html=True)


# =========================================================
# HEADER
# =========================================================

h1, h2, h3 = st.columns([2, 4, 2], vertical_alignment="center")
with h1:
    md('<div class="brand">CINÉ<b>MA</b></div><div class="tagline">Movie intelligence, tuned to your taste</div>')
with h2:
    st.radio("Navigation", VIEWS, horizontal=True, label_visibility="collapsed",
             key="view", on_change=on_nav)
with h3:
    online = st.session_state["online"]
    state, label = ("on", "API online") if online else (("off", "API offline") if online is False else ("", "API idle"))
    md(f'<span class="status"><span class="dot {state}"></span>{label}</span>')
    account_email = (st.session_state.get("auth_user") or {}).get("email", "Signed in")
    st.caption(f"Account: {account_email}")
    st.button(
        "Sign out",
        key="sign_out",
        use_container_width=True,
        on_click=sign_out,
    )
    st.toggle("Light mode", key="light")

# =========================================================
# SEARCH
# =========================================================

with st.form("movie_search", clear_on_submit=False):
    sc, bc = st.columns([6, 1])
    query = sc.text_input("Search", placeholder="Search by title: Dune, Interstellar, Inception…",
                          label_visibility="collapsed")
    submitted = bc.form_submit_button("Search", use_container_width=True)

if submitted:
    if query.strip():
        set_search(query.strip())
    else:
        st.warning("Enter a movie title to search.")


# =========================================================
# PAGES
# =========================================================

def page_details(selected_id: int):
    st.button("← Back", on_click=go_back)
    try:
        with st.spinner("Loading movie details…"):
            d = api_get(f"/movie/id/tmdb_id/{int(selected_id)}")
        if not isinstance(d, dict):
            raise ValueError("Invalid movie details response.")
        st.session_state["online"] = True
    except Exception as exc:
        st.session_state["online"] = not isinstance(exc, requests.ConnectionError)
        st.error("Couldn't load movie details. " + error_message(exc))
        return

    extra = batch_details((int(selected_id),)).get(int(selected_id), {})
    runtime = to_int(d.get("runtime") or d.get("duration")) or extra.get("runtime")
    tmdb_rating = pick_rating(d) or extra.get("vote_average")
    imdb_id = d.get("imdb_id") or extra.get("imdb_id")
    imdb = imdb_lookup(str(imdb_id)) if imdb_id else None
    vote_count = d.get("vote_count") or extra.get("vote_count")

    movie = normalise_movie({**d, "tmdb_id": selected_id})
    movie["runtime"] = runtime
    movie["vote_average"] = tmdb_rating
    title = str(movie["title"])

    back = d.get("backdrop_url") if valid_img(d.get("backdrop_url")) else movie["poster_url"]
    if valid_img(back):
        md(f'<div class="backdrop" style="background-image:url(\'{html.escape(str(back), quote=True)}\')"></div>')

    left, right = st.columns([1, 2.4], gap="large")
    with left:
        md(poster_html(movie["poster_url"], title, "poster-detail"))

    with right:
        main_rating = imdb["rating"] if imdb else tmdb_rating
        ring_label = "IMDb" if imdb else "TMDB"
        ring = (f'<div class="ring-wrap"><div class="ring" style="--p:{(main_rating or 0) * 10:.0f}">'
                f'<span>{safe_rating(main_rating)}</span></div><small>{ring_label}</small></div>')
        md(f'<div style="display:flex;justify-content:space-between;align-items:flex-start;gap:1rem">'
           f'<div class="detail-title">{esc(title)}</div>{ring}</div>')
        if d.get("tagline"):
            md(f'<div class="tagline-quote">“{esc(d.get("tagline"))}”</div>')

        genres = [g.get("name") for g in (d.get("genres") or []) if isinstance(g, dict) and g.get("name")]
        if genres:
            md("".join(f'<span class="chip hot">{esc(g)}</span>' for g in genres))

        facts = [("Release", safe_year(d.get("release_date")))]
        if runtime:
            facts.append(("Duration", fmt_runtime(runtime)))
        facts.append(("TMDB rating", f"★ {safe_rating(tmdb_rating)}"))
        if imdb:
            facts.append(("IMDb rating", f"★ {imdb['rating']:.1f}"))
        if vote_count:
            facts.append(("Votes", f"{to_int(vote_count):,}"))
        if d.get("original_language"):
            facts.append(("Language", str(d["original_language"]).upper()))
        md('<div class="facts">' + "".join(
            f'<div class="fact"><small>{esc(k)}</small><b>{esc(v)}</b></div>' for k, v in facts) + "</div>")

        md(f'<div class="overview">{esc(d.get("overview"), "No synopsis is available.")}</div>')

        cast = d.get("cast") or []
        names = [c.get("name") if isinstance(c, dict) else c for c in cast][:8]
        if names:
            md('<div class="sec"><p>Top cast</p></div>' + "".join(f'<span class="chip">{esc(n)}</span>' for n in names if n))

        st.write("")
        saved = selected_id in saved_ids()
        a, b, c = st.columns([1, 1, 1])
        a.button("✓ In watchlist" if saved else "＋ Add to watchlist", key="detail_save",
                 on_click=toggle_save, args=(movie,), type="secondary" if saved else "primary",
                 use_container_width=True)
        yt = "https://www.youtube.com/results?search_query=" + quote_plus(f"{title} official trailer")
        b.markdown(f'<a class="btn-link" href="{yt}" target="_blank" rel="noopener">▶ Watch trailer</a>',
                   unsafe_allow_html=True)
        if d.get("homepage"):
            c.markdown(f'<a class="btn-link" href="{html.escape(str(d["homepage"]), quote=True)}" '
                       f'target="_blank" rel="noopener">Official site</a>', unsafe_allow_html=True)

    st.markdown("---")
    render_where_to_watch(int(selected_id), title)

    st.markdown("---")
    section("Similar stories", "Content-based matches ranked by TF-IDF similarity.")
    try:
        recs = [r for r in get_list(api_get("/recommend/tfidf", {"title": title, "limit": 12}))
                if isinstance(r, dict)]
        if recs:
            for i, rec in enumerate(recs):
                try:
                    score = max(0.0, min(1.0, float(rec.get("score", 0))))
                except (TypeError, ValueError):
                    score = 0.0
                c1, c2 = st.columns([6, 1.3], vertical_alignment="center")
                c1.markdown(compact(f"""<div class="rec-row"><div style="display:flex;justify-content:space-between;gap:1rem">
                    <span class="row-title">{esc(rec.get("title"), "Untitled")}</span>
                    <span class="row-meta">{score:.0%} match</span></div>
                    <div class="bar"><i style="width:{score * 100:.0f}%"></i></div></div>"""),
                            unsafe_allow_html=True)
                c2.button("Find", key=f"sim_{i}", on_click=set_search, args=(str(rec.get("title", "")),))
        else:
            st.info("No similar movies were found.")
    except Exception as exc:
        st.warning("Similarity recommendations are unavailable. " + error_message(exc))

    st.markdown("---")
    section("More in these genres")
    try:
        render_grid(get_list(api_get("/recommend/genre", {"tmdb_id": int(selected_id), "limit": 12})),
                    "genre", compact_cards=True)
    except Exception as exc:
        st.warning("Genre recommendations are unavailable. " + error_message(exc))


def page_search():
    q = st.session_state["search_query"]
    section(f'Results for “{q}”')
    st.button("Clear search", on_click=clear_search)
    filter_bar()
    try:
        with st.spinner("Searching…"):
            raw = get_list(api_get("/tmdb/search", {"query": q, "page": 1}))
            found = [normalise_movie(m) for m in raw if isinstance(m, dict)]
        st.session_state["online"] = True
        results = apply_filters(found)
        if results:
            st.caption(f"{len(results)} results")
            render_grid(results[:30], "search")
        else:
            st.info("No results match. Check the spelling or loosen the filters.")
    except Exception as exc:
        st.session_state["online"] = not isinstance(exc, requests.ConnectionError)
        st.error(error_message(exc))


def page_discover():
    st.radio("Category", list(CATEGORIES), horizontal=True, label_visibility="collapsed",
             key="category", on_change=on_category)
    category = st.session_state["category"]
    try:
        with st.spinner(f"Loading {category.lower()}…"):
            movies = fetch_movies(CATEGORIES[category], st.session_state["limit"])
        st.session_state["online"] = True
    except Exception as exc:
        st.session_state["online"] = not isinstance(exc, requests.ConnectionError)
        st.error("Couldn't load movies. " + error_message(exc))
        with st.expander("Troubleshooting"):
            st.code("uvicorn main:app --reload --port 8000\nstreamlit run app.py")
        return

    if not movies:
        st.info("No movies are available in this category yet.")
        return

    featured = max((m for m in movies if valid_img(m["poster_url"])),
                   key=lambda m: rnum(m.get("vote_average")) or 0, default=movies[0])
    bg = html.escape(str(featured["poster_url"] or ""), quote=True)
    rt_chip = f'<span class="chip">⏱ {esc(fmt_runtime(featured.get("runtime")))}</span>' if featured.get("runtime") else ""
    md(f"""
    <div class="hero" style="--bg-img:url('{bg}')">
      <div class="hero-copy">
        <div class="hero-kicker">FEATURED IN {esc(category.upper())}</div>
        <h1>{esc(featured["title"])}</h1>
        <p>{esc(featured.get("overview"), "Open the details for the full story, cast and similar picks.")}</p>
        <span class="chip hot">★ {safe_rating(featured.get("vote_average"))}</span>
        <span class="chip">{esc(safe_year(featured.get("release_date")))}</span>
        {rt_chip}
      </div>
      {poster_html(featured["poster_url"], featured["title"], "hero-poster")}
    </div>""")

    a, b, c, _ = st.columns([1.2, 1.2, 1.4, 4])
    a.button("Open details", key="hero_open", on_click=open_movie, args=(featured,),
             type="primary", disabled=not featured.get("tmdb_id"))
    b.button("＋ Save", key="hero_save", on_click=toggle_save, args=(featured,))
    pick = c.button("🎲 Surprise me", key="surprise")
    if pick:
        choice = random.choice([m for m in movies if m.get("tmdb_id")] or movies)
        open_movie(choice)
        st.rerun()

    recent = st.session_state["recent"]
    if recent:
        st.caption("Recently viewed")
        cols = st.columns(min(len(recent), 6))
        for i, r in enumerate(recent):
            cols[i].button(str(r["title"])[:22], key=f"recent_{i}", on_click=open_by_id,
                           args=(r["tmdb_id"], r["title"]), use_container_width=True)

    section(category, f"{len(movies)} titles loaded")
    filter_bar()
    render_grid(apply_filters(movies), f"home_{CATEGORIES[category]}")

    if len(movies) >= st.session_state["limit"] and st.session_state["limit"] < PAGE_MAX:
        _, mid_col, _ = st.columns([2, 1, 2])
        mid_col.button("Load more", on_click=more, use_container_width=True)


def page_watchlist():
    wl = st.session_state["watchlist"]
    watched = [m for m in wl if m.get("watched")]
    rated = [rnum(m.get("vote_average")) for m in wl if rnum(m.get("vote_average"))]
    todo_minutes = sum(to_int(m.get("runtime")) for m in wl if not m.get("watched"))
    section("Your watchlist", "Track what you plan to watch and what you've finished.")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Saved", len(wl))
    m2.metric("Watched", f"{len(watched)}" + (f" ({len(watched) / len(wl):.0%})" if wl else ""))
    m3.metric("Avg rating", f"{sum(rated) / len(rated):.1f}" if rated else "–")
    m4.metric("Time left to watch", fmt_runtime(todo_minutes) or "–")

    if not wl:
        st.info("Your watchlist is empty. Choose Save on any movie in Discover to start building it.")
    else:
        st.radio("Show", ["All", "To watch", "Watched"], horizontal=True,
                 label_visibility="collapsed", key="wl_filter")
        view = {"All": wl, "To watch": [m for m in wl if not m.get("watched")], "Watched": watched}[
            st.session_state["wl_filter"]]
        for i, m in enumerate(view):
            mid = m.get("tmdb_id")
            c1, c2, c3, c4, c5 = st.columns([1, 5, 2, 2, 2], gap="small", vertical_alignment="center")
            c1.markdown(compact(poster_html(m.get("poster_url"), m.get("title"), "thumb")), unsafe_allow_html=True)
            c2.markdown(compact(f"""<div class="row-title">{esc(m.get("title"))}</div>
                <div class="row-meta">{esc(meta_line(m))} · ★ {safe_rating(m.get("vote_average"))} · added {esc(m.get("added"), "–")}</div>"""),
                        unsafe_allow_html=True)
            c3.button("Details", key=f"wl_d_{i}_{mid}", on_click=open_movie, args=(m,),
                      use_container_width=True, disabled=not mid)
            c4.button("✓ Watched" if m.get("watched") else "Mark watched", key=f"wl_w_{i}_{mid}",
                      on_click=toggle_watched, args=(mid,), use_container_width=True,
                      type="secondary" if m.get("watched") else "primary")
            c5.button("Remove", key=f"wl_r_{i}_{mid}", on_click=toggle_save, args=(m,),
                      use_container_width=True)

    with st.expander("Export, import and reset"):
        x1, x2 = st.columns(2)
        if wl:
            x1.download_button("Download JSON", json.dumps(wl, indent=2), "watchlist.json",
                               "application/json", use_container_width=True)
            x2.download_button("Download CSV", pd.DataFrame(wl).to_csv(index=False), "watchlist.csv",
                               "text/csv", use_container_width=True)
        up = st.file_uploader("Import a watchlist (JSON)", type="json")
        if up is not None and st.button("Merge imported movies"):
            try:
                data = json.load(up)
                have = saved_ids()
                added = [m for m in data if isinstance(m, dict) and m.get("tmdb_id") not in have]
                st.session_state["watchlist"] = wl + added
                st.toast(f"Imported {len(added)} movies")
                st.rerun()
            except Exception:
                st.error("That file isn't a valid watchlist export.")
        if wl:
            st.button("Clear entire watchlist", on_click=clear_watchlist)


def page_insights():
    section("Insights", "Rating and release patterns across live TMDB collections.")
    stats, pool = {}, []
    try:
        with st.spinner("Crunching the numbers…"):
            for label, key in CATEGORIES.items():
                items = fetch_movies(key, 20, enrich_data=False)
                rs = [rnum(m["vote_average"]) for m in items if rnum(m["vote_average"])]
                if rs:
                    stats[label] = round(sum(rs) / len(rs), 2)
                if label == st.session_state["category"]:
                    pool = enrich(items, need_runtime=True)
        st.session_state["online"] = True
    except Exception as exc:
        st.session_state["online"] = not isinstance(exc, requests.ConnectionError)
        st.error("Insights need the backend. " + error_message(exc))
        return

    ratings = [rnum(m["vote_average"]) for m in pool if rnum(m["vote_average"])]
    runtimes = [m["runtime"] for m in pool if m.get("runtime")]
    top = max(pool, key=lambda m: rnum(m["vote_average"]) or 0, default=None)
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Titles analysed", len(pool))
    k2.metric("Average rating", f"{sum(ratings) / len(ratings):.2f}" if ratings else "–")
    k3.metric("Highest rated", f"★ {safe_rating(top['vote_average'])}" if top else "–")
    k4.metric("Avg duration", fmt_runtime(sum(runtimes) / len(runtimes)) if runtimes else "–")
    k5.metric("Saved by you", len(st.session_state["watchlist"]))
    if top:
        st.caption(f"Top title in {st.session_state['category']}: {top['title']}")

    accent = (LIGHT if st.session_state["light"] else DARK)["accent"]
    data = (LIGHT if st.session_state["light"] else DARK)["data"]
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Average rating by collection**")
        if stats:
            st.bar_chart(pd.Series(stats, name="Avg rating"), color=accent)
    with c2:
        st.markdown(f"**Rating distribution · {st.session_state['category']}**")
        if ratings:
            dist = ((pd.Series(ratings) * 2).round() / 2).value_counts().sort_index()
            dist.index = dist.index.astype(str)
            st.bar_chart(dist, color=data)

    years = pd.Series([year_num(m) for m in pool if year_num(m)])
    wl = st.session_state["watchlist"]
    c3, c4 = st.columns(2)
    with c3:
        st.markdown(f"**Release years · {st.session_state['category']}**")
        if not years.empty:
            ydist = years.value_counts().sort_index()
            ydist.index = ydist.index.astype(str)
            st.bar_chart(ydist, color=accent)
    with c4:
        st.markdown("**Your watchlist by decade**")
        decades = pd.Series([f"{year_num(m) // 10 * 10}s" for m in wl if year_num(m)])
        if not decades.empty:
            st.bar_chart(decades.value_counts().sort_index(), color=data)
        else:
            st.caption("Save a few movies to see your taste profile.")

    if pool:
        st.markdown("**Top 10 in this collection**")
        df = pd.DataFrame(pool)[["title", "release_date", "runtime", "vote_average"]].copy()
        df["runtime"] = df["runtime"].apply(fmt_runtime)
        df.columns = ["Title", "Release", "Duration", "Rating"]
        st.dataframe(df.sort_values("Rating", ascending=False).head(10), hide_index=True,
                     use_container_width=True)

    st.button("Refresh data", on_click=api_get.clear)


# =========================================================
# ROUTER
# =========================================================

scroll_to_top_if_requested()

selected = st.session_state["selected_movie_id"]
if selected:
    page_details(int(selected))
elif st.session_state["search_query"]:
    page_search()
elif st.session_state["view"] == "Watchlist":
    page_watchlist()
elif st.session_state["view"] == "Insights":
    page_insights()
else:
    page_discover()

try:
    persist_user_data()
except Exception as _save_exc:
    st.warning("Your latest changes could not be synced to your account. They remain in this session. " + error_message(_save_exc))

# =========================================================
# FOOTER
# =========================================================

st.markdown("---")
md("""<div style="text-align:center;padding:.6rem 0">
<div class="brand" style="font-size:1.2rem">CINÉ<b>MA</b></div>
<p class="mono" style="color:var(--muted);font-size:.72rem;margin-top:.5rem">Movie data provided by TMDB. Streaming availability via JustWatch. This product is not endorsed or certified by TMDB.</p>
</div>""")