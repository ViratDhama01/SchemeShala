from __future__ import annotations

from pathlib import Path
from typing import Optional, List

import pandas as pd
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# ================================
# Config
# ================================
APP_NAME = "SchemeSarthi API 🚀"
DATA_FILE = Path(__file__).with_name("gov.csv")  # same folder as this file
DEFAULT_LIMIT = 5
MAX_LIMIT = 50

# Columns we try to use if present
TEXT_COL_CANDIDATES: List[str] = [
    "title",
    "name",
    "schemeName",
    "description",
    "eligibility",
    "schemeCategory",
    "category",
    "level",
    "state",
    "department",
]

NUM_COL_CANDIDATES = {
    "minAge": ("age", ">="),
    "maxAge": ("age", "<="),
    "incomeLimit": ("income", ">="),  # if dataset has a max income threshold
}

# ================================
# Data loading & normalization
# ================================

def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    # strip spaces and unify casing
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

    # Try to map common variants to our canonical names
    rename_map = {}
    for c in df.columns:
        lc = c.lower()
        if lc == "eligibility" and c != "eligibility":
            rename_map[c] = "eligibility"
        if lc == "scheme name" or lc == "schemename":
            rename_map[c] = "schemeName"
        if lc == "cat" or lc == "categories":
            rename_map[c] = "schemeCategory"
    if rename_map:
        df = df.rename(columns=rename_map)

    # Ensure the text columns exist (create empty if missing)
    for col in TEXT_COL_CANDIDATES:
        if col not in df.columns:
            df[col] = ""

    # Build a searchable blob once for quick filtering/scoring
    text_cols_present = [c for c in TEXT_COL_CANDIDATES if c in df.columns]
    df["__blob"] = (
        df[text_cols_present]
        .fillna("")
        .astype(str)
        .agg(" ".join, axis=1)
        .str.lower()
    )
    return df


def _load_dataframe(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"CSV not found at {path.resolve()} — place 'gov.csv' next to app.py"
        )
    try:
        df = pd.read_csv(path, engine="python", on_bad_lines="skip")
    except Exception:
        # Last-resort fallback: ignore bad bytes
        df = pd.read_csv(
            path, engine="python", on_bad_lines="skip", encoding_errors="ignore"
        )
    return _normalize_columns(df)


# Load once when the app starts
try:
    df = _load_dataframe(DATA_FILE)
except Exception as e:
    # Create a tiny placeholder DF so the app still starts and returns a helpful message
    df = pd.DataFrame(
        {
            "title": ["Data failed to load"],
            "description": [str(e)],
            "eligibility": [""],
            "schemeCategory": [""],
            "level": [""],
            "__blob": [""],
        }
    )


# ================================
# API model
# ================================
class Profile(BaseModel):
    age: Optional[int] = None
    income: Optional[float] = None
    occupation: Optional[str] = None
    education: Optional[str] = None
    location: Optional[str] = None
    category: Optional[str] = None


# ================================
# App init
# ================================
app = FastAPI(title=APP_NAME)

# CORS (allow everything in dev; tighten for prod)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ================================
# Utility: simple scoring recommender
# ================================

def _keywords_from_profile(p: Profile) -> List[str]:
    kws: List[str] = []
    for s in [p.occupation, p.education, p.location, p.category]:
        if s:
            s = str(s).strip().lower()
            if s:
                kws.append(s)
    return kws


def _apply_numeric_filters(p: Profile, data: pd.DataFrame) -> pd.DataFrame:
    # If dataset has numeric constraints, apply them defensively
    out = data
    if p.age is not None:
        if "minAge" in out.columns:
            out = out[(out["minAge"].fillna(-1).astype(float) <= float(p.age))]
        if "maxAge" in out.columns:
            out = out[(out["maxAge"].fillna(10**9).astype(float) >= float(p.age))]
    if p.income is not None and "incomeLimit" in out.columns:
        # Keep rows whose incomeLimit is >= user's income (i.e., user qualifies)
        out = out[(out["incomeLimit"].fillna(10**12).astype(float) >= float(p.income))]
    return out


def _score_rows(p: Profile, data: pd.DataFrame) -> pd.DataFrame:
    kws = _keywords_from_profile(p)
    if not kws:
        out = data.copy()
        out["__score"] = 0
        return out

    # Basic scoring: +2 per keyword present in blob; +1 if present in specific columns
    out = data.copy()
    blob = out["__blob"].fillna("")

    score = pd.Series(0, index=out.index)
    for kw in kws:
        contains_kw = blob.str.contains(kw, na=False)
        score = score + (contains_kw.astype(int) * 2)
        # Light extra weight for category/level/eligibility column hits
        for col in ("schemeCategory", "level", "eligibility"):
            if col in out.columns:
                score = score + out[col].fillna("").astype(str).str.lower().str.contains(kw, na=False).astype(int)

    out["__score"] = score
    return out


def recommend(profile: Profile, data: pd.DataFrame, limit: int = DEFAULT_LIMIT) -> pd.DataFrame:
    # 1) numeric filters if available
    filtered = _apply_numeric_filters(profile, data)

    # 2) text scoring
    scored = _score_rows(profile, filtered)

    # 3) If all scores are zero (or profile empty), fall back to top rows
    if scored["__score"].max() == 0:
        return scored.head(limit)

    # 4) Sort by score desc, then by title for stability
    recs = scored.sort_values(["__score", "title"], ascending=[False, True]).head(limit)
    return recs


# ================================
# Routes
# ================================
@app.get("/")
def root():
    return {"message": f"Welcome to {APP_NAME}", "rows": len(df)}


@app.get("/health")
def health():
    ok = bool(len(df)) and "__blob" in df.columns
    return {"status": "ok" if ok else "degraded"}


@app.get("/columns")
def columns():
    return {"columns": list(df.columns)}


@app.post("/recommend")
def post_recommend(profile: Profile, limit: int = Query(DEFAULT_LIMIT, ge=1, le=MAX_LIMIT)):
    """Return up to `limit` recommended schemes for a given profile."""
    recs = recommend(profile, df, limit=limit)

    # Choose a clean subset of columns for output if present
    preferred_cols = [
        "title",
        "schemeName",
        "schemeCategory",
        "level",
        "eligibility",
        "description",
        "state",
        "department",
        "__score",
    ]
    present = [c for c in preferred_cols if c in recs.columns]
    if not present:
        present = list(recs.columns)

    return recs[present].to_dict(orient="records")


# ================================
# Dev server
# ================================
if __name__ == "__main__":
    # Example: uvicorn app:app --reload --port 8000
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
