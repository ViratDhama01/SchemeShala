from pathlib import Path
from typing import Optional, List
import pandas as pd
import streamlit as st

# ================================
# Config
# ================================
APP_NAME = "SchemeSarthi 🚀"
DATA_FILE = Path(__file__).with_name("gov.csv")  # must be in the same folder
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
    "incomeLimit": ("income", ">="),  # max income threshold
}

# ================================
# Data loading & normalization
# ================================
def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

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

    for col in TEXT_COL_CANDIDATES:
        if col not in df.columns:
            df[col] = ""

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
        df = pd.read_csv(
            path, engine="python", on_bad_lines="skip", encoding_errors="ignore"
        )
    return _normalize_columns(df)


# Load data
try:
    df = _load_dataframe(DATA_FILE)
except Exception as e:
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
# Profile class (simple, no Pydantic)
# ================================
class Profile:
    def __init__(
        self,
        age: Optional[int] = None,
        income: Optional[float] = None,
        occupation: Optional[str] = None,
        education: Optional[str] = None,
        location: Optional[str] = None,
        category: Optional[str] = None,
    ):
        self.age = age
        self.income = income
        self.occupation = occupation
        self.education = education
        self.location = location
        self.category = category


# ================================
# Recommender system
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
    out = data
    if p.age is not None:
        if "minAge" in out.columns:
            out = out[(out["minAge"].fillna(-1).astype(float) <= float(p.age))]
        if "maxAge" in out.columns:
            out = out[(out["maxAge"].fillna(10**9).astype(float) >= float(p.age))]
    if p.income is not None and "incomeLimit" in out.columns:
        out = out[(out["incomeLimit"].fillna(10**12).astype(float) >= float(p.income))]
    return out


def _score_rows(p: Profile, data: pd.DataFrame) -> pd.DataFrame:
    kws = _keywords_from_profile(p)
    if not kws:
        out = data.copy()
        out["__score"] = 0
        return out

    out = data.copy()
    blob = out["__blob"].fillna("")
    score = pd.Series(0, index=out.index)
    for kw in kws:
        contains_kw = blob.str.contains(kw, na=False)
        score = score + (contains_kw.astype(int) * 2)
        for col in ("schemeCategory", "level", "eligibility"):
            if col in out.columns:
                score = score + out[col].fillna("").astype(str).str.lower().str.contains(kw, na=False).astype(int)
    out["__score"] = score
    return out


def recommend(profile: Profile, data: pd.DataFrame, limit: int = DEFAULT_LIMIT) -> pd.DataFrame:
    filtered = _apply_numeric_filters(profile, data)
    scored = _score_rows(profile, filtered)
    if scored["__score"].max() == 0:
        return scored.head(limit)
    recs = scored.sort_values(["__score", "title"], ascending=[False, True]).head(limit)
    return recs


# ================================
# Streamlit App
# ================================
st.title(APP_NAME)

st.sidebar.header("Your Profile")
age = st.sidebar.number_input("Age", min_value=0, step=1, value=21)
income = st.sidebar.number_input("Income", min_value=0, step=1000, value=200000)
occupation = st.sidebar.text_input("Occupation", "Student")
education = st.sidebar.text_input("Education", "")
location = st.sidebar.text_input("Location", "Rural")
category = st.sidebar.text_input("Category", "Education")
limit = st.sidebar.slider("Number of recommendations", 1, MAX_LIMIT, DEFAULT_LIMIT)

profile = Profile(age=age, income=income, occupation=occupation, education=education, location=location, category=category)

st.subheader("Recommended Schemes")
recs = recommend(profile, df, limit=limit)

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

st.dataframe(recs[present])
