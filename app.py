import streamlit as st
import pandas as pd
from pathlib import Path
from typing import Optional, List

# ================================
# Config
# ================================
APP_NAME = "🌐 SchemeSarthi"
DATA_FILE = Path(__file__).with_name("gov.csv")
DEFAULT_LIMIT = 5
MAX_LIMIT = 50

# Columns we try to use if present
TEXT_COL_CANDIDATES: List[str] = [
    "title", "name", "schemeName", "description", "eligibility",
    "schemeCategory", "category", "level", "state", "department"
]

# ================================
# Data loading & normalization
# ================================
@st.cache_data
def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

    rename_map = {}
    for c in df.columns:
        lc = c.lower()
        if lc == "scheme name" or lc == "schemename":
            rename_map[c] = "schemeName"
        if lc == "cat" or lc == "categories":
            rename_map[c] = "schemeCategory"
    if rename_map:
        df = df.rename(columns=rename_map)

    # Ensure the text columns exist
    for col in TEXT_COL_CANDIDATES:
        if col not in df.columns:
            df[col] = ""

    # Searchable blob
    text_cols_present = [c for c in TEXT_COL_CANDIDATES if c in df.columns]
    df["__blob"] = (
        df[text_cols_present].fillna("").astype(str).agg(" ".join, axis=1).str.lower()
    )
    return df


@st.cache_data
def _load_dataframe(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"CSV not found at {path.resolve()}")
    try:
        df = pd.read_csv(path, engine="python", on_bad_lines="skip")
    except Exception:
        df = pd.read_csv(path, engine="python", on_bad_lines="skip", encoding_errors="ignore")
    return _normalize_columns(df)


# Load once
try:
    df = _load_dataframe(DATA_FILE)
except Exception as e:
    df = pd.DataFrame(
        {"title": ["Data failed to load"], "description": [str(e)], "eligibility": [""],
         "schemeCategory": [""], "level": [""], "state": [""], "department": [""], "__blob": [""]}
    )

# ================================
# Profile model (for filters)
# ================================
class Profile:
    def __init__(self, age: Optional[int] = None, income: Optional[float] = None,
                 occupation: Optional[str] = None, education: Optional[str] = None,
                 location: Optional[str] = None, category: Optional[str] = None, state: Optional[str] = None):
        self.age = age
        self.income = income
        self.occupation = occupation
        self.education = education
        self.location = location
        self.category = category
        self.state = state


# ================================
# Recommender Logic
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
# Streamlit UI
# ================================
st.set_page_config(page_title=APP_NAME, page_icon="🚀", layout="wide")
st.title(APP_NAME)
st.markdown("### Discover schemes tailored for you 🎯")

# Sidebar filters
st.sidebar.header("🔍 Filters")

age = st.sidebar.number_input("Enter your age", min_value=0, max_value=120, value=25)
income = st.sidebar.number_input("Enter your income", min_value=0, step=1000)
occupation = st.sidebar.text_input("Occupation")
education = st.sidebar.text_input("Education")
category = st.sidebar.text_input("Category")

# State selection (dropdown)
states = sorted(df["state"].dropna().unique().tolist())
states.insert(0, "Select your state")
user_state = st.sidebar.selectbox("Select your state", states)

# Keyword search
search_query = st.sidebar.text_input("Search schemes (keyword)", "")

# Limit slider
limit = st.sidebar.slider("Number of results", 1, MAX_LIMIT, DEFAULT_LIMIT)

# Build profile
profile = Profile(age=age, income=income, occupation=occupation,
                  education=education, category=category, state=None)

# Filter by state (state + Central)
filtered_df = df.copy()
if user_state != "Select your state":
    filtered_df = filtered_df[
        (filtered_df["state"].str.contains(user_state, case=False, na=False)) |
        (filtered_df["level"].str.contains("central", case=False, na=False))
    ]

# Apply keyword search
if search_query.strip():
    filtered_df = filtered_df[filtered_df["__blob"].str.contains(search_query.lower(), na=False)]

# Get recommendations
recs = recommend(profile, filtered_df, limit=limit)

# ================================
# Display results
# ================================
if recs.empty:
    st.warning("⚠️ No schemes found for your selection. Try changing filters.")
else:
    for _, row in recs.iterrows():
        with st.container():
            st.markdown(
                f"""
                <div style="padding:15px; margin-bottom:12px; border-radius:12px; 
                background-color:#f9f9f9; box-shadow:0 2px 6px rgba(0,0,0,0.1);">
                    <h3 style="margin-bottom:8px; color:#2c3e50;">{row.get('schemeName', 'Unnamed Scheme')}</h3>
                    <p><b>📝 Description:</b> {row.get('description', 'N/A')}</p>
                    <p><b>✅ Eligibility:</b> {row.get('eligibility', 'N/A')}</p>
                    <p><b>🏷️ Category:</b> {row.get('schemeCategory', '')} | <b>🌍 Level:</b> {row.get('level', '')}</p>
                    <p><b>📍 State:</b> {row.get('state', '')} | <b>🏢 Department:</b> {row.get('department', '')}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
