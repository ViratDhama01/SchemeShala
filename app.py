# app.py
from pathlib import Path
from typing import Optional, List
import pandas as pd
import streamlit as st
import os
from datetime import datetime

# ================================
# Config
# ================================
APP_NAME = "🌐 SchemeSarthi"
DATA_FILE = Path(__file__).with_name("gov.csv")
USERS_DB = Path(__file__).with_name("users_db.csv")
ADMIN_PASSWORD = "change_this_password"  # <-- change this before pushing to production
DEFAULT_LIMIT = 5
MAX_LIMIT = 50

# Columns we try to use if present
TEXT_COL_CANDIDATES: List[str] = [
    "title", "name", "schemeName", "description", "eligibility",
    "schemeCategory", "category", "level", "state", "department", "tags", "benefits"
]

# Full list of Indian states + UTs
ALL_STATES = [
    "Andhra Pradesh","Arunachal Pradesh","Assam","Bihar","Chhattisgarh","Goa",
    "Gujarat","Haryana","Himachal Pradesh","Jharkhand","Karnataka","Kerala",
    "Madhya Pradesh","Maharashtra","Manipur","Meghalaya","Mizoram","Nagaland",
    "Odisha","Punjab","Rajasthan","Sikkim","Tamil Nadu","Telangana","Tripura",
    "Uttar Pradesh","Uttarakhand","West Bengal",
    # Union Territories
    "Andaman and Nicobar Islands","Chandigarh","Dadra and Nagar Haveli and Daman and Diu",
    "Delhi","Jammu and Kashmir","Ladakh","Lakshadweep","Puducherry"
]

# Occupations (expandable)
OCCUPATIONS = [
    "Any","Farmer","Student","Government Employee","Private Employee","Self-Employed",
    "Professional (Doctor)","Professional (Engineer)","Teacher","Trader/Shopkeeper",
    "Artisan","Goldsmith","Driver","Construction Worker","Daily Wage Labourer",
    "Housewife / Homemaker","Entrepreneur","Unemployed","Retired"
]

# Education options
EDUCATION_LEVELS = ["Any", "Below 8th", "8th", "10th", "12th", "Diploma", "Graduation",
                    "Post Graduation", "PhD", "Other"]

# Category options
CATEGORIES = ["Any", "General", "OBC", "SC", "ST", "EWS"]

# ================================
# Helpers: load & normalize
# ================================
@st.cache_data
def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # strip and normalize column names
    df.columns = [c.strip() for c in df.columns]

    # rename common variants
    rename_map = {}
    for c in df.columns:
        lc = c.lower()
        if lc in ("scheme name", "schemename"):
            rename_map[c] = "schemeName"
        if lc in ("cat", "categories"):
            rename_map[c] = "schemeCategory"
        if lc == "benefit" or lc == "benefits":
            rename_map[c] = "benefits"
        if lc == "tags" and c != "tags":
            rename_map[c] = "tags"
    if rename_map:
        df = df.rename(columns=rename_map)

    # ensure text columns exist
    for col in TEXT_COL_CANDIDATES:
        if col not in df.columns:
            df[col] = ""

    # create blob for simple text matching
    text_cols_present = [c for c in TEXT_COL_CANDIDATES if c in df.columns]
    df["__blob"] = df[text_cols_present].fillna("").astype(str).agg(" ".join, axis=1).str.lower()

    # Ensure state and level columns exist
    if "state" not in df.columns:
        df["state"] = ""
    if "level" not in df.columns:
        df["level"] = ""

    # Standardize schemeName visibility: prefer schemeName, then title, then name
    def _preferred_name(row):
        for col in ("schemeName", "title", "name"):
            val = row.get(col, "")
            if isinstance(val, str) and val.strip():
                return val.strip()
        return "Unnamed Scheme"
    df["__display_name"] = df.apply(_preferred_name, axis=1)

    return df

@st.cache_data
def _load_dataframe(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()  # handled later
    try:
        df = pd.read_csv(path, engine="python", on_bad_lines="skip")
    except Exception:
        df = pd.read_csv(path, engine="python", on_bad_lines="skip", encoding_errors="ignore")
    df = df.fillna("")
    return _normalize_columns(df)

df = _load_dataframe(DATA_FILE)

# ================================
# Simple scoring recommender (keeps your original logic but with occupation & category boosts)
# ================================
def _keywords_from_profile(profile: dict) -> List[str]:
    kws = []
    for k in ("occupation", "education", "location", "category"):
        v = profile.get(k)
        if v and isinstance(v, str):
            s = v.strip().lower()
            if s and s != "any":
                kws.append(s)
    return kws

def _apply_numeric_filters(profile: dict, data: pd.DataFrame) -> pd.DataFrame:
    out = data
    age = profile.get("age")
    income = profile.get("income")
    if age is not None:
        if "minAge" in out.columns:
            out = out[out["minAge"].fillna(-1).astype(float) <= float(age)]
        if "maxAge" in out.columns:
            out = out[out["maxAge"].fillna(10**9).astype(float) >= float(age)]
    if income is not None and "incomeLimit" in out.columns:
        out = out[out["incomeLimit"].fillna(10**12).astype(float) >= float(income)]
    return out

def _score_rows(profile: dict, data: pd.DataFrame) -> pd.DataFrame:
    kws = _keywords_from_profile(profile)
    out = data.copy()
    blob = out["__blob"].fillna("")

    # base score
    score = pd.Series(0, index=out.index)

    # keyword matches: +2
    for kw in kws:
        score += blob.str.contains(kw, na=False).astype(int) * 2

    # occupation priority: +5 if occupation appears in blob or eligibility
    occ = profile.get("occupation", "")
    if occ and occ.lower() != "any":
        occ_kw = occ.strip().lower()
        score += blob.str.contains(occ_kw, na=False).astype(int) * 5
        for col in ("eligibility", "schemeCategory", "description"):
            if col in out.columns:
                score += out[col].astype(str).str.lower().str.contains(occ_kw, na=False).astype(int) * 3

    # category boost (e.g., OBC/SC/ST) +3 if category is mentioned in eligibility or blob
    cat = profile.get("category", "")
    if cat and cat.lower() != "any":
        cat_kw = cat.strip().lower()
        score += blob.str.contains(cat_kw, na=False).astype(int) * 3
        if "eligibility" in out.columns:
            score += out["eligibility"].astype(str).str.lower().str.contains(cat_kw, na=False).astype(int) * 2

    # education matching boost
    edu = profile.get("education", "")
    if edu and edu.lower() != "any":
        edu_kw = edu.strip().lower()
        score += blob.str.contains(edu_kw, na=False).astype(int) * 2
        if "eligibility" in out.columns:
            score += out["eligibility"].astype(str).str.lower().str.contains(edu_kw, na=False).astype(int)

    out["__score"] = score
    return out

def recommend(profile: dict, data: pd.DataFrame, limit: int = DEFAULT_LIMIT) -> pd.DataFrame:
    filtered = _apply_numeric_filters(profile, data)
    scored = _score_rows(profile, filtered)
    if scored["__score"].max() == 0:
        return scored.head(limit)
    recs = scored.sort_values(["__score", "__display_name"], ascending=[False, True]).head(limit)
    return recs

# ================================
# UI & Interaction
# ================================
st.set_page_config(page_title=APP_NAME, page_icon="🚀", layout="wide")
st.markdown(
    """
    <style>
    /* Card style that adapts to dark/light mode */
    .scheme-card {
      padding: 16px;
      margin-bottom: 12px;
      border-radius: 12px;
      box-shadow: 0 2px 6px rgba(0,0,0,0.08);
      transition: transform 0.08s ease;
    }
    .scheme-card:hover { transform: translateY(-4px); }
    /* Light mode */
    :root { --card-bg: #ffffff; --muted: #555555; --accent: #0b6ef6; }
    /* Dark mode via prefers-color-scheme */
    @media (prefers-color-scheme: dark) {
      :root { --card-bg: #0f1720; --muted: #aab3bf; --accent: #66a3ff; }
      .scheme-card { box-shadow: 0 2px 10px rgba(0,0,0,0.6); }
    }
    .scheme-card-inner { background: var(--card-bg); padding:12px; border-radius:10px; }
    .scheme-name { color: var(--accent); margin-bottom:6px; }
    .muted { color: var(--muted); font-size: 14px; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title(APP_NAME)
st.markdown("### Discover government schemes tailored to your profile")

# Sidebar: User profile + filters + save details
st.sidebar.header("👤 Your Profile & Filters")

# Personal details section (saved to users_db)
st.sidebar.subheader("Personal details (optional save)")
user_name = st.sidebar.text_input("Full name")
user_age = st.sidebar.number_input("Age", min_value=0, max_value=120, value=25, step=1)
user_contact = st.sidebar.text_input("Contact number (optional)")
user_email = st.sidebar.text_input("Email (optional)")

if st.sidebar.button("Save my details"):
    # append to users_db.csv
    USERS_DB.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "timestamp": datetime.utcnow().isoformat(),
        "name": user_name,
        "age": user_age,
        "contact": user_contact,
        "email": user_email
    }
    cols = ["timestamp", "name", "age", "contact", "email"]
    if not USERS_DB.exists():
        pd.DataFrame([row])[cols].to_csv(USERS_DB, index=False)
        st.sidebar.success("Saved (created new users_db.csv).")
    else:
        pd.DataFrame([row])[cols].to_csv(USERS_DB, mode="a", header=False, index=False)
        st.sidebar.success("Saved to users_db.csv.")

st.sidebar.markdown("---")
st.sidebar.subheader("Profile fields for recommendation")

age = st.sidebar.number_input("Your age (for eligibility)", min_value=0, max_value=120, value=user_age, step=1)
income = st.sidebar.number_input("Your annual income (approx)", min_value=0, step=1000, value=0)
occupation = st.sidebar.selectbox("Occupation", OCCUPATIONS, index=0)
education = st.sidebar.selectbox("Highest education", EDUCATION_LEVELS, index=0)
category = st.sidebar.selectbox("Social category", CATEGORIES, index=0)

st.sidebar.markdown("---")
# State select: include all states (and fallback)
states_for_select = ["Any"] + sorted([s for s in ALL_STATES if s])
# Also add unique states from df for completeness
if not df.empty:
    df_states = sorted([s for s in df["state"].dropna().unique().tolist() if s and s not in states_for_select])
    if df_states:
        states_for_select = ["Any"] + sorted(set(states_for_select + df_states))
user_state = st.sidebar.selectbox("Select your state", states_for_select, index=0)

st.sidebar.markdown("---")
limit = st.sidebar.slider("Number of recommendations", min_value=1, max_value=MAX_LIMIT, value=DEFAULT_LIMIT)
search_query = st.sidebar.text_input("Search (keyword)", "")

st.sidebar.markdown("---")
# Admin area to view users_db
st.sidebar.subheader("🔒 Admin (view saved users)")
admin_pw = st.sidebar.text_input("Admin password", type="password")
if admin_pw and admin_pw == ADMIN_PASSWORD:
    if USERS_DB.exists():
        st.sidebar.success("Admin authenticated — showing users DB")
        try:
            users_df = pd.read_csv(USERS_DB)
            st.sidebar.dataframe(users_df.tail(50))
        except Exception as e:
            st.sidebar.error(f"Could not load users DB: {e}")
    else:
        st.sidebar.info("No users_db.csv found yet.")
elif admin_pw:
    st.sidebar.error("Wrong admin password")

# ================================
# Build profile dict & filter dataset
# ================================
profile = {
    "age": age,
    "income": income,
    "occupation": occupation,
    "education": education,
    "category": category,
    "location": user_state if user_state and user_state != "Any" else None
}

# start with full df
results = df.copy()
if results.empty:
    st.warning("No gov.csv found or it's empty. Please upload gov.csv in the app folder.")
else:
    # State filter: show selected state schemes + central schemes
    if profile["location"]:
        results = results[
            results["state"].str.contains(profile["location"], case=False, na=False) |
            results["level"].str.contains("central", case=False, na=False)
        ]

    # Category filter quick pruning if user selected specific category
    if profile["category"] and profile["category"].lower() != "any":
        cat_kw = profile["category"].lower()
        # Keep rows that mention category in eligibility or have blank eligibility (so we don't overfilter)
        results = results[
            results["eligibility"].astype(str).str.lower().str.contains(cat_kw, na=False) |
            results["__blob"].str.contains(cat_kw, na=False)
        ]

    # keyword search
    if search_query and search_query.strip():
        results = results[results["__blob"].str.contains(search_query.strip().lower(), na=False)]

    # apply numeric & scoring + recommend
    recs = recommend(profile, results, limit=limit)

    # Display header with counts
    st.markdown(f"#### Showing top {len(recs)} results (from {len(results)} matched schemes)")

    if recs.empty:
        st.warning("No schemes match your criteria. Try broadening the filters or change keywords.")
    else:
        # Display recommended schemes as cards
        for _, row in recs.iterrows():
            display_name = row.get("__display_name") or row.get("schemeName") or row.get("title") or row.get("name") or "Unnamed Scheme"
            desc = row.get("description", "")
            eligibility = row.get("eligibility", "")
            benefits = row.get("benefits", "")
            cat = row.get("schemeCategory", "")
            level = row.get("level", "")
            state = row.get("state", "")
            department = row.get("department", "")
            score = row.get("__score", 0)

            # Card HTML (adapts to dark/light via CSS above)
            st.markdown(
                f"""
                <div class="scheme-card">
                  <div class="scheme-card-inner">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                      <div style="flex:1">
                        <div class="scheme-name"><strong>{display_name}</strong></div>
                        <div class="muted">🏷️ {cat or 'N/A'} • 🌍 {level or 'N/A'} • 📍 {state or 'N/A'}</div>
                      </div>
                      <div style="text-align:right; margin-left:12px;">
                        <div class="muted">Score: {int(score)}</div>
                      </div>
                    </div>
                    <hr/>
                    <div style="margin-top:6px;">
                      <div><b>📝 Description:</b> {desc if desc else 'N/A'}</div>
                      <div style="margin-top:4px;"><b>✅ Eligibility:</b> {eligibility if eligibility else 'N/A'}</div>
                      <div style="margin-top:4px;"><b>💡 Benefits:</b> {benefits if benefits else 'N/A'}</div>
                      <div style="margin-top:8px; font-size:13px;" class="muted">Department: {department or 'N/A'}</div>
                    </div>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            # ... keep all your code above unchanged ...

    if recs.empty:
        st.warning("No schemes match your criteria. Try broadening the filters or change keywords.")
    else:
        # Display recommended schemes as cards
        for _, row in recs.iterrows():
            # --- Fix: ensure scheme name always visible ---
            display_name = (
                row.get("schemeName")
                or row.get("title")
                or row.get("name")
                or row.get("__display_name")
                or "Unknown Scheme"
            )

            # --- Fix: ensure description always visible ---
            desc = (
                row.get("description")
                or row.get("about")
                or row.get("details")
                or "No description available."
            )

            eligibility = row.get("eligibility", "")
            benefits = row.get("benefits", "")
            cat = row.get("schemeCategory", "")
            level = row.get("level", "")
            state = row.get("state", "")
            department = row.get("department", "")
            score = row.get("__score", 0)

            # Card HTML (adapts to dark/light via CSS above)
            st.markdown(
                f"""
                <div class="scheme-card">
                  <div class="scheme-card-inner">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                      <div style="flex:1">
                        <div class="scheme-name"><strong>{display_name}</strong></div>
                        <div class="muted">🏷️ {cat or 'N/A'} • 🌍 {level or 'N/A'} • 📍 {state or 'N/A'}</div>
                      </div>
                      <div style="text-align:right; margin-left:12px;">
                        <div class="muted">Score: {int(score)}</div>
                      </div>
                    </div>
                    <hr/>
                    <div style="margin-top:6px;">
                      <div><b>📝 Description:</b> {desc}</div>
                      <div style="margin-top:4px;"><b>✅ Eligibility:</b> {eligibility if eligibility else 'N/A'}</div>
                      <div style="margin-top:4px;"><b>💡 Benefits:</b> {benefits if benefits else 'N/A'}</div>
                      <div style="margin-top:8px; font-size:13px;" class="muted">Department: {department or 'N/A'}</div>
                    </div>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

# Footer / helpful notes
st.markdown("---")
st.caption("Tip: Fill your social category, occupation and state for more accurate results. "
           "Saved user details are stored in `users_db.csv` (owner-only view).")

# Footer / helpful notes
st.markdown("---")
st.caption("Tip: Fill your social category, occupation and state for more accurate results. "
           "Saved user details are stored in `users_db.csv` (owner-only view).")
