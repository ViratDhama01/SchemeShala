import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ================================
# Load Data
# ================================
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("gov.csv", engine="python", on_bad_lines="skip")
    except Exception:
        df = pd.read_csv("gov.csv", engine="python", on_bad_lines="skip", encoding_errors="ignore")
    df = df.fillna("")
    return df

df = load_data()

# Combine text features for ML
if not df.empty:
    df["combined_text"] = (
        df["scheme_name"].astype(str) + " " +
        df["benefits"].astype(str) + " " +
        df["eligibility"].astype(str) + " " +
        df["tags"].astype(str)
    )

    vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
    X = vectorizer.fit_transform(df["combined_text"])
else:
    vectorizer, X = None, None

# ================================
# Helper Functions
# ================================
def filter_schemes(profile, df):
    filtered = df.copy()

    if profile.get("category"):
        if "schemeCategory" in df.columns:
            filtered = filtered[
                filtered["schemeCategory"].str.contains(profile["category"], case=False, na=False)
            ]

    if profile.get("location"):
        if "level" in df.columns:
            filtered = filtered[
                filtered["level"].str.contains(profile["location"], case=False, na=False)
            ]

    if profile.get("income"):
        if "eligibility" in df.columns:
            filtered = filtered[
                filtered["eligibility"].str.contains("income", case=False, na=False)
            ]

    return filtered


def recommend(profile_text, top_n=5):
    if vectorizer is None or X is None:
        return pd.DataFrame()

    user_vec = vectorizer.transform([profile_text])
    sim = cosine_similarity(user_vec, X).flatten()
    idx = sim.argsort()[-top_n:][::-1]
    return df.iloc[idx][["scheme_name", "benefits", "eligibility"]]

# ================================
# Streamlit App
# ================================
st.title("SchemeSarthi 🚀")
st.write("Find government schemes based on your profile!")

# User profile input
profile = {
    "age": st.number_input("Age", min_value=0, max_value=120, value=25),
    "income": st.number_input("Income", min_value=0, value=50000),
    "location": st.text_input("Location (e.g., Rural, Urban, State)"),
    "category": st.text_input("Category (e.g., Farmer, Student, Women)"),
    "occupation": st.text_input("Occupation"),
    "education": st.text_input("Education"),
}

if st.button("Get Scheme Recommendations"):
    if df.empty:
        st.warning("No data loaded. Please upload gov.csv file.")
    else:
        # Step 1: Apply filters
        filtered = filter_schemes(profile, df)

        # Step 2: Build profile text for ML recommendation
        profile_text = " ".join([str(v) for v in profile.values() if v])
        if not profile_text.strip():
            profile_text = "government scheme"

        recs = recommend(profile_text, top_n=5)

        # Final result: intersection of filters + recs
        if not filtered.empty and not recs.empty:
            final = pd.merge(
                filtered, recs, on="scheme_name", how="inner"
            )
        elif not recs.empty:
            final = recs
        else:
            final = filtered

        st.subheader("Recommended Schemes")
        if final.empty:
            st.write("No matching schemes found.")
        else:
            for _, row in final.iterrows():
                st.markdown(f"### {row['scheme_name']}")
                st.write(f"**Benefits:** {row['benefits']}")
                st.write(f"**Eligibility:** {row['eligibility']}")
                st.markdown("---")
