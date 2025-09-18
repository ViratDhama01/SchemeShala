import streamlit as st
import pandas as pd
from pathlib import Path

# ================================
# Config
# ================================
APP_NAME = "SchemeSarthi 🚀"
DATA_FILE = Path(__file__).with_name("gov.csv")  # CSV in the same folder

# ================================
# Data loading
# ================================
@st.cache_data
def load_data(path: Path):
    if not path.exists():
        st.error(f"CSV not found at {path.resolve()} — please upload 'gov.csv'")
        return pd.DataFrame()
    try:
        df = pd.read_csv(path, engine="python", on_bad_lines="skip")
    except Exception:
        df = pd.read_csv(
            path, engine="python", on_bad_lines="skip", encoding_errors="ignore"
        )
    return df.fillna("")

df = load_data(DATA_FILE)

# ================================
# App UI
# ================================
st.title(APP_NAME)
st.write("Find government schemes based on your profile!")

# User inputs
age = st.number_input("Age", min_value=0, max_value=120, value=25)
income = st.number_input("Income", min_value=0, value=50000)
location = st.text_input("Location (e.g., State, Rural, Urban)")
category = st.text_input("Category (e.g., Farmer, Student, Woman)")
occupation = st.text_input("Occupation (optional)")
education = st.text_input("Education (optional)")

if st.button("Recommend Schemes"):
    if df.empty:
        st.warning("No data loaded. Please check gov.csv file.")
    else:
        filtered = df.copy()

        # Basic filtering rules
        if "schemeCategory" in df.columns and category:
            filtered = filtered[
                filtered["schemeCategory"].str.contains(category, case=False, na=False)
            ]
        if "level" in df.columns and location:
            filtered = filtered[
                filtered["level"].str.contains(location, case=False, na=False)
            ]
        if "eligibility" in df.columns and income:
            # Example: keep schemes mentioning "income"
            filtered = filtered[
                filtered["eligibility"].str.contains("income", case=False, na=False)
            ]

        st.subheader("Top Recommendations")
        st.dataframe(filtered.head(5))
