"""
Global Stability Explorer - Streamlit App
-----------------------------------------
HOW TO RUN THIS APP:

1. Make sure you have Python 3 installed.

2. Install the required packages (run this in the terminal, NOT inside Python):
       pip install streamlit pandas matplotlib seaborn plotly

3. Run the Streamlit app:
       streamlit run app.py

4. The app will open automatically in your browser.
   If not, look for the URL printed in the terminal (usually http://localhost:8501).
"""

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# ---------------------------------------------------------
# 1. CONFIG
# ---------------------------------------------------------

st.set_page_config(page_title="Global Stability Explorer - Scatterplot", layout="wide")

CSV_FILES = {
    "economy": "data/raw/economy_data.csv",
    "demographics": "data/raw/demographics_data.csv",
    "energy_emissions": "data/raw/energy_data.csv",
    "communications": "data/raw/communications_data.csv",
    "transportation": "data/raw/transportation_data.csv",
    "geography": "data/raw/geography_data.csv",
    "government_civics": "data/raw/government_and_civics_data.csv",
}

COUNTRY_COL = "Country"  # all files must have this column


# ---------------------------------------------------------
# 2. LOAD AND MERGE DATA
# ---------------------------------------------------------

@st.cache_data
def load_and_merge(files_dict):
    dfs = []
    for name, path in files_dict.items():
        try:
            df_part = pd.read_csv(path)
        except FileNotFoundError:
            st.warning(f"File not found: {path} (dataset '{name}')")
            continue

        if COUNTRY_COL not in df_part.columns:
            st.warning(f"Column '{COUNTRY_COL}' not found in file: {path}")
            continue

        non_key_cols = [c for c in df_part.columns if c != COUNTRY_COL]
        df_part = df_part[[COUNTRY_COL] + non_key_cols]
        # prefix to avoid clashes
        df_part = df_part.rename(columns={c: f"{name}_{c}" for c in non_key_cols})

        dfs.append(df_part)

    if not dfs:
        return pd.DataFrame()

    merged = dfs[0]
    for df_next in dfs[1:]:
        merged = pd.merge(merged, df_next, on=COUNTRY_COL, how="outer")

    return merged


st.title("Global Stability Explorer – Interactive Scatterplot")
st.markdown(
    """
Each point in the scatterplot is a **country**.  
We merge the CIA datasets on the `Country` column and then choose indicators for the axes.
"""
)

df = load_and_merge(CSV_FILES)

if df.empty:
    st.error("Merged dataframe is empty. Please check the CSV filenames and paths.")
    st.stop()

st.write(f"Total rows (countries) after merge: **{len(df)}**")


# ---------------------------------------------------------
# 3. BUILD PRETTY LABELS FOR COLUMNS
# ---------------------------------------------------------

def pretty_name(col: str) -> str:
    """Turn 'economy_GDP_per_capita' into 'GDP Per Capita (economy)'."""
    if col == COUNTRY_COL:
        return "Country"
    if "_" in col:
        prefix, rest = col.split("_", 1)
        rest_clean = rest.replace("_", " ").strip().title()
        return f"{rest_clean} ({prefix})"
    return col.replace("_", " ").strip().title()

all_cols = df.columns.tolist()
numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
non_numeric_cols = df.select_dtypes(exclude=["int64", "float64"]).columns.tolist()

# Remove Country from non-numeric list if present
if COUNTRY_COL in non_numeric_cols:
    non_numeric_cols.remove(COUNTRY_COL)

# Mapping from pretty label -> real column name
pretty_numeric = {pretty_name(c): c for c in numeric_cols}
pretty_categorical = {pretty_name(c): c for c in non_numeric_cols}

pretty_numeric_options = list(pretty_numeric.keys())
pretty_categorical_options = ["None"] + list(pretty_categorical.keys())

if len(numeric_cols) < 2:
    st.error("Not enough numeric columns found to build a scatterplot.")
    st.stop()


# ---------------------------------------------------------
# 4. SIDEBAR CONTROLS (USING PRETTY LABELS)
# ---------------------------------------------------------

st.sidebar.header("Scatterplot settings")

x_pretty = st.sidebar.selectbox("X-axis (numeric indicator)", pretty_numeric_options, index=0)
y_pretty = st.sidebar.selectbox(
    "Y-axis (numeric indicator)",
    pretty_numeric_options,
    index=1 if len(pretty_numeric_options) > 1 else 0,
)

color_pretty = st.sidebar.selectbox("Color by (categorical)", pretty_categorical_options, index=0)

size_pretty_options = ["None"] + pretty_numeric_options
size_pretty = st.sidebar.selectbox("Size by (numeric)", size_pretty_options, index=0)

# Map back to real column names
x_col = pretty_numeric[x_pretty]
y_col = pretty_numeric[y_pretty]
color_col = None if color_pretty == "None" else pretty_categorical[color_pretty]
size_col = None if size_pretty == "None" else pretty_numeric[size_pretty]

# Drop rows with missing values for chosen x/y
plot_df = df.dropna(subset=[x_col, y_col])

st.write(f"Rows used after dropping missing values in **{x_pretty}** and **{y_pretty}**: **{len(plot_df)}**")

if len(plot_df) == 0:
    st.warning("No data left after filtering missing values for selected axes.")
    st.stop()


# ---------------------------------------------------------
# 5. PLOTLY INTERACTIVE SCATTERPLOT
# ---------------------------------------------------------

st.subheader("Interactive scatterplot (Plotly)")

hover_data = {x_col: True, y_col: True}
if COUNTRY_COL in plot_df.columns:
    hover_data[COUNTRY_COL] = True
if color_col is not None:
    hover_data[color_col] = True
if size_col is not None:
    hover_data[size_col] = True

fig = px.scatter(
    plot_df,
    x=x_col,
    y=y_col,
    color=color_col,
    size=size_col,
    hover_name=COUNTRY_COL if COUNTRY_COL in plot_df.columns else None,
    hover_data=hover_data,
    title=f"{y_pretty} vs {x_pretty}",
)

fig.update_layout(
    margin=dict(l=20, r=20, t=40, b=20),
    legend_title_text=color_pretty if color_col is not None else "Legend",
)

st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------
# 6. STATIC SCATTERPLOT (MATPLOTLIB / SEABORN)
# ---------------------------------------------------------

st.subheader("Static scatterplot (matplotlib / seaborn)")
st.markdown("This static plot can be useful for exporting figures to the report.")

plt.figure(figsize=(6, 4))
sns.scatterplot(
    data=plot_df,
    x=x_col,
    y=y_col,
    hue=color_col if color_col is not None else None,
    size=size_col if size_col is not None else None,
    alpha=0.8,
)
plt.title(f"{y_pretty} vs {x_pretty}")
plt.tight_layout()
st.pyplot(plt.gcf())