"""
Global Stability Explorer - Streamlit App
-----------------------------------------
HOW TO RUN THIS APP:

1. Make sure you have Python 3 installed.

2. Install the required packages (run this in the terminal, NOT inside Python):
       pip install streamlit pandas matplotlib seaborn plotly

3. Run the Streamlit app:
       streamlit run src\app.py\app.py

4. The app will open automatically in your browser.
   If not, look for the URL printed in the terminal (usually http://localhost:8501).
"""

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px


# ---------------------------------------------------------
# 1. CONFIG
# ---------------------------------------------------------

st.set_page_config(page_title="Global Stability Explorer", layout="wide")

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
        df_part = df_part.rename(columns={c: f"{name}_{c}" for c in non_key_cols})
        dfs.append(df_part)

    if not dfs:
        return pd.DataFrame()

    merged = dfs[0]
    for df_next in dfs[1:]:
        merged = pd.merge(merged, df_next, on=COUNTRY_COL, how="outer")

    return merged


def pretty_name(col: str) -> str:
    """Turn 'economy_GDP_per_capita' into 'GDP Per Capita (economy)'."""
    if col == COUNTRY_COL:
        return "Country"
    if "_" in col:
        prefix, rest = col.split("_", 1)
        rest_clean = rest.replace("_", " ").strip().title()
        return f"{rest_clean} ({prefix})"
    return col.replace("_", " ").strip().title()


def zscore(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    mu = s.mean(skipna=True)
    sd = s.std(skipna=True)
    if sd == 0 or np.isnan(sd):
        return pd.Series([np.nan] * len(s), index=s.index)
    return (s - mu) / sd


def build_stability_score(df_in: pd.DataFrame, active_domains: list[str]) -> pd.DataFrame:
    """
    Simple, explainable composite stability score.
    - Uses a small set of indicators if they exist.
    - Standardizes via z-scores.
    - Flips sign for 'bad' indicators (higher = less stable).
    - Averages available components per country.
    - Rescales to 0..100 for interpretability.
    """
    df = df_in.copy()

    candidates_pos = [
        "economy_Real_GDP_Per_Capita_USD",
        "economy_Real_GDP_PPP_Billion_USD",
        "economy_Real_GDP_Growth_Rate_percent",
    ]
    candidates_neg = [
        "economy_Unemployment_Rate_percent",
        "economy_Population_Below_Poverty_Line_percent",
    ]
    candidates_dem = [
        "demographics_Population_Growth_Rate",
    ]

    def domain_ok(col: str) -> bool:
        if "_" not in col:
            return False
        dom = col.split("_", 1)[0]
        return dom in active_domains

    pos_cols = [c for c in candidates_pos if c in df.columns and domain_ok(c)]
    neg_cols = [c for c in candidates_neg if c in df.columns and domain_ok(c)]
    dem_cols = [c for c in candidates_dem if c in df.columns and domain_ok(c)]

    if len(pos_cols) + len(neg_cols) + len(dem_cols) == 0:
        df["stability_score"] = np.nan
        return df

    parts = []
    for c in pos_cols:
        parts.append(zscore(df[c]))
    for c in neg_cols:
        parts.append(-zscore(df[c]))
    for c in dem_cols:
        parts.append(-zscore(df[c]).abs())

    comp = pd.concat(parts, axis=1)
    raw = comp.mean(axis=1, skipna=True)

    raw_min = raw.min(skipna=True)
    raw_max = raw.max(skipna=True)
    if pd.isna(raw_min) or pd.isna(raw_max) or raw_max == raw_min:
        df["stability_score"] = np.nan
        return df

    df["stability_score"] = (raw - raw_min) / (raw_max - raw_min) * 100.0
    return df


# ---------------------------------------------------------
# 3. APP HEADER + LOAD
# ---------------------------------------------------------

st.title("Global Stability Explorer – Interactive Explorer")
st.markdown(
    """
Each point in the scatterplot is a **country**.  
We merge CIA datasets on the `Country` column and explore indicators with linked views:
scatterplot + table + country profile (and compare).
"""
)

df = load_and_merge(CSV_FILES)

df[COUNTRY_COL] = df[COUNTRY_COL].astype(str).str.strip()
df = df[~df[COUNTRY_COL].str.upper().str.contains("WORLD|GLOBAL|TOTAL", na=False)].copy()

if df.empty:
    st.error("Merged dataframe is empty. Please check the CSV filenames and paths.")
    st.stop()

st.write(f"Total rows (countries) after merge: **{len(df)}**")

pop_cols = [c for c in df.columns if "Total_Population" in c]
for c in pop_cols:
    df[c] = df[c].astype(str).str.replace(",", "")
    df[c] = pd.to_numeric(df[c], errors="coerce")


# ---------------------------------------------------------
# 4. DOMAIN FILTERING + COLUMN OPTIONS
# ---------------------------------------------------------

all_domains = sorted(list(CSV_FILES.keys()))
st.sidebar.header("Explorer settings")

active_domains = st.sidebar.multiselect(
    "Active domains (affects indicator dropdowns + stability score)",
    all_domains,
    default=["economy", "demographics", "government_civics"],
)

numeric_cols_all = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
non_numeric_cols_all = df.select_dtypes(exclude=["int64", "float64"]).columns.tolist()
if COUNTRY_COL in non_numeric_cols_all:
    non_numeric_cols_all.remove(COUNTRY_COL)

def is_active_domain_col(col: str) -> bool:
    if "_" not in col:
        return False
    dom = col.split("_", 1)[0]
    return dom in active_domains

numeric_cols = [c for c in numeric_cols_all if is_active_domain_col(c)]
non_numeric_cols = [c for c in non_numeric_cols_all if is_active_domain_col(c)]

df = build_stability_score(df, active_domains)
if "stability_score" in df.columns:
    numeric_cols_with_score = list(dict.fromkeys(numeric_cols + ["stability_score"]))
else:
    numeric_cols_with_score = numeric_cols

pretty_numeric = {pretty_name(c): c for c in numeric_cols_with_score}
pretty_categorical = {pretty_name(c): c for c in non_numeric_cols}

pretty_numeric_options = list(pretty_numeric.keys())
pretty_categorical_options = ["None", "Country", "Stability Score"] + list(pretty_categorical.keys())

if len(pretty_numeric_options) < 2:
    st.error(
        "Not enough numeric columns in the selected domains to build a scatterplot. "
        "Select more domains in the sidebar."
    )
    st.stop()


# ---------------------------------------------------------
# 5. SIDEBAR CONTROLS
# ---------------------------------------------------------

st.sidebar.subheader("Scatterplot settings")

x_pretty = st.sidebar.selectbox("X-axis (numeric indicator)", pretty_numeric_options, index=0)
y_pretty = st.sidebar.selectbox(
    "Y-axis (numeric indicator)",
    pretty_numeric_options,
    index=1 if len(pretty_numeric_options) > 1 else 0,
)

color_pretty = st.sidebar.selectbox("Color by", pretty_categorical_options, index=0)

size_pretty_options = ["None"] + pretty_numeric_options
size_pretty = st.sidebar.selectbox("Size by (numeric)", size_pretty_options, index=0)

x_col = pretty_numeric[x_pretty]
y_col = pretty_numeric[y_pretty]

if color_pretty == "None":
    color_col = None
elif color_pretty == "Country":
    color_col = COUNTRY_COL
elif color_pretty == "Stability Score":
    color_col = "stability_score"
else:
    color_col = pretty_categorical[color_pretty]

size_col = None if size_pretty == "None" else pretty_numeric[size_pretty]

all_countries = sorted(df[COUNTRY_COL].dropna().unique())
exclude_countries = st.sidebar.multiselect("Exclude countries (optional)", all_countries, default=[])

highlight_countries = st.sidebar.multiselect(
    "Highlight countries (linked to profile/table)",
    all_countries,
    default=[],
)

st.sidebar.subheader("Compare (optional)")
compare_a = st.sidebar.selectbox("Country A", ["None"] + all_countries, index=0)
compare_b = st.sidebar.selectbox("Country B", ["None"] + all_countries, index=0)


# ---------------------------------------------------------
# 6. FILTER DATA FOR THE CURRENT VIEW
# ---------------------------------------------------------

required_cols = [x_col, y_col]
if size_col is not None:
    required_cols.append(size_col)
if color_col in numeric_cols_with_score:
    required_cols.append(color_col)

plot_df = df.dropna(subset=list(dict.fromkeys(required_cols))).copy()

if exclude_countries:
    plot_df = plot_df[~plot_df[COUNTRY_COL].isin(exclude_countries)]

st.write(
    f"Rows used after dropping missing values in **{x_pretty}**, **{y_pretty}**"
    + (f" and **{size_pretty}**" if size_col is not None else "")
    + (f" excluding {len(exclude_countries)} countries" if exclude_countries else "")
    + f": **{len(plot_df)}**"
)

if len(plot_df) == 0:
    st.warning("No data left after filtering missing values / exclusions.")
    st.stop()


# ---------------------------------------------------------
# 7. LINKED LAYOUT: SCATTER + PROFILE
# ---------------------------------------------------------

left, right = st.columns([2.2, 1.0], gap="large")

with left:
    st.subheader("Interactive scatterplot (Plotly)")

    # ---- Plot-side menu (requested) ----
    opts = st.columns([1.2, 1.0, 1.0])
    with opts[0]:
        only_show_highlighted = st.checkbox(
            "Only show highlighted",
            value=False,
            help="If enabled, the plot (and correlation metric) uses only highlighted countries (if any)."
        )
    with opts[1]:
        show_trendline = st.checkbox(
            "Show trendline",
            value=False,
            help="Adds an OLS trendline to the currently visible points."
        )
    with opts[2]:
        # Small UX helper: show how many highlights exist
        st.caption(f"Highlighted: {len(highlight_countries)}")

    # Decide what the plot should actually show
    plot_view_df = plot_df.copy()
    if only_show_highlighted:
        if highlight_countries:
            plot_view_df = plot_view_df[plot_view_df[COUNTRY_COL].isin(highlight_countries)].copy()
        else:
            st.info("No highlighted countries selected. Showing all countries in current view.")
            plot_view_df = plot_df.copy()

    if plot_view_df.empty:
        st.warning("No data to show for the current plot options.")
        st.stop()

    # correlation (matches what's currently visible in the plot)
    x_vals = pd.to_numeric(plot_view_df[x_col], errors="coerce")
    y_vals = pd.to_numeric(plot_view_df[y_col], errors="coerce")
    corr = x_vals.corr(y_vals)

    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        st.metric("Pearson r (visible)", f"{corr:.3f}" if pd.notna(corr) else "n/a")
    with c2:
        st.metric("Countries visible", f"{plot_view_df[COUNTRY_COL].nunique()}")
    with c3:
        st.metric("Rows visible", f"{len(plot_view_df)}")

    hover_data = {x_col: True, y_col: True, COUNTRY_COL: True}
    if color_col is not None:
        hover_data[color_col] = True
    if size_col is not None:
        hover_data[size_col] = True
    if "stability_score" in plot_view_df.columns:
        hover_data["stability_score"] = True

    # Outlier detection (compute on the FULL filtered set so table/profile stay consistent)
    zx = zscore(plot_df[x_col])
    zy = zscore(plot_df[y_col])
    outlier_mask = (zx.abs() > 2.5) | (zy.abs() > 2.5)
    plot_df["_is_outlier"] = outlier_mask.fillna(False)

    # Base figure (uses plot_view_df now)
    fig = px.scatter(
        plot_view_df,
        x=x_col,
        y=y_col,
        color=color_col,
        size=size_col,
        hover_name=COUNTRY_COL,
        hover_data=hover_data,
        title=f"{y_pretty} vs {x_pretty}",
        trendline="ols" if show_trendline else None,
    )

    # Highlight overlay (only when NOT already limiting to highlighted,
    # otherwise it becomes redundant noise)
    if (highlight_countries) and (not only_show_highlighted):
        sel = plot_df[plot_df[COUNTRY_COL].isin(highlight_countries)]
        if not sel.empty:
            fig2 = px.scatter(
                sel,
                x=x_col,
                y=y_col,
                size=size_col,
                hover_name=COUNTRY_COL,
                hover_data=hover_data,
            )
            for tr in fig2.data:
                tr.name = "Highlighted"
                tr.showlegend = True
                tr.marker.line.width = 2
            fig.add_traces(fig2.data)

    # If coloring by Country: show only top-5 legend by default, with toggle
    updatemenus = []
    if color_col == COUNTRY_COL:
        showlegend_all = [True] * len(fig.data)

        # Rank by size if provided; else by frequency (using what’s visible)
        if size_col is not None and size_col in plot_view_df.columns:
            ranking = (
                plot_view_df[[COUNTRY_COL, size_col]]
                .groupby(COUNTRY_COL)[size_col]
                .mean()
                .sort_values(ascending=False)
            )
            top_countries = list(ranking.head(5).index)
        else:
            top_countries = list(plot_view_df[COUNTRY_COL].value_counts().head(5).index)

        showlegend_top5 = []
        for tr in fig.data:
            showlegend_top5.append((tr.name in top_countries) if tr.name != "Highlighted" else True)

        for tr, show in zip(fig.data, showlegend_top5):
            tr.showlegend = show

        updatemenus = [
            dict(
                type="buttons",
                direction="right",
                x=1,
                y=1.15,
                showactive=True,
                buttons=[
                    dict(label="Top 5 in legend", method="restyle", args=[{"showlegend": showlegend_top5}]),
                    dict(label="All legend items", method="restyle", args=[{"showlegend": showlegend_all}]),
                ],
            )
        ]

    fig.update_layout(
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis_title=pretty_name(x_col),
        yaxis_title=pretty_name(y_col),
        legend_title_text=color_pretty if color_col is not None else "Legend",
        updatemenus=updatemenus,
    )

    st.plotly_chart(fig, use_container_width=True)

with right:
    st.subheader("Country profile (linked)")

    default_profile = highlight_countries[0] if highlight_countries else (all_countries[0] if all_countries else None)
    profile_country = st.selectbox(
        "Profile country",
        all_countries,
        index=all_countries.index(default_profile) if default_profile in all_countries else 0
    )

    row = plot_df[plot_df[COUNTRY_COL] == profile_country]
    if row.empty:
        st.info("Selected country is not in the current filtered view (missing data or excluded).")
    else:
        r = row.iloc[0]
        st.write(f"**{COUNTRY_COL}:** {r[COUNTRY_COL]}")
        if "stability_score" in plot_df.columns:
            st.metric("Stability score (0–100)", f"{r['stability_score']:.1f}" if pd.notna(r["stability_score"]) else "n/a")

        st.markdown("**Selected indicators:**")
        metrics = [
            (pretty_name(x_col), r.get(x_col, np.nan)),
            (pretty_name(y_col), r.get(y_col, np.nan)),
        ]
        if size_col is not None:
            metrics.append((pretty_name(size_col), r.get(size_col, np.nan)))
        if color_col is not None and color_col not in (COUNTRY_COL,):
            metrics.append((pretty_name(color_col), r.get(color_col, np.nan)))

        for label, val in metrics:
            if pd.isna(val):
                st.write(f"- {label}: n/a")
            else:
                st.write(f"- {label}: {val}")

        is_out = bool(row["_is_outlier"].iloc[0]) if "_is_outlier" in row.columns else False
        st.write("**Outlier (in current view):** " + ("✅ Yes" if is_out else "No"))

    st.subheader("Compare (A vs B)")
    if compare_a != "None" and compare_b != "None" and compare_a != compare_b:
        a = plot_df[plot_df[COUNTRY_COL] == compare_a]
        b = plot_df[plot_df[COUNTRY_COL] == compare_b]
        if a.empty or b.empty:
            st.info("One of the selected countries is not in the current view (missing data or excluded).")
        else:
            a = a.iloc[0]
            b = b.iloc[0]

            def fmt(v):
                return "n/a" if pd.isna(v) else v

            st.write(f"**{compare_a} vs {compare_b}**")
            comp_rows = []
            for col in [x_col, y_col] + ([size_col] if size_col is not None else []) + (["stability_score"] if "stability_score" in plot_df.columns else []):
                if col is None:
                    continue
                comp_rows.append(
                    {
                        "Indicator": pretty_name(col),
                        compare_a: fmt(a.get(col, np.nan)),
                        compare_b: fmt(b.get(col, np.nan)),
                        "Δ (A-B)": (a.get(col, np.nan) - b.get(col, np.nan))
                        if (pd.notna(a.get(col, np.nan)) and pd.notna(b.get(col, np.nan)))
                        else "n/a",
                    }
                )
            st.dataframe(pd.DataFrame(comp_rows), use_container_width=True, hide_index=True)
    else:
        st.caption("Select two different countries in the sidebar to compare.")


# ---------------------------------------------------------
# 8. LINKED TABLE VIEW
# ---------------------------------------------------------

st.subheader("Country table (linked to current filters)")
st.caption("If you highlight countries, the table shows only those. Otherwise it shows the full filtered set.")

if highlight_countries:
    table_source = plot_df[plot_df[COUNTRY_COL].isin(highlight_countries)].copy()
    st.write(f"Showing **{len(table_source)}** highlighted countries.")
else:
    table_source = plot_df.copy()
    st.write(f"Showing **{len(table_source)}** countries in the current filtered view.")

table_cols = [COUNTRY_COL, x_col, y_col]

if size_col is not None:
    table_cols.append(size_col)

if "stability_score" in table_source.columns and "stability_score" not in table_cols:
    table_cols.append("stability_score")

if "_is_outlier" in table_source.columns:
    table_cols.append("_is_outlier")

table_cols = list(dict.fromkeys(table_cols))
table_df = table_source[table_cols].copy()

rename_map = {}
used = set()
for c in table_df.columns:
    if c in (COUNTRY_COL, "_is_outlier"):
        continue
    base = pretty_name(c)
    new = base
    k = 2
    while new in used:
        new = f"{base} ({k})"
        k += 1
    rename_map[c] = new
    used.add(new)

table_df = table_df.rename(columns=rename_map)
table_df = table_df.rename(columns={"_is_outlier": "Outlier (current view)"})

st.dataframe(table_df, use_container_width=True, hide_index=True)


# ---------------------------------------------------------
# 9. STATIC SCATTERPLOT (optional)
# ---------------------------------------------------------

st.subheader("Static scatterplot (matplotlib / seaborn)")
st.markdown("Useful for exporting figures to the report.")

plt.figure(figsize=(6, 4))
sns.scatterplot(
    data=plot_df,
    x=x_col,
    y=y_col,
    hue=color_col if (color_col is not None and color_col != "stability_score") else None,
    size=size_col if size_col is not None else None,
    alpha=0.8,
)
plt.title(f"{y_pretty} vs {x_pretty}")
plt.tight_layout()
st.pyplot(plt.gcf())
