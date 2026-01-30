"""
Global Stability Explorer - Streamlit App
-----------------------------------------
HOW TO RUN THIS APP:

1. Make sure you have Python 3 installed.

2. Install the required packages (run this in the terminal, NOT inside Python):
       pip install streamlit pandas matplotlib seaborn plotly numpy

3. Run the Streamlit app (from the project root):
       streamlit run src/app.py

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

COUNTRY_COL = "Country"

# Color theme: continuous GREEN
CONTINUOUS_GREEN_SCALE = "Greens"
HIGHLIGHT_GREEN = "#2ECC71"   # nice, readable green
DARK_BG = "#111111"


# ---------------------------------------------------------
# 2. HELPERS
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
    if col == COUNTRY_COL:
        return "Country"
    if col == "stability_score":
        return "Score (stability)"
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


# --- Country naming for maps (keep simple + targeted) ---
NAME_FIX = {
    "SWITZERLAND": "Switzerland",
    "BAHAMAS, THE": "Bahamas",
    "GAMBIA, THE": "Gambia",
    "CZECHIA": "Czech Republic",
    "RUSSIA": "Russian Federation",
    "KOREA, SOUTH": "South Korea",
    "KOREA, NORTH": "North Korea",
    "CONGO, DEMOCRATIC REPUBLIC OF THE": "Democratic Republic of the Congo",
    "CONGO, REPUBLIC OF THE": "Republic of the Congo",
    "BOLIVIA": "Bolivia",
    "VENEZUELA": "Venezuela",
    "TANZANIA": "Tanzania",
    "LAOS": "Laos",
    "BRUNEI": "Brunei",
    "VIETNAM": "Vietnam",
    "UNITED STATES": "United States",
    "UNITED KINGDOM": "United Kingdom",
}


def to_map_name(country: str) -> str:
    raw = str(country).strip()
    up = raw.upper()
    if up in NAME_FIX:
        return NAME_FIX[up]
    return raw.title()


@st.cache_data
def gapminder_iso_map():
    """
    Name->ISO3 mapping from Plotly gapminder (fast, no extra packages).
    We supplement it with a small manual ISO3 table for common territories.
    """
    gm = px.data.gapminder()[["country", "iso_alpha"]].drop_duplicates().copy()
    gm["key"] = gm["country"].astype(str).str.strip().str.upper()
    base = dict(zip(gm["key"], gm["iso_alpha"]))

    # Manual ISO3 for frequent CIA entries / territories
    manual = {
        "SWITZERLAND": "CHE",
        "BAHAMAS": "BHS",
        "GAMBIA": "GMB",
        "CZECH REPUBLIC": "CZE",
        "RUSSIAN FEDERATION": "RUS",
        "SOUTH KOREA": "KOR",
        "NORTH KOREA": "PRK",
        "DEMOCRATIC REPUBLIC OF THE CONGO": "COD",
        "REPUBLIC OF THE CONGO": "COG",
        "BOLIVIA": "BOL",
        "VENEZUELA": "VEN",
        "TANZANIA": "TZA",
        "LAOS": "LAO",
        "BRUNEI": "BRN",
        "VIETNAM": "VNM",
        "AMERICAN SAMOA": "ASM",
        "ANGUILLA": "AIA",
        "ANTIGUA AND BARBUDA": "ATG",
        "ARUBA": "ABW",
        "BERMUDA": "BMU",
        "CAYMAN ISLANDS": "CYM",
        "CURACAO": "CUW",
        "FAROE ISLANDS": "FRO",
        "FRENCH POLYNESIA": "PYF",
        "GIBRALTAR": "GIB",
        "GREENLAND": "GRL",
        "GUADELOUPE": "GLP",
        "GUAM": "GUM",
        "HONG KONG": "HKG",
        "MACAU": "MAC",
        "MARTINIQUE": "MTQ",
        "MAYOTTE": "MYT",
        "MONTSERRAT": "MSR",
        "NEW CALEDONIA": "NCL",
        "NORTHERN MARIANA ISLANDS": "MNP",
        "PUERTO RICO": "PRI",
        "REUNION": "REU",
        "SAINT MARTIN": "MAF",
        "SINT MAARTEN": "SXM",
        "TOKELAU": "TKL",
        "TURKS AND CAICOS ISLANDS": "TCA",
        "US VIRGIN ISLANDS": "VIR",
        "VIRGIN ISLANDS, BRITISH": "VGB",
        "WALLIS AND FUTUNA": "WLF",
        "WESTERN SAHARA": "ESH",
        "PALESTINE": "PSE",
        "WEST BANK": "PSE",
        "GAZA STRIP": "PSE",
        "KOSOVO": "XKX",  # commonly used pseudo-ISO3
    }
    base.update(manual)
    return base


def to_iso3(country: str) -> str | None:
    """
    Best-effort ISO3: use name fixes then lookup in gapminder/manual.
    """
    iso_map = gapminder_iso_map()
    up = str(country).strip().upper()

    # apply name fixes first (in uppercase)
    if up in NAME_FIX:
        key = NAME_FIX[up].strip().upper()
    else:
        key = up

    # a couple of extra normalizations
    key = key.replace(", THE", "").replace("  ", " ").strip()

    return iso_map.get(key)


# ---------------------------------------------------------
# 3. HEADER + LOAD
# ---------------------------------------------------------

st.title("Global Stability Explorer – Interactive Explorer")
st.markdown(
    """
Each point is a **country**.  
Explore indicators with linked views: **scatterplot → profile → table → world map**.
"""
)

df = load_and_merge(CSV_FILES)

# Remove aggregate rows like WORLD / GLOBAL / TOTAL
df[COUNTRY_COL] = df[COUNTRY_COL].astype(str).str.strip()
df = df[~df[COUNTRY_COL].str.upper().str.contains("WORLD|GLOBAL|TOTAL", na=False)].copy()

if df.empty:
    st.error("Merged dataframe is empty. Please check the CSV filenames and paths.")
    st.stop()

st.write(f"Total rows (countries) after merge: **{len(df)}**")

# Ensure population is numeric (if present)
pop_cols = [c for c in df.columns if "Total_Population" in c]
for c in pop_cols:
    df[c] = df[c].astype(str).str.replace(",", "")
    df[c] = pd.to_numeric(df[c], errors="coerce")

# ---------------------------------------------------------
# CLEAN NUMERIC COLUMNS (CRITICAL FOR GDP / SWITZERLAND FIX)
# ---------------------------------------------------------

def maybe_to_numeric(series: pd.Series) -> pd.Series:
    # Only attempt conversion on object/string columns
    if series.dtype != "object":
        return series

    s = series.astype(str).str.strip()

    # common missing tokens
    s = s.replace(
        {"": np.nan, "nan": np.nan, "N/A": np.nan, "n/a": np.nan, "..": np.nan, "—": np.nan}
    )

    # remove separators/symbols
    s_clean = (
        s.str.replace(",", "", regex=False)
        .str.replace("%", "", regex=False)
        .str.replace("$", "", regex=False)
    )

    # detect numeric-looking values
    looks_numeric = s_clean.str.match(r"^-?\d+(\.\d+)?$", na=False)
    ratio = looks_numeric.mean()

    # convert only if mostly numeric
    if ratio >= 0.80:
        return pd.to_numeric(s_clean, errors="coerce")

    return series


for c in df.columns:
    if c == COUNTRY_COL:
        continue
    df[c] = maybe_to_numeric(df[c])

# ---------------------------------------------------------
# 4. SIDEBAR: DOMAIN + PLOT CONTROLS
# ---------------------------------------------------------

all_domains = sorted(list(CSV_FILES.keys()))
st.sidebar.header("Explorer settings")

active_domains = st.sidebar.multiselect(
    "Active domains",
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
    return col.split("_", 1)[0] in active_domains


numeric_cols = [c for c in numeric_cols_all if is_active_domain_col(c)]
non_numeric_cols = [c for c in non_numeric_cols_all if is_active_domain_col(c)]

df = build_stability_score(df, active_domains)
numeric_cols_with_score = list(
    dict.fromkeys(numeric_cols + (["stability_score"] if "stability_score" in df.columns else []))
)

pretty_numeric = {pretty_name(c): c for c in numeric_cols_with_score}
pretty_numeric_options = list(pretty_numeric.keys())

pretty_categorical = {pretty_name(c): c for c in non_numeric_cols}
pretty_categorical_options = ["None", "Country", "Stability Score"] + list(pretty_categorical.keys())

if len(pretty_numeric_options) < 2:
    st.error("Not enough numeric columns in selected domains to build a scatterplot.")
    st.stop()

st.sidebar.subheader("Scatterplot settings")

x_pretty = st.sidebar.selectbox("X-axis (numeric)", pretty_numeric_options, index=0)
y_pretty = st.sidebar.selectbox(
    "Y-axis (numeric)",
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
    "Highlight countries",
    all_countries,
    default=[],
)

st.sidebar.subheader("Compare (optional)")
compare_a = st.sidebar.selectbox("Country A", ["None"] + all_countries, index=0)
compare_b = st.sidebar.selectbox("Country B", ["None"] + all_countries, index=0)


# ---------------------------------------------------------
# 5. FILTERED VIEW FOR SCATTER/TABLE
# ---------------------------------------------------------

required_cols = [x_col, y_col]
if size_col is not None:
    required_cols.append(size_col)
if color_col in numeric_cols_with_score:
    required_cols.append(color_col)

plot_df = df.dropna(subset=list(dict.fromkeys(required_cols))).copy()

if exclude_countries:
    plot_df = plot_df[~plot_df[COUNTRY_COL].isin(exclude_countries)].copy()

st.write(
    f"Rows used after dropping missing values in **{x_pretty}**, **{y_pretty}**"
    + (f" and **{size_pretty}**" if size_col is not None else "")
    + (f" excluding {len(exclude_countries)} countries" if exclude_countries else "")
    + f": **{len(plot_df)}**"
)

if plot_df.empty:
    st.warning("No data left after filtering missing values / exclusions.")
    st.stop()


# ---------------------------------------------------------
# 6. TOP: SCATTER + PROFILE (LINKED)
# ---------------------------------------------------------

left, right = st.columns([2.2, 1.0], gap="large")

with left:
    st.subheader("Interactive scatterplot")

    opts = st.columns([1.2, 1.0, 1.0])
    with opts[0]:
        only_show_highlighted = st.checkbox(
            "Only show highlighted",
            value=False,
            help="If enabled, the plot (and Pearson r) uses only highlighted countries (if any).",
        )
    with opts[1]:
        show_trendline = st.checkbox(
            "Show trendline",
            value=False,
            help="Adds an OLS trendline to the currently visible points.",
        )
    with opts[2]:
        st.caption(f"Highlighted: {len(highlight_countries)}")

    plot_view_df = plot_df.copy()
    if only_show_highlighted and highlight_countries:
        plot_view_df = plot_view_df[plot_view_df[COUNTRY_COL].isin(highlight_countries)].copy()

    if plot_view_df.empty:
        st.warning("No data to show for the current plot options.")
        st.stop()

    x_vals = pd.to_numeric(plot_view_df[x_col], errors="coerce")
    y_vals = pd.to_numeric(plot_view_df[y_col], errors="coerce")
    corr = x_vals.corr(y_vals)

    m1, m2, m3 = st.columns([1, 1, 1])
    with m1:
        st.metric("Pearson coefficient", f"{corr:.3f}" if pd.notna(corr) else "n/a")
    with m2:
        st.metric("Countries visible", f"{plot_view_df[COUNTRY_COL].nunique()}")
    with m3:
        st.metric("Rows visible", f"{len(plot_view_df)}")

    hover_data = {x_col: True, y_col: True, COUNTRY_COL: True}
    if color_col is not None:
        hover_data[color_col] = True
    if size_col is not None:
        hover_data[size_col] = True
    if "stability_score" in plot_view_df.columns:
        hover_data["stability_score"] = True

    # Outliers computed on full filtered set (plot_df)
    zx = zscore(plot_df[x_col])
    zy = zscore(plot_df[y_col])
    plot_df["_is_outlier"] = ((zx.abs() > 2.5) | (zy.abs() > 2.5)).fillna(False)

    # If color is numeric -> continuous green scale
    # If color is categorical -> discrete green shades
    green_discrete = px.colors.sequential.Greens[2:]

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
        color_continuous_scale=CONTINUOUS_GREEN_SCALE,
        color_discrete_sequence=green_discrete,
    )

    # Highlight overlay (only if not already showing only highlighted)
    if highlight_countries and not only_show_highlighted:
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

    fig.update_layout(
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis_title=pretty_name(x_col),
        yaxis_title=pretty_name(y_col),
        legend_title_text=color_pretty if color_col is not None else "Legend",
    )

    st.plotly_chart(fig, use_container_width=True)

    # ---------------------------------------------------------
    # COMPARE (A vs B) - moved UNDER the scatterplot
    # ---------------------------------------------------------
    st.subheader("Compare countries (A vs B)")
    st.caption("Comparison uses the full dataset (even if a country is missing from the current scatter view).")

    if compare_a != "None" and compare_b != "None" and compare_a != compare_b:
        a = df[df[COUNTRY_COL] == compare_a]
        b = df[df[COUNTRY_COL] == compare_b]

        if a.empty or b.empty:
            st.info("One of the selected countries is not in the dataset.")
        else:
            a = a.iloc[0]
            b = b.iloc[0]

            def fmt(v):
                return "n/a" if pd.isna(v) else v

            compare_cols = [x_col, y_col]
            if size_col is not None:
                compare_cols.append(size_col)
            if "stability_score" in df.columns:
                compare_cols.append("stability_score")

            # de-duplicate while preserving order
            compare_cols = [c for c in dict.fromkeys(compare_cols) if c is not None]

            rows = []
            for col in compare_cols:
                va, vb = a.get(col, np.nan), b.get(col, np.nan)
                rows.append(
                    {
                        "Indicator": pretty_name(col),
                        compare_a: fmt(va),
                        compare_b: fmt(vb),
                        "Δ (A-B)": (va - vb) if (pd.notna(va) and pd.notna(vb)) else "n/a",
                    }
                )

            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.caption("Select two different countries in the sidebar to compare.")

with right:
    st.subheader("Country profile")

    default_profile = (
        highlight_countries[0] if highlight_countries else (all_countries[0] if all_countries else None)
    )
    profile_country = st.selectbox(
        "Profile country",
        all_countries,
        index=all_countries.index(default_profile) if default_profile in all_countries else 0,
    )

    row_view = plot_df[plot_df[COUNTRY_COL] == profile_country]
    row_full = df[df[COUNTRY_COL] == profile_country]

    if row_full.empty:
        st.warning("Country not found in the dataset.")
    else:
        if row_view.empty:
            st.info(
                "Selected country is not in the current filtered view for the chosen axes (missing X/Y data), "
                "but its profile is shown from the full dataset."
            )

        r = row_full.iloc[0]

        st.write(f"**{COUNTRY_COL}:** {r[COUNTRY_COL]}")
        if "stability_score" in df.columns:
            st.metric("Score (stability)", f"{r['stability_score']:.1f}" if pd.notna(r["stability_score"]) else "n/a")

        st.markdown("**Selected indicators:**")

        def show_metric(label, col):
            val = r.get(col, np.nan)
            st.write(f"- {label}: {'n/a' if pd.isna(val) else val}")

        show_metric(pretty_name(x_col), x_col)
        show_metric(pretty_name(y_col), y_col)

        if size_col is not None:
            show_metric(pretty_name(size_col), size_col)

        if color_col is not None and color_col not in (COUNTRY_COL,):
            show_metric(pretty_name(color_col), color_col)

        if not row_view.empty and "_is_outlier" in row_view.columns:
            is_out = bool(row_view["_is_outlier"].iloc[0])
            st.write("**Outlier (current view):** " + ("✅ Yes" if is_out else "No"))
        else:
            st.write("**Outlier (current view):** n/a")

# ---------------------------------------------------------
# 7. TABLE (LINKED)
# ---------------------------------------------------------

st.subheader("Country table")
st.caption("If you highlight countries, the table shows only those. Otherwise it shows the full filtered set.")

table_source = plot_df.copy()
if highlight_countries:
    table_source = table_source[table_source[COUNTRY_COL].isin(highlight_countries)].copy()

table_cols = [COUNTRY_COL, x_col, y_col]
if size_col is not None:
    table_cols.append(size_col)
if "stability_score" in table_source.columns and "stability_score" not in table_cols:
    table_cols.append("stability_score")
if "_is_outlier" in table_source.columns:
    table_cols.append("_is_outlier")

table_cols = list(dict.fromkeys(table_cols))
table_df = table_source[table_cols].copy()

# Pretty rename (unique)
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
# 8. WORLD MAP (FULL-WIDTH MAP, INFO BELOW)
# ---------------------------------------------------------

st.subheader("World map")

# ---- controls (above map, compact) ----
ctrl1, ctrl2, ctrl3 = st.columns([2.2, 1.2, 1.2])
with ctrl1:
    map_metric_pretty = st.selectbox(
        "Map color metric",
        options=["Highlight only"] + pretty_numeric_options,
        index=1 if len(pretty_numeric_options) > 0 else 0,
    )
with ctrl2:
    dim_non_highlighted = st.checkbox(
        "Dim non-highlighted to black",
        value=bool(highlight_countries),
    )
with ctrl3:
    use_log_scale = st.checkbox(
        "Log scale (better for GDP/population)",
        value=True,
    )

# ---- base data for map (FULL df, minus exclusions only) ----
map_base = df.copy()
if exclude_countries:
    map_base = map_base[~map_base[COUNTRY_COL].isin(exclude_countries)].copy()

map_df = map_base[[COUNTRY_COL]].drop_duplicates().copy()
map_df["map_name"] = map_df[COUNTRY_COL].apply(to_map_name)
map_df["iso3"] = map_df[COUNTRY_COL].apply(to_iso3)

# ---- build map values ----
if map_metric_pretty == "Highlight only":
    map_df["map_value"] = map_df[COUNTRY_COL].isin(highlight_countries).astype(int)
    color_scale = [[0.0, DARK_BG], [1.0, HIGHLIGHT_GREEN]]
    show_scale = False
    colorbar_title = None
else:
    metric_col = pretty_numeric[map_metric_pretty]
    vals = map_base[[COUNTRY_COL, metric_col]].copy()
    vals[metric_col] = pd.to_numeric(vals[metric_col], errors="coerce")
    vals = vals.groupby(COUNTRY_COL, as_index=False)[metric_col].mean()

    map_df = map_df.merge(vals, on=COUNTRY_COL, how="left")

    if dim_non_highlighted and highlight_countries:
        map_df["map_value"] = np.where(
            map_df[COUNTRY_COL].isin(highlight_countries),
            map_df[metric_col],
            np.nan,
        )
    else:
        map_df["map_value"] = map_df[metric_col]

    if use_log_scale:
        map_df["map_value"] = np.log10(map_df["map_value"] + 1)

    color_scale = CONTINUOUS_GREEN_SCALE
    show_scale = True
    colorbar_title = f"{map_metric_pretty}" + (" (log10+1)" if use_log_scale else "")

# ---- availability stats ----
available_count = int(map_df["map_value"].notna().sum())
missing_count = int(map_df["map_value"].isna().sum())

missing_for_metric = (
    map_df[pd.isna(map_df["map_value"])][COUNTRY_COL]
    .dropna()
    .astype(str)
    .sort_values()
    .tolist()
    if map_metric_pretty != "Highlight only"
    else []
)

# ---- split ISO vs fallback ----
iso_df = map_df.dropna(subset=["iso3"]).copy()
name_df = map_df[map_df["iso3"].isna()].copy()

# ---- build choropleth (ISO first) ----
map_fig = px.choropleth(
    iso_df,
    locations="iso3",
    color="map_value",
    hover_name=COUNTRY_COL,
    color_continuous_scale=color_scale,
)

for tr in map_fig.data:
    tr.coloraxis = "coloraxis"

# ---- fallback for non-ISO names ----
if not name_df.empty:
    fallback = px.choropleth(
        name_df,
        locations="map_name",
        locationmode="country names",
        color="map_value",
        hover_name=COUNTRY_COL,
        color_continuous_scale=color_scale,
    )
    for tr in fallback.data:
        tr.coloraxis = "coloraxis"
        tr.showlegend = False
    map_fig.add_traces(fallback.data)

# ---- highlight overlay (ISO only, guaranteed visible) ----
if highlight_countries:
    hi = pd.DataFrame({COUNTRY_COL: highlight_countries})
    hi["iso3"] = hi[COUNTRY_COL].apply(to_iso3)
    hi = hi.dropna(subset=["iso3"])
    if not hi.empty:
        overlay = px.choropleth(
            hi,
            locations="iso3",
            color_discrete_sequence=[HIGHLIGHT_GREEN],
        )
        for tr in overlay.data:
            tr.showlegend = False
            tr.hovertemplate = "%{location}<extra></extra>"
        map_fig.add_traces(overlay.data)

# Clean layout: full-width map, no Plotly colorbar (we’ll show a neat legend below)
map_fig.update_layout(
    margin=dict(l=0, r=0, t=10, b=0),
    geo=dict(
        showframe=False,
        showcoastlines=True,
        coastlinecolor="#333333",
        bgcolor="rgba(0,0,0,0)",
        landcolor=DARK_BG,
        oceancolor="#0b0f1a",
        showocean=True,
    ),
    coloraxis=dict(showscale=False),  # hide built-in colorbar
)

st.plotly_chart(
    map_fig,
    use_container_width=True,
    config={
        "displayModeBar": True,
        "scrollZoom": True,
        "displaylogo": False,
    },
)

# ---- Neat legend below the map (replaces the Plotly colorbar) ----
if map_metric_pretty == "Highlight only":
    st.caption("Legend: highlighted countries are shown in green; others are dark.")
else:
    # Show min/max from original (non-log) values so it’s interpretable
    metric_col = pretty_numeric[map_metric_pretty]
    raw_vals = map_base[[COUNTRY_COL, metric_col]].copy()
    raw_vals[metric_col] = pd.to_numeric(raw_vals[metric_col], errors="coerce")
    vmin = raw_vals[metric_col].min(skipna=True)
    vmax = raw_vals[metric_col].max(skipna=True)

    left_leg, right_leg = st.columns([3.0, 1.2])
    with left_leg:
        st.caption(
            f"**{map_metric_pretty}**" + (" (map shows log10(value+1))" if use_log_scale else "")
        )
    with right_leg:
        if pd.notna(vmin) and pd.notna(vmax):
            st.caption(f"min: **{vmin:,.2f}** · max: **{vmax:,.2f}**")
        else:
            st.caption("min/max: n/a")

# ---- Horizontal colorbar under the legend (GREEN, labels follow log toggle) ----
if map_metric_pretty != "Highlight only":
    greens_css = (
        "linear-gradient(90deg, "
        "#f7fcf5, #e5f5e0, #c7e9c0, #a1d99b, "
        "#74c476, #41ab5d, #238b45, #006d2c, #00441b)"
    )

    if pd.notna(vmin) and pd.notna(vmax):
        if use_log_scale:
            vmin_s = np.log10(vmin + 1)
            vmax_s = np.log10(vmax + 1)
            vmid_s = (vmin_s + vmax_s) / 2
            min_txt = f"{vmin_s:.2f}"
            mid_txt = f"{vmid_s:.2f}"
            max_txt = f"{vmax_s:.2f}"
        else:
            vmid = (vmin + vmax) / 2
            min_txt = f"{vmin:,.2f}"
            mid_txt = f"{vmid:,.2f}"
            max_txt = f"{vmax:,.2f}"
    else:
        min_txt, mid_txt, max_txt = "n/a", "n/a", "n/a"

    st.markdown(
        f"""
        <div style="margin-top:8px;">
          <div style="
              height:12px;
              width:100%;
              background:{greens_css};
              border-radius:8px;
              border:1px solid rgba(255,255,255,0.18);
          "></div>
          <div style="
              display:flex;
              justify-content:space-between;
              font-size:12px;
              opacity:0.85;
              margin-top:4px;
          ">
            <span>{min_txt}</span>
            <span>{mid_txt}</span>
            <span>{max_txt}</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ---- availability info BELOW the map ----
st.markdown("### Data availability")

k1, k2 = st.columns(2)
with k1:
    st.metric("With data", available_count)
with k2:
    st.metric("Missing", missing_count)

if map_metric_pretty != "Highlight only" and missing_for_metric:
    with st.expander("Show missing countries"):
        q = st.text_input("Search", placeholder="Type to filter...")
        shown = missing_for_metric
        if q.strip():
            q_up = q.strip().upper()
            shown = [c for c in shown if q_up in c.upper()]

        st.caption(f"{len(shown)} missing countries")
        st.dataframe(
            pd.DataFrame({"Missing countries": shown}),
            use_container_width=True,
            hide_index=True,
            height=300,
        )