import os
import pickle
import polars as pl
import re
import streamlit as st
from data import Progress
from rapidfuzz.process import cdist
from rapidfuzz.distance import DamerauLevenshtein
import numpy as np

# Root directory where projects are stored
PROJECTS_DIR = f"{os.path.dirname(os.path.abspath(__file__))}/projects"

def load_progress(project_path):
    pkl = os.path.join(PROJECTS_DIR, project_path, "progress.pkl")
    if not os.path.exists(pkl):
        return 0.0

    with open(pkl, "rb") as f:
        pro = pickle.load(f)

    try:
        country = float(pro.get_country()/pro.get_size()[1]/100)
        # region = float(pro.get_region())
        # value = (country + region) / 2
    except (TypeError, ValueError):
        country = 0.0

    return max(0.0, country)



def preprocess(file, file_stem):
    try:
        df = pl.read_csv(file)\
            .with_row_index("Index")\
            .with_columns(
                pl.col("Location Original Text").map_elements(
                    lambda x: re.compile(r"^((?:[^,]+,\s*)*[^,]+)(?:,\s*\1)+$").sub(r"\1", x) if x is not None else None, return_dtype=pl.Utf8
                    ),
                pl.col("Location Original Text").str.replace_all(r"\s*,\s*", ",").str.split(",").alias("loc_array")
                )\
            .select([
                pl.col("Index"),
                pl.col("Folder Name").alias("Folder"),
                pl.col("Location Original Text").alias("Original"),
                pl.col("loc_array").list.get(pl.lit(-4), null_on_oob=True).alias("Village"),
                pl.col("loc_array").list.get(pl.lit(-3), null_on_oob=True).alias("District"),
                pl.col("loc_array").list.get(pl.lit(-2), null_on_oob=True).alias("Region"),
                pl.col("loc_array").list.get(pl.lit(-1), null_on_oob=True).alias("Country")
            ])\
            .with_columns(
                pl.col("Village").str.replace_all(r"[^\p{L}\s]", "").str.strip_chars().str.to_titlecase(),
                pl.col("District").str.replace_all(r"[^\p{L}\s]", "").str.strip_chars().str.to_titlecase(),
                pl.col("Region").str.replace_all(r"[^\p{L}\s]", "").str.strip_chars().str.to_titlecase(),
                pl.col("Country").str.replace_all(r"[^\p{L}\s]", "").str.strip_chars().str.to_titlecase(),
                pl.lit(0).alias("Flagged")
            )

        # Create subdirectory and save new project inside
        if not file_stem:
            raw_filename = file.name
            file_stem = os.path.splitext(raw_filename)[0]
        dir_name = f"{PROJECTS_DIR}/{file_stem}"
        os.makedirs(dir_name, exist_ok=True)
        df.write_parquet(f"{dir_name}/processed_data.parquet")


        pro = Progress(df.shape, file_stem)
        with open(f"{dir_name}/progress.pkl", "wb") as f:
            pickle.dump(pro, f)
        
        country_check(dir_name)

        # st.table(df.head())
        
    except Exception as e:
        print("Error in Preprocessing")
        print(e)



# --- Helper Function: Generate Circular Progress SVG ---
def get_progress_svg(percentage, size=80, color="#4CAF50"):
    """
    Generates an SVG string for a circular progress bar.
    """
    radius = 30
    circumference = 2 * 3.14159 * radius
    stroke_dashoffset = circumference - (percentage / 100) * circumference
    
    return f"""
    <svg width="{size}" height="{size}" viewBox="0 0 {size} {size}">
        <circle cx="{size/2}" cy="{size/2}" r="{radius}" 
            fill="none" stroke="#e6e6e6" stroke-width="6" />
        <circle cx="{size/2}" cy="{size/2}" r="{radius}" 
            fill="none" stroke="{color}" stroke-width="6"
            stroke-dasharray="{circumference}"
            stroke-dashoffset="{stroke_dashoffset}"
            transform="rotate(-90 {size/2} {size/2})" 
            stroke-linecap="round" />
        <text x="50%" y="50%" text-anchor="middle" dy=".3em" 
            font-size="14px" fill="white" font-family="sans-serif" font-weight="bold">
            {int(percentage)}%
        </text>
    </svg>
    """

# --- Helper Function: Create Clickable Card ---
def clickable_card(title, progress, target_page):
    """
    Renders a clickable card using HTML/CSS.
    target_page: The URL relative path (e.g., 'project_alpha')
    """
    # Define colors based on progress (optional)
    color = "#4CAF50" if progress >= 75 else "#2196F3" if progress >= 50 else "#FFC107"
    
    svg_code = get_progress_svg(progress, size=80, color=color)
    
    # CSS for the card
    card_style = """
    <style>
        .card-link {
            text-decoration: none !important;
            color: inherit !important;
        }
        .project-card {
            border: 1px solid #e0e0e0;
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 15px;
            background-color: black; /*white before*/
            box-shadow: 0 2px 5px rgba(0,0,0,0.05);
            transition: transform 0.2s ease, box-shadow 0.2s ease;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            height: 180px;
            text-align: center;
        }
        .project-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
            border-color: #2196F3;
        }
        .card-title {
            margin-top: 15px;
            font-family: sans-serif;
            font-weight: 600;
            font-size: 16px;
            color: white; /*#333 before*/
        }
    </style>
    """
    
    # The HTML Structure
    html_code = f"""
    {card_style}
    <a href="{target_page}" target="_self" class="card-link">
    <div class="project-card">
    {svg_code}
    <div class="card-title">{title}</div>
    </div>
    </a>
    """
    
    st.markdown(html_code, unsafe_allow_html=True)




def country_check(dir_name):
    try:
        df = pl.scan_parquet(f"{dir_name}/processed_data.parquet")
        # co = pl.read_parquet("country.parquet/**/*.parquet")

        co = st.session_state.countries


        print("Country count: ", co.count())


        df = df.with_columns(
            pl.when(~pl.col("Country").is_in(co["all_names"]))
            .then(1)
            .otherwise(0)
            .alias("Flagged")
        ).collect()


        flagged_df = (
            df.filter(pl.col("Flagged") == 1)
            .select("Country")
            .unique()
        )

        queries = flagged_df["Country"].to_list()
        choices = co["all_names"].to_list()

        if queries:
            mat = cdist(queries, choices, score_cutoff=.75, scorer=DamerauLevenshtein.normalized_similarity, workers=-1)
            max_idx = mat.argmax(axis=1)
            max_score = mat.max(axis=1)
            best_matches = pl.DataFrame({
                "Country": queries,
                "Suggested_Raw": [choices[i] for i in max_idx],
                "MatchScore": max_score,
            })

            best_matches = best_matches.with_columns(
                pl.when(pl.col("MatchScore") >= 0.75)
                .then(pl.col("Suggested_Raw"))
                .otherwise(None)
                .alias("Suggested")
            ).drop("Suggested_Raw")

            df = df.join(best_matches, on="Country", how="left")
        else:
            df = df.with_columns([
                pl.lit(None).alias("Suggested"),
                pl.lit(0.0).alias("MatchScore")
            ])


        # trying to go through all the values that didn't find a match and put the single values in the next column over and split the other
        # values by space then put them in their columns
        df = df\
            .with_columns(
                pl.when((pl.col("Flagged") == 1) & (pl.col("MatchScore") == 0))
                .then(pl.col("Original").str.split(" "))
                .otherwise(None)
                .alias("loc_array")
            )\
            .with_columns(
                pl.when(pl.col("loc_array").is_not_null())
                .then(pl.when(pl.col("loc_array").list.len() == 1)
                      .then(pl.col("loc_array").list.get(pl.lit(-1), null_on_oob=True))
                      .otherwise(pl.col("loc_array").list.get(pl.lit(-2), null_on_oob=True)))
                .otherwise(pl.col("Region")).alias("Region"),

                pl.when(pl.col("loc_array").is_not_null())
                .then(pl.when(pl.col("loc_array").list.len() > 1)
                      .then(pl.col("loc_array").list.get(pl.lit(-1), null_on_oob=True))
                      .otherwise(None))
                .otherwise(pl.col("Country")).alias("Country"),

                pl.when(pl.col("loc_array").is_not_null())
                .then(pl.when(pl.col("loc_array").list.len() > 2)
                      .then(pl.col("loc_array").list.get(pl.lit(-3), null_on_oob=True))
                      .otherwise(None))
                .otherwise(pl.col("District")).alias("District"),

                pl.when(pl.col("loc_array").is_not_null())
                .then(pl.when(pl.col("loc_array").list.len() > 3)
                      .then(pl.col("loc_array").list.get(pl.lit(-4), null_on_oob=True))
                      .otherwise(None))
                .otherwise(pl.col("Village")).alias("Village")
            ).with_columns(pl.when(pl.col("Flagged").is_null()).then(pl.lit(0)).otherwise(pl.col("Flagged")))
        

        flagged_df = (
            df.filter((pl.col("Flagged") == 1) & (pl.col("Country").is_not_null()) & (pl.col("MatchScore") == 0))
            .select("Country")
            .unique()
        )

        queries = flagged_df["Country"].to_list()

        if queries:
            mat = cdist(queries, choices, score_cutoff=.75, scorer=DamerauLevenshtein.normalized_similarity, workers=-1)
            max_idx = mat.argmax(axis=1)
            max_score = mat.max(axis=1)
            best_matches = pl.DataFrame({
                "Country": queries,
                "Suggested_Raw": [choices[i] for i in max_idx],
                "MatchScore": max_score,
            })

            best_matches = best_matches.with_columns(
                pl.when(pl.col("MatchScore") >= 0.75)
                .then(pl.col("Suggested_Raw"))
                .otherwise(None)
                .alias("Suggested")
            ).drop("Suggested_Raw")

            df = df.join(best_matches, on="Country", how="left")



        df.write_parquet(f"{dir_name}/processed_data.parquet")

        with open(f"{dir_name}/progress.pkl", "rb") as f:
            progress = pickle.load(f)

        num = df.unique(["Country"]).select(pl.col("Flagged").sum()).item()
        progress.set_country(
            num
        )

        with open(f"{dir_name}/progress.pkl", "wb") as f:
            pickle.dump(progress, f)
        
        # print("COUNTRY PRINTED", progress.get_country())
        # print("NUM", num)
        # st.dataframe(df.show(5))
    except Exception as e:
        print(e)


@st.cache_resource
def load_countries():
    if "allPlaces" not in st.session_state:
        st.session_state.allPlaces = load_places()
    return st.session_state.allPlaces\
            .filter(pl.col("feature_code").str.contains("PCL"))\
            .with_columns(
                all_names = pl.concat_list(
                    pl.col("name"),
                    pl.col("asciiname"),
                    pl.col("alternatenames").str.split(",")
                ).list.unique()
            )\
            .explode("all_names")\
            .collect()


@st.cache_resource
def load_places():
    return pl.scan_parquet(
        "allCountries/*/*.parquet"
    )\
    .filter(pl.col("feature_class").is_in(["P", "A"]))


@st.cache_resource
def load_project(dir: str):
    return pl.scan_parquet(f"{PROJECTS_DIR}/{dir}/processed_data.parquet")

def filter_prefix(lf: pl.LazyFrame, column: str, prefix: str) -> pl.DataFrame:
    if len(prefix) > 0:
        return (
            lf
            .filter(pl.col(column).str.to_lowercase().str.contains(prefix.lower()))
        )
    else:
        return lf







