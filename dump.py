import os
import pickle
import polars as pl
import re
import streamlit as st
from data import Progress
from rapidfuzz.process import cdist
from rapidfuzz.distance import DamerauLevenshtein, Levenshtein
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
        country = float(pro.get_flagged()/pro.get_size()[1]/100)
        # region = float(pro.get_region())
        # value = (country + region) / 2
    except (TypeError, ValueError):
        country = 0.0

    return max(0.0, country)



def preprocess(file, file_stem):
    try:
        df = pl.read_csv(file)\
            .with_row_index("Index")\
            .with_columns(split = pl.col("Location Original Text").str.split(r"\s*[,;.]+\s*"))\
            .with_columns(
                # if split array is = 1 split by space
                pl.when(pl.col("split").list.len() == 1)
                .then(pl.col("split").list.first().str.split(r"\s+"))
                .otherwise(pl.col("split"))
                .alias("Separated")
            ).drop("split")\
            .with_columns(
                # drop duplicates, ignoring case
                pl.col("Separated")
                .list.eval(pl.element().str.to_titlecase())
                .list.unique(maintain_order=True)
                .alias("Separated")
            )\
            .with_columns(
                # create flagged column
                pl.col("Separated").list.get(-1, null_on_oob=True).alias("To_analyze"),
                pl.lit(0).alias("Flagged"),
                pl.lit([]).alias("Analyzed"),
                pl.lit(-1).alias("Current_index")
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

        print (f"Queries {len(queries)}, Choices {len(choices)}")



        df = df.with_columns(
            pl.when(pl.col("To_analyze").is_not_null() & ~pl.col("To_analyze").is_in(co["all_names"]))
            .then(1)
            .otherwise(0)
            .alias("Flagged"),
            pl.col("Separated").list.slice(0, pl.col("Separated").list.len() - 1).alias("Separated")
        )\
        # .with_columns(
        #         Analyzed = pl.when(pl.col("Flagged") == 0)
        #             .then(
        #                 pl.concat_list([
        #                     pl.col("To_analyze").cast(pl.List(pl.Utf8)), 
        #                     pl.col("Analyzed").fill_null([])
        #                 ])
        #             )
        #             .otherwise(pl.col("Analyzed"))
        #     )\
        # .collect()


        flagged_df = (
            df.filter(pl.col("Flagged") == 1)
            .select("To_analyze")
            .unique()
        )

        queries = flagged_df["To_analyze"].to_list()
        choices = co["all_names"].to_list()

        if queries:
            mat = cdist(queries, choices, score_cutoff=.75, scorer=Levenshtein.normalized_similarity, workers=-1)
            max_idx = mat.argmax(axis=1)
            max_score = mat.max(axis=1)
            best_matches = pl.DataFrame({
                "To_analyze": queries,
                "Suggested_Raw": [choices[i] for i in max_idx],
                "MatchScore": max_score,
            })

            best_matches = best_matches.with_columns(
                pl.when(pl.col("MatchScore") >= 0.75)
                .then(pl.col("Suggested_Raw"))
                .otherwise(None)
                .alias("Suggested")
            ).drop("Suggested_Raw")

            print("HERE")

            df = df.join(best_matches, on="To_analyze", how="left")
        else:
            df = df.with_columns([
                pl.lit(None).alias("Suggested"),
                pl.lit(0.0).alias("MatchScore")
            ])



        df.write_parquet(f"{dir_name}/processed_data.parquet")

        with open(f"{dir_name}/progress.pkl", "rb") as f:
            progress = pickle.load(f)

        num = df.unique(["To_analyze"]).select(pl.col("Flagged").sum()).item()
        progress.set_unique_flagged(
            num
        )
        progress.set_flagged(df.select(pl.col("Flagged").sum()).item())

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
        "allCountries.parquet/*/*.parquet"
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







