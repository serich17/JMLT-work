import os
import pickle
import polars as pl
import re
import streamlit as st
from data import Progress
from rapidfuzz.process import cdist
from rapidfuzz.distance import Levenshtein
import time

# Root directory where projects are stored
PROJECTS_DIR = f"{os.path.dirname(os.path.abspath(__file__))}/projects"
SAVE_INTERVAL = 180  # seconds
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
        co = st.session_state.countries["alternateName"].to_list()
        country_pattern = ""
        for c in co:
            country_pattern += f"|{c}"
        country_pattern = country_pattern[1:]

        pat_full_repeat = re.compile(r"^((?:[^,]+,\s*)*[^,]+)(?:,\s*\1)+$", re.I) # if the whole string repeats itself twice, replace with first match
        pat_phrase_repeat = re.compile(r"(\b[a-zA-Z\s]{5,}\b),\s+(\1)(?!,)", re.I) # if a word repeats itself, but the second match doesn't have a comma, replace the whole thing with the first
        pat_period = re.compile(r"(?i)(?:\.(?![a-z]\s)|(?<=\.[a-z])\s)(?=[^.]{3,}(?:\.|$))") # replace all periods that have three or more characters inbetween with a comma

        df = pl.read_csv(file, encoding="utf8-lossy")\
            .with_row_index("Index")\
            .with_columns(
                pl.col("Location Original Text")
                .map_elements(
                    lambda x: (
                        pat_period.sub(
                            ",",
                        pat_phrase_repeat.sub(
                            r"\1,",
                            pat_full_repeat.sub(r"\1", x)
                        )
                        ) if x is not None else None
                    ),
                    return_dtype=pl.Utf8
                )
                .alias("Preprocess")
            )\
            .with_columns(
                pl.col("Preprocess").str.replace_all(rf"(?i)(^|[^,\s])\s*({country_pattern})$", r"$1, $2") # if one of the countries is on the end of the string, add a comma beforehand
            )\
            .with_columns(pl.col("Preprocess").str.replace_all(r"\s*,\s*", ",").str.split(",").alias("split") #, then split into an array
                          )\
            .with_columns(
                # if split array is = 1 split by space or title case
                pl.when(pl.col("split").list.len() == 1)
                    .then(
                        pl.when(
                            pl.col("split")
                            .list.first()
                            .str.contains(r"[a-z][A-Z]")
                        )
                        .then(
                            pl.col("split")
                            .list.first()
                            .str.replace_all(r"\s+", "")
                            .str.replace_all(r"([a-z])([A-Z])", r"$1,$2")
                            .str.split(",")
                        )
                        .otherwise(
                            pl.col("split")
                            .list.first()
                            .str.split(r"\s+")
                        )
                    )
                    .otherwise(pl.col("split"))
                    .alias("Separated")
            ).drop("split")\
            .with_columns(
                # drop duplicates, ignoring case
                pl.col("Separated")
                .list.eval(pl.element().str.to_titlecase().filter(pl.element() != ""))
                .list.unique(maintain_order=True)
                .alias("Separated")
            )\
            .drop("Preprocess")\
            .with_columns(
                # create flagged column
                pl.col("Separated").alias("Initial_separated"),
                pl.col("Separated").list.get(-1, null_on_oob=True).alias("To_analyze"),
                pl.lit(0).alias("Flagged"),
                pl.lit([]).alias("Analyzed"),
                pl.lit(-1).alias("Current_index")
            )\
            # .with_columns(pl.col("To_analyze").str.replace_all(r"(?i)\b([A-Za-z ]{3,}?)\s*(?:(?<=\S)\s*)\1\b", r"${1}")).alias("To_analyze")
        df = df.rename({"Location Original Text": "Original"})

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
        # USING 2000 alternate name file instead 

        df = pl.scan_parquet(f"{dir_name}/processed_data.parquet")
        # co = pl.read_parquet("country.parquet/**/*.parquet")
        print("right here")
        # places = st.session_state.allPlaces.select("alternateName").collect(engine="streaming").to_series()
        # print(places.limit(5))

        co = st.session_state.countries


        print("Country count: ", co.count())

        co_map = {name.lower(): name for name in co["alternateName"] if name is not None}

        df = df.with_columns(
            pl.when(pl.col("To_analyze").is_not_null() & ~pl.col("To_analyze").str.to_lowercase().is_in(co_map.keys()))
            .then(1)
            .otherwise(0)
            .alias("Flagged"),
            pl.col("Separated").list.slice(0, pl.col("Separated").list.len() - 1).alias("Separated"), # remove the last item from the list
            pl.col("To_analyze")\
            .str.to_lowercase()\
            .replace(co_map, default=pl.col("To_analyze")).alias("To_analyze")
        )\
        .with_columns(
                Analyzed = pl.when(pl.col("Flagged") == 0)
                    .then(
                        pl.concat_list([
                            pl.col("To_analyze").cast(pl.List(pl.Utf8)), 
                            pl.col("Analyzed").fill_null([])
                        ])
                    )
                    .otherwise(pl.col("Analyzed")),
                To_analyze = pl.when(pl.col("Flagged") == 0)
                    .then(None).otherwise(pl.col("To_analyze"))
            )\
        .collect()


        flagged_df = (
            df.filter(pl.col("Flagged") == 1)
            .select("To_analyze")
            .unique()
        )

        queries = flagged_df["To_analyze"].to_list()
        choices = co["alternateName"].unique().to_list()

        print (f"Queries {len(queries)}, Choices {len(choices)}")

        if queries:
            mat = cdist(queries, choices, scorer=Levenshtein.normalized_similarity, workers=-1, scorer_kwargs={"weights":(3,1,2)})
            max_idx = mat.argmax(axis=1)
            max_score = mat.max(axis=1)
            best_matches = pl.DataFrame({
                "To_analyze": queries,
                "Suggested_Raw": [choices[i] for i in max_idx],
                "MatchScore": max_score,
            })

            best_matches = best_matches.with_columns(
                pl.when(pl.col("MatchScore") >= 0)
                .then(pl.col("Suggested_Raw"))
                .otherwise(None)
                .alias("Suggested")
            ).drop("Suggested_Raw")

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

    except Exception as e:
        print(e)


# @st.cache_resource
# def load_countries():
#     if "allPlaces" not in st.session_state:
#         st.session_state.allPlaces = load_places()
#     return st.session_state.allPlaces\
#             .filter(pl.col("feature_code").str.contains("PCL"))\
#             .with_columns(
#                 all_names = pl.concat_list(
#                     pl.col("name"),
#                     pl.col("asciiname"),
#                     pl.col("alternatenames").str.split(",")
#                 ).list.unique()
#             )\
#             .explode("all_names")\
#             .collect()

@st.cache_resource
def load_countries():
    return pl.scan_parquet("colonialCountries/*/*.parquet").collect()


@st.cache_resource
def load_places():
    return pl.scan_parquet(
        "allCountries2/*/*.parquet", hive_partitioning=True
    )\
    .filter(pl.col("feature_class").is_in(["P", "A"]))


def load_project(dir: str):
    Lock = st.session_state.lock
    with Lock:
        df = (pl.scan_parquet(f"{PROJECTS_DIR}/{dir}/processed_data.parquet"))
    return df 

def filter_prefix(lf: pl.LazyFrame, column: str, prefix: str, isUnique: bool, onlyFlagged: bool, single_list: bool) -> pl.DataFrame:
    if len(prefix) > 0:
        lf = lf.filter(pl.col(column).cast(pl.String).str.to_lowercase().str.contains(prefix.lower()))
    if onlyFlagged:
        lf = lf.filter(pl.col("Flagged") == 1)
    if isUnique:
        lf = lf.with_columns(
            pl.col("To_analyze").count().over("To_analyze").alias("Unique Count")
        )\
        .unique(["To_analyze"])
    if single_list:
        lf = lf.filter(pl.col("Initial_separated").list.len() < 2)
    return lf


def filter_update(filter_state):
    option = filter_state[0]
    start = filter_state[1]
    isUnique = filter_state[2]
    onlyFlagged = filter_state[3]
    single_item = filter_state[4]
    if "last_filter_state" not in st.session_state:
        st.session_state.last_filter_state = None
    if filter_state != st.session_state.last_filter_state:
        st.session_state.base_df = filter_prefix(
            st.session_state.project_df.lazy(),
            option,
            start,
            isUnique,
            onlyFlagged,
            single_item
        ).collect(engine="streaming")
        st.session_state.last_filter_state = filter_state

def update_changed_df(projectid, change_same, progress):
    Lock = st.session_state.lock
    with st.session_state.save_lock:
        pending = st.session_state.pending_changes
        deletes = st.session_state.pending_deletes

        if not pending and not deletes:
            return

        df = st.session_state.project_df

          # Build a map of old To_analyze value → new value from the pending changes
        to_analyze_map = {}
        for index_val, changes in pending.items():
            if "To_analyze" in changes:
                original_val = (
                    df.filter(pl.col("Index") == index_val)
                    .select("To_analyze")
                    .collect()
                    .item()
                )
                to_analyze_map[original_val] = changes["To_analyze"]

        if change_same:
            # Expanded indexes = all rows that had a matching original value
            with Lock:
                expanded_indexes = (
                    df.filter(pl.col("To_analyze").is_in(list(to_analyze_map.keys())))
                    .select("Index")
                    .collect()
                    .to_series()
                    .to_list()
                )
            
            # Apply To_analyze changes to all rows matching the original value
            for old_val, new_val in to_analyze_map.items():
                df = df.with_columns(
                    pl.when(pl.col("To_analyze") == old_val)
                    .then(pl.lit(new_val))
                    .otherwise(pl.col("To_analyze"))
                    .alias("To_analyze")
                )
            
        else:
            # Only apply To_analyze changes to the original indexes
            for index_val, changes in pending.items():
                if "To_analyze" in changes:
                    df = df.with_columns(
                        pl.when(pl.col("Index") == index_val)
                        .then(pl.lit(changes["To_analyze"]))
                        .otherwise(pl.col("To_analyze"))
                        .alias("To_analyze")
                    )
            expanded_indexes = list(pending.keys())

        # Apply all other column changes only to original indexes
        for index_val, changes in pending.items():
            for col, val in changes.items():
                if col == "To_analyze":
                    continue
                df = df.with_columns(
                    pl.when(pl.col("Index") == index_val)
                    .then(pl.lit(val))
                    .otherwise(pl.col(col))
                    .alias(col)
                )

        for index_val in deletes:
            df = df.filter(pl.col("Index") != index_val)

        df = fix_flagged(expanded_indexes, df, False, False, False, False, "")  # change_same already handled above
        with Lock:
            df = df.collect(engine="streaming")
        save_file(df, projectid)
        st.session_state.project_df = load_project(projectid)
        st.session_state.pending_changes = {}
        st.session_state.pending_deletes = set()
        st.session_state["editor"]["edited_rows"] = {}
        st.session_state["editor"]["deleted_rows"] = []
        with Lock:
            st.session_state.base_df = filter_prefix(
                st.session_state.project_df,
                st.session_state.last_filter_state[0],
                st.session_state.last_filter_state[1],
                st.session_state.last_filter_state[2],
                st.session_state.last_filter_state[3],
                st.session_state.last_filter_state[4]
            ).collect(streaming=True)
        update_progress(Lock, progress, projectid, df)

def safe_write_parquet(df, path):
    tmp = path + ".tmp"
    df.write_parquet(tmp)
    os.replace(tmp, path)

def save_file(df, projectid):
    Lock = st.session_state.lock
    with Lock:
        safe_write_parquet(
            df,
            f"{PROJECTS_DIR}/{projectid}/processed_data.parquet"
        )
        st.session_state.last_save = time.time()



def options(keys):
    # Choose what to see
    key3 = keys[0]
    key4 = keys[1]

    col1, col2 = st.columns(2)
    with col1:
        start = st.text_input("Contains...", key=key3)
    with col2:
        search_cols = list(filter(lambda x: x not in ["Flagged"], st.session_state.project_df.collect_schema()))
        option = st.selectbox(
            "Select a column to filter",
            search_cols,
            index= search_cols.index("To_analyze"),
            key=key4
        )
            
    return (option, start)

def approve_rows(indexes, change_same, projectid, filter_state, progress, suggested, prepend, swap=False, swap_val=""):
    Lock = st.session_state.lock
    df = fix_flagged(indexes, st.session_state.project_df, suggested, change_same, prepend, swap, swap_val)


    with Lock:
        df = df.collect(engine="streaming")

    save_file(df, projectid)
    st.session_state.project_df = load_project(projectid)
    with Lock:
        st.session_state.base_df = filter_prefix(
                st.session_state.project_df,
                filter_state[0],
                filter_state[1],
                filter_state[2],
                filter_state[3],
                filter_state[4]
            ).collect(engine="streaming")
    st.session_state.last_filter_state = filter_state

    update_progress(Lock, progress, projectid, df)

def fix_flagged(indexes, df, suggested, change_same, prepend, swap, swap_val):
    Lock = st.session_state.lock
    base_mask = pl.col("Index").is_in(indexes)

    if change_same:
        with Lock:
            vals = (
                df.filter(base_mask)\
                .select("To_analyze")
                .unique()
                .collect()
                .to_series()
                .to_list()
            )

        matched_indexes = df.filter(pl.col("To_analyze").is_in(vals))\
            .select("Index")\
            .collect()\
            .to_series()\
            .to_list()
        mask = pl.col("Index").is_in(matched_indexes)
    else:
        mask = base_mask
    
    if swap:
        df = df.with_columns(
            pl.when(mask)
            .then(
                pl.lit(swap_val)
            )
            .otherwise(pl.col("To_analyze"))
            .alias("To_analyze")
        )

    if prepend:
        df = df.with_columns(
            pl.when(mask)
            .then(
                pl.concat_list(
                    "Separated",
                    "To_analyze"
                )
            )
            .otherwise(pl.col("Separated"))
            .alias("Separated"),
        )
        
    else:
        if suggested:
            df = df.with_columns(
                pl.when(mask)
                .then(
                    pl.concat_list([
                        pl.col("Suggested"),
                        pl.col("Analyzed")
                    ])
                )
                .otherwise(pl.col("Analyzed"))
                .alias("Analyzed"),
            )
        else:
            df = df.with_columns(
                pl.when(mask)
                .then(
                    pl.concat_list([
                        pl.col("To_analyze"),
                        pl.col("Analyzed")
                    ])
                )
                .otherwise(pl.col("Analyzed"))
                .alias("Analyzed")
            )

    df = df.with_columns(
        [
            pl.when(mask)
            .then(None)
            .otherwise(pl.col("To_analyze"))
            .alias("To_analyze"),
            pl.when(mask)
            .then(None)
            .otherwise(pl.col("Suggested"))
            .alias("Suggested"),
            pl.when(mask)
            .then(0)
            .otherwise(pl.col("Flagged"))
            .alias("Flagged")
        ]
    )
    return df

def update_progress(Lock, progress, projectid, df):
    with Lock:
        counts = (
            df.filter(pl.col("Flagged") == 1)
            .select(
                total_flagged=pl.len(),
                unique_flagged=pl.col("To_analyze").n_unique()
            )
        )
    progress.set_flagged(counts["total_flagged"][0])
    progress.set_unique_flagged(counts["unique_flagged"][0])
    with st.session_state.lock_class:
        with open(f"{PROJECTS_DIR}/{projectid}/progress.pkl", "wb") as f:
            pickle.dump(progress, f)

def prep_df_map():
    df = st.session_state["allPlaces"]
    levels = ["ADM1", "ADM2", "ADM3", "ADM4", "ADM5", "ADMD", "PPL", "PPLA", "PPLA2", "PPLA3", "PPLA4", "PPLA5", "PPLC", "PPLF", "PPLL", "PPLR", "PPLS", "PPLX", "STLMT"]
    with st.expander("Filter Options"):
        admins = st.multiselect(
            "Administrative Level Codes",
            options= levels,
            default= levels,
        )
        st.markdown("[Admin Code Lookup](https://www.geonames.org/export/codes.html)")
    par1, par2 = st.columns(2)
    with par1:
        type_search = st.selectbox(
            "Search Type",
            ["Starts With", "Ends With", "Contains", "Exact Match"],
            index= 3
        )
    with par2:
        mapped = st.text_input("Search Key", key=15654564564)
    df = df.drop_nulls(subset=["latitude", "longitude"]).filter(pl.col("feature_code").is_in(admins))
    match type_search:
        case "Starts With":
            df = df.filter(pl.col("name_lower").str.starts_with(mapped.lower()))
        case "Ends With":
            df = df.filter(pl.col("name_lower").str.ends_with(mapped.lower()))
        case "Contains":
            df = df.filter(pl.col("name_lower").str.contains(mapped.lower()))
        case "Exact Match":
            df = df.filter(pl.col("name_lower") == (mapped.lower()))
    return df



def transition_next_admin_settings(project_id):
    levels = ["ADM1", "ADM2", "ADM3", "ADM4", "ADM5", "ADMD", "PPL", "PPLA", "PPLA2", "PPLA3", "PPLA4", "PPLA5", "PPLC", "PPLF", "PPLL", "PPLR", "PPLS", "PPLX", "STLMT"]
    with st.expander("String matcher"):
        st.text("Set how small to filter the checking database to")
        st.warning("If the set is too big, it will overwhelm the ram and cpu with the rapidfuzz checks")
        admins = st.multiselect(
            "Administrative Level Codes",
            options= levels,
            default= [],
        )
        st.session_state.transition_state = (admins, 5)
        if st.button("Check sizes"):
            with open(f"projects/{project_id}/progress.pkl", "rb") as f:
                pro = pickle.load(f)
            row_count = st.session_state.allPlaces\
                .filter(pl.col("feature_code").is_in(st.session_state.transition_state[0])).select(pl.col("alternateName")).select(pl.count()).collect().item()
            st.text(f"df: {row_count} X flagged: {pro.get_unique_flagged()} = {pro.get_unique_flagged()*row_count:,}")

        if len(admins) > 0:
            st.button("Run matcher", on_click=matcher, args=[project_id])
        st.markdown("[Admin Code Lookup](https://www.geonames.org/export/codes.html)")
    with st.expander("Increase Admin level/refill To_analyze"):
        st.text("Move to next admin level, and pull the next item on the queue out of Separated and onto To_analyze")
        st.text("When it pulls out the next item, it will check if it is in the places database, and if so it will automatically put it in Analyzed")
        st.button("Next transition", on_click=next_admin_transition, args=[project_id])
        pass

    st.session_state.transition_state = (admins, 5) # 5 is unneeded


def next_admin_transition(projectid):
    project = st.session_state.project_df
    admin_db = st.session_state.allPlaces.select(pl.col("alternateName")).collect(engine="streaming")["alternateName"].str.to_lowercase()
    with st.session_state.lock_class:
        with open(f"{PROJECTS_DIR}/{projectid}/progress.pkl", "rb") as f:
            progress = pickle.load(f)


    df = project\
        .with_columns(
            # When to_analyze is none so if someone does it before everything is analyzed on the previous it won't overwrite them
            pl.lit(1).alias("Flagged"),
            pl.when(pl.col("To_analyze").is_null())
            .then(pl.col("Separated").list.get(-1, null_on_oob=True))
            .otherwise(pl.col("To_analyze"))
            .alias("To_analyze")
        )\
        .with_columns(
            pl.col("Separated").list.slice(0, pl.col("Separated").list.len() - 1).alias("Separated"),
            # pl.col("To_analyze")\
            #     .str.to_lowercase()\
            #     .replace(admin_db, default=pl.col("To_analyze")).alias("To_analyze")
        )\
        .with_columns(
            pl.when(pl.col("To_analyze").str.to_lowercase().is_in(admin_db))
            .then(pl.lit(0))
            .otherwise(pl.col("Flagged"))
            .alias("Flagged")
        )\
        .with_columns(
            Analyzed = pl.when(pl.col("Flagged") == 0)
                    .then(
                        pl.concat_list([
                            pl.col("To_analyze").cast(pl.List(pl.Utf8)), 
                            pl.col("Analyzed").fill_null([])
                        ])
                    )
                    .otherwise(pl.col("Analyzed")),
            
        )\
        .with_columns(
            To_analyze = pl.when(pl.col("Flagged") == 0)
                .then(None).otherwise(pl.col("To_analyze"))
        )

        # .with_columns(
        #     pl.when(pl.col("Flagged") == 0 & pl.col("To_analyze").is_not_null())
        #     .then()
        # )\


    df = df.collect(engine="streaming")

    save_file(df, projectid)

    num = df.unique(["To_analyze"]).select(pl.col("Flagged").sum()).item()
    progress.set_unique_flagged(
        num
    )
    progress.set_flagged(df.select(pl.col("Flagged").sum()).item())
    progress.set_curr_level()

    with st.session_state.lock_class:
        with open(f"{PROJECTS_DIR}/{projectid}/progress.pkl", "wb") as f:
            pickle.dump(progress, f)


def matcher(projectid):
    state = st.session_state.transition_state
    df = st.session_state.project_df
    admin_db = st.session_state.allPlaces\
        .filter(pl.col("feature_code").is_in(state[0])).select(pl.col("alternateName")).collect(engine="streaming")["alternateName"]
    admin_db = {x.lower():x for x in admin_db.to_list()}
    flagged_df = (
            df.filter(pl.col("Flagged") == 1)
            .select("To_analyze")
            .unique()
        ).collect()

    queries = flagged_df["To_analyze"].to_list()
    choices = list(admin_db.values())

    print (f"Queries {len(queries)}, Choices {len(choices)}")

    if queries:
        mat = cdist(queries, choices, scorer=Levenshtein.normalized_similarity, workers=-1, scorer_kwargs={"weights":(3,1,5)})
        max_idx = mat.argmax(axis=1)
        max_score = mat.max(axis=1)
        best_matches = pl.DataFrame({
            "To_analyze": queries,
            "Suggested_Raw": [choices[i] for i in max_idx],
            "MatchScore": max_score,
        })
        best_matches = best_matches.with_columns(
            pl.when(pl.col("MatchScore") >= 0)
            .then(pl.col("Suggested_Raw"))
            .otherwise(None)
            .alias("Suggested")
        ).drop("Suggested_Raw")
        df = df.drop(["Suggested", "MatchScore"]).join(best_matches.lazy(), on="To_analyze", how="left")
    else:
        df = df.with_columns([
            pl.lit(None).alias("Suggested"),
            pl.lit(0.0).alias("MatchScore")
        ])

    df = df.collect(engine="streaming")

    save_file(df, projectid)