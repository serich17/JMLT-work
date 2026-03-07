import streamlit as st
import polars as pl
from funcs import *


project_id = st.query_params.get("project_id")
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECTS_DIR = os.path.join(PARENT_DIR, "projects")


if "allPlaces" not in st.session_state:
    st.session_state.allPlaces = load_places()

if "countries" not in st.session_state:
    st.session_state["countries"] = load_countries()

if "project" not in st.session_state:
    st.session_state["project"] = load_project(project_id)



if not project_id:
    st.title("No Project Linked")
    st.button("Connect Project") and st.switch_page("main.py")
else:
    st.title(project_id)
    with open(f"{PROJECTS_DIR}/{project_id}/progress.pkl", "rb") as f:
        progress = pickle.load(f)
    
    # Show metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Unique Flagged", f"{progress.get_unique_flagged():,}", )
    col2.metric("Total Flagged", f"{progress.get_flagged():,}")
    col3.metric("Project Size (Rows)", f"{progress.get_size()[0]:,}")
    


    col1, col2 = st.columns(2)

   
    with col1:
        start = st.text_input("Contains...")
    with col2:
        option = st.selectbox(
            "Select a column to filter",
            list(filter(lambda x: x not in ["Index", "Flagged"], st.session_state.project.collect_schema().names())),
            placeholder="Country"
        )
    
    cols_selected = st.multiselect(
        "Select which columns to show in table",
        st.session_state.project.collect_schema().names(),
        default=["Original", "To_analyze"]
    )

    filtered = filter_prefix(st.session_state.project, option, start)

    edited_df = st.data_editor(
        filtered\
        .filter(~pl.col("Preprocess").str.contains(",") & pl.col("Preprocess").str.contains(" ")).collect(engine="streaming"),
        width="stretch",
        key="editor",
        column_order= cols_selected
    )

    # num_rows = "delete"
    # column_order =("one", "two")
    # column_config = {"col1":"Col 1"}
    # disabled=[cols that you can't edit]

    print(filtered\
        .filter(pl.col("Flagged")==1).unique(["To_analyze"]).filter(pl.col("Separated").list.len() > 1).collect(engine="streaming").describe())


    # df = st.session_state["allPlaces"]

    # st.dataframe(df.filter(pl.col("feature_code").str.contains("PCL")))
    # print(df.filter(pl.col("feature_code").str.contains("PCL")).collect().count())



    
    # df = df.with_columns([
    #     pl.col("latitude").cast(pl.Float64, strict=False),
    #     pl.col("longitude").cast(pl.Float64, strict=False)
    # ]).collect()

    # # # 3. Drop rows where coordinates are now null (optional but recommended)
    # # df = df.drop_nulls(subset=["latitude", "longitude"]).collect()

    # # # 4. Streamlit's st.map still prefers a Pandas DataFrame or a specific format, 
    # # # so we convert it back just for the map rendering.
    # st.map(df)


    
    # e1f = pl.scan_parquet(f"{PROJECTS_DIR}/{project_id}/processed_data.parquet")


    # # e1f = pl.scan_parquet("allCountries.parquet/**/*.parquet")\
    # #         .filter(pl.col("name_lower").str.starts_with("pla"))\

    # st.dataframe(e1f)

