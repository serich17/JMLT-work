# Import all libraries needed
import polars as pl
import streamlit as st
import os
from funcs import *

# Load lazyframes

if "allPlaces" not in st.session_state:
    st.session_state.allPlaces = load_places()
    print(st.session_state.allPlaces.describe())

if "countries" not in st.session_state:
    st.session_state["countries"] = load_countries()

if "file_active" not in st.session_state:
    st.session_state.file_active = False
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0


# Root directory where projects are stored
PROJECTS_DIR = f"{os.path.dirname(os.path.abspath(__file__))}/projects"


st.title("Open a Project")

projects = [
    d for d in os.listdir(PROJECTS_DIR)
    if os.path.isdir(os.path.join(PROJECTS_DIR, d))
]

# Create a Grid Layout
cols = st.columns(3)

for i, project in enumerate(projects):
    with cols[i % 3]:
        clickable_card(
            title=project,
            progress=load_progress(project),
            target_page=f'/project?project_id={project}' # This must match your Streamlit page URL
        )







uploaded_file = st.file_uploader(
    "Upload CSV", 
    type="csv", 
    accept_multiple_files=False,
    key=f"uploader_{st.session_state.uploader_key}"
)


if uploaded_file:
    st.session_state.file_active = True
    name = st.text_input("Project Name")
    if st.button("Create Project"):
        st.write(f"Processing {name}...")
        # Do preprocessing here
        preprocess(uploaded_file, name)
        st.session_state.uploader_key += 1
        st.rerun()
        st.success("Created Project")

else:
    st.session_state.file_active = False
    st.info("Please upload a file to begin")




# if DATA_CONNECTED:
#     df = pl.scan_parquet("allCountries.parquet/**/*.parquet")\
#         .filter(pl.col("name_lower").str.starts_with("sant "))\
#         # .sort(by=["name_lower"], descending=True)

#     df.show(20)
#     st.title("Data Connected")
#     st.write("Let's gooo!")
