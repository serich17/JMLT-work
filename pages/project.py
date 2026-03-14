
import streamlit as st
import polars as pl
from funcs import *
import threading
import time

SAVE_INTERVAL = 180  # seconds


project_id = st.query_params.get("project_id")
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECTS_DIR = os.path.join(PARENT_DIR, "projects")
print("here")
if not project_id:
    st.title("No Project Linked")
    st.button("Connect Project") and st.switch_page("main.py")
else:
    if "lock" not in st.session_state:
        st.session_state.lock = threading.Lock()
    if "lock_class" not in st.session_state:
        st.session_state.lock_class = threading.Lock()
    if "save_lock" not in st.session_state:
        st.session_state.save_lock = threading.Lock()
    Lock = st.session_state.lock
    # initialize timer
    if "last_save" not in st.session_state:
        st.session_state.last_save = time.time()

    if "allPlaces" not in st.session_state:
        st.session_state.allPlaces = load_places()

    if "countries" not in st.session_state:
        st.session_state["countries"] = load_countries()

    if "project_df" not in st.session_state:
        st.session_state["project_df"] = load_project(project_id)

    # Initialize in session state
    if "pending_changes" not in st.session_state:
        st.session_state.pending_changes = {}  # {index_val: {col: val}}
    if "pending_deletes" not in st.session_state:
        st.session_state.pending_deletes = set()
    st.title(project_id)
    with st.session_state.lock_class:
        with open(f"{PROJECTS_DIR}/{project_id}/progress.pkl", "rb") as f:
            progress = pickle.load(f)
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Unique Flagged", f"{progress.get_unique_flagged():,}", )
    col2.metric("Total Flagged", f"{progress.get_flagged():,}")
    col3.metric("Project Size (Rows)", f"{progress.get_size()[0]:,}")

    approve, editor, map, export = st.tabs(["Approve Suggestions", "Data Editor", "Search Map", "Export CSV"], on_change="rerun")

    if approve.open:
        with approve:
            @st.fragment
            def approve_code():
                change_all_same = st.checkbox("Apply Changes to Duplicate Values in To_analyze", value=True)
                opts = options((21321325,56465478))

                filter_state = (opts[0], opts[1], True, True)
                filter_update(filter_state)

                cols_selected = st.multiselect(
                    "Select which columns to show in table",
                    st.session_state.base_df.schema.keys(),
                    default=["Original", "To_analyze", "Suggested", "MatchScore"],
                    key = 65645165
                )
                @st.fragment
                def approval_table():
                    event = st.dataframe(
                        st.session_state.base_df,
                        column_order=cols_selected,
                        on_select="rerun",
                        selection_mode=["multi-row"],
                        width="stretch",
                        hide_index=False
                    )
                    selected_rows = event.selection.rows

                    if len(selected_rows) > 0:
                        selected_indexes = (
                            st.session_state.base_df[selected_rows]        
                            .get_column("Index")
                            .to_list()
                        )
                        if st.button(f"Approve {len(selected_rows)} Suggestions"):
                            approve_rows(selected_indexes, change_all_same, project_id, filter_state, progress)
                            st.rerun()
                        
                approval_table()
            approve_code()

    if editor.open:
        with editor:
                # Translate and accumulate after every rerun
            # Safe — returns empty dict/list if editor hasn't rendered yet
            editor_state = st.session_state.get("editor", {})

            for row_pos, changes in editor_state.get("edited_rows", {}).items():
                index_val = st.session_state.base_df[row_pos, "Index"]
                if index_val not in st.session_state.pending_changes:
                    st.session_state.pending_changes[index_val] = {}
                st.session_state.pending_changes[index_val].update(changes)

            for row_pos in editor_state.get("deleted_rows", []):
                index_val = st.session_state.base_df[row_pos, "Index"]
                st.session_state.pending_deletes.add(index_val)

            # Clear after translating so they don't get re-processed next rerun
            if "editor" in st.session_state:
                st.session_state["editor"]["edited_rows"] = {}
                st.session_state["editor"]["deleted_rows"] = []
            
            dat_fil, save_sets = st.columns(2)
            with dat_fil:
                st.subheader("Data Filter Settings")
                isUnique = st.checkbox("Show Unique Values Only", value=True)
                onlyFlagged = st.checkbox("Show Only Flagged", value=True)

            with save_sets:
                st.subheader("Save Settings")
                change_all_same = st.checkbox("Apply Changes to Duplicate Values in To_analyze", value=True, key=6989685)
            
            opts = options((565412,567489))
            filter_state = (opts[0], opts[1], isUnique, onlyFlagged)
            filter_update(filter_state)
            cols_selected = st.multiselect(
                "Select which columns to show in table",
                st.session_state.base_df.schema.keys(),
                default=["Original", "To_analyze"]
            )
            # print(st.session_state.base_df)
            
            with st.session_state.save_lock:
                edited_df = st.data_editor(
                    st.session_state.base_df,
                    width="stretch",
                    key="editor",
                    column_order = cols_selected,
                    column_config={"Select": st.column_config.CheckboxColumn(required=True, pinned=True)},
                    num_rows="delete",
                    hide_index=False,
                    disabled=[x for x in st.session_state.base_df.columns if x not in ["To_analyze", "Separated", "Analyzed"]]
                )
            

            if (time.time() - st.session_state.last_save >= SAVE_INTERVAL):
                update_changed_df(project_id, change_all_same, progress)
                    
                
            st.button("Save Changes", on_click=update_changed_df, args=[project_id, change_all_same, progress])
                
            


    with map:
        df = st.session_state["allPlaces"]
        number = st.slider("Number of results to show on map", 100, 10000, step=10)
        mapped = st.text_input("Contains...", key=15654564564)
        
        with Lock:
            df = df.drop_nulls(subset=["latitude", "longitude"]).filter(pl.col("name_lower").str.contains(mapped.lower())).limit(number).with_columns([
                pl.col("latitude").cast(pl.Float64, strict=False),
                pl.col("longitude").cast(pl.Float64, strict=False)
            ]).collect()
        # # 3. Drop rows where coordinates are now null (optional but recommended)
        # df = df.drop_nulls(subset=["latitude", "longitude"]).collect()
        # # 4. Streamlit's st.map still prefers a Pandas DataFrame or a specific format, 
        # # so we convert it back just for the map rendering.
        st.map(df)
    # e1f = pl.scan_parquet(f"{PROJECTS_DIR}/{project_id}/processed_data.parquet")
    # # e1f = pl.scan_parquet("allCountries.parquet/**/*.parquet")\
    # #         .filter(pl.col("name_lower").str.starts_with("pla"))\
    # st.dataframe(e1f)

