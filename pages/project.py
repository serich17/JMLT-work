from threading import Lock
import streamlit as st
import polars as pl
from funcs import *
import time

SAVE_INTERVAL = 180  # seconds


project_id = st.query_params.get("project_id")
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECTS_DIR = os.path.join(PARENT_DIR, "projects")
if not project_id:
    st.title("No Project Linked")
    st.button("Connect Project") and st.switch_page("main.py")
else:
    if "lock" not in st.session_state:
        st.session_state.lock = Lock()
    if "lock_class" not in st.session_state:
        st.session_state.lock_class = Lock()
    if "save_lock" not in st.session_state:
        st.session_state.save_lock = Lock()
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
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Unique Flagged", f"{progress.get_unique_flagged():,}", border=True)
    col2.metric("Total Flagged", f"{progress.get_flagged():,}", border=True)
    col3.metric("Current Admin Level", f"{progress.get_curr_level()}", border=True)
    col4.metric("Project Size (Rows)", f"{progress.get_size()[0]:,}", border=True)

    approve, editor, map, next_admin, export = st.tabs(["Approve Suggestions", "Data Editor", "Search Map", "Next Admin", "Export CSV"], on_change="rerun")

    if approve.open:
        with approve:
            with st.expander("Filter Options"):
                dat_fil, save_sets = st.columns(2)
                with dat_fil:
                    st.subheader("Data Filter Settings")
                    single_list = st.checkbox("Rows Containing Initial Separated List of Length 1", value=False, key=212321)
                with save_sets:
                    st.subheader("Save Settings")
                    change_all_same = st.checkbox("Apply Changes to Duplicate Values in To_analyze", value=True)
            opts = options((21321325,56465478))
            filter_state = (opts[0], opts[1], True, True, single_list)
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
                    st.text(f"{len(selected_rows)} Rows Selected")
                    b1, b2, b3 = st.columns(3)
                    with b1:
                        if st.button(f"Use Suggested"):
                            approve_rows(selected_indexes, change_all_same, project_id, filter_state, progress, True, False)
                            st.rerun()
                        replace_with = st.text_input(label="Replace with", label_visibility="collapsed")
                        
                    with b2:
                        if st.button(f"Approve As Is"):
                            approve_rows(selected_indexes, change_all_same, project_id, filter_state, progress, False, False)
                            st.rerun()
                        if st.button("Replace Selected With") and len(replace_with) > 0:
                            approve_rows(selected_indexes, change_all_same, project_id, filter_state, progress, False, False, True, replace_with)
                            st.rerun()
                    with b3:
                        if st.button(f"Mark Not {progress.get_curr_level()}"):
                            approve_rows(selected_indexes, change_all_same, project_id, filter_state, progress, False, True)
                            st.rerun()
                    
            approval_table()
            

    if editor.open:
        with editor:
            with st.expander("Filter Options"):
                dat_fil, save_sets = st.columns(2)
                with dat_fil:
                    st.subheader("Data Filter Settings")
                    isUnique = st.checkbox("Show Unique Values Only", value=True)
                    onlyFlagged = st.checkbox("Show Only Flagged", value=True)
                    single_list = st.checkbox("Rows Containing Initial Separated List of Length 1", value=False, key=212)

                with save_sets:
                    st.subheader("Save Settings")
                    change_all_same = st.checkbox("Apply Changes to Duplicate Values in To_analyze", value=True, key=6989685)
            
            opts = options((565412,567489))
            filter_state = (opts[0], opts[1], isUnique, onlyFlagged, single_list)
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
            

            if (time.time() - st.session_state.last_save >= SAVE_INTERVAL):
                update_changed_df(project_id, change_all_same, progress)
                    
                
            st.button("Save Changes", on_click=update_changed_df, args=[project_id, change_all_same, progress])
                
            

    if map.open:
        with map:
            st.text("This lookup uses the allplaces database, and filters to the Admin codes listed below.")
            number = st.slider("Number of results to show on map", 100, 10000, step=10)
            df = prep_df_map()
            
            data_load_state = st.text('Loading map...')
            with Lock:
                df = df.limit(number).with_columns([
                    pl.col("latitude").cast(pl.Float64, strict=False),
                    pl.col("longitude").cast(pl.Float64, strict=False)
                ]).collect()
                co = pl.scan_parquet("country/*/*.parquet").with_columns(pl.col("ISO").alias("country_code")).select("Country", "country_code").collect()
                df = df.join(co, on="country_code", how="left")
            if len(df) > 0:
                st.map(df)
            else:
                st.warning("No results found")
            data_load_state.text("Done!")
            st.dataframe(df.select("name", "asciiname", "alternatenames", "feature_code", "Country"))
                
    if next_admin.open:
        with next_admin:
            st.text("This tab is for moving on to the next admin level, and running checks in the databases, and doing match scores with rapidfuzz")
            st.warning("Only move to the next level after there are no more flagged rows in the current level")

            st.header("Settings")
            transition_next_admin_settings(project_id)

            




            
    if export.open:
        with export:
            @st.fragment
            def export_tab():
                st.header("Export the CSV")
                st.text("This export function will combine the original Folder Name column with the items in the Analyzed column into a single exported CSV, more for the final product")
                st.text("If you want to export something else, use the built in export csv in the tables and select the columns you want")
                if st.button("Prepare Export"):
                    # 1. Get the LazyFrame
                    lf = st.session_state.project_df
                    
                    # 2. Materialize to a DataFrame to calculate max width and for downloading
                    df = lf.collect()
                    
                    # 3. Calculate max width (must be done on a collected DF or via a separate aggregation)
                    max_width = df.select(pl.col("Analyzed").list.len().max()).item()
                    
                    # 4. Expand the list column
                    # Note: Using .unnest("Analyzed") to match your column name
                    expanded = (
                        df.select(["Folder Name", "Analyzed"])
                        .with_columns(
                            pl.col("Analyzed").list.to_struct(upper_bound=max_width)
                        )
                        .unnest("Analyzed")
                    )
                    
                    # 5. Provide the download button
                    st.download_button(
                        label="Download CSV",
                        data=expanded.write_csv(),
                        file_name="output.csv",
                        mime="text/csv"
                    )
            export_tab()
