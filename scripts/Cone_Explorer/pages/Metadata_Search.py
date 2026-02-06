import json
from pathlib import Path
import pandas as pd
import streamlit as st
from st_keyup import st_keyup
import plotly.express as px
import plotly.graph_objects as go
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # .../Scripts

sys.path.append(str(PROJECT_ROOT))
from Cone_Explorer.const import (
    PARSED_METADATA_PATH,
    PREPARED_METADATA_PATH,
    SCRIPT_DIR
)

st.set_page_config(page_title="Cone Metadata Search", page_icon="🔎", layout="wide")

st.title("Cone Metadata Search")

# Initialize the queue in session state if it doesn't exist
if "test_queue" not in st.session_state:
    st.session_state.test_queue = []

st.write("Select the test status you would like to view")
test_types = ['SmURFed', "All (Parsed Versions)", "Not SmURFed"]
selected_type = st.selectbox("Choose SmURF status", test_types)
if selected_type == "Not SmURFed":
    metadata_name_map = {}
    for p in PARSED_METADATA_PATH.rglob("*.json"):
        with open(p, 'r') as f:
            metadata = json.load(f)
        # Check if 'SmURF' is absent, None, or empty string
        smurf_value = metadata.get('SmURF', None)
        if not smurf_value:  # Covers None, '', [], {}, etc.
            metadata_name_map[p.stem] = p
elif selected_type == "All (Parsed Versions)":
        # Get the paths to all the test metadata files
        metadata_name_map = {p.stem: p for p in list(PARSED_METADATA_PATH.rglob("*.json"))}
else:
    # Get the paths to all the test metadata files
    metadata_name_map = {p.stem: p for p in list(PREPARED_METADATA_PATH.rglob("*.json"))}

avg_tests = []
for test_name, test_path in metadata_name_map.items():
    if 'Average'  in test_name:
        avg_tests.append(test_name)
for avg in avg_tests:
    del metadata_name_map[avg]
# Get the metadata for each test
test_metadata = []
for metadata_path in metadata_name_map.values():
    m = json.load(open(metadata_path))
    m["File name"] = metadata_path.stem
    m = {k: str(v) for k, v in m.items()}
    test_metadata.append(m)

if test_metadata != []:
    # Create a dataframe with all the test metadata so that it can be easily displayed
    metadata_df = pd.DataFrame(test_metadata).set_index("File name")
    query = st_keyup("Search test metadata:", placeholder="e.g. 'PMMA'")

    if query:
        # Filter the metadata based on the query
        mask = metadata_df.map(lambda x: query.lower() in str(x).lower()).any(axis=1)
        metadata_df = metadata_df[mask]

    st.divider()
    
    # Create checkbox column as a separate Series, then concat (avoids fragmentation warning)
    queue_checkbox = pd.Series(
        metadata_df.index.isin(st.session_state.test_queue),
        index=metadata_df.index,
        name="Add to Queue"
    )
    
    # Concatenate checkbox column with metadata and create a fresh copy
    metadata_df = pd.concat([queue_checkbox, metadata_df], axis=1).copy()

    # Queue management controls
    st.divider()
    col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
    
    with col1:
        st.metric("Currently Displayed", len(metadata_df))
    with col2:
        st.metric("Tests in Queue", len(st.session_state.test_queue))
    with col3:
        if st.button("Select All Displayed", use_container_width=True):
            for test in metadata_df.index.tolist():
                if test not in st.session_state.test_queue:
                    st.session_state.test_queue.append(test)
            st.rerun()
    with col4:
        if st.button("Clear Entire Queue", use_container_width=True):
            st.session_state.test_queue = []
            st.rerun()

    # Show all queued tests in expander
    if st.session_state.test_queue:
        with st.expander(f"📋 View Queue ({len(st.session_state.test_queue)} tests)"):
            for test in st.session_state.test_queue:
                col_name, col_status, col_remove = st.columns([4, 1, 1])
                with col_name:
                    st.text(test)
                with col_status:
                    # Show if test is visible in current filter
                    if test in metadata_df.index:
                        st.caption("✅ Shown")
                    else:
                        st.caption("🔍 Hidden")
                with col_remove:
                    if st.button("Remove", key=f"remove_{test}"):
                        st.session_state.test_queue.remove(test)
                        st.rerun()

    # Display the editable dataframe with checkboxes
    edited_df = st.data_editor(
        metadata_df.reset_index(),
        use_container_width=True,
        column_config={
            "Add to Queue": st.column_config.CheckboxColumn(
                "Add to Queue",
                help="Check to add test to SmURF Editor queue",
                default=False,
            ),
        },
        disabled=[col for col in metadata_df.reset_index().columns if col != "Add to Queue"],
        hide_index=True,
    )

    # Set index back for downstream operations
    edited_df = edited_df.set_index("File name")
    
    # Update the queue based on checkbox changes
    new_queue = []
    for test_name in edited_df.index:
        if edited_df.loc[test_name, "Add to Queue"]:
            new_queue.append(test_name)
    
    # Also keep any queued tests that aren't currently displayed (from previous searches)
    for test in st.session_state.test_queue:
        if test not in edited_df.index and test not in new_queue:
            new_queue.append(test)
    
    # Update session state if changed
    if set(new_queue) != set(st.session_state.test_queue):
        st.session_state.test_queue = new_queue
        st.rerun()

    # ===================== END QUEUE FUNCTIONALITY =====================
    st.markdown("#### Notes")
    readme = SCRIPT_DIR / "README.md"
    section_title = "### Metadata Search"

    # Read the README file
    with open(readme, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # Find start and end indices for the subsection
    start_idx, end_idx = None, None
    for i, line in enumerate(lines):
        if line.strip() == section_title:
            start_idx = i +1
            break

    if start_idx is not None:
        for j in range(start_idx + 1, len(lines)):
            if lines[j].startswith("### ") or lines[j].startswith("## "):
                end_idx = j
                break
        # If no further section, use end of file
        if end_idx is None:
            end_idx = len(lines)
        subsection = "".join(lines[start_idx:end_idx])
        st.markdown(subsection)
else:
    st.error(f"There are no tests with the status {selected_type} available.")