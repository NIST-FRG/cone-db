import streamlit as st
from const import INPUT_DATA_PATH, OUTPUT_DATA_PATH

st.set_page_config(page_title="NIST Cone Data Explorer", page_icon="🔬")

st.title("NIST Cone Data Explorer")

st.markdown(
    """
### Process data:
- Edit metadata for cone calorimeter tests
- Rename files to match metadata
- Add metadata fields for reports, comments, and other metadata
### View/visualize data:
- Compare data across tests
- Graph multiple columns from the same test
"""
)

# Make sure the INPUT_DATA_PATH and OUTPUT_DATA_PATH exist, if not create them
if not INPUT_DATA_PATH.exists():
    INPUT_DATA_PATH.mkdir(parents=True, exist_ok=True)
if not OUTPUT_DATA_PATH.exists():
    OUTPUT_DATA_PATH.mkdir(parents=True, exist_ok=True)
