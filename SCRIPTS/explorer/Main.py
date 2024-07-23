import streamlit as st

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
