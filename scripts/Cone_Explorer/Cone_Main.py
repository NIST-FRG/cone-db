import streamlit as st
import json
import shutil
import os
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]  # .../Scripts

sys.path.append(str(PROJECT_ROOT))
from Cone_Explorer.const import (
    INPUT_DATA_PATH, PARSED_METADATA_PATH, 
    PARSED_DATA_PATH, PREPARED_DATA_PATH
)


st.set_page_config(page_title="NIST Cone Data Explorer", page_icon="🔬")

st.title("NIST Cone Data Explorer")


SCRIPT_DIR = Path(__file__).resolve().parent
readme_path = SCRIPT_DIR / "README.md"
with open (readme_path, 'r') as f:
    st.markdown(f.read())
# Make sure the INPUT_DATA_PATH and OUTPUT_DATA_PATH exist, if not create them
if not INPUT_DATA_PATH.exists():
    INPUT_DATA_PATH.mkdir(parents=True, exist_ok=True)
