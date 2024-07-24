import json
from pathlib import Path

import pandas as pd

import streamlit as st

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from const import INPUT_DATA_PATH, OUTPUT_DATA_PATH

st.set_page_config(page_title="Cone Data Processor", page_icon="📊")

st.title("Cone Data Processor")

# Get the paths to all the test files
test_name_map = {p.stem: p for p in list(INPUT_DATA_PATH.rglob("*.csv"))}

# Get the paths to all the metadata files
metadata_name_map = {p.stem: p for p in list(INPUT_DATA_PATH.rglob("*.json"))}


# Initialize some session state variables
if "columns" not in st.session_state:
    st.session_state.columns = []
if "index" not in st.session_state:
    st.session_state.index = 0

st.subheader(list(test_name_map.keys())[st.session_state.index])

# Progress bar
progress_bar = st.progress(
    (st.session_state.index + 1) / len(test_name_map),
    text=f"Test {st.session_state.index + 1} of {len(test_name_map)}",
)


# File controls
def next_file():
    if st.session_state.index + 1 >= len(test_name_map):
        return
    st.session_state.index += 1


def prev_file():
    if st.session_state.index - 1 < 0:
        return
    st.session_state.index -= 1


def set_file():
    st.session_state.index = list(test_name_map.values()).index(
        test_name_map[file_selection]
    )


col1, col2 = st.columns(2)
col1.button("Previous file", use_container_width=True, on_click=prev_file)
col2.button("Next file", use_container_width=True, on_click=next_file)

col3, col4 = st.columns([0.8, 0.2], vertical_alignment="bottom")
file_selection = col3.selectbox(
    "Select test",
    list(test_name_map.keys()),
    index=st.session_state.index,
)
col4.button("Go", on_click=set_file, use_container_width=True)

test_data = pd.read_csv(
    test_name_map[list(test_name_map.keys())[st.session_state.index]]
).set_index("Time (s)")

test_metadata = json.load(
    open(metadata_name_map[list(test_name_map.keys())[st.session_state.index]])
)


def save_data():
    test_data.to_csv(
        OUTPUT_DATA_PATH
        / Path(list(test_name_map.keys())[st.session_state.index]).with_suffix(".csv")
    )


def save_metadata():
    json.dump(
        test_metadata,
        open(
            OUTPUT_DATA_PATH
            / Path(list(test_name_map.keys())[st.session_state.index]).with_suffix(
                ".json"
            ),
            "w",
        ),
        indent=4,
    )


def save_test():
    save_data()
    save_metadata()


save_test()
st.info("Tests are saved automatically!")

st.divider()

st.markdown("#### View data")
columns_to_graph = st.multiselect(
    "Select column(s) from test to graph",
    options=test_data.columns,
    max_selections=2,
)

fig = make_subplots(specs=[[{"secondary_y": True}]])
if len(columns_to_graph) >= 1:
    fig.add_trace(
        go.Scatter(y=test_data[columns_to_graph[0]], name=columns_to_graph[0])
    )
    fig.update_yaxes(title_text=columns_to_graph[0], secondary_y=False)
    if len(columns_to_graph) == 2:
        fig.add_trace(
            go.Scatter(
                y=test_data[columns_to_graph[1]], name=columns_to_graph[1] + " (sec.)"
            ),
            secondary_y=True,
        )
        fig.update_yaxes(title_text=columns_to_graph[1], secondary_y=True)

st.plotly_chart(fig)

with st.expander("**View data in table format**"):
    st.dataframe(test_data, use_container_width=True)

st.divider()

st.markdown("#### Edit metadata")


def make_metadata_df():
    # since streamlit's data editor requires all rows in a column to have the same type, we'll convert everything to strings.
    converted_metadata = {k: str(v or "") for k, v in test_metadata.items()}
    converted_metadata = pd.DataFrame(
        converted_metadata.items(), columns=["property", "value"]
    ).set_index("property")
    return converted_metadata


# now, convert the strings back into the correct types
def fix_types(key, value):
    original_type = type(test_metadata[key])
    try:
        if value in [""]:
            return (key, None)
        if original_type is int:
            return (key, int(value))
        elif original_type is float:
            return (key, float(value))
        elif original_type is bool:
            if value.lower() == "true":
                return (key, True)
            elif value.lower() == "false":
                return (key, False)
            return (key, None)
        elif original_type is str:
            return (key, str(value))
    except Exception as e:
        st.error(f"Type conversion was not successful for {key}: {e}")
        return (key, None)


st.markdown(
    "Metadata file is updated automatically. Leave the cell empty for `None` and use `True` and `False` for their respective Boolean equivalents."
)

metadata_df = make_metadata_df()
edited_metadata = st.data_editor(metadata_df, use_container_width=True).to_dict()[
    "value"
]

edited_metadata = dict(map(lambda kv: fix_types(kv[0], kv[1]), edited_metadata.items()))

edited_metadata["report_citation_key"] = st.text_input("Report citation key (BibTeX)")

test_metadata = edited_metadata

save_test()

st.markdown("#### :red[Danger zone]")

st.markdown(
    "*Note: Because changes are applied only to the copy saved to the output folder, the original remains unmodified, meaning that any changes made in this section will **not** be reflected in other sections of the app.*"
)

tab1, tab2, tab3 = st.tabs(
    ["Replace columns with null values", "Delete columns", "Delete test"]
)

with tab1:
    columns_to_delete = st.multiselect(
        "Select column(s) to replace with null values",
        options=test_data.columns,
    )

    # replaces columns with null values
    def null_columns():
        for column in columns_to_delete:
            test_data[column] = None
        print(test_data)
        save_test()
        st.success("Columns replaced with null values")

    st.button("Replace columns", on_click=null_columns)

with tab2:
    columns_to_delete = st.multiselect(
        "Select column(s) to delete",
        options=test_data.columns,
    )

    def delete_columns():
        for column in columns_to_delete:
            del test_data[column]
        save_test()
        st.success("Columns deleted")

    st.button("Delete columns", on_click=delete_columns)

with tab3:

    def delete_test():
        # # if the test file exists, delete it
        # if test_name_map[list(test_name_map.keys())[st.session_state.index]].exists():
        #     test_name_map[list(test_name_map.keys())[st.session_state.index]].unlink()
        # # if the metadata file exists, delete it
        # if metadata_name_map[
        #     list(test_name_map.keys())[st.session_state.index]
        # ].exists():
        #     metadata_name_map[
        #         list(test_name_map.keys())[st.session_state.index]
        #     ].unlink()

        # if either of the output files exist, delete them
        if (
            OUTPUT_DATA_PATH
            / Path(list(test_name_map.keys())[st.session_state.index]).with_suffix(
                ".csv"
            )
        ).exists():
            (
                OUTPUT_DATA_PATH
                / Path(list(test_name_map.keys())[st.session_state.index]).with_suffix(
                    ".csv"
                )
            ).unlink()
        if (
            OUTPUT_DATA_PATH
            / Path(list(test_name_map.keys())[st.session_state.index]).with_suffix(
                ".json"
            )
        ).exists():
            (
                OUTPUT_DATA_PATH
                / Path(list(test_name_map.keys())[st.session_state.index]).with_suffix(
                    ".json"
                )
            ).unlink()

        st.error("Test deleted")
        next_file()

    st.markdown(
        "**Note: the original (input) file is not deleted, only the output file generated by this app.**"
    )
    st.button(
        "Delete output & continue to next test", on_click=delete_test, type="primary"
    )
