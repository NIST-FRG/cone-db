# NIST Cone Data Explorer

#### A web-based app for processing + visualizing cone calorimeter test data.

## Setup
Place .csv & .json files from the parsing scripts (e.g. FTT.py, midas.py, etc.) in the `data` folder. Folder structure doesn't matter, but make sure the CSV data file and the JSON metadata file for each test have the same name and are in the same folder.

## Usage
From the root of the repository, run the following command:

```bash
streamlit run scripts/cone-explorer/Main.py 
```

(if the above doesn't work, you can also try `python -m streamlit run scripts/cone-explorer/Main.py`)

## Tools

### Metadata Editor
Allows for editing metadata fields through a spreadsheet/table-like interface, as well as bulk deleting files.  

The metadata editor loads metadata from the `data` folder and upon clicking **Save**, saves changes in-place (i.e. it modifies the original JSON files in the `data` folder). If you've changed the contents of the `data` folder while the app is running, use the **Reload** button so that your changes are reflected in the editor.

Files can be deleted by checking the **\*\* DELETE FILE** checkbox next to their name and then clicking the **Delete files** button in the sidebar. 

To export the final metadata files, click the **Export metadata** button in the sidebar. This will save all the tests with a `material_id` field to the `output` folder, sorted into folders by year.

#### Material IDs
Material IDs should use the format `<material_name>:<report_identifier>`. Spaces in a material name should be replaced with underscores. For example, the material ID for a material named "Material A" with a report identifier of "smith2015characterization" would be `Material_A:smith2015characterization`.

### Search
Searches metadata files for a given text query
### View
Plots data for multiple tests at a time.

## General troubleshooting
If you encounter any unexpected errors, try interrupting the program (Ctrl+C) and restarting it. Note you will lose any unsaved changes in the metadata editor.