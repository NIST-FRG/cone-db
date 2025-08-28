# NIST Cone Data Explorer

#### A web-based app for processing + visualizing cone calorimeter test data.

## Setup
Prior to opening the explorer, preparsing and parsing of data should have been performed. Please run these scripts if this is not the case, or pull down the data from GitHub. Upon opening the explorer (see Usage), click the import data button on the `Main` page to pull in parsed data.

## Usage
From the root of the repository, run the following command:

```bash
streamlit run scripts/cone-explorer/Main.py 
``

(if the above doesn't work, you can also try `python -m streamlit run scripts/cone-explorer/Main.py`)

If you're running the app for the first time and you get an error about missing packages, you may need to install streamlit again.
```bash
pip install streamlit
```
### SmURF
The major use of this data explorer will be 

## Tools

### Compare Tests

##### Overview

Grants the user the ability to view and compare the data from multiple tests.

##### Filter Options

Select `SmURF Filter` to select a date range of tests; start and end date must be in the format **YYYY-MM-DD**.

##### Selecting Tests to View

Type out the material you wish to view, followed by the version of that material if more than one exist.
Select the specific tests you would like to view, or click one of the three checkboxes.

`Select All Tests for []` will select all tests for the material(s) and version(s) you have selected in brackets.

`Select All Average Tests` will select all average test file for each version of each material.

`Select All Manually Reviewed and Approved Tests` will select all tests that have been manually reviewed and approved.

##### Data Plotting Options

For TGA, curves of Mass (mg), Normalized Mass (g/g_0), Mass Loss Rate (mg/s), and Normalized Mass Loss Rate (1/s) can be plotted against Temperature (K).

Average curves only exist for Normalized Data. If an average test exists, click `View Uncertainty in Average (data type)` to plot its expanded uncertainty (2 sigma).

Plots can zoom in and out, pan, and save the plot as a .png file. Click on the name of a test to hide its curve; double click to isolate a curve.



























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