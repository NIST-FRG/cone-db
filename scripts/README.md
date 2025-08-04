# Cone Calorimeter Scripts

This folder contains scripts and explorers to parse, prepare, and process experimental data across various sources of Cone Calorimeter tests within the Cone Database (within the Material Flammability Database)

Required: Python 3.11 or higher


## Raw Experimental Data
**Legacy Cone Calorimeter Data** : 
- Raw PDF and LlamaParsed Markdown files found in [LlamaTime Shared Google Drive](https://drive.google.com/drive/u/0/folders/0AKNwSqPWrWJYUk9PVA) in the “Scanned Files” folder
- Scripts ending in “_md_A” support files existing in **Llama Time/Scanned Files/A/md_A**
- **<ins> Pull all format A md files and place into **cone-db/data/raw/md_A**. Please create this folder if it doesn’t exist. </ins>**



# Running Scripts
**<ins> Clone this cone-db repository.</ins>**
Firstly, navigate the terminal to **/cone-db/scripts/**


## Pre-parsing
Run:
```
python PH_preparse_md_A.py
```
- Input LlamaParsed files at **../data/raw/md_A**
- This script applies pre-parsing to all format A files. Pre-parsing entails extracting the data tables and remaining metadata into two separate files.
- Data Table columns are compiled as a singular dataframe and exported as a csv file.
    - Performs restrictive parsing, removing unnecessary whitespace, dashes, stars, etc  
    - Output Directory : **\cone-db\data\pre-parsed\md_A\test####_pdf_name.csv**
    - Data Columns :

| Time (s) | Q-Dot (kW/m²) | Sum Q (MJ/m²) | M-Dot (g/s·m²) | Mass Loss (kg/m²) | HT Comb (MJ/kg) | Ex Area (m²/kg) | CO₂ (kg/kg) | CO (kg/kg) | H₂O (kg/kg) | H'carbs (kg/kg) | HCl (kg/kg) |
|----------|----------------|----------------|----------------|-------------------|------------------|------------------|--------------|-------------|--------------|------------------|--------------|

- Metadata contains information on testing conditions and parameters, material data, additional measured values, etc.  
    - Output Directory: **\cone-db\metadata\md_A\preparsed\test####_pdf_name.json**

 

## Parsing

Run:
```
python PH_parse_md_A.py
```

- The parsing script selects the crucial columns/data features of the pre-parsed data table (.csv), most informative in determining material flammability. It copies all pre-parsed metadata files to the parsed folder, as there’s no additional parsing needed. 
- Input : 
    - Pre-parsed data table files : **\cone-db\data\pre-parsed\md_A\test####_pdf_name.csv]**
    - Metadata files : **\cone-db\metadata\md_A\preparsed\test####_pdf_name.json**
- Updated Data Table
    - Output Directory: /cone-db/data/parsed/md_A/test####_pdf_name.csv
    - Selected columns:

| Time (s) | HRR (kW/m2) | CO2 (Vol %) | CO (Vol %) | MLR (g/s-m2) | Ex Area (m2/kg) |
|-----------|------------|-------------|------------|--------------|-----------------|

- Copying all metadata 
    - Output Directory: /cone-db/data/parsed/md_A/test####_pdf_name.json
- Both Parsed Metadata and Data Table files are sent to **\cone-db\scripts\cone-explorer\data\parsed\md_A** ready to be prepared.


## Prepare

<ins>Navigate the terminal to /cone-db/scripts/cone-explorer</ins>

Run:
```
streamlit run Main.py
```

- The preparation stage for the Legacy Cone Data consists of performing initial data tables and metadata processing. The manual preparation includes connecting all tests to publications, fixing errors in metadata fields, and renaming .csv data table and .json metadata files. 
- The manual preparation is executed through the NIST Cone Data Explorer.
    - Input : 
        - Parsed Metadata and Data Table files **\cone-db\scripts\cone-explorer\data\parsed\md_A**

### NIST Cone Data Explorer

#### Metadata Editor
- Loads all metadata and HRR curves as an organized dataframe, with each row representing a test and columns listing material information, testing conditions, and additional metadata per test.
- **<ins>Editable cells</ins>** : double-click any cell to edit, remove all unsaved edits by pressing the “Reload” button
- **<ins>Saving Updated Metadata</ins>** : Press the “Save” button to save all metadata field changes to their respective parsed metadata files.
- **<ins>Select Columns</ins>** : Select from the dropdown to limit displayed columns. Remove all selected columns to display all metadata fields.
- **<ins>Exporting & Renaming Test files</ins>** : Press the “Export” button to export the current version of metadata as newly prepared metadata files. During the export process, the corresponding metadata and data table files will be copied and renamed to “Matl-ID_HF_Orientation_Test# and placed into the output prepared directory.
    - Output Prepared Directory : \cone-db\scripts\cone-explorer\data\prepared\md_A

#### Cone Metadata Search
- Displays all metadata fields for each test.
- Search bar above can be used to filter a key value or word applied to all fields.

#### Cone Data Viewer
- Plot any number of tests to compare across any three curves: Heat Release Rate (kW/m2) vs. Time (s), Mass Loss Rate (g/s-m2) vs. Time (s), Total Heat Release (MJ/m2) vs. Time (s)
- **_Still need to Fix into displaying columns of parsed data table_**


**_Main page containing setup and page functionality descriptions needs updating_**


## Autoprocess

<ins>Navigate the terminal to \cone-db\scripts.</ins>

Run:
```
python autoprocess_md_A.py
```

- The autoprocessing script calculates additional key features and thermophysical properties related to material flammability. Listed below is the current collection of programmatically calculated values as “Property/Value (unit) [metadata_field_name]”:
    - Average Mass Loss Rate (g/s-m2) **[MLRPUA]**
    - Residue Yield (g/g) **[mf/m0_g/g]**
    - Peak HRR (kW/m2) **[peak_q_dot_kw/m2]**

Full List of Additional Calculated Features : https://docs.google.com/document/d/1uqdBjGeKTKFXSLZ6MUrCiqjBiaWAMfYQEK9GYorT1Aw/edit?tab=t.0


# External Cone-DB 

<ins>Navigate the terminal to \cone-db\scripts\cone-db-landing.</ins>

Run:
```
streamlit run Main.py 
```

The external Cone-DB contains pages to view and analyze the data tables and metadata in tabular and visualized plotted formats. Pages include the Cone Data Table and Cone Data Explorer. 

## Cone Data Table
- This page displays the critical measured and calculated thermophysical properties and key features of material flammability as a sortable filterable data table. 
- Search Bar : displaying tests for a specified material

## Cone Data Explorer
- Plot any number of tests to compare across any three curves:
    - Heat Release Rate (kW/m2) vs. Time (s)
    - Mass Loss Rate (g/s-m2) vs. Time (s)
    - Total Heat Release (MJ/m2) vs. Time (s)


