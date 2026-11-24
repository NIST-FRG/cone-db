import pandas as pd
import re
from pathlib import Path
import json
from datetime import datetime
import numpy as np
import traceback
import shutil
from utils import colorize


# Path handling: relative to this script's location
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

INPUT_DIR = PROJECT_ROOT / "data" / "raw" / "Alex_Morgan"
OUTPUT_DIR = PROJECT_ROOT / "data" / "preparsed" / "Box" / "Alex_Morgan"
LOG_FILE = PROJECT_ROOT / "preparse_AM_log.json"


# NIST/BFRL/MPG table columns in the order they appear in the file.
# The second row under the heading contains units, and is skipped.
NIST_COLUMNS = [
    "Scan",
    "HRRPUA (kW/m2)",
    "HT Comb (MJ/kg)",
    "Extinction Area (m2/kg)",
    "MLRPUA (g/s-m2)",
    "CO2 (kg/kg)",
    "CO (kg/kg)",
    "Duct Mass Flow (kg/s)",
    "Duct Volume Flow (L/s)",
    "Smoke Mass Flow (kg/s)",
    "Smoke Volume Flow (L/s)",
    "Mass (g)",
    "K Smoke (1/m)",
    "Exhaust Pressure (Pa)",
    "Stack Temperature (C)",
    "Smoke Temperature (C)",
    "O2 (%)",
    "CO2 (%)",
    "CO (%)",
    "Soot Flow (g/s)",
    "Time (s)",
    "Aux 1 (V)",
    "Aux 2 (V)",
    "Aux 3 (V)",
]


#region parse_dir
def parse_dir(input_dir):
    """Find and parse Markdown or text files under input_dir."""
    input_dir = Path(input_dir)
    paths = sorted(
        path for path in input_dir.glob("**/*")
        if path.is_file() and path.suffix.lower() in {".md", ".txt", ".cdt", ".xcd"}
    )

    print(paths)
    print(colorize(f"Found {len(paths)} files to parse", "purple"))

    files_parsed = 0
    files_parsed_fully = 0
    files_parsed_partial = 0

    for path in paths:
        files_parsed += 1
        pct = None

        try:
            pct = parse_file(path)
        except Exception as e:
            try:
                with open(LOG_FILE, "r", encoding="utf-8") as f:
                    logfile = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                logfile = {}

            logfile[str(path)] = str(e)
            with open(LOG_FILE, "w", encoding="utf-8") as f:
                json.dump(logfile, f, indent=4)

            print(colorize(f" - Error parsing {path}: {e}\n", "red"))

        if pct == 100:
            print(colorize(f"Parsed {path} successfully\n", "green"))
            files_parsed_fully += 1

            # Keep the original move behavior.
            out_path = Path(str(path).replace("Alex_Mogran", "Alex_Morgan_Preparsed"))
            if out_path == path:
                out_path = path.with_name(path.stem + "_preparsed" + path.suffix)

            #out_path.parent.mkdir(parents=True, exist_ok=True)
            #if path.exists():
            #    shutil.move(str(path), str(out_path))

        elif pct is None or pct == 0:
            print(colorize(f"{path} could not be parsed", "red"))
        else:
            print(colorize(
                f"{pct}% of tests in {path} parsed successfully\n", "yellow"
            ))
            files_parsed_partial += 1

    if files_parsed:
        print(colorize(
            f"Files pre-parsed fully: {files_parsed_fully}/{files_parsed} "
            f"({files_parsed_fully / files_parsed * 100:.1f}%)",
            "blue",
        ))
        print(colorize(
            f"Files pre-parsed partially: {files_parsed_partial}/{files_parsed} "
            f"({files_parsed_partial / files_parsed * 100:.1f}%)",
            "blue",
        ))
    else:
        print(colorize("No input files found.", "yellow"))


#region parse_file
def parse_file(file_path):
    """Parse one input file using the existing test-processing flow."""
    file_path = Path(file_path)
    print(colorize(f"Parsing {file_path.name}:", "yellow"))

    with open(file_path, "r", encoding="utf-8", errors="replace") as file:
        lines = file.readlines()
    print(f"Read {len(lines)} lines from file")

    try:
        tests = get_tests(lines)
        numtests = len(tests)
        parsed = 0
    except Exception as e:
        _write_log(file_path.name, str(e))
        print(colorize(f" - Error parsing {file_path.name}: {e}\n", "red"))
        return None

    if numtests == 0:
        _write_log(file_path.name, "No tests found")
        return 0

    for test in tests:
        try:
            test_data_df, metadata = get_data(tests[test])
            data_df, test_filename = parse_data(
                test_data_df, test, file_path.name
            )
            data_df = (
                data_df.replace([np.inf, -np.inf], np.nan)
                .dropna(how="all")
            )

            output_path = OUTPUT_DIR / f"{test_filename}.csv"

            if output_path.exists():
                old_df = pd.read_csv(output_path)
                if old_df.equals(data_df):
                    print(colorize(
                        f"{test_filename} already exists and is identical. "
                        "Skipping generation.",
                        "blue",
                    ))
                    parsed += 1
                    continue
                print(colorize(
                    f"{test_filename} already exists but differs. "
                    "Overwriting with new data.",
                    "yellow",
                ))

            status = parse_metadata(metadata, test_filename)
            if status is None:
                OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
                data_df.to_csv(output_path, index=False)
                print(colorize(f"Generated {output_path}", "blue"))
                parsed += 1
            elif status in {"SmURF", "Bad"}:
                parsed += 1

        except Exception as e:
            tb_list = traceback.extract_tb(e.__traceback__)
            if tb_list:
                fail = tb_list[-1]
                location = f"{Path(fail.filename).name}:{fail.lineno} ({fail.name})"
            else:
                location = "unknown location"

            _write_log(
                f"{file_path.name}-{test}",
                f"{e} @ {location}",
            )
            print(colorize(f" - Error parsing {test}: {e}\n", "red"))

    return (parsed / numtests) * 100


def _write_log(key, value):
    """Add a message to the run log."""
    try:
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            logfile = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        logfile = {}

    logfile[key] = value
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        json.dump(logfile, f, indent=4)


#region get_tests
def get_tests(file_contents):
    """
    An NIST/BFRL/MPG file contains one test, so return it in the same
    dictionary structure the original parse_file function expects.
    """
    if any("NIST/BFRL/MPG" in str(line).upper() for line in file_contents):
        tests = {
            "Test 0001": [str(line).rstrip("\n") for line in file_contents]
        }
        print(colorize("Found 1 NIST/BFRL/MPG test: ['Test 0001']", "cyan"))
        return tests

    # Preserve support for the original test-header style.
    tests = {}
    test_number = -1

    for line in file_contents:
        line_upper = str(line).upper().strip()
        test_match = re.search(r"\((\d{3,4}[A-Z]?)\)\s*$", line_upper)

        if test_match is None:
            test_match = re.search(r"TEST\s+(\d{4})", line_upper)

        if test_match is not None:
            raw = test_match.group(1)
            test_number = f"Test {raw.zfill(4)}"
            tests.setdefault(test_number, [])

        if test_number != -1:
            tests[test_number].append(str(line).rstrip("\n"))

    print(colorize(f"Found {len(tests)} tests: {list(tests.keys())}", "cyan"))
    return tests


#region get_data
def get_data(data):
    """
    Split a NIST/BFRL/MPG test into time-series data and metadata lines.
    Returns (dataframe, metadata_lines), matching the original interface.
    """
    lines = [str(line).rstrip("\n") for line in data]

    header_index = None
    for i, line in enumerate(lines):
        if re.match(r"^\s*Scan\s+HRR\b", line, flags=re.IGNORECASE):
            header_index = i
            break

    if header_index is None:
        raise ValueError(
            "Could not find the NIST data table header beginning 'Scan HRR'."
        )

    # Keep every non-table line available for the existing metadata writer.
    metadata = lines[:header_index]

    # Header is followed by a units row, which is intentionally skipped.
    records = []
    for line_number, line in enumerate(lines[header_index + 2:], start=header_index + 3):
        if not line.strip():
            continue

        fields = line.split()

        # Skip non-data trailing lines.
        if not fields or not re.fullmatch(r"\d+", fields[0]):
            continue

        if len(fields) != len(NIST_COLUMNS):
            raise ValueError(
                f"Expected {len(NIST_COLUMNS)} values in table row at line "
                f"{line_number}, found {len(fields)}."
            )

        records.append(fields)

    if not records:
        raise ValueError("The NIST table header was found, but no data rows were found.")

    data_df = pd.DataFrame(records, columns=NIST_COLUMNS)
    for column in data_df.columns:
        data_df[column] = pd.to_numeric(data_df[column], errors="coerce")

    return data_df, metadata


#region parse_data
def parse_data(data_df, test, file_name):
    """
    Clean the NIST time series and retain standardized data-column names.
    """
    data_df = data_df.copy()

    for column in data_df.columns:
        data_df[column] = pd.to_numeric(data_df[column], errors="coerce")

    data_df = data_df.dropna(how="all").reset_index(drop=True)

    print(data_df)
    if "Time (s)" not in data_df.columns:
        raise ValueError("The parsed table does not contain a Time (s) column.")

    times = data_df["Time (s)"].dropna().to_numpy()
    if len(times) == 0:
        raise ValueError("No numeric time values were found.")

    if not np.isclose(times[0], 0) and not np.isclose(times[0],1):
        print(times[0])
        raise ValueError("Test does not start at 0 or 1 seconds.")

    if len(times) > 1:
        increments = np.diff(times)
        expected_step = np.median(increments)

        if expected_step <= 0:
            raise ValueError("Time values are not increasing.")

        continuous_mask = (
            (increments >= expected_step * 0.1)
            & (increments <= expected_step * 5)
        )

        if not np.all(continuous_mask):
            bad_indices = np.where(~continuous_mask)[0]
            print(f"Expected time step (median): {expected_step}")
            for idx in bad_indices:
                print(
                    f"Discontinuity between {times[idx]} and {times[idx + 1]} "
                    f"(increment {increments[idx]})"
                )
            raise ValueError(
                "Test does not have continuous time data; please review the input."
            )

    # The source filename is used in the same test_filename pattern as before.
    test_name = test.casefold().replace(" ", "")
    test_filename = file_name[:8]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(colorize(f"Final columns: {list(data_df.columns)}", "green"))

    return data_df, test_filename


#region parse_metadata
def parse_metadata(input, test_name):
    """
    Build the same metadata JSON schema used by the original script,
    extracting the corresponding values from the NIST/BFRL/MPG report.
    """
    meta_path = OUTPUT_DIR / f"{test_name}.json"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Preserve the existing behavior of not overwriting manually classified
    # SmURF or bad-data metadata.
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            existing = json.load(f)

        if existing.get("SmURF") is not None:
            print(colorize(
                f'{existing.get("Original Testname")} has already been '
                f'SmURFed to {existing.get("Testname")} on '
                f'{existing.get("SmURF")}. Skipping Preparsing.',
                "blue",
            ))
            return "SmURF"

        if existing.get("Bad Data") is not None:
            print(colorize(
                f'{existing.get("Original Testname")} was deemed bad on '
                f'{existing.get("Bad Data")}. Skipping Preparsing.',
                "purple",
            ))
            return "Bad"

    expected_keys = [
        "Material ID", "Material Name", "Sample Mass (g)", "Residual Mass (g)",
        "Replicate", "Original Testname", "Testname", "Thickness (mm)",
        "Sample Description", "Specimen Prep", "Instrument", "Test Date",
        "Test Time", "Operator", "Director", "Sponsor", "Institution",
        "Report Name", "Original Source", "Preparsed", "Parsed",
        "Auto Prepared", "Manually Prepared", "SmURF", "Bad Data",
        "Autoprocessed", "Manually Reviewed Series", "Pass Review", "Published",
        "Heat Flux (kW/m2)", "Orientation", "C Factor", "Surface Area (m2)",
        "Grid", "Edge Frame", "Ignition Source", "Separation (mm)",
        "Test Start Time (s)", "Test End Time (s)", "MLR EOT Mass (g/m2)",
        "End of test criterion", "Heat of Combustion O2 (MJ/kg)",
        "OD Correction Factor", "Substrate", "Non-scrubbed",
        "Duct Diameter (m)", "O2 Delay Time (s)", "CO2 Delay Time (s)",
        "CO Delay Time (s)", "Ambient Temperature (°C)",
        "Barometric Pressure (Pa)", "Relative Humidity (%)",
        "X_O2 Initial", "X_CO2 Initial", "X_CO Initial",
        "t_ignition (s)", "t_ignition Outlier", "m_ignition (g)",
        "m_ignition Outlier", "Residue Yield (%)", "Residue Yield Outlier",
        "Heat Release Rate Outlier", "Average HRRPUA 60s (kW/m2)",
        "Average HRRPUA 60s Outlier", "Average HRRPUA 180s (kW/m2)",
        "Average HRRPUA 180s Outlier", "Average HRRPUA 300s (kW/m2)",
        "Average HRRPUA 300s Outlier", "Steady Burning MLRPUA (g/s-m2)",
        "Steady Burning MLRPUA Outlier", "Peak MLRPUA (g/s-m2)",
        "Peak MLRPUA Outlier", "Steady Burning HRRPUA (kW/m2)",
        "Steady Burning HRRPUA Outlier", "Peak HRRPUA (kW/m2)",
        "Peak HRRPUA Outlier", "Total Heat Release (MJ/m2)",
        "Total Heat Release Outlier", "Effective Heat Of Combustion (kJ/g)",
        "Effective Heat Of Combustion Outlier",
        "Average Specific Extinction Area (m2/kg)",
        "Average Specific Extinction Area Outlier",
        "Smoke Production Pre-ignition (m2/m2)",
        "Smoke Production Pre-ignition Outlier",
        "Smoke Production Post-ignition (m2/m2)",
        "Smoke Production Post-ignition Outlier",
        "Smoke Production Total (m2/m2)",
        "Smoke Production Total Outlier", "Y_Soot (g/g)", "Y_Soot Outlier",
        "Y_CO2 (g/g)", "Y_CO2 Outlier", "Y_CO (g/g)", "Y_CO Outlier",
        "Fire Growth Potential (m2/J)", "Fire Growth Potential Outlier",
        "Ignition Energy (MJ/m2)", "Ignition Energy Outlier",
        "t_flameout (s)", "t_flameout Outlier", "Comments", "Data Corrections",
    ]

    metadata_json = {key: None for key in expected_keys}
    metadata_json["Comments"] = []
    metadata_json["Data Corrections"] = []

    lines = [str(line).strip() for line in input]

    def numeric_after_colon(line):
        if ":" not in line:
            return None
        match = re.search(
            r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][-+]?\d+)?",
            line.split(":", 1)[1],
        )
        return float(match.group(0)) if match else None

    def set_if_value(key, value):
        if value is not None:
            metadata_json[key] = value

    material_lines = []
    reading_material = False

    for i, line in enumerate(lines):
        if not line:
            continue

        upper = line.upper()

        # Test date, time, and operator share a line in this format.
        date_time_operator = re.search(
            r"Test Date\s*:\s*(.*?)\s+Test Time\s*:\s*(.*?)\s+"
            r"Operator\s*:\s*(.*)",
            line,
            flags=re.IGNORECASE,
        )
        if date_time_operator:
            date_text = date_time_operator.group(1).strip()
            try:
                metadata_json["Test Date"] = datetime.strptime(
                    date_text, "%d-%b-%y"
                ).strftime("%Y-%m-%d")
            except ValueError:
                metadata_json["Test Date"] = date_text

            raw_time = date_time_operator.group(2).strip()
            try:
                normalized_time = raw_time.lower().replace(".", "").upper()
                metadata_json["Test Time"] = datetime.strptime(
                    normalized_time.upper(), "%I:%M %p"
                ).strftime("%H:%M")
            except ValueError:
                metadata_json["Test Time"] = raw_time
            metadata_json["Operator"] = date_time_operator.group(3).strip()
            continue

        # Material name and description are listed on following lines.
        if re.match(r"^Sample Material\s*:", line, flags=re.IGNORECASE):
            reading_material = True
            continue

        if reading_material:
            if line.startswith("-"):
                reading_material = False
            else:
                material_lines.append(line)
                continue

        if line.startswith("-"):
            reading_material = False

        if upper.startswith("SAMPLE SURFACE AREA"):
            set_if_value("Surface Area (m2)", numeric_after_colon(line))
        elif upper.startswith("RADIANT HEAT FLUX"):
            set_if_value("Heat Flux (kW/m2)", numeric_after_colon(line))
        elif upper.startswith("SAMPLE ORIENTATION"):
            orientation = line.split(":", 1)[1].strip() if ":" in line else ""
            metadata_json["Orientation"] = orientation.upper() or None
        elif upper.startswith("CALIBRATION FACTOR"):
            set_if_value("C Factor", numeric_after_colon(line))
        elif upper.startswith("CONVERSION FACTOR"):
            set_if_value(
                "Heat of Combustion O2 (MJ/kg)",
                numeric_after_colon(line),
            )
        elif upper.startswith("BASELINE O2 READING"):
            set_if_value("X_O2 Initial", numeric_after_colon(line))
        elif upper.startswith("BASELINE CO2 READING"):
            set_if_value("X_CO2 Initial", numeric_after_colon(line))
        elif upper.startswith("BASELINE CO READING"):
            set_if_value("X_CO Initial", numeric_after_colon(line))
        elif upper.startswith("ENTERED INITIAL SPECIMEN MASS"):
            set_if_value("Sample Mass (g)", numeric_after_colon(line))
        elif upper.startswith("MEASURED FINAL SPECIMEN MASS"):
            set_if_value("Residual Mass (g)", numeric_after_colon(line))
        elif upper.startswith("TIME TO SUSTAINED IGNITION"):
            value = numeric_after_colon(line)
            if value is not None:
                set_if_value("t_ignition (s)", int(round(value)))
 

    if material_lines:
        metadata_json["Material Name"] = material_lines[0]
        if len(material_lines) > 1:
            metadata_json["Sample Description"] = " ".join(material_lines[1:])

        # Parse thickness from descriptions such as "Samp.Thk. 1.6mm".
        thickness_match = re.search(
            r"(?:THK\.?|THICKNESS)\s*([0-9]+(?:\.[0-9]+)?)\s*MM",
            " ".join(material_lines),
            flags=re.IGNORECASE,
        )
        if thickness_match:
            metadata_json["Thickness (mm)"] = float(thickness_match.group(1))

        # Parse sample/pan mass when present in the notes.
        all_notes = " ".join(lines)
        sample_mass_match = re.search(
            r"(?:SAMP\.?\s*WT\.?|INITIAL\s+WEIGHT)\s*=?\s*"
            r"([0-9]+(?:\.[0-9]+)?)\s*G",
            all_notes,
            flags=re.IGNORECASE,
        )
        if sample_mass_match and metadata_json["Sample Mass (g)"] is None:
            metadata_json["Sample Mass (g)"] = float(sample_mass_match.group(1))

    # Populate the existing output metadata fields and provenance.
    metadata_json["Original Testname"] = test_name
    metadata_json["Instrument"] = "NBS Cone Calorimeter"
    metadata_json["Preparsed"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    metadata_json["Original Source"] = "Box/Alex_Morgan"
    metadata_json["Comments"] = lines
    metadata_json["Data Corrections"] = []

    if any("edge control" in line.lower() for line in lines):
        metadata_json["Edge Frame"] = True
        metadata_json['Surface Area (m2)'] = 0.0088
    if metadata_json["Surface Area (m2)"] == 0.01:
        metadata_json["Edge Frame"] = False
    elif (
        metadata_json["Surface Area (m2)"] is not None
        and 0.008 < metadata_json["Surface Area (m2)"] <= 0.0095
    ):
        metadata_json["Edge Frame"] = True

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata_json, f, indent=4)

    print(colorize(f"Generated {meta_path}", "blue"))
    return None


#region helpers
def get_number(item, num_type):
    """Extract an integer, float, or exponent number from a string."""
    number = None

    if num_type == "int":
        match = re.search(r"[-+]?\d+", str(item))
        if match:
            number = int(match.group())
    elif num_type == "flt":
        match = re.search(
            r"[-+]?(?:\d+\.\d*|\.\d+|\d+)",
            str(item),
        )
        if match:
            number = float(match.group())
    elif num_type == "exp":
        match = re.search(
            r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)[Ee][-+]?\d+",
            str(item),
        )
        if match:
            number = match.group()

    return number


def get_field(item):
    """Return the text after the first colon or equals sign."""
    for separator in (":", "="):
        if separator in item:
            return item.split(separator, 1)[1].strip()
    return item.strip()


#region main
if __name__ == "__main__":
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(LOG_FILE, "w", encoding="utf-8") as f:
        json.dump({}, f, indent=4)

    print("✅ preparse_AM_log.json created.")
    parse_dir(INPUT_DIR)