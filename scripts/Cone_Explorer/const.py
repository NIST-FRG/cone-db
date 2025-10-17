from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent         # .../coneDB/scripts/PH_Cone_Explorer
PROJECT_ROOT = SCRIPT_DIR.parent.parent             # .../coneDB 


# Global constants
INPUT_DATA_PATH = PROJECT_ROOT / "scripts"/ "PH_cone-explorer"/ "data" / "parsed"
OUTPUT_DATA_PATH = PROJECT_ROOT / "scripts"/ "PH_cone-explorer"/ "data" / "prepared-final"
PARSED_DATA_PATH = PROJECT_ROOT / "Exp-Data_Parsed"
PARSED_METADATA_PATH = PROJECT_ROOT / "Metadata"/ "Parsed"
PREPARED_DATA_PATH = PROJECT_ROOT / "Exp-Data_Prepared-Final"
PREPARED_METADATA_PATH = PROJECT_ROOT / "Metadata"/ "Prepared-Final"
