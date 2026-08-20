import os
import json
import shutil
from glob import glob
from pathlib import Path
from datetime import datetime

def get_repo_paths():
    """Get paths to both repositories"""
    script_dir = Path(__file__).parent
    cone_db = script_dir.parent
    nist_frg = cone_db.parent
    matl_flam = nist_frg / "Matl-Flam-DB-Developers"
    
    return cone_db, matl_flam

def parse_smurf_date(smurf_value):
    """Parse SmURF datetime string to datetime object"""
    if not smurf_value:
        return None
    try:
        for fmt in ["%Y-%m-%d %H:%M:%S", "%m/%d/%Y %H:%M:%S", "%Y-%m-%dT%H:%M:%S"]:
            try:
                return datetime.strptime(smurf_value, fmt)
            except ValueError:
                continue
        return None
    except Exception:
        return None

def get_parse_script(og_source):
    """Determine which parsing script to use based on source"""
    if "FTT" in og_source:
        return "Parse_Cone-FTT.py"
    elif "md_A" in og_source:
        return "Parse_Cone-mdA.py"
    elif "md_B" in og_source:
        return "Parse_Cone-mdB.py"
    elif "md_C" in og_source:
        return "Parse_Cone-mdC.py"
    else:
        return "Unknown"

def gather_test_metadata(cone_db):
    """Step 1: Locate all test metadata files"""
    metadata_path = cone_db / "Metadata" / "Prepared-Final"
    json_files = glob(str(metadata_path / "**" / "*.json"), recursive=True)
    
    file_groups = []
    
    for json_file in json_files:
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        testname = data.get("Testname")
        mat_id = data.get("Material ID")
        og_testname = data.get("Original Testname")
        og_source = data.get("Original Source", "")
        smurf = data.get("SmURF")
        
        folder = og_source.split("/")[0] if "/" in og_source else og_source
        
        group = {
            "metadata_json": Path(json_file),
            "testname": testname,
            "mat_id": mat_id,
            "og_testname": og_testname,
            "og_source": og_source,
            "folder": folder,
            "smurf": smurf,
            "prepared_csv": cone_db / "Exp-Data_Prepared-Final" / folder / f"{testname}.csv",
            "material_json": cone_db / "Metadata" / "Materials" / f"{mat_id}.json",
            "parsed_csv": cone_db / "Exp-Data_Parsed" / og_source / f"{og_testname}.csv",
        }
        
        file_groups.append(group)
    
    return file_groups

def determine_all_operations(file_groups, cone_db, matl_flam):
    """Determine all operations for all files"""
    operations = []
    checked_materials = {}
    missing_files = []
    
    for group in file_groups:
        testname = group["testname"]
        folder = group["folder"]
        mat_id = group["mat_id"]
        
        dest_metadata = matl_flam / "Metadata" / "Cone" / folder / f"{testname}.json"
        dest_material = matl_flam / "Metadata" / "Materials" / f"{mat_id}.json"
        source_material = cone_db / "Metadata" / "Materials" / f"{mat_id}.json"
        dest_parsed = matl_flam / "Exp-Data_Parsed" / "Cone" / folder / f"{testname}.csv"
        dest_prepared = matl_flam / "Exp-Data_Prepared-Final" / "Cone" / folder / f"{testname}.csv"
        
        op = {
            "group": group,
            "dest_metadata": dest_metadata,
            "dest_parsed": dest_parsed,
            "dest_prepared": dest_prepared,
            "dest_material": dest_material,
            "source_material": source_material,
            "metadata_action": None,
            "parsed_action": None,
            "prepared_action": None,
            "material_action": None,
            "parse_script": None,
            "skip_all": False,
            "missing_reasons": [],
        }
        
        # Check for missing source files
        missing = []
        
        if not group["parsed_csv"].exists():
            parse_script = get_parse_script(group["og_source"])
            op["parse_script"] = parse_script
            missing.append(f"Parsed CSV (run {parse_script})")
        
        if not group["prepared_csv"].exists():
            missing.append("Prepared CSV")
        
        # Check material (only once per mat_id)
        # Material is OK if it exists in EITHER source OR destination
        if mat_id not in checked_materials:
            source_exists = source_material.exists()
            dest_exists = dest_material.exists()
            
            if dest_exists:
                # Already in destination - no action needed
                checked_materials[mat_id] = "SKIP"
            elif source_exists:
                # In source but not destination - copy it
                checked_materials[mat_id] = "COPY"
            else:
                # Missing from both - error
                checked_materials[mat_id] = "MISSING"
        
        # Only add to missing if material doesn't exist in EITHER location
        if checked_materials[mat_id] == "MISSING":
            missing.append(f"Material metadata ({mat_id}.json)")
        
        # If any file is missing, skip entire test
        if missing:
            op["skip_all"] = True
            op["missing_reasons"] = missing
            op["metadata_action"] = "SKIP"
            op["parsed_action"] = "SKIP"
            op["prepared_action"] = "SKIP"
            op["material_action"] = "SKIP"
            missing_files.append({
                "testname": testname,
                "og_source": group["og_source"],
                "parse_script": op["parse_script"],
                "missing": missing,
            })
            operations.append(op)
            continue
        
        # Determine metadata action
        if not dest_metadata.exists():
            op["metadata_action"] = "COPY"
        else:
            with open(dest_metadata, 'r') as f:
                dest_data = json.load(f)
            
            source_smurf = parse_smurf_date(group["smurf"])
            dest_smurf = parse_smurf_date(dest_data.get("SmURF"))
            
            if source_smurf and dest_smurf and source_smurf > dest_smurf:
                op["metadata_action"] = "OVERWRITE"
            else:
                op["metadata_action"] = "SKIP"
        
        metadata_action = op["metadata_action"]
        
        # Determine parsed action
        if metadata_action in ["COPY", "OVERWRITE"]:
            op["parsed_action"] = "COPY"
        elif not dest_parsed.exists():
            op["parsed_action"] = "COPY"
        else:
            op["parsed_action"] = "SKIP"
        
        # Determine prepared action
        if metadata_action in ["COPY", "OVERWRITE"]:
            op["prepared_action"] = "COPY"
        elif not dest_prepared.exists():
            op["prepared_action"] = "COPY"
        else:
            op["prepared_action"] = "SKIP"
        
        # Material action
        op["material_action"] = checked_materials[mat_id]
        
        operations.append(op)
    
    return operations, checked_materials, missing_files

def print_operations(operations):
    """Print condensed output - one line per test"""
    print("\n" + "="*90)
    print("OPERATIONS SUMMARY")
    print("="*90)
    print(f"{'Test Name':<40} {'Meta':^8} {'Parsed':^8} {'Prep':^8} {'Mat':^8} {'Status':^10}")
    print("-"*90)
    
    for op in operations:
        testname = op["group"]["testname"]
        
        if op["skip_all"]:
            print(f"{testname:<40} {'—':^8} {'—':^8} {'—':^8} {'—':^8} {'MISSING':^10}")
        else:
            meta = op["metadata_action"][:4] if op["metadata_action"] else "—"
            parsed = op["parsed_action"][:4] if op["parsed_action"] else "—"
            prep = op["prepared_action"][:4] if op["prepared_action"] else "—"
            mat = op["material_action"][:4] if op["material_action"] else "—"
            print(f"{testname:<40} {meta:^8} {parsed:^8} {prep:^8} {mat:^8} {'OK':^10}")

def print_summary(operations, checked_materials, missing_files):
    """Print summary counts"""
    print("\n" + "="*90)
    print("COUNTS")
    print("="*90)
    
    # Filter out skipped tests
    valid_ops = [op for op in operations if not op["skip_all"]]
    skipped_ops = [op for op in operations if op["skip_all"]]
    
    counts = {
        "metadata": {"COPY": 0, "OVERWRITE": 0, "SKIP": 0},
        "parsed": {"COPY": 0, "SKIP": 0},
        "prepared": {"COPY": 0, "SKIP": 0},
    }
    
    for op in valid_ops:
        counts["metadata"][op["metadata_action"]] += 1
        counts["parsed"][op["parsed_action"]] += 1
        counts["prepared"][op["prepared_action"]] += 1
    
    # Count materials (only from valid operations)
    valid_mat_ids = set(op["group"]["mat_id"] for op in valid_ops)
    mat_counts = {"COPY": 0, "SKIP": 0}
    for mat_id in valid_mat_ids:
        action = checked_materials.get(mat_id)
        if action in ["COPY", "SKIP"]:
            mat_counts[action] += 1
    
    print(f"Total tests: {len(operations)}")
    print(f"Valid tests: {len(valid_ops)}")
    print(f"Skipped (missing files): {len(skipped_ops)}")
    print()
    print(f"Test Metadata:  Copy: {counts['metadata']['COPY']}, Overwrite: {counts['metadata']['OVERWRITE']}, Skip: {counts['metadata']['SKIP']}")
    print(f"Parsed CSV:     Copy: {counts['parsed']['COPY']}, Skip: {counts['parsed']['SKIP']}")
    print(f"Prepared CSV:   Copy: {counts['prepared']['COPY']}, Skip: {counts['prepared']['SKIP']}")
    print(f"Material Meta:  Copy: {mat_counts['COPY']}, Skip: {mat_counts['SKIP']}")
    
    # Print missing files section
    if missing_files:
        print("\n" + "="*90)
        print("ACTION REQUIRED: Tests skipped due to missing files")
        print("="*90)
        
        # Group by parse script
        by_script = {}
        missing_materials = set()
        
        for item in missing_files:
            for reason in item["missing"]:
                if "Parsed CSV" in reason:
                    script = item["parse_script"] or "Unknown"
                    if script not in by_script:
                        by_script[script] = []
                    by_script[script].append(item["testname"])
                elif "Material metadata" in reason:
                    mat_id = reason.split("(")[1].split(".")[0]
                    missing_materials.add(mat_id)
        
        if by_script:
            print("\nMissing parsed data - run these scripts:")
            for script, tests in by_script.items():
                print(f"\n  {script}: {len(tests)} files")
                for test in tests:
                    print(f"    - {test}")
        
        if missing_materials:
            print("\nMissing material metadata (not in source OR destination) - generate these files:")
            for mat_id in sorted(missing_materials):
                print(f"  - {mat_id}.json")

def execute_operations(operations, checked_materials):
    """Execute copy operations"""
    print("\n" + "="*90)
    print("EXECUTING")
    print("="*90)
    
    copied_materials = set()
    copied_count = 0
    
    for op in operations:
        # Skip if any file was missing
        if op["skip_all"]:
            continue
        
        group = op["group"]
        testname = group["testname"]
        mat_id = group["mat_id"]
        
        actions = []
        
        # Copy metadata
        if op["metadata_action"] in ["COPY", "OVERWRITE"]:
            op["dest_metadata"].parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(group["metadata_json"], op["dest_metadata"])
            actions.append(f"Meta:{op['metadata_action'][:4]}")
        
        # Copy parsed CSV
        if op["parsed_action"] == "COPY":
            op["dest_parsed"].parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(group["parsed_csv"], op["dest_parsed"])
            actions.append("Parsed:COPY")
        
        # Copy prepared CSV
        if op["prepared_action"] == "COPY":
            op["dest_prepared"].parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(group["prepared_csv"], op["dest_prepared"])
            actions.append("Prep:COPY")
        
        # Copy material metadata (only once per mat_id)
        if mat_id not in copied_materials and checked_materials.get(mat_id) == "COPY":
            op["dest_material"].parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(op["source_material"], op["dest_material"])
            actions.append(f"Mat:COPY({mat_id})")
            copied_materials.add(mat_id)
        
        if actions:
            print(f"{testname}: {', '.join(actions)}")
            copied_count += 1
    
    return copied_count

def main():
    cone_db, matl_flam = get_repo_paths()
    
    print(f"Cone DB: {cone_db}")
    print(f"Matl-Flam: {matl_flam}")
    
    # Step 1: Locate all test metadata
    file_groups = gather_test_metadata(cone_db)
    print(f"\nFound {len(file_groups)} test metadata files")
    
    # Step 2 & 3: Determine all operations
    operations, checked_materials, missing_files = determine_all_operations(file_groups, cone_db, matl_flam)
    
    # Print condensed operations
    print_operations(operations)
    
    # Print summary
    print_summary(operations, checked_materials, missing_files)
    
    # Check if anything to do
    valid_ops = [op for op in operations if not op["skip_all"]]
    actionable = [op for op in valid_ops if 
                  op["metadata_action"] in ["COPY", "OVERWRITE"] or
                  op["parsed_action"] == "COPY" or
                  op["prepared_action"] == "COPY"]
    
    if not actionable:
        print("\nNo files to copy. Everything is up to date or missing source files.")
        return
    
    # Confirm before executing
    response = input("\nProceed with export? (y/n): ").strip().lower()
    if response != 'y':
        print("Export cancelled.")
        return
    
    # Execute operations
    copied_count = execute_operations(operations, checked_materials)
    
    print("\n" + "="*90)
    print(f"Export complete! Processed {copied_count} tests.")
    print("="*90)

if __name__ == "__main__":
    main()