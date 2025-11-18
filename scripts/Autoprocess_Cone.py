#!/usr/bin/env python3
"""

Script for fitting all Cone data in database

"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from utils import colorize


def get_series_names(folder_path):
    """Get unique series names from CSV files in folder"""
    folder = Path(folder_path)
    series_set = set()
    
    # Find all CSV files recursively
    for csv_file in folder.rglob("*.csv"):
        stem = csv_file.stem
        if "Average" not in stem and 'Empty' not in stem:
            # Extract series name by removing replicate number
            series_name = stem.rsplit('_R', 1)[0] if '_R' in stem else stem.rsplit('_r', 1)[0]
            series_set.add(series_name)
    
    return sorted(list(series_set))  # Return sorted list for consistent ordering

if __name__ == "__main__":
    # Set up command-line arguments
    SCRIPT_DIR = Path(__file__).parent
    PROJECT_ROOT = SCRIPT_DIR.parent
    parser = argparse.ArgumentParser(description='Process all Cone series that need updating')
    parser.add_argument('--root', type=str, default=str(PROJECT_ROOT),
                        help='Project root directory')
    parser.add_argument('--prepared', type=str,
                        help='Prepared data directory (default: ROOT/Exp-Data_Prepared-Final/)')
    parser.add_argument('--metadata', type=str,
                        help='Metadata directory (default: ROOT/Metadata/Prepared-Final)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show what would be processed without actually running')
    parser.add_argument('--filter', type=str,
                        help='Only process series matching this pattern')
    
    args = parser.parse_args()
    
    # Set up directories
    ROOT_DIR = Path(args.root)
    prepared_dir = Path(args.prepared) if args.prepared else ROOT_DIR / "Exp-Data_Prepared-Final" 
    metadata_dir = Path(args.metadata) if args.metadata else ROOT_DIR / "Metadata" / "Prepared-Final"
    
    # Get all series names
    cone_tests = get_series_names(prepared_dir)
    
    # Apply filter if provided
    if args.filter:
        cone_tests = [t for t in cone_tests if args.filter in t]
    
    print(f"Found {len(cone_tests)} Cone series to check")
    
    # Process each series
    processed_count = 0
    processed_successfully_count = 0
    skipped_count = 0
    
    for test in cone_tests:
        # Initialize counts
        noauto_count = 0
        new_manual_review_count = 0
        
        # Check all metadata files for this series
        for json_file in metadata_dir.rglob(f"{test}_[rR]*.json"):
            try:
                with open(json_file, "r") as f:
                    check = json.load(f)
                
                if not check.get("Autoprocessed"):
                    noauto_count += 1
                    
                if check.get("Manually Reviewed Series"):
                    if check["Manually Reviewed Series"] > check.get("Autoprocessed", "") and check.get('Pass Review'):
                        new_manual_review_count += 1
                        
            except (json.JSONDecodeError, KeyError) as e:
                print(colorize(f"Error reading {json_file}: {e}", "red"))
                continue
        
        # Determine action
        if noauto_count != 0:
            print(colorize(f"Autoprocessing {test}, new data has been added to the database", "blue"))
        elif new_manual_review_count != 0:
            print(colorize(f"Re-Autoprocessing {test}, series has been manually reviewed since last autoprocess", "purple"))
        else:
            print(colorize(f"Skipping {test}, all data has already been autoprocessed since last manual review", "yellow"))
            skipped_count += 1
            continue
        
        # Run autoprocessing
        if not args.dry_run:
            cmd = [sys.executable, str(PROJECT_ROOT / "scripts" / "Autoprocess_Cone_IndSeries.py"), test, "--root", str(PROJECT_ROOT)]
            processed_count+=1
            result = subprocess.run(cmd, text=True)
            if result.returncode != 0:
                print(colorize(f"Error processing {test}: {result.stderr}\n", "red"))
            else:
                processed_successfully_count += 1
                print(colorize("Autoprocessed successfully\n", "green"))
        else:
            print(f"  [DRY RUN] Would run: python Autoprocess_Cone_IndSeries.py {test}")
            processed_successfully_count += 1
    
    if processed_count != 0:
        # Summary
        print(colorize(f"Test series autoprocessed successfully: {processed_successfully_count}/{processed_count} ({(processed_successfully_count/processed_count) * 100}%), {skipped_count} series skipped", "purple"))
    else:
        print(colorize(f"All {skipped_count} series skipped and up to date", "purple"))