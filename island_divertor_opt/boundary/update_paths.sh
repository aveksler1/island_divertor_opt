#!/bin/bash

# Define the root directory to start the search
BASE_DIR="."

# Define the replacements in pairs [SEARCH_STR, REPLACE_STR]
# Using | as a delimiter in sed so we don't have to escape the slashes /
OLD_ROOT="/global/homes/a/aveksler/research/divertor_paper/data/input_files"
NEW_ROOT="/global/cfs/cdirs/m4680/av3249/research/divertor_paper/data/input_files"

echo "Updating file paths inside JSON configuration files..."

# Use find to locate all fwd and bwd files recursively
find "$BASE_DIR" -type f \( -name "classical_divertor_fwd_islands_*" -o -name "classical_divertor_bwd_islands_*" \) | while read -r file; do
    
    # 1. Update mgrid_file path
    sed -i "s|${OLD_ROOT}/mgrid_fpp_jpp_vacuum.nc|${NEW_ROOT}/mgrid_fpp_jpp_vacuum.nc|g" "$file"
    
    # 2. Update island_boundary_file path
    sed -i "s|${OLD_ROOT}/island_boundary_halfdegree.csv|${NEW_ROOT}/island_boundary_halfdegree.csv|g" "$file"
    
    # 3. Update o_points_file path
    sed -i "s|${OLD_ROOT}/island_opoints.csv|${NEW_ROOT}/island_opoints.csv|g" "$file"

    echo "Updated: $file"
done

echo "Done. All paths have been migrated to the research directory."