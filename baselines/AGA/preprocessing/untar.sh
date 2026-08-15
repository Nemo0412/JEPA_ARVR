#!/bin/bash

# Use the first argument as the target directory, or default to the current directory (.)
TARGET_DIR="${1:-.}"

echo "Searching for .tar files in ${TARGET_DIR}..."

# Find all .tar files and read them line by line
find "$TARGET_DIR" -type f -name "*.tar" | while read -r tar_file; do
    # Get the directory path containing the tar file
    dir_path=$(dirname "$tar_file")
    
    echo "Extracting: $tar_file  ->  $dir_path/"
    
    # Extract the tar file (-x) into its specific directory (-C)
    tar -xf "$tar_file" -C "$dir_path"
    
    # Optional check: If you want to delete the .tar file after successful extraction, 
    # uncomment the lines below:
    # if [ $? -eq 0 ]; then
    #     rm "$tar_file"
    # fi
done

echo "All extractions complete!"

