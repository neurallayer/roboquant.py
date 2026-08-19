#!/bin/bash
[[ ! -f "LICENSE" ]] && echo "run the script from the project root directory like this: ./bin/create_docs.sh" && exit 1

files=`find docs -name "*.md"`

for file in $files; do
    if [ -f "$file" ]; then
        nbfile="${file/\.py/.ipynb}"
        echo "Processing: $file"
        uv run jupytext --to ipynb "$file"
    fi
done
