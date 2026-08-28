#!/bin/bash
[[ ! -f "LICENSE" ]] && echo "run the script from the project root directory like this: ./bin/create_docs.sh" && exit 1

cd scratch
uv run --with jupyter jupyter lab test.ipynb
