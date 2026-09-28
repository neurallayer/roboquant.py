#!/bin/bash
[[ ! -f "LICENSE" ]] && echo "run the script from the project root directory like this: ./bin/create_docs.sh" && exit 1

# generate the the website and start listening 
cd docs
uv run --with jupyter jupyter book start --execute
