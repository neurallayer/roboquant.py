#!/bin/bash
[[ ! -f "LICENSE" ]] && echo "run the script from the project root directory like this: ./bin/create_docs.sh" && exit 1

uv run pdoc --html --config show_source_code=False --output-dir ./build roboquant