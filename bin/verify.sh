#!/bin/bash
[[ ! -f "LICENSE" ]] && echo "run the script from the project root directory like this: ./bin/verify.sh" && exit 1
export PYRIGHT_PYTHON_FORCE_VERSION="latest"

uv sync --quiet --all-extras --dev

# QA
echo "Running ruff" 
uvx ruff check || exit 1

echo "Running ty" 
uvx ty check roboquant tests 

# echo "Running pyright" 
# uv run pyright tests roboquant || exit 1

echo "Running unittest" 
uv run python -m unittest discover -q -s tests/unit || exit 1

echo "All tests passed"