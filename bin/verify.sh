[[ ! -f "LICENSE" ]] && echo "run the script from the project root directory like this: ./bin/verify.sh" && exit 1

uv sync --all-extras --dev

# QA
echo "Running ruff" 
uvx ruff check || exit 1
echo "Running unittest" 
uv run python -m unittest discover -s tests/unit || exit 1

echo "All tests passed"