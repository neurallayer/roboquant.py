
[[ ! -f "LICENSE" ]] && echo "run the script from the project root directory like this: ./bin/local_install.sh" && exit 1

rm -rf ./runs

uv sync --all-extras

# QA
uvx ruff check || exit 1
uv run python -m unittest discover -s tests/unit || exit 1

# Build
rm -rf dist
uv build || exit 1

# Install
uv pip install .

