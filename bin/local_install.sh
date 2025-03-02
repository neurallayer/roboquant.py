
[[ ! -f "LICENSE" ]] && echo "run the script from the project root directory like this: ./bin/local_install.sh" && exit 1
export PYRIGHT_PYTHON_FORCE_VERSION="latest"

rm -rf ./runs

uv sync --all-extras --dev

# QA
uvx ruff check || exit 1
uv run python -m unittest discover -s tests/unit || exit 1

# Build the package
rm -rf dist
uv build || exit 1

# Install the just build package
uv pip install dist/*.whl

