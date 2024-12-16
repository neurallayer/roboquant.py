
[[ ! -f "LICENSE" ]] && echo "run the script from the project root directory like this: ./bin/publish.sh" && exit 1

rm -rf ./runs

uv sync --all-extras --dev

# QA
uvx ruff check
uv run python -m unittest discover -s tests/unit || exit 1

# Build
rm -rf dist
uv build || exit 1

# Publish
read -p "Publish (y/n)? " ANSWER
if [ "$ANSWER" = "y" ]; then
  uv publish; exit 0
else
  echo "Not published"; exit 1
fi
