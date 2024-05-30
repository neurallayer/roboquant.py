
[[ ! -f "LICENSE" ]] && echo "run the script from the project root directory like this: ./bin/publish.sh" && exit 1

source .venv/bin/activate

rm -rf ./runs

# QA
flake8 || exit 1
pylint roboquant tests samples || exit 1
python -m unittest discover -s tests/unit || exit 1

# Build
rm -rf dist
python -m build || exit 1

# Publish
read -p "Publish (y/n)? " ANSWER
if [ "$ANSWER" = "y" ]; then
  twine upload dist/*; exit 0
else
  echo "Not published"; exit 1
fi
