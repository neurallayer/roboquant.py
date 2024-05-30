
[[ ! -f "LICENSE" ]] && echo "run the script from the project root directory like this: ./bin/local_install.sh" && exit 1

source .venv/bin/activate

rm -rf ./runs

# QA
flake8 roboquant tests || exit 1
pylint roboquant tests samples || exit 1
python -m unittest discover -s tests/unit || exit 1

# Build
rm -rf dist
python -m build || exit 1

# Install
pip install .
