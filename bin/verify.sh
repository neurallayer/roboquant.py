
[[ ! -f "LICENSE" ]] && echo "run the script from the project root directory like this: ./bin/verify.sh" && exit 1

source .venv/bin/activate

# QA
flake8 || exit 1
pylint roboquant tests || exit 1
python -m unittest discover -s tests/unit || exit 1

echo "All tests passed"