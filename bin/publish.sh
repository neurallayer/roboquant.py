source .venv/bin/activate

# QA
flake8 roboquant tests || exit 1
pylint roboquant tests || exit 1
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
