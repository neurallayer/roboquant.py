
[[ ! -f "LICENSE" ]] && echo "run the script from the project root directory like this: ./bin/publish.sh" && exit 1
export PYRIGHT_PYTHON_FORCE_VERSION="latest"

rm -rf ./runs

./bin/verify.sh

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
