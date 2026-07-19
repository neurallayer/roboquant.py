# Building from source
Roboquant.py uses `uv` as the main tool for handling package dependencies and the below code snippets assume `uv` is already
installed.

```shell
gti clone https://github.com/neurallayer/roboquant.py.git
cd roboquant.py
uv sync --all-extras --dev
```

You should have all the dependencies installed and be ready to build and locally install *roboquant*:

```shell
uv build
uv pip install dist/*.whl
```

## Shell scripts

For convenience there are some shell scripts in the ./bin directory

### Build & validate the code

```shell
./bin/verify.sh
```

The validation will run the following three tools:
1. ruff (linter and code formatter)
2. pyright (static type checker)
3. unit test

### Publish

This only works if UV_PUBLISH_TOKEN is set and is limited to the 
core maintainers of roboquant.

```shell
./bin/publish.sh
```