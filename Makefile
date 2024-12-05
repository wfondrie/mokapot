
DOCKER_IMAGE_NAME=wfondrie/mokapot
DOCKER_IMAGE_TAG=latest

test:
	uv run --all-extras pytest --durations=0

check: ruff-lint format pre-commit
	@echo "All checks passed"

build: build-wheel build-docker
	@echo "Build completed"

build-wheel:
	uv run --with build python -m build --wheel .

build-docker:
	uv run --with build python -m build --wheel --outdir dist .
	docker build -t $(DOCKER_IMAGE_NAME):$(DOCKER_IMAGE_TAG) .

pre-commit:
	pre-commit run --all-files

ruff-lint:
	uv run ruff check .

ruff-lint-ci:
	uv run ruff check . --output-format=github

format:
	uv run ruff format .
