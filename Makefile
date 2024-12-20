
# TODO: Make these over-rideable
DOCKER_IMAGE_NAME=wfondrie/mokapot
DOCKER_IMAGE_TAG=latest

test:
	uv run --all-extras pytest --durations=0 --slow-last

testff:
	# Test btut fails fast
	uv run --all-extras pytest --durations=0 --slow-last --last-failed -xs

unit-test:
	uv run --all-extras pytest --durations=0 ./tests/unit_tests

check: ruff-lint format pre-commit
	@echo "All checks passed"

build: build-wheel build-sdist build-docker
	@echo "Build completed"

build-wheel:
	uv run --with build python -m build --wheel .

build-sdist:
	uv run --with build python -m build --sdist .

build-docker:
	uv run --with build python -m build --wheel --outdir dist .
	docker build -t $(DOCKER_IMAGE_NAME):$(DOCKER_IMAGE_TAG) .
	docker run --rm $(DOCKER_IMAGE_NAME):$(DOCKER_IMAGE_TAG) mokapot --help

pre-commit:
	pre-commit run --all-files

ruff-lint:
	uv run ruff check .

lint-ci:
	uv run --no-project --with ruff ruff check . --output-format=github

format:
	uv run ruff format .

format-ci:
	uv run --no-project --with ruff ruff format --check .
