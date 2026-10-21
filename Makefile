DATA_DIR ?= data
ARTIFACTS ?= artifacts

.PHONY: install train serve test docker-build docker-up docker-down clean

install:        ## install the package with dev dependencies (editable)
	pip install -e ".[dev]"

train:          ## train the model and write artifacts
	starcast-train --data-dir $(DATA_DIR) --out $(ARTIFACTS)

serve:          ## run the API locally with auto-reload
	ARTIFACT_DIR=$(ARTIFACTS) uvicorn starcast.api.main:app --reload --port 8000

test:           ## run the test suite
	pytest -q

docker-build:   ## build the Docker image (train first)
	docker compose build

docker-up:      ## start the API container on :8000
	docker compose up -d

docker-down:    ## stop the API container
	docker compose down

clean:
	rm -rf .pytest_cache build *.egg-info src/*.egg-info
	find . -name __pycache__ -prune -exec rm -rf {} +
