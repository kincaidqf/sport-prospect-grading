.PHONY: build build-gpu up up-gpu train notebook shell clean

# ── CPU / Mac ──────────────────────────────────────────────────────────────────
build:
	docker compose -f docker/docker-compose.yml build

up:
	docker compose -f docker/docker-compose.yml up

train:
	docker compose -f docker/docker-compose.yml run --rm nba-ml \
		uv run python src/main.py --config src/config/config.yaml $(ARGS)

notebook:
	docker compose -f docker/docker-compose.yml run --rm --service-ports nba-ml \
		uv run jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root

shell:
	docker compose -f docker/docker-compose.yml run --rm nba-ml bash

# ── GPU / RTX 5090 ─────────────────────────────────────────────────────────────
build-gpu:
	docker compose -f docker/docker-compose.yml -f docker/docker-compose.gpu.yml build

up-gpu:
	docker compose -f docker/docker-compose.yml -f docker/docker-compose.gpu.yml up

train-gpu:
	docker compose -f docker/docker-compose.yml -f docker/docker-compose.gpu.yml \
		run --rm nba-ml python src/main.py --config src/config/config.yaml $(ARGS)

shell-gpu:
	docker compose -f docker/docker-compose.yml -f docker/docker-compose.gpu.yml \
		run --rm nba-ml bash

# ── Misc ───────────────────────────────────────────────────────────────────────
clean:
	docker compose -f docker/docker-compose.yml down --rmi local --volumes
