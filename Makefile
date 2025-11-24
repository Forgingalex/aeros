.PHONY: help install train generate-data export-onnx test clean run-api run-web docker-build docker-up docker-down

help:
	@echo "AEROS Makefile Commands:"
	@echo "  make install          - Install Python dependencies"
	@echo "  make generate-data    - Generate synthetic corridor dataset"
	@echo "  make train            - Train the heading estimation model"
	@echo "  make export-onnx      - Export PyTorch model to ONNX"
	@echo "  make test             - Run tests"
	@echo "  make run-api          - Run FastAPI server"
	@echo "  make run-web          - Run React dashboard"
	@echo "  make docker-build     - Build Docker images"
	@echo "  make docker-up        - Start Docker containers"
	@echo "  make docker-down      - Stop Docker containers"
	@echo "  make clean            - Clean generated files"

install:
	pip install -r requirements.txt
	cd web && npm install

generate-data:
	python scripts/generate_synthetic_data.py --output-dir data/synthetic --num-samples 10000

train:
	python scripts/train.py

export-onnx:
	python scripts/export_onnx.py --checkpoint models/checkpoints/best_model.pth --output models/heading_model.onnx

test:
	pytest tests/ -v --cov=src --cov-report=html

run-api:
	uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

run-web:
	cd web && npm start

docker-build:
	docker-compose build

docker-up:
	docker-compose up -d

docker-down:
	docker-compose down

clean:
	find . -type d -name __pycache__ -exec rm -r {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name ".pytest_cache" -exec rm -r {} +
	find . -type d -name ".mypy_cache" -exec rm -r {} +
	rm -rf web/build
	rm -rf web/node_modules
	rm -rf htmlcov
	rm -rf .coverage

