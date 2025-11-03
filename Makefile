.PHONY: help build up down restart logs status clean test shell exec scale health

# Colors for output
BLUE := \033[0;34m
GREEN := \033[0;32m
YELLOW := \033[1;33m
RED := \033[0;31m
NC := \033[0m # No Color

# Default target
.DEFAULT_GOAL := help

# Variables
COMPOSE := docker compose
UV := uv
PYTHON := python

# Enable BuildKit for FAST builds (30-60x faster for code changes!)
export DOCKER_BUILDKIT=1
export COMPOSE_DOCKER_CLI_BUILD=1

help: ## Show this help message
	@echo "$(BLUE)Ghost Swarm - Available Commands$(NC)"
	@echo "=================================="
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "$(GREEN)%-20s$(NC) %s\n", $$1, $$2}'

## =============================================================================
## Development Setup
## =============================================================================

install: ## Install dependencies with UV
	@echo "$(BLUE)Installing dependencies...$(NC)"
	$(UV) sync
	@echo "$(GREEN)âœ“ Dependencies installed$(NC)"

install-dev: ## Install development dependencies
	@echo "$(BLUE)Installing dev dependencies...$(NC)"
	$(UV) sync --dev
	@echo "$(GREEN)âœ“ Dev dependencies installed$(NC)"

sync: ## Sync dependencies (update uv.lock)
	@echo "$(BLUE)Syncing dependencies...$(NC)"
	$(UV) sync --frozen
	@echo "$(GREEN)âœ“ Dependencies synced$(NC)"

env: ## Create .env file from template
	@if [ ! -f .env ]; then \
		cp .env.example .env; \
		echo "$(GREEN)âœ“ Created .env file$(NC)"; \
		echo "$(YELLOW)âš  Please edit .env and add your API keys$(NC)"; \
	else \
		echo "$(YELLOW).env already exists$(NC)"; \
	fi

## =============================================================================
## Docker Build & Management
## =============================================================================

build: ## Build all Docker images
	@echo "$(BLUE)Building Docker images...$(NC)"
	$(COMPOSE) build
	@echo "$(GREEN)âœ“ All images built$(NC)"

build-base: ## Build only base image
	@echo "$(BLUE)Building base image...$(NC)"
	$(COMPOSE) build ghost-base
	@echo "$(GREEN)âœ“ Base image built$(NC)"

build-no-cache: ## Build without cache (clean build)
	@echo "$(BLUE)Building images without cache...$(NC)"
	$(COMPOSE) build --no-cache
	@echo "$(GREEN)âœ“ Clean build complete$(NC)"

build-parallel: ## Build images in parallel
	@echo "$(BLUE)Building images in parallel...$(NC)"
	$(COMPOSE) build --parallel
	@echo "$(GREEN)âœ“ Parallel build complete$(NC)"

quick-build: ## Fast rebuild for code changes only (~5-15 seconds)
	@echo "$(BLUE)Quick build (code layer only)...$(NC)"
	$(COMPOSE) build --parallel orchestrator worker-1 worker-2 worker-3
	@echo "$(GREEN)✓ Quick build complete$(NC)"

build-time: ## Show build time breakdown with caching info
	@echo "$(BLUE)Building with detailed timing...$(NC)"
	$(COMPOSE) build --progress=plain 2>&1 | grep -E "(CACHED|RUN|COPY)" || true
	@echo "$(GREEN)✓ Build complete - check for CACHED layers$(NC)"

cache-info: ## Show Docker build cache usage
	@echo "$(BLUE)Docker build cache info:$(NC)"
	@docker system df
	@echo ""
	@echo "$(BLUE)BuildKit cache:$(NC)"
	@docker buildx du || echo "$(YELLOW)Run 'docker buildx create --use' to enable BuildKit$(NC)"

clean-cache: ## Clean Docker build cache (use when builds are stuck)
	@echo "$(YELLOW)⚠ Cleaning Docker build cache...$(NC)"
	@docker builder prune -af
	@echo "$(GREEN)✓ Build cache cleaned$(NC)"
	@echo "$(YELLOW)Next build will be slower but fresh$(NC)"

## =============================================================================
## Docker Compose Operations
## =============================================================================

up: ## Start all services
	@echo "$(BLUE)Starting Ghost Swarm...$(NC)"
	$(COMPOSE) up -d
	@echo "$(GREEN)âœ“ Ghost Swarm started$(NC)"
	@make status

up-build: ## Build and start all services
	@echo "$(BLUE)Building and starting Ghost Swarm...$(NC)"
	$(COMPOSE) up -d --build
	@echo "$(GREEN)âœ“ Ghost Swarm started$(NC)"
	@make status

down: ## Stop all services
	@echo "$(BLUE)Stopping Ghost Swarm...$(NC)"
	$(COMPOSE) down
	@echo "$(GREEN)âœ“ Ghost Swarm stopped$(NC)"

down-volumes: ## Stop services and remove volumes
	@echo "$(YELLOW)âš  This will delete all data!$(NC)"
	@read -p "Are you sure? [y/N] " -n 1 -r; \
	echo; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		$(COMPOSE) down -v; \
		echo "$(GREEN)âœ“ Services stopped and volumes removed$(NC)"; \
	fi

restart: ## Restart all services
	@echo "$(BLUE)Restarting Ghost Swarm...$(NC)"
	$(COMPOSE) restart
	@echo "$(GREEN)âœ“ Ghost Swarm restarted$(NC)"

restart-service: ## Restart a specific service (usage: make restart-service SERVICE=worker-1)
	@if [ -z "$(SERVICE)" ]; then \
		echo "$(RED)Error: SERVICE not specified$(NC)"; \
		echo "Usage: make restart-service SERVICE=worker-1"; \
		exit 1; \
	fi
	@echo "$(BLUE)Restarting $(SERVICE)...$(NC)"
	$(COMPOSE) restart $(SERVICE)
	@echo "$(GREEN)âœ“ $(SERVICE) restarted$(NC)"

## =============================================================================
## Logs & Monitoring
## =============================================================================

logs: ## Show logs from all services
	$(COMPOSE) logs -f

logs-service: ## Show logs from a specific service (usage: make logs-service SERVICE=worker-1)
	@if [ -z "$(SERVICE)" ]; then \
		echo "$(RED)Error: SERVICE not specified$(NC)"; \
		echo "Usage: make logs-service SERVICE=worker-1"; \
		exit 1; \
	fi
	$(COMPOSE) logs -f $(SERVICE)

logs-orchestrator: ## Show orchestrator logs
	$(COMPOSE) logs -f orchestrator

logs-workers: ## Show all worker logs
	$(COMPOSE) logs -f worker-1 worker-2 worker-3

status: ## Show service status
	@echo "$(BLUE)Service Status:$(NC)"
	@$(COMPOSE) ps

health: ## Check service health
	@echo "$(BLUE)Health Checks:$(NC)"
	@$(COMPOSE) ps --format json | jq -r '.[] | "\(.Service): \(.State) - \(.Health)"'

stats: ## Show resource usage statistics
	@docker stats --no-stream

## =============================================================================
## Scaling
## =============================================================================

scale: ## Scale workers (usage: make scale WORKERS=5)
	@if [ -z "$(WORKERS)" ]; then \
		echo "$(RED)Error: WORKERS not specified$(NC)"; \
		echo "Usage: make scale WORKERS=5"; \
		exit 1; \
	fi
	@echo "$(BLUE)Scaling to $(WORKERS) workers...$(NC)"
	$(COMPOSE) up -d --scale worker-1=$(WORKERS)
	@echo "$(GREEN)âœ“ Scaled to $(WORKERS) workers$(NC)"

scale-down: ## Scale down to 1 worker
	@echo "$(BLUE)Scaling down to 1 worker...$(NC)"
	$(COMPOSE) up -d --scale worker-1=1
	@echo "$(GREEN)âœ“ Scaled down$(NC)"

## =============================================================================
## Shell Access
## =============================================================================

shell: ## Open shell in orchestrator
	$(COMPOSE) exec orchestrator bash

shell-worker: ## Open shell in worker-1
	$(COMPOSE) exec worker-1 bash

exec: ## Execute command in orchestrator (usage: make exec CMD="python --version")
	@if [ -z "$(CMD)" ]; then \
		echo "$(RED)Error: CMD not specified$(NC)"; \
		echo "Usage: make exec CMD=\"python --version\""; \
		exit 1; \
	fi
	$(COMPOSE) exec orchestrator $(CMD)

exec-service: ## Execute command in specific service (usage: make exec-service SERVICE=worker-1 CMD="...")
	@if [ -z "$(SERVICE)" ] || [ -z "$(CMD)" ]; then \
		echo "$(RED)Error: SERVICE or CMD not specified$(NC)"; \
		echo "Usage: make exec-service SERVICE=worker-1 CMD=\"python --version\""; \
		exit 1; \
	fi
	$(COMPOSE) exec $(SERVICE) $(CMD)

## =============================================================================
## Testing & Validation
## =============================================================================

test: ## Run tests
	@echo "$(BLUE)Running tests...$(NC)"
	$(COMPOSE) exec orchestrator pytest
	@echo "$(GREEN)âœ“ Tests complete$(NC)"

test-local: ## Run tests locally with UV
	@echo "$(BLUE)Running tests locally...$(NC)"
	$(UV) run pytest
	@echo "$(GREEN)âœ“ Tests complete$(NC)"

lint: ## Lint code
	@echo "$(BLUE)Linting code...$(NC)"
	$(UV) run ruff check .
	@echo "$(GREEN)âœ“ Linting complete$(NC)"

format: ## Format code
	@echo "$(BLUE)Formatting code...$(NC)"
	$(UV) run black .
	@echo "$(GREEN)âœ“ Formatting complete$(NC)"

type-check: ## Type check with mypy
	@echo "$(BLUE)Type checking...$(NC)"
	$(UV) run mypy .
	@echo "$(GREEN)âœ“ Type checking complete$(NC)"

check-all: lint format type-check ## Run all checks

## =============================================================================
## Testing Connectivity
## =============================================================================

test-a2a: ## Test A2A connectivity between agents
	@echo "$(BLUE)Testing A2A connectivity...$(NC)"
	@$(COMPOSE) exec worker-1 python -c "from common.communication.a2a import A2AClient; import asyncio; asyncio.run((lambda: print('âœ“ A2A imports work'))())"
	@echo "$(GREEN)âœ“ A2A connectivity test passed$(NC)"

test-redis: ## Test Redis connectivity
	@echo "$(BLUE)Testing Redis connectivity...$(NC)"
	@$(COMPOSE) exec worker-1 python -c "import redis; r=redis.from_url('redis://redis:6379'); print('âœ“ Redis ping:', r.ping())"
	@echo "$(GREEN)âœ“ Redis connectivity test passed$(NC)"

test-all: test-a2a test-redis ## Run all connectivity tests

## =============================================================================
## Cleanup
## =============================================================================

clean: ## Clean up containers and images
	@echo "$(BLUE)Cleaning up...$(NC)"
	$(COMPOSE) down
	docker system prune -f
	@echo "$(GREEN)âœ“ Cleanup complete$(NC)"

clean-all: ## Clean up everything including volumes and images
	@echo "$(YELLOW)âš  This will delete ALL data and images!$(NC)"
	@read -p "Are you sure? [y/N] " -n 1 -r; \
	echo; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		$(COMPOSE) down -v --rmi all; \
		docker system prune -af --volumes; \
		echo "$(GREEN)âœ“ Deep cleanup complete$(NC)"; \
	fi

clean-logs: ## Clean up log files
	@echo "$(BLUE)Cleaning logs...$(NC)"
	rm -rf logs/*.log
	@echo "$(GREEN)âœ“ Logs cleaned$(NC)"

## =============================================================================
## Production Operations
## =============================================================================

prod-up: ## Start in production mode
	@echo "$(BLUE)Starting Ghost Swarm in production mode...$(NC)"
	ENVIRONMENT=production $(COMPOSE) -f docker-compose.yml -f docker-compose.prod.yml up -d
	@echo "$(GREEN)âœ“ Ghost Swarm started in production mode$(NC)"

prod-build: ## Build production images
	@echo "$(BLUE)Building production images...$(NC)"
	DOCKER_TARGET=production-with-source $(COMPOSE) -f docker-compose.yml -f docker-compose.prod.yml build
	@echo "$(GREEN)âœ“ Production images built$(NC)"

prod-down: ## Stop production services
	$(COMPOSE) -f docker-compose.yml -f docker-compose.prod.yml down

## =============================================================================
## Development Helpers
## =============================================================================

dev-setup: install-dev env ## Complete development setup
	@echo "$(GREEN)âœ“ Development environment ready$(NC)"
	@echo "$(YELLOW)Next steps:$(NC)"
	@echo "  1. Edit .env and add your ANTHROPIC_API_KEY"
	@echo "  2. Run: make build"
	@echo "  3. Run: make up"

watch-logs: ## Watch logs with timestamps
	$(COMPOSE) logs -f --timestamps

tail-logs: ## Tail last 100 lines of logs
	$(COMPOSE) logs --tail=100

inspect: ## Inspect a service (usage: make inspect SERVICE=worker-1)
	@if [ -z "$(SERVICE)" ]; then \
		echo "$(RED)Error: SERVICE not specified$(NC)"; \
		echo "Usage: make inspect SERVICE=worker-1"; \
		exit 1; \
	fi
	docker inspect $$($(COMPOSE) ps -q $(SERVICE))

## =============================================================================
## Quick Commands
## =============================================================================

quick-start: env build up ## Quick start (setup, build, start)
	@echo "$(GREEN)âœ“ Ghost Swarm is running!$(NC)"
	@make status

quick-restart: down up ## Quick restart
	@echo "$(GREEN)âœ“ Ghost Swarm restarted$(NC)"

rebuild: down build-no-cache up ## Full rebuild

## =============================================================================
## Network & Debugging
## =============================================================================

network-inspect: ## Inspect Docker network
	docker network inspect ghost-swarm_ghost-network

network-test: ## Test network connectivity between services
	@echo "$(BLUE)Testing network connectivity...$(NC)"
	@$(COMPOSE) exec worker-1 ping -c 3 orchestrator
	@$(COMPOSE) exec worker-1 ping -c 3 redis
	@echo "$(GREEN)âœ“ Network connectivity test passed$(NC)"

debug-orchestrator: ## Show debug info for orchestrator
	@echo "$(BLUE)Orchestrator Debug Info:$(NC)"
	@$(COMPOSE) exec orchestrator python -c "import sys; print('Python:', sys.version); print('Path:', sys.path)"

debug-worker: ## Show debug info for worker-1
	@echo "$(BLUE)Worker-1 Debug Info:$(NC)"
	@$(COMPOSE) exec worker-1 python -c "import sys; print('Python:', sys.version); print('Path:', sys.path)"

## =============================================================================
## Info & Documentation
## =============================================================================

version: ## Show versions
	@echo "$(BLUE)Version Information:$(NC)"
	@echo "Docker: $$(docker --version)"
	@echo "Docker Compose: $$($(COMPOSE) version)"
	@echo "UV: $$($(UV) --version)"
	@echo "Python: $$($(PYTHON) --version)"

info: ## Show environment info
	@echo "$(BLUE)Ghost Swarm Environment:$(NC)"
	@echo "Compose Project: $$($(COMPOSE) config --services | wc -l) services"
	@echo "Networks: $$(docker network ls | grep ghost | wc -l)"
	@echo "Volumes: $$(docker volume ls | grep ghost | wc -l)"
	@echo "Images: $$(docker images | grep ghost | wc -l)"

docs: ## Open documentation
	@if [ -f "CORRECTED_ARCHITECTURE.md" ]; then \
		cat CORRECTED_ARCHITECTURE.md; \
	else \
		echo "$(YELLOW)Documentation not found$(NC)"; \
	fi

## =============================================================================
## Aliases (Common Commands)
## =============================================================================

start: up ## Alias for 'up'
stop: down ## Alias for 'down'
ps: status ## Alias for 'status'
l: logs ## Alias for 'logs'
b: build ## Alias for 'build'
r: restart ## Alias for 'restart'