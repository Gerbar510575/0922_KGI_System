COMPOSE=infra/docker-compose.yml

.PHONY: up down build logs ps reindex restart clean

up:
\t@docker compose -f $(COMPOSE) up -d qdrant redis rag gateway advisor report market ui

down:
\t@docker compose -f $(COMPOSE) down

build:
\t@docker compose -f $(COMPOSE) build

logs:
\t@docker compose -f $(COMPOSE) logs -f

ps:
\t@docker compose -f $(COMPOSE) ps

reindex:
\t@docker compose -f $(COMPOSE) run --rm -e GENAI_API_KEY rag python -m indexer_gemini

restart:
\t@docker compose -f $(COMPOSE) restart rag gateway advisor report market ui

clean:
\t@docker compose -f $(COMPOSE) down -v
